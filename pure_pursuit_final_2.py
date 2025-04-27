#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math
import numpy as np

from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

#
# ------------------------------ Utility Classes/Functions ------------------------------
#

class FixedQueue:
    """
    A simple fixed-length queue to compute a rolling average (for filtering range_rate).
    """
    def __init__(self, size):
        self.size = size
        self.data = []

    def push(self, value):
        if len(self.data) >= self.size:
            self.data.pop(0)
        self.data.append(value)

    def get_mean(self):
        if len(self.data) == 0:
            return 0.0
        return sum(self.data) / len(self.data)

def compute_yaw_list(x_list, y_list):
    """
    Compute approximate yaw for each waypoint based on direction from i -> i+1.
    """
    n_points = len(x_list)
    yaw_list = np.zeros(n_points, dtype=float)
    for i in range(n_points - 1):
        dx = x_list[i+1] - x_list[i]
        dy = y_list[i+1] - y_list[i]
        yaw_list[i] = math.atan2(dy, dx)
    yaw_list[-1] = yaw_list[-2] if n_points > 1 else 0.0
    return yaw_list

def compute_speed_list(n_points, default_speed=2.0):
    """
    Return a constant speed array of length n_points.
    """
    speed_list = np.full(n_points, default_speed, dtype=float)
    return speed_list

def get_lookahead(
    curr_pos,
    curr_yaw,
    xyv_list,
    yaw_list,
    lookahead_dist,
    lookahead_points,
    lookbehind_points,
    slope
):
    """
    Find a target waypoint for Pure Pursuit from xyv_list: Nx3 => [x, y, v].
    """
    waypoints_xy = xyv_list[:, :2]
    v_list = xyv_list[:, 2]

    # 1) Closest waypoint
    dists = np.sqrt(np.sum((waypoints_xy - curr_pos) ** 2, axis=1))
    closest_idx = np.argmin(dists)

    # 2) Nominal target index
    target_idx = closest_idx + lookahead_points
    target_idx = max(0, target_idx - lookbehind_points)
    if target_idx >= len(waypoints_xy):
        target_idx = len(waypoints_xy) - 1

    # 3) Attenuate lookahead distance if big yaw difference
    path_yaw = yaw_list[target_idx]
    yaw_diff = abs(path_yaw - curr_yaw)
    yaw_diff = min(yaw_diff, math.pi)
    L_mod = lookahead_dist * (1.0 - slope * (yaw_diff / math.pi))
    L_mod = max(0.4, L_mod)

    # 4) Advance until distance >= L_mod
    while True:
        if target_idx >= len(waypoints_xy) - 1:
            break
        dist_to_wp = np.linalg.norm(waypoints_xy[target_idx] - curr_pos)
        if dist_to_wp >= L_mod:
            break
        target_idx += 1

    target_idx = min(target_idx, len(waypoints_xy) - 1)
    target_point = waypoints_xy[target_idx]
    target_speed = v_list[target_idx]

    # 5) Heading error
    dx = target_point[0] - curr_pos[0]
    dy = target_point[1] - curr_pos[1]
    heading_to_point = math.atan2(dy, dx)
    error = heading_to_point - curr_yaw
    # Wrap to [-pi, pi]
    if error > math.pi:
        error -= 2.0 * math.pi
    elif error < -math.pi:
        error += 2.0 * math.pi

    return error, target_speed, target_point, target_idx

def generate_side_path(global_x, global_y, shift_meters=0.5):
    """
    Naive function that "shifts" the global path sideways by shift_meters.
    In a real system, you'd do a more robust local re-planning.
    We assume a 2D path, compute local normals or a simple perpendicular shift.

    For each segment, we find the normal vector (perpendicular) and shift by shift_meters.

    :param global_x: (N,) array of x coords
    :param global_y: (N,) array of y coords
    :param shift_meters: shift to the left or right (+ => left if your coordinate system has +Y left)
    :return: (side_x, side_y) shifted path
    """
    n = len(global_x)
    side_x = np.zeros(n)
    side_y = np.zeros(n)

    for i in range(n - 1):
        dx = global_x[i+1] - global_x[i]
        dy = global_y[i+1] - global_y[i]
        # direction angle
        ang = math.atan2(dy, dx)
        # normal angle
        normal_ang = ang + math.pi / 2.0
        # shift coords
        side_x[i] = global_x[i] + shift_meters * math.cos(normal_ang)
        side_y[i] = global_y[i] + shift_meters * math.sin(normal_ang)

    # last point: just copy the shift from the second-last
    side_x[-1] = global_x[-1] + shift_meters * math.cos(normal_ang)
    side_y[-1] = global_y[-1] + shift_meters * math.sin(normal_ang)

    return side_x, side_y

#
# ------------------------------ Main Node Class ------------------------------
#

class PurePursuitOvertakeAvoid(Node):
    """
    A single-node demonstration combining:
      - Basic obstacle detection from LIDAR
      - Simple "overtake" logic (shift path sideways)
      - Pure Pursuit path following
    """

    def __init__(self):
        super().__init__('pure_pursuit_overtake_avoid_node')

        # Declare params
        self.declare_parameter('lookahead_distance', 1.2)
        self.declare_parameter('lookahead_points', 5)
        self.declare_parameter('lookbehind_points', 1)
        self.declare_parameter('L_slope_atten', 0.5)
        self.declare_parameter('max_steer', 0.48)
        self.declare_parameter('kp', 0.6)
        self.declare_parameter('ki', 0.0)
        self.declare_parameter('kd', 0.005)
        self.declare_parameter('steer_alpha', 1.0)

        # Distances used for naive "obstacle in front?" logic
        self.declare_parameter('distance_threshold', 3.0)      # if obstacle < 3m ahead, consider avoidance
        self.declare_parameter('overtake_shift', 1.0)          # how much to shift path for "overtake"
        self.declare_parameter('overtake_speed', 3.0)          # speed on overtake path
        self.declare_parameter('stop_speed_if_blocked', 0.0)   # fallback speed if blocked
        self.declare_parameter('queue_size', 10)

        # Internal states
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_steer = 0.0
        self.speed = 0.0

        # Rolling queue for range_rate smoothing
        queue_size = self.get_parameter('queue_size').get_parameter_value().integer_value
        self.range_rate_queue = FixedQueue(queue_size)
        self.filtered_range_rate = 0.0

        # For LIDAR-based "front obstacle" detection
        self.last_cared_ranges = 0.0
        self.last_time = self.get_clock().now().nanoseconds / 1e9

        # Load or define a "global path" in x,y form
        # For demonstration, let's define a big circle or load from CSV
        # Here, we'll just define a small example path (like a line or curve).
        # Replace with your real waypoints if desired.
        t = np.linspace(0, 2*math.pi, 200)
        radius = 5.0
        global_x = radius * np.cos(t)
        global_y = radius * np.sin(t)
        # Or if you have a CSV: 
        # global_waypoints = np.loadtxt('/path/to/2d_waypoints.csv', delimiter=',', skiprows=0)
        # global_x = global_waypoints[:,0]
        # global_y = global_waypoints[:,1]

        # Compute speed / yaw for the global path
        self.global_yaw = compute_yaw_list(global_x, global_y)
        self.global_speed = compute_speed_list(len(global_x), default_speed=2.0)
        # Combine into Nx3
        self.global_xyv = np.column_stack((global_x, global_y, self.global_speed))

        # We'll also precompute the "overtake path" (shifted to one side).
        # In a real system, you'd generate it on-the-fly. Here, we do it once.
        shift_meters = self.get_parameter('overtake_shift').get_parameter_value().double_value
        overtake_speed = self.get_parameter('overtake_speed').get_parameter_value().double_value

        overtake_x, overtake_y = generate_side_path(global_x, global_y, shift_meters)
        self.overtake_yaw = compute_yaw_list(overtake_x, overtake_y)
        overtake_speed_array = compute_speed_list(len(overtake_x), default_speed=overtake_speed)
        self.overtake_xyv = np.column_stack((overtake_x, overtake_y, overtake_speed_array))

        # A flag to indicate which path we are currently following
        # 0 => global path, 1 => overtake path
        self.current_path_id = 0

        self.get_logger().info("Loaded global path and overtake path. Ready.")

        # Subscriptions
        self.odom_sub_pose = self.create_subscription(PoseStamped, '/pf/viz/inferred_pose', self.pose_callback, 10)
        self.odom_sub_speed = self.create_subscription(Odometry, '/odom', self.speed_callback, 10)
        self.scan_sub_ = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # Publishers
        self.drive_pub_ = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.path_pub_ = self.create_publisher(Marker, '/path_marker', 10)
        self.target_pub_ = self.create_publisher(Marker, '/target_marker', 10)

    # ------------------------------------------------------------------------------
    # Callback: LIDAR
    # ------------------------------------------------------------------------------
    def scan_callback(self, scan_msg: LaserScan):
        n_ranges = len(scan_msg.ranges)
        ranges_np = np.array(scan_msg.ranges)

        # Focus on center portion, e.g. -15 to +15 deg
        lower_idx = int(n_ranges * 8.0 / 18.0)
        upper_idx = int(n_ranges * 10.0 / 18.0)
        cared_ranges = ranges_np[lower_idx:upper_idx]

        # Replace out-of-range data with large number
        cared_ranges = np.where(
            ((cared_ranges >= scan_msg.range_min) & (cared_ranges <= scan_msg.range_max)),
            cared_ranges, 30.0
        )

        mean_cared = np.mean(cared_ranges)
        now_time = self.get_clock().now().nanoseconds / 1e9
        dt = now_time - self.last_time
        if dt < 1e-9:
            return

        range_diff = mean_cared - self.last_cared_ranges
        self.last_cared_ranges = mean_cared
        self.last_time = now_time

        self.range_rate_queue.push(range_diff / dt)
        self.filtered_range_rate = self.range_rate_queue.get_mean()

    # ------------------------------------------------------------------------------
    # Callback: Odom Speed
    # ------------------------------------------------------------------------------
    def speed_callback(self, odom_msg: Odometry):
        self.speed = odom_msg.twist.twist.linear.x

    # ------------------------------------------------------------------------------
    # Callback: Pose
    # ------------------------------------------------------------------------------
    def pose_callback(self, pose_msg: PoseStamped):
        # Current pose
        px = pose_msg.pose.position.x
        py = pose_msg.pose.position.y
        curr_quat = pose_msg.pose.orientation

        curr_yaw = math.atan2(
            2 * (curr_quat.w * curr_quat.z + curr_quat.x * curr_quat.y),
            1 - 2 * (curr_quat.y**2 + curr_quat.z**2)
        )
        curr_pos = np.array([px, py])

        # Get relevant parameters
        L = self.get_parameter('lookahead_distance').get_parameter_value().double_value
        lookahead_points = self.get_parameter('lookahead_points').get_parameter_value().integer_value
        lookbehind_points = self.get_parameter('lookbehind_points').get_parameter_value().integer_value
        slope = self.get_parameter('L_slope_atten').get_parameter_value().double_value

        distance_threshold = self.get_parameter('distance_threshold').get_parameter_value().double_value
        stop_speed = self.get_parameter('stop_speed_if_blocked').get_parameter_value().double_value

        # 1) Check if there's an obstacle close in front
        #    We'll do a naive check: if self.last_cared_ranges < distance_threshold, we are "blocked."
        #    If blocked, attempt an "overtake path." If already on the overtake path, maybe do nothing.
        #    If the overtake path is also blocked, we might slow down to 0 or a lower speed.
        blocked_ahead = (self.last_cared_ranges < distance_threshold)

        # Basic state machine:
        # - If currently on the global path and blocked, switch to overtake path.
        # - If currently on the overtake path and blocked, slow down or stop.
        # - If on overtake path, check if safe to return to global path (optional).
        #   For simplicity, we won't implement returning logic here; you could do it
        #   if you detect no obstacles for a while.

        if self.current_path_id == 0:
            # On global path
            if blocked_ahead:
                self.get_logger().info("Blocked ahead => Attempting overtake path.")
                self.current_path_id = 1  # switch to overtake
        else:
            # On overtake path
            if blocked_ahead:
                self.get_logger().info("Overtake path also blocked => Slowing down.")
                # We'll artificially reduce the speed in the 'get_lookahead' step
                # (the path itself has a speed, but we'll clamp it to something smaller).
                pass
            # Optionally: Check if no longer blocked, return to global path
            # This is left out for simplicity.

        # 2) Based on current_path_id, pick the relevant path
        if self.current_path_id == 0:
            # Use global path
            xyv_list = self.global_xyv
            yaw_list = self.global_yaw
        else:
            # Use overtake path
            xyv_list = self.overtake_xyv
            yaw_list = self.overtake_yaw

        # 3) Run Pure Pursuit
        error, target_speed, target_point, _ = get_lookahead(
            curr_pos,
            curr_yaw,
            xyv_list,
            yaw_list,
            L,
            lookahead_points,
            lookbehind_points,
            slope
        )

        # If path is blocked, clamp the speed
        if blocked_ahead:
            target_speed = min(target_speed, stop_speed)

        steer = self.get_steer(error)

        # 4) Publish drive
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = float(target_speed)
        drive_msg.drive.steering_angle = float(steer)
        self.drive_pub_.publish(drive_msg)

        # 5) Visualization
        self.visualize_path(xyv_list)
        self.visualize_target(target_point)

    # ------------------------------------------------------------------------------
    # Pure Pursuit Steering
    # ------------------------------------------------------------------------------
    def get_steer(self, error):
        kp = self.get_parameter('kp').get_parameter_value().double_value
        ki = self.get_parameter('ki').get_parameter_value().double_value
        kd = self.get_parameter('kd').get_parameter_value().double_value
        alpha = self.get_parameter('steer_alpha').get_parameter_value().double_value
        max_steer = self.get_parameter('max_steer').get_parameter_value().double_value

        d_error = error - self.prev_error
        self.prev_error = error
        self.integral += error

        raw_steer = kp * error + ki * self.integral + kd * d_error
        raw_steer = np.clip(raw_steer, -max_steer, max_steer)

        # Optional steering smoothing
        new_steer = alpha * raw_steer + (1.0 - alpha) * self.prev_steer
        self.prev_steer = new_steer
        return new_steer

    # ------------------------------------------------------------------------------
    # Visualization: publish current path as a line strip
    # ------------------------------------------------------------------------------
    def visualize_path(self, xyv_list):
        path_marker = Marker()
        path_marker.header.frame_id = 'map'
        path_marker.ns = 'pp_path'
        path_marker.id = 0
        path_marker.type = Marker.LINE_STRIP
        path_marker.action = Marker.ADD
        path_marker.scale.x = 0.05
        path_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)

        for row in xyv_list:
            pt = Point()
            pt.x = float(row[0])
            pt.y = float(row[1])
            pt.z = 0.0
            path_marker.points.append(pt)

        self.path_pub_.publish(path_marker)

    # ------------------------------------------------------------------------------
    # Visualization: publish target waypoint as a sphere
    # ------------------------------------------------------------------------------
    def visualize_target(self, target_xy):
        target_marker = Marker()
        target_marker.header.frame_id = 'map'
        target_marker.ns = 'pp_target'
        target_marker.id = 1
        target_marker.type = Marker.SPHERE
        target_marker.action = Marker.ADD
        target_marker.scale.x = 0.3
        target_marker.scale.y = 0.3
        target_marker.scale.z = 0.3
        target_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
        target_marker.pose.position.x = float(target_xy[0])
        target_marker.pose.position.y = float(target_xy[1])
        target_marker.pose.position.z = 0.0

        self.target_pub_.publish(target_marker)


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitOvertakeAvoid()
    node.get_logger().info("Starting Pure Pursuit with Overtake and Avoid demo.")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
