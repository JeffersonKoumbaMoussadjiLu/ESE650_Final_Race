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


# -------------------------------------------------------------------------
# Utility Classes/Functions
# -------------------------------------------------------------------------

class FixedQueue:
    """
    Simple rolling-average queue for smoothing range_rate or speed data.
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
    Compute approximate yaw for each waypoint based on direction i->i+1.
    """
    n_points = len(x_list)
    yaw_list = np.zeros(n_points, dtype=float)
    for i in range(n_points - 1):
        dx = x_list[i+1] - x_list[i]
        dy = y_list[i+1] - y_list[i]
        yaw_list[i] = math.atan2(dy, dx)
    if n_points > 1:
        yaw_list[-1] = yaw_list[-2]
    return yaw_list

def compute_speed_list(n_points, speed=2.0):
    """
    Return a constant speed array of length n_points.
    """
    return np.full(n_points, speed, dtype=float)

def get_lookahead(
    curr_pos,
    curr_yaw,
    xyv_list,        # Nx3 => columns: [x, y, speed]
    yaw_list,        # Nx1 array of path yaws
    lookahead_dist,
    lookahead_points,
    lookbehind_points,
    slope
):
    """
    Finds the target waypoint from xyv_list for pure pursuit.
    Returns (heading_error, target_speed, target_point, target_idx).
    """
    waypoints_xy = xyv_list[:, :2]
    v_list = xyv_list[:, 2]

    # 1) Closest waypoint
    dists = np.sqrt(np.sum((waypoints_xy - curr_pos) ** 2, axis=1))
    closest_idx = np.argmin(dists)

    # 2) Baseline target index
    target_idx = closest_idx + lookahead_points
    target_idx = max(0, target_idx - lookbehind_points)
    if target_idx >= len(waypoints_xy):
        target_idx = len(waypoints_xy) - 1

    # 3) Adjust the lookahead distance if yaw difference is large
    path_yaw = yaw_list[target_idx]
    yaw_diff = abs(path_yaw - curr_yaw)
    yaw_diff = min(yaw_diff, math.pi)
    L_mod = lookahead_dist * (1.0 - slope * (yaw_diff / math.pi))
    L_mod = max(0.5, L_mod)

    # 4) Move forward until distance >= L_mod or end
    while True:
        if target_idx >= len(waypoints_xy) - 1:
            break
        if np.linalg.norm(waypoints_xy[target_idx] - curr_pos) >= L_mod:
            break
        target_idx += 1

    target_idx = min(target_idx, len(waypoints_xy) - 1)
    target_point = waypoints_xy[target_idx]
    target_speed = v_list[target_idx]

    # 5) Heading error
    dx = target_point[0] - curr_pos[0]
    dy = target_point[1] - curr_pos[1]
    heading_to_point = math.atan2(dy, dx)
    heading_error = heading_to_point - curr_yaw

    # Wrap error to [-pi, pi]
    if heading_error > math.pi:
        heading_error -= 2.0 * math.pi
    elif heading_error < -math.pi:
        heading_error += 2.0 * math.pi

    return heading_error, target_speed, target_point, target_idx

def generate_side_path(x_list, y_list, shift=0.5):
    """
    Naive function to shift the path laterally by 'shift' meters.
    For each segment, we find a normal vector and offset by 'shift'.
    This simulates a "side lane" or "overtaking" path.
    """
    n = len(x_list)
    side_x = np.zeros(n, dtype=float)
    side_y = np.zeros(n, dtype=float)

    for i in range(n - 1):
        dx = x_list[i+1] - x_list[i]
        dy = y_list[i+1] - y_list[i]
        heading = math.atan2(dy, dx)
        normal_angle = heading + math.pi / 2.0  # shift to the left
        side_x[i] = x_list[i] + shift * math.cos(normal_angle)
        side_y[i] = y_list[i] + shift * math.sin(normal_angle)

    # Last point: same shift as second last
    side_x[-1] = x_list[-1] + shift * math.cos(normal_angle)
    side_y[-1] = y_list[-1] + shift * math.sin(normal_angle)

    return side_x, side_y

# -------------------------------------------------------------------------
# Node Class
# -------------------------------------------------------------------------

class PurePursuitAvoid(Node):
    """
    Extended Pure Pursuit with:
     - Obstacle detection (front LIDAR)
     - Slowing down when blocked
     - Optional side path usage to bypass obstacles
    """

    def __init__(self):
        super().__init__('pure_pursuit_avoid_node')

        # Declare parameters
        self.declare_parameter('lookahead_distance', 1.2)
        self.declare_parameter('lookahead_points', 8)
        self.declare_parameter('lookbehind_points', 2)
        self.declare_parameter('L_slope_atten', 0.7)

        # PID gains for heading error
        self.declare_parameter('kp', 0.6)
        self.declare_parameter('ki', 0.0)
        self.declare_parameter('kd', 0.005)
        self.declare_parameter('steer_alpha', 1.0)  # Steering smoothing factor
        self.declare_parameter('max_control', 0.48) # max steering angle

        # Obstacle avoidance
        self.declare_parameter('distance_threshold', 1.0)
        self.declare_parameter('speed_tolerance', 1.0)
        self.declare_parameter('queue_size', 10)

        # Speed settings
        self.base_speed = 4.0           # default full speed
        self.slow_speed = 1.0          # speed when blocked & no side path
        self.side_path_speed = 3.0      # speed on side path
        self.side_shift = 0.7          # shift in meters for side path

        # Internal states
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_steer = 0.0
        self.last_time = self.get_clock().now().nanoseconds / 1e9

        self.flag = False  # True if real, else sim
        self.get_logger().info(f"Real-world test? {self.flag}")

        # ------------------ Load Main Path from CSV ------------------
        # Adjust CSV path as needed
        csv_loc = '/home/jeff/sim_ws/src/pure_pursuit/scripts/smoothed_final_waypoints.csv'
        main_waypoints = np.loadtxt(csv_loc, delimiter=',', skiprows=0)
        main_x = main_waypoints[:, 0]
        main_y = main_waypoints[:, 1]
        n_points = len(main_x)

        # Build speed & yaw for main path
        main_speed = compute_speed_list(n_points, self.base_speed)
        main_yaw = compute_yaw_list(main_x, main_y)
        self.main_xyv_list = np.column_stack((main_x, main_y, main_speed))
        self.main_yaw_list = main_yaw

        # ------------------ Build Side Path (overtake) ------------------
        side_x, side_y = generate_side_path(main_x, main_y, shift=self.side_shift)
        side_speed = compute_speed_list(n_points, self.side_path_speed)
        side_yaw = compute_yaw_list(side_x, side_y)
        self.side_xyv_list = np.column_stack((side_x, side_y, side_speed))
        self.side_yaw_list = side_yaw

        # Which path are we using now? 0 = main path, 1 = side path
        self.current_path_id = 0

        # For color-visualization
        self.v_max = self.base_speed
        self.v_min = min(self.slow_speed, self.base_speed)

        # Subscriptions
        if self.flag:
            odom_topic_pose = '/pf/viz/inferred_pose'
            self.odom_sub_ = self.create_subscription(
                PoseStamped, odom_topic_pose, self.pose_callback, 10)

            odom_topic_speed = '/odom'
            self.odom_sub_speed = self.create_subscription(
                Odometry, odom_topic_speed, self.speed_callback, 10)
        else:
            odom_topic_pose = '/ego_racecar/odom'
            self.odom_sub_ = self.create_subscription(
                Odometry, odom_topic_pose, self.pose_callback, 10)

        self.scan_sub_ = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # Publishers
        self.drive_pub_ = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.waypoint_pub_ = self.create_publisher(Marker, '/waypoint', 10)
        self.waypoint_path_pub_ = self.create_publisher(Marker, '/waypoint_path', 10)

        # LIDAR-based obstacle detection
        queue_size = self.get_parameter('queue_size').get_parameter_value().integer_value
        self.range_rate_queue = FixedQueue(queue_size)
        self.mean_cared_ranges = 99.0
        self.last_cared_ranges = 99.0
        self.range_rate = 0.0
        self.filtered_range_rate = 0.0

        self.speed = 0.0
        self.target_point = np.array([0.0, 0.0])
        self.curr_target_idx = 0

        self.get_logger().info("Extended Pure Pursuit node is ready.")

    # -------------------------------------------------------------------------
    # ROS Callbacks
    # -------------------------------------------------------------------------
    def scan_callback(self, scan_msg: LaserScan):
        """
        We check the front sector (e.g. -15° to +15°) for obstacles,
        compute the average distance and track how it's changing.
        """
        n_ranges = len(scan_msg.ranges)
        ranges_np = np.array(scan_msg.ranges)
        lower_idx = int(n_ranges * 8.0 / 18.0)
        upper_idx = int(n_ranges * 10.0 / 18.0)
        front_ranges = ranges_np[lower_idx:upper_idx]

        # Replace out-of-range with large number
        front_ranges = np.where(
            ((front_ranges >= scan_msg.range_min) & (front_ranges <= scan_msg.range_max)),
            front_ranges, 30.0
        )

        self.mean_cared_ranges = np.mean(front_ranges)
        now_time = self.get_clock().now().nanoseconds / 1e9
        dt = now_time - self.last_time
        if dt < 1e-9:
            return

        range_diff = self.mean_cared_ranges - self.last_cared_ranges
        self.last_cared_ranges = self.mean_cared_ranges
        self.last_time = now_time

        # Rolling average for rate
        self.range_rate_queue.push(range_diff / dt)
        self.filtered_range_rate = self.range_rate_queue.get_mean()

    def speed_callback(self, odom_msg: Odometry):
        self.speed = odom_msg.twist.twist.linear.x

    def pose_callback(self, pose_msg):
        """
        If flagged for real, read from PoseStamped, else from Odometry.
        Then do pure pursuit on either main path or side path (if we decided).
        """
        if self.flag:
            curr_x = pose_msg.pose.position.x
            curr_y = pose_msg.pose.position.y
            curr_quat = pose_msg.pose.orientation
        else:
            curr_x = pose_msg.pose.pose.position.x
            curr_y = pose_msg.pose.pose.position.y
            curr_quat = pose_msg.pose.pose.orientation

        # Convert to yaw
        curr_yaw = math.atan2(
            2 * (curr_quat.w * curr_quat.z + curr_quat.x * curr_quat.y),
            1 - 2 * (curr_quat.y**2 + curr_quat.z**2)
        )
        curr_pos = np.array([curr_x, curr_y])

        # Retrieve relevant parameters
        L = self.get_parameter('lookahead_distance').get_parameter_value().double_value
        lookahead_points = self.get_parameter('lookahead_points').get_parameter_value().integer_value
        lookbehind_points = self.get_parameter('lookbehind_points').get_parameter_value().integer_value
        slope = self.get_parameter('L_slope_atten').get_parameter_value().double_value

        distance_threshold = self.get_parameter('distance_threshold').get_parameter_value().double_value
        speed_tolerance = self.get_parameter('speed_tolerance').get_parameter_value().double_value

        # ---------------------------------------------------------------------
        # Decide which path to use: main or side?
        # If we are "blocked" in front, attempt side path
        # If side path also blocked, we do slow speed
        # Otherwise, use main path at normal speed
        # ---------------------------------------------------------------------
        blocked_main = self.is_path_blocked(distance_threshold, speed_tolerance)
        if blocked_main:
            # Check if side path is blocked
            blocked_side = self.is_path_blocked(distance_threshold, speed_tolerance, side=True)
            if not blocked_side:
                # Switch to side path
                self.current_path_id = 1
            else:
                # Both blocked => we will slow down on whichever path we're on
                pass
        else:
            # Main path is clear => use main
            self.current_path_id = 0

        # Select XYV & yaw based on current path
        if self.current_path_id == 0:
            xyv_list = self.main_xyv_list
            yaw_list = self.main_yaw_list
        else:
            xyv_list = self.side_xyv_list
            yaw_list = self.side_yaw_list

        # Run pure pursuit
        error, base_target_speed, target_point, idx = get_lookahead(
            curr_pos, curr_yaw,
            xyv_list, yaw_list,
            L, lookahead_points, lookbehind_points, slope
        )
        self.target_point = target_point
        self.curr_target_idx = idx

        # Now handle speed logic
        # If blocked, slow down
        # If on side path & not blocked, can maintain side_path_speed
        # If on main path & not blocked, maintain base_speed
        if self.current_path_id == 0:
            # main path
            if blocked_main:
                # slow down
                target_speed = self.slow_speed
            else:
                # normal
                target_speed = self.base_speed
        else:
            # side path
            # check if side path is also blocked
            if self.is_path_blocked(distance_threshold, speed_tolerance, side=True):
                target_speed = self.slow_speed
            else:
                target_speed = self.side_path_speed

        # Build Ackermann command
        ack_msg = AckermannDriveStamped()
        ack_msg.drive.speed = float(target_speed)
        ack_msg.drive.steering_angle = float(self.get_steer(error))
        self.drive_pub_.publish(ack_msg)

        # Visualization
        self.visualize_waypoints()

    # -------------------------------------------------------------------------
    # Helper: Check if path is blocked
    # -------------------------------------------------------------------------
    def is_path_blocked(self, dist_thresh, speed_tol, side=False):
        """
        Return True if the path is considered blocked in front.
        We say "blocked" if:
          1) The LIDAR front distance is < dist_thresh
          2) The rate is negative => obstacle is approaching
          3) The absolute rate is < speed_tol => not a fleeting measurement

        Optionally, we do a "side check" by shifting our front sector slightly
        if you want to see if the side path is also blocked. This is minimal here.
        """
        # Simple check: (range_rate < 0) => distance is decreasing
        # and (mean_cared_ranges < dist_thresh)
        # and (abs(range_rate) < speed_tol)
        if self.filtered_range_rate < 0.0 and self.mean_cared_ranges < dist_thresh and abs(self.filtered_range_rate) < speed_tol:
            return True
        return False

    # -------------------------------------------------------------------------
    # Steering PID
    # -------------------------------------------------------------------------
    def get_steer(self, error):
        kp = self.get_parameter('kp').get_parameter_value().double_value
        ki = self.get_parameter('ki').get_parameter_value().double_value
        kd = self.get_parameter('kd').get_parameter_value().double_value
        alpha = self.get_parameter('steer_alpha').get_parameter_value().double_value
        max_steer = self.get_parameter('max_control').get_parameter_value().double_value

        d_error = error - self.prev_error
        self.prev_error = error
        self.integral += error

        raw_steer = kp * error + ki * self.integral + kd * d_error
        raw_steer = np.clip(raw_steer, -max_steer, max_steer)

        new_steer = alpha * raw_steer + (1.0 - alpha) * self.prev_steer
        self.prev_steer = new_steer

        return new_steer

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------
    def visualize_waypoints(self):
        """
        Publish a line strip for the "active" path,
        and a sphere for the current target waypoint.
        """
        marker_path = Marker()
        marker_path.header.frame_id = 'map'
        marker_path.id = 0
        marker_path.type = Marker.LINE_STRIP
        marker_path.action = Marker.ADD
        marker_path.scale.x = 0.05
        marker_path.color.a = 1.0
        marker_path.color.g = 1.0  # green line
        marker_path.pose.orientation.w = 1.0

        if self.current_path_id == 0:
            # Main path
            xyv_list = self.main_xyv_list
        else:
            # Side path
            xyv_list = self.side_xyv_list

        for row in xyv_list:
            pt = Point()
            pt.x = float(row[0])
            pt.y = float(row[1])
            marker_path.points.append(pt)

        self.waypoint_path_pub_.publish(marker_path)

        # Target sphere
        marker_target = Marker()
        marker_target.header.frame_id = 'map'
        marker_target.id = 1
        marker_target.type = Marker.SPHERE
        marker_target.action = Marker.ADD
        marker_target.scale.x = 0.3
        marker_target.scale.y = 0.3
        marker_target.scale.z = 0.3
        marker_target.color.a = 1.0
        marker_target.color.r = 1.0  # red sphere
        marker_target.pose.orientation.w = 1.0
        marker_target.pose.position.x = float(self.target_point[0])
        marker_target.pose.position.y = float(self.target_point[1])
        self.waypoint_pub_.publish(marker_target)

# -------------------------------------------------------------------------
# main
# -------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitAvoid()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
