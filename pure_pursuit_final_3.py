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


# ------------------------------------------------------------------------
# Utility Classes/Functions
# ------------------------------------------------------------------------

class FixedQueue:
    """
    A simple fixed-length queue to compute a rolling average.
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
    Compute approximate yaw for each waypoint based on direction
    from waypoint i->i+1.
    """
    n_points = len(x_list)
    yaw_list = np.zeros(n_points, dtype=float)
    for i in range(n_points - 1):
        dx = x_list[i+1] - x_list[i]
        dy = y_list[i+1] - y_list[i]
        yaw_list[i] = math.atan2(dy, dx)
    # For the last waypoint, repeat the previous yaw
    if n_points > 1:
        yaw_list[-1] = yaw_list[-2]
    return yaw_list


def compute_speed_list(x_list, y_list, base_speed):
    """
    Return a constant speed array of length len(x_list).
    Uses a 'base_speed' parameter for easy tuning.
    """
    n_points = len(x_list)
    return np.full(n_points, base_speed, dtype=float)


def get_lookahead(
    curr_pos,
    curr_yaw,
    xyv_list,
    yaw_list,
    v_list,
    lookahead_dist,
    lookahead_points,
    lookbehind_points,
    slope
):
    """
    1) Find the index of the closest waypoint.
    2) Adjust lookahead distance if path yaw differs significantly from current heading.
    3) Return heading error, target speed, target point, target index.
    """
    waypoints_xy = xyv_list[:, :2]

    # 1) Closest waypoint by distance
    dists = np.sqrt(np.sum((waypoints_xy - curr_pos) ** 2, axis=1))
    closest_idx = np.argmin(dists)

    # 2) Decide target index
    target_idx = closest_idx + lookahead_points
    target_idx = max(0, target_idx - lookbehind_points)
    if target_idx >= waypoints_xy.shape[0]:
        target_idx = waypoints_xy.shape[0] - 1

    # 3) Modulate lookahead dist if big yaw difference
    path_yaw = yaw_list[target_idx]
    yaw_diff = abs(path_yaw - curr_yaw)
    yaw_diff = min(yaw_diff, math.pi)
    L_mod = lookahead_dist * (1.0 - slope * (yaw_diff / math.pi))
    L_mod = max(0.5, L_mod)

    # 4) Advance while distance < L_mod
    while True:
        if target_idx >= waypoints_xy.shape[0] - 1:
            break
        if np.linalg.norm(waypoints_xy[target_idx] - curr_pos) >= L_mod:
            break
        target_idx += 1

    target_idx = min(target_idx, waypoints_xy.shape[0] - 1)
    target_point = waypoints_xy[target_idx]
    target_speed = v_list[target_idx]

    # 5) Heading error
    dx = target_point[0] - curr_pos[0]
    dy = target_point[1] - curr_pos[1]
    heading_to_point = math.atan2(dy, dx)
    error = heading_to_point - curr_yaw
    # Wrap [-pi, pi]
    if error > math.pi:
        error -= 2 * math.pi
    elif error < -math.pi:
        error += 2 * math.pi

    return error, target_speed, target_point, target_idx


# ------------------------------------------------------------------------
# Main Node
# ------------------------------------------------------------------------

WIDTH = 0.2032
WHEEL_LENGTH = 0.0381
MAX_STEER = 0.48

csv_loc = '/home/jeff/sim_ws/src/pure_pursuit/scripts/smoothed_final_waypoints.csv'

class PurePursuit(Node):
    """
    Basic Pure Pursuit with bounding box slowdown, plus tunable speeds.
    """

    def __init__(self):
        super().__init__('pure_pursuit_node')

        # Real or sim toggle
        self.flag = False
        self.get_logger().info(f"Real-world test? {self.flag}")

        #
        # Declare ROS parameters, including speed parameters
        #
        self.declare_parameter('lookahead_distance', 1.2)
        self.declare_parameter('lookahead_points', 8)
        self.declare_parameter('lookbehind_points', 2)
        self.declare_parameter('L_slope_atten', 0.7)

        self.declare_parameter('kp', 0.6)
        self.declare_parameter('ki', 0.0)
        self.declare_parameter('kd', 0.005)
        self.declare_parameter('steer_alpha', 1.0)
        self.declare_parameter('max_control', MAX_STEER)

        # Distances
        self.declare_parameter('distance_threshold', 0.5)
        self.declare_parameter('speed_tolerance', 10.0)
        self.declare_parameter('queue_size', 10)

        # Speed parameters
        # base_speed => for normal CSV waypoints
        # blocked_speed => cap if obstacle detected
        # box_speed => cap if inside bounding box
        self.declare_parameter('base_speed', 4.5)
        self.declare_parameter('blocked_speed', 1.5)
        self.declare_parameter('box_speed', 2.0)

        # PID states
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_steer = 0.0

        # Load CSV
        waypoints = np.loadtxt(csv_loc, delimiter=',', skiprows=0)
        self.x_list = waypoints[:, 0]
        self.y_list = waypoints[:, 1]

        # Retrieve 'base_speed' param to build speed list
        base_speed = self.get_parameter('base_speed').get_parameter_value().double_value
        self.v_list = compute_speed_list(self.x_list, self.y_list, base_speed)

        self.yaw_list = compute_yaw_list(self.x_list, self.y_list)
        self.xyv_list = np.column_stack((self.x_list, self.y_list, self.v_list))

        self.v_max = np.max(self.v_list)
        self.v_min = np.min(self.v_list)

        self.get_logger().info(f"Loaded {len(self.x_list)} CSV waypoints with base_speed={base_speed:.2f} m/s.")

        #
        # Subscriptions
        #
        if self.flag:
            # Real
            odom_topic_pose = '/pf/viz/inferred_pose'
            odom_topic_speed = '/odom'
            self.odom_sub_ = self.create_subscription(PoseStamped, odom_topic_pose, self.pose_callback, 10)
            self.odom_sub_speed = self.create_subscription(Odometry, odom_topic_speed, self.speed_callback, 10)
        else:
            # Sim
            odom_topic_pose = '/ego_racecar/odom'
            self.odom_sub_ = self.create_subscription(Odometry, odom_topic_pose, self.pose_callback, 10)
            self.odom_sub_speed = None  # optional
        self.scan_sub_ = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        #
        # Publishers
        #
        drive_topic = '/drive'
        waypoint_topic = '/waypoint'
        waypoint_path_topic = '/waypoint_path'
        self.drive_pub_ = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.waypoint_pub_ = self.create_publisher(Marker, waypoint_topic, 10)
        self.waypoint_path_pub_ = self.create_publisher(Marker, waypoint_path_topic, 10)

        #
        # Rolling queue for smoothing range rate
        #
        queue_size = self.get_parameter('queue_size').get_parameter_value().integer_value
        self.RRqueue = FixedQueue(queue_size)
        self.last_cared_ranges = 0.0
        self.last_time = self.get_clock().now().nanoseconds / 1e9
        self.range_rate = 0.0
        self.filtered_range_rate = 0.0
        self.mean_cared_ranges = 999.0
        self.speed = 0.0

        #
        # Visualization placeholders
        #
        self.target_point = np.array([0.0, 0.0])
        self.curr_target_idx = 0

        #
        # Bounding box corners => if inside, cap speed to 'box_speed'
        #
        self.box_x_min = -21.0432
        self.box_x_max = -15.8303
        self.box_y_min = -4.71817
        self.box_y_max =  8.26059

        self.get_logger().info("Pure Pursuit node started with tunable speeds.")

    # --------------------------------------------------------------------------
    # LIDAR callback
    # --------------------------------------------------------------------------
    def scan_callback(self, scan_msg: LaserScan):
        n_ranges = len(scan_msg.ranges)
        ranges_np = np.array(scan_msg.ranges)

        # Focus on a limited front sector: -15..+15 deg
        lower_idx = int(n_ranges * 8 / 18)
        upper_idx = int(n_ranges * 10 / 18)
        cared = ranges_np[lower_idx:upper_idx]

        cared = np.where(
            ((cared >= scan_msg.range_min) & (cared <= scan_msg.range_max)),
            cared, 30.0
        )

        self.mean_cared_ranges = np.mean(cared)
        now_time = self.get_clock().now().nanoseconds / 1e9
        dt = now_time - self.last_time
        if dt < 1e-9:
            return

        range_diff = self.mean_cared_ranges - self.last_cared_ranges
        self.last_cared_ranges = self.mean_cared_ranges
        self.last_time = now_time

        self.RRqueue.push(range_diff / dt)
        self.filtered_range_rate = self.RRqueue.get_mean()

    # --------------------------------------------------------------------------
    # Speed callback
    # --------------------------------------------------------------------------
    def speed_callback(self, odom_msg: Odometry):
        self.speed = odom_msg.twist.twist.linear.x

    # --------------------------------------------------------------------------
    # Pose callback
    # --------------------------------------------------------------------------
    def pose_callback(self, msg):
        if self.flag:
            # Real => PoseStamped
            curr_x = msg.pose.position.x
            curr_y = msg.pose.position.y
            curr_quat = msg.pose.orientation
        else:
            # Sim => Odometry
            curr_x = msg.pose.pose.position.x
            curr_y = msg.pose.pose.position.y
            curr_quat = msg.pose.pose.orientation
            # Also read speed from same message
            self.speed = msg.twist.twist.linear.x

        # Convert quaternion to yaw
        curr_yaw = math.atan2(
            2*(curr_quat.w * curr_quat.z + curr_quat.x * curr_quat.y),
            1 - 2*(curr_quat.y**2 + curr_quat.z**2)
        )
        curr_pos = np.array([curr_x, curr_y])

        # Retrieve relevant parameters
        L = self.get_parameter('lookahead_distance').get_parameter_value().double_value
        lookahead_points = self.get_parameter('lookahead_points').get_parameter_value().integer_value
        lookbehind_points = self.get_parameter('lookbehind_points').get_parameter_value().integer_value
        slope = self.get_parameter('L_slope_atten').get_parameter_value().double_value
        distance_threshold = self.get_parameter('distance_threshold').get_parameter_value().double_value
        speed_tolerance = self.get_parameter('speed_tolerance').get_parameter_value().double_value

        # Additional speed params
        blocked_speed = self.get_parameter('blocked_speed').get_parameter_value().double_value
        box_speed = self.get_parameter('box_speed').get_parameter_value().double_value

        # 1) get lookahead
        error, target_v, target_point, idx = get_lookahead(
            curr_pos, curr_yaw,
            self.xyv_list, self.yaw_list, self.v_list,
            L, lookahead_points, lookbehind_points, slope
        )
        self.target_point = target_point
        self.curr_target_idx = idx

        # 2) If something is blocking => reduce speed
        if (self.filtered_range_rate < 0 and
            self.mean_cared_ranges < distance_threshold and
            abs(self.filtered_range_rate) < speed_tolerance):
            target_v = min(target_v, blocked_speed)

        # 3) If inside bounding box => clamp speed to box_speed
        if self.is_in_bounding_box(curr_x, curr_y):
            target_v = min(target_v, box_speed)

        # Steering
        steer = self.get_steer(error)

        # Publish
        ack_msg = AckermannDriveStamped()
        ack_msg.drive.speed = float(target_v)
        ack_msg.drive.steering_angle = float(steer)
        print(f"Speed: {float(target_v)}, Steering Angle: {float(steer)}",float(target_v), float(steer))
        self.drive_pub_.publish(ack_msg)

        # Visualization
        self.visualize_waypoints()

    # --------------------------------------------------------------------------
    # Helper: bounding box check
    # --------------------------------------------------------------------------
    def is_in_bounding_box(self, x, y):
        return (self.box_x_min <= x <= self.box_x_max) and (self.box_y_min <= y <= self.box_y_max)

    # --------------------------------------------------------------------------
    # Steering
    # --------------------------------------------------------------------------
    def get_steer(self, error):
        kp = self.get_parameter('kp').get_parameter_value().double_value
        ki = self.get_parameter('ki').get_parameter_value().double_value
        kd = self.get_parameter('kd').get_parameter_value().double_value
        alpha = self.get_parameter('steer_alpha').get_parameter_value().double_value
        max_control = self.get_parameter('max_control').get_parameter_value().double_value

        d_error = error - self.prev_error
        self.prev_error = error
        self.integral += error

        raw_steer = kp * error + ki * self.integral + kd * d_error
        raw_steer = np.clip(raw_steer, -max_control, max_control)

        new_steer = alpha * raw_steer + (1.0 - alpha) * self.prev_steer
        self.prev_steer = new_steer
        return new_steer

    # --------------------------------------------------------------------------
    # Visualization
    # --------------------------------------------------------------------------
    def visualize_waypoints(self):
        path_marker = Marker()
        path_marker.header.frame_id = 'map'
        path_marker.id = 0
        path_marker.ns = 'pursuit_waypoint_path'
        path_marker.type = Marker.LINE_STRIP
        path_marker.action = Marker.ADD

        path_marker.points = []
        path_marker.colors = []

        for i in range(self.x_list.shape[0]):
            pt = Point()
            pt.x = float(self.x_list[i])
            pt.y = float(self.y_list[i])
            pt.z = 0.0
            path_marker.points.append(pt)

            speed = self.v_list[i]
            if (self.v_max - self.v_min) < 1e-9:
                normalized_s = 0.5
            else:
                normalized_s = (speed - self.v_min)/(self.v_max - self.v_min)
            c = ColorRGBA()
            c.a = 1.0
            c.r = 1.0 - normalized_s
            c.g = normalized_s
            c.b = 0.0
            path_marker.colors.append(c)

        path_marker.scale.x = 0.05
        path_marker.scale.y = 0.05
        path_marker.scale.z = 0.05
        path_marker.pose.orientation.w = 1.0
        self.waypoint_path_pub_.publish(path_marker)

        # Target sphere
        target_marker = Marker()
        target_marker.header.frame_id = 'map'
        target_marker.id = 1
        target_marker.ns = 'pursuit_waypoint_target'
        target_marker.type = Marker.SPHERE
        target_marker.action = Marker.ADD
        target_marker.scale.x = 0.3
        target_marker.scale.y = 0.3
        target_marker.scale.z = 0.3
        target_marker.pose.orientation.w = 1.0
        target_marker.pose.position.x = float(self.target_point[0])
        target_marker.pose.position.y = float(self.target_point[1])
        target_marker.pose.position.z = 0.0

        target_speed = self.v_list[self.curr_target_idx]
        if (self.v_max - self.v_min) < 1e-9:
            normalized_s = 0.5
        else:
            normalized_s = (target_speed - self.v_min)/(self.v_max - self.v_min)

        target_marker.color.a = 1.0
        target_marker.color.r = 1.0 - normalized_s
        target_marker.color.g = normalized_s
        target_marker.color.b = 0.0
        self.waypoint_pub_.publish(target_marker)


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuit()
    node.get_logger().info("PurePursuit with tunable speeds, bounding box slowdown, etc. Spinning...")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
