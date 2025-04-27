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


# ------------------------------------------------------------------------------
# Below are minimal stand-in implementations for what you'd typically
# keep in a separate util.py, adapted for your new 2-column waypoints.
# ------------------------------------------------------------------------------

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
    Compute approximate yaw for each waypoint based on the
    direction from waypoint i to waypoint i+1.
    For the last waypoint, we just repeat the second-to-last yaw.
    """
    n_points = len(x_list)
    yaw_list = np.zeros(n_points, dtype=float)
    for i in range(n_points - 1):
        dx = x_list[i+1] - x_list[i]
        dy = y_list[i+1] - y_list[i]
        yaw_list[i] = math.atan2(dy, dx)
    # For the last waypoint, repeat the previous yaw
    yaw_list[-1] = yaw_list[-2] if n_points > 1 else 0.0
    return yaw_list


def compute_speed_list(x_list, y_list):
    """
    Return a default speed array. For example, a constant speed = 2.0 m/s.
    Or you could compute speed based on curvature, distance, etc.
    """
    n_points = len(x_list)
    # Example: assign constant speed = 2.0 for all waypoints
    speed_list = np.full(n_points, 2.0, dtype=float)
    return speed_list


def get_lookahead(
    curr_pos,          # np.array([x, y]) for the current vehicle position
    curr_yaw,          # current vehicle yaw (rad)
    xyv_list,          # Nx3 array of [x, y, v]
    yaw_list,          # Nx1 array of yaws
    v_list,            # Nx1 array of speeds
    lookahead_dist,    # nominal distance L
    lookahead_points,  # how many points ahead from the closest
    lookbehind_points, # how many points behind the closest to consider
    slope
):
    """
    1) Find the index of the closest waypoint.
    2) Adjust lookahead distance if the path yaw is significantly different
       than the current heading (slope parameter).
    3) Return:
       - error (heading difference),
       - target_v (waypoint speed),
       - target_point ([x, y]),
       - target_idx (which waypoint is chosen).
    Modify as needed for your application.
    """
    waypoints_xy = xyv_list[:, :2]  # first two columns are x,y

    # 1) Find closest waypoint by Euclidean distance
    dists = np.sqrt(np.sum((waypoints_xy - curr_pos) ** 2, axis=1))
    closest_idx = np.argmin(dists)

    # 2) Decide the "target" index
    target_idx = closest_idx + lookahead_points
    # add some “look behind” offset if desired
    target_idx = max(0, target_idx - lookbehind_points)
    if target_idx >= waypoints_xy.shape[0]:
        target_idx = waypoints_xy.shape[0] - 1

    # 3) Optionally modulate L by how different the path yaw is from current yaw
    path_yaw = yaw_list[target_idx]
    yaw_diff = abs(path_yaw - curr_yaw)
    yaw_diff = min(yaw_diff, math.pi)  # keep it in [0..pi]
    # Example “attenuation”: if yaw_diff is large, reduce lookahead
    L_mod = lookahead_dist * (1.0 - slope * (yaw_diff / math.pi))
    L_mod = max(0.5, L_mod)  # clamp to some minimum

    # 4) Now actually find a point that is ~L_mod away from the car
    while True:
        if target_idx >= waypoints_xy.shape[0] - 1:
            break
        if np.linalg.norm(waypoints_xy[target_idx] - curr_pos) >= L_mod:
            break
        target_idx += 1

    target_idx = min(target_idx, waypoints_xy.shape[0] - 1)
    target_point = waypoints_xy[target_idx]
    target_v = v_list[target_idx]  # speed from the array

    # 5) Compute heading error: difference between current yaw and direction to the target
    dx = target_point[0] - curr_pos[0]
    dy = target_point[1] - curr_pos[1]
    heading_to_point = math.atan2(dy, dx)
    error = heading_to_point - curr_yaw
    # wrap error to (-pi, pi)
    if error > math.pi:
        error -= 2 * math.pi
    elif error < -math.pi:
        error += 2 * math.pi

    return error, target_v, target_point, target_idx


# ------------------------------------------------------------------------------
# End of "util.py" stand-ins.
# ------------------------------------------------------------------------------


# Constants (these match your snippet or your xacro parameters).
WIDTH = 0.2032         # (m)
WHEEL_LENGTH = 0.0381  # (m)
MAX_STEER = 0.48       # (rad)

# Change this to your CSV with 2 columns [x, y].
csv_loc = '/home/nvidia/f1tenth_ws/2_column_waypoints.csv'


class PurePursuit(Node):
    """
    Implement Pure Pursuit on the car, adapted for 2-column waypoints (x, y only).
    """

    def __init__(self):
        super().__init__('pure_pursuit_node')

        # Declare ROS parameters
        self.declare_parameter('lookahead_distance', 1.2)
        self.declare_parameter('lookahead_points', 8)
        self.declare_parameter('lookbehind_points', 2)
        self.declare_parameter('L_slope_atten', 0.7)
        self.declare_parameter('kp', 0.6)
        self.declare_parameter('ki', 0.0)
        self.declare_parameter('kd', 0.005)
        self.declare_parameter("max_control", MAX_STEER)
        self.declare_parameter("steer_alpha", 1.0)
        self.declare_parameter("distance_threshold", 10.0)
        self.declare_parameter("speed_tolerance", 10.0)
        self.declare_parameter("queue_size", 10)

        # PID states
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_steer = 0.0

        # Flag: set True if using real-world style topics, else sim
        self.flag = True
        self.get_logger().info(f"Real-world test? {self.flag}")

        # 1) Load CSV waypoints that only have x,y
        #    If your CSV has no header, remove skiprows or set skiprows=0.
        waypoints = np.loadtxt(csv_loc, delimiter=',', skiprows=0)
        # Splitting them out
        self.x_list = waypoints[:, 0]
        self.y_list = waypoints[:, 1]

        # 2) We must have a default or computed speed list
        self.v_list = compute_speed_list(self.x_list, self.y_list)

        # 3) We must have a yaw list
        self.yaw_list = compute_yaw_list(self.x_list, self.y_list)

        # For get_lookahead, we combine x,y,v into Nx3
        self.xyv_list = np.column_stack((self.x_list, self.y_list, self.v_list))

        # For color-visualization, define these for min/max speeds
        self.v_max = np.max(self.v_list)
        self.v_min = np.min(self.v_list)
        self.get_logger().info("2-column waypoints loaded and processed (x, y, speed, yaw).")

        # Subscriptions
        if self.flag:
            odom_topic_pose = '/pf/viz/inferred_pose'
            self.odom_sub_ = self.create_subscription(
                PoseStamped, odom_topic_pose, self.pose_callback, 10)
            self.get_logger().info(f"Subscribed to: {odom_topic_pose}")

            odom_topic_speed = '/odom'
            self.odom_sub_speed = self.create_subscription(
                Odometry, odom_topic_speed, self.speed_callback, 10)
            self.get_logger().info(f"Subscribed to: {odom_topic_speed}")
        else:
            odom_topic_pose = '/ego_racecar/odom'
            self.odom_sub_ = self.create_subscription(
                Odometry, odom_topic_pose, self.pose_callback, 10)
            self.get_logger().info(f"Subscribed to: {odom_topic_pose}")

        self.scan_sub_ = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.get_logger().info("Subscribed to: /scan")

        # Publishers
        drive_topic = '/drive'
        waypoint_topic = '/waypoint'
        waypoint_path_topic = '/waypoint_path'
        self.drive_pub_ = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.get_logger().info(f"Publishing drive commands on: {drive_topic}")

        self.waypoint_pub_ = self.create_publisher(Marker, waypoint_topic, 10)
        self.get_logger().info(f"Publishing waypoint marker on: {waypoint_topic}")

        self.waypoint_path_pub_ = self.create_publisher(Marker, waypoint_path_topic, 10)
        self.get_logger().info(f"Publishing path marker on: {waypoint_path_topic}")

        # Variables for range rate calculations
        self.last_cared_ranges = 0.0
        self.last_time = self.get_clock().now().nanoseconds / 1e9
        self.opp_speed = 0.0
        self.range_rate = 0.0
        self.speed = 0.0

        # Rolling queue for smoothing
        queue_size = self.get_parameter('queue_size').get_parameter_value().integer_value
        self.RRqueue = FixedQueue(queue_size)
        self.filtered_range_rate = 0.0

        # Visualization placeholders
        self.target_point = np.array([0.0, 0.0])
        self.curr_target_idx = 0

    # ------------------------------------------------------------------------------
    # ROS Callbacks
    # ------------------------------------------------------------------------------

    def scan_callback(self, scan_msg: LaserScan):
        n_ranges = len(scan_msg.ranges)
        ranges_np = np.array(scan_msg.ranges)
        # Focus on a limited front sector, e.g. -15 to +15 deg
        lower_idx = int(n_ranges * 8.0 / 18.0)
        upper_idx = int(n_ranges * 10.0 / 18.0)
        cared_ranges = ranges_np[lower_idx:upper_idx]

        # Replace out-of-range data with some large number
        cared_ranges = np.where(
            ((cared_ranges >= scan_msg.range_min) & (cared_ranges <= scan_msg.range_max)),
            cared_ranges, 30.0
        )

        self.mean_cared_ranges = np.mean(cared_ranges)

        # Range rate approximation
        now_time = self.get_clock().now().nanoseconds / 1e9
        dt = now_time - self.last_time
        range_diff = self.mean_cared_ranges - self.last_cared_ranges
        if dt > 1e-9:
            self.range_rate = range_diff / dt
        else:
            self.range_rate = 0.0

        self.last_cared_ranges = self.mean_cared_ranges
        self.last_time = now_time

        # Push into rolling queue for smoothing
        self.RRqueue.push(self.range_rate)
        self.filtered_range_rate = self.RRqueue.get_mean()

    def speed_callback(self, speed_msg: Odometry):
        self.speed = speed_msg.twist.twist.linear.x

    def pose_callback(self, pose_msg):
        """
        Extract current pose from either a PoseStamped (flag==True) or
        an Odometry (flag==False) message.
        """
        if self.flag:
            curr_x = pose_msg.pose.position.x
            curr_y = pose_msg.pose.position.y
            curr_quat = pose_msg.pose.orientation
        else:
            curr_x = pose_msg.pose.pose.position.x
            curr_y = pose_msg.pose.pose.position.y
            curr_quat = pose_msg.pose.pose.orientation

        curr_pos = np.array([curr_x, curr_y])

        # Convert quaternion to yaw
        curr_yaw = math.atan2(
            2 * (curr_quat.w * curr_quat.z + curr_quat.x * curr_quat.y),
            1 - 2 * (curr_quat.y**2 + curr_quat.z**2)
        )

        # Retrieve parameters
        L = self.get_parameter('lookahead_distance').get_parameter_value().double_value
        lookahead_points = self.get_parameter('lookahead_points').get_parameter_value().integer_value
        lookbehind_points = self.get_parameter('lookbehind_points').get_parameter_value().integer_value
        slope = self.get_parameter('L_slope_atten').get_parameter_value().double_value

        distance_threshold = self.get_parameter('distance_threshold').get_parameter_value().double_value
        speed_tolerance = self.get_parameter('speed_tolerance').get_parameter_value().double_value

        # Get lookahead info
        error, target_v, target_point, curr_target_idx = get_lookahead(
            curr_pos, curr_yaw,
            self.xyv_list,
            self.yaw_list,
            self.v_list,
            L,
            lookahead_points,
            lookbehind_points,
            slope
        )

        self.target_point = target_point
        self.curr_target_idx = curr_target_idx

        # If something is blocking (filtered_range_rate < 0, meaning distance is shrinking)
        # and the mean front distance is less than threshold, reduce speed.
        if (self.filtered_range_rate < 0 and
            self.mean_cared_ranges < distance_threshold and
            abs(self.filtered_range_rate) < speed_tolerance):
            # Force zero or a smaller speed
            target_v = 0.0

        # Build Ackermann command
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = target_v
        drive_msg.drive.steering_angle = self.get_steer(error)

        # Publish the command
        self.drive_pub_.publish(drive_msg)

        # Visualize waypoints
        self.visualize_waypoints()

    # ------------------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------------------

    def get_steer(self, error):
        """
        Simple PID control around the heading error.
        """
        kp = self.get_parameter('kp').get_parameter_value().double_value
        ki = self.get_parameter('ki').get_parameter_value().double_value
        kd = self.get_parameter('kd').get_parameter_value().double_value
        max_control = self.get_parameter('max_control').get_parameter_value().double_value
        alpha = self.get_parameter('steer_alpha').get_parameter_value().double_value

        d_error = error - self.prev_error
        self.prev_error = error
        self.integral += error

        raw_steer = kp * error + ki * self.integral + kd * d_error
        raw_steer = np.clip(raw_steer, -max_control, max_control)

        # Optional low-pass filter on steering
        new_steer = alpha * raw_steer + (1.0 - alpha) * self.prev_steer
        self.prev_steer = new_steer

        return new_steer

    def visualize_waypoints(self):
        """
        Publish markers showing:
         - The entire path (with color-coded speeds).
         - The currently selected target point in a larger marker.
        """
        # 1) Publish the path as a line strip
        path_marker = Marker()
        path_marker.header.frame_id = 'map'
        path_marker.id = 0
        path_marker.ns = 'pursuit_waypoint_path'
        path_marker.type = Marker.LINE_STRIP
        path_marker.action = Marker.ADD

        path_marker.points = []
        path_marker.colors = []

        length = self.x_list.shape[0]
        for i in range(length):
            pt = Point()
            pt.x = float(self.x_list[i])
            pt.y = float(self.y_list[i])
            pt.z = 0.0
            path_marker.points.append(pt)

            # Color by speed
            speed = self.v_list[i]
            # Protect against zero range for (v_max - v_min)
            if (self.v_max - self.v_min) < 1e-9:
                normalized_s = 0.5
            else:
                normalized_s = (speed - self.v_min) / (self.v_max - self.v_min)
            color = ColorRGBA()
            color.a = 1.0
            color.r = 1.0 - normalized_s
            color.g = normalized_s
            color.b = 0.0
            path_marker.colors.append(color)

        # Slight thickness for line
        path_marker.scale.x = 0.05
        path_marker.scale.y = 0.05
        path_marker.scale.z = 0.05
        path_marker.pose.orientation.w = 1.0

        self.waypoint_path_pub_.publish(path_marker)

        # 2) Publish a single sphere for the current target
        target_marker = Marker()
        target_marker.header.frame_id = 'map'
        target_marker.id = 1
        target_marker.ns = 'pursuit_waypoint_target'
        target_marker.type = Marker.SPHERE
        target_marker.action = Marker.ADD

        target_marker.pose.position.x = float(self.target_point[0])
        target_marker.pose.position.y = float(self.target_point[1])
        target_marker.pose.position.z = 0.0

        target_speed = self.v_list[self.curr_target_idx]
        if (self.v_max - self.v_min) < 1e-9:
            normalized_s = 0.5
        else:
            normalized_s = (target_speed - self.v_min) / (self.v_max - self.v_min)

        target_marker.color.a = 1.0
        target_marker.color.r = 1.0 - normalized_s
        target_marker.color.g = normalized_s
        target_marker.color.b = 0.0

        target_marker.scale.x = 0.3
        target_marker.scale.y = 0.3
        target_marker.scale.z = 0.3
        target_marker.pose.orientation.w = 1.0

        self.waypoint_pub_.publish(target_marker)


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuit()
    node.get_logger().info("PurePursuit node initialized. Spinning...")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
