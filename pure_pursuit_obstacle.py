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
    """A simple fixed-length queue to compute a rolling average."""
    def __init__(self, size):
        self.size = size
        self.data = []

    def push(self, value):
        if len(self.data) >= self.size:
            self.data.pop(0)
        self.data.append(value)

    def get_mean(self):
        return sum(self.data)/len(self.data) if self.data else 0.0

def compute_yaw_list(x_list, y_list):
    """Compute approximate yaw for each waypoint based on direction between consecutive points."""
    n_points = len(x_list)
    yaw_list = np.zeros(n_points, dtype=float)
    for i in range(n_points - 1):
        dx = x_list[i+1] - x_list[i]
        dy = y_list[i+1] - y_list[i]
        yaw_list[i] = math.atan2(dy, dx)
    if n_points > 1:
        yaw_list[-1] = yaw_list[-2]
    return yaw_list

def compute_speed_list(x_list, y_list, base_speed):
    """Generate constant speed array matching waypoint count."""
    return np.full(len(x_list), base_speed, dtype=float)

def get_lookahead(curr_pos, curr_yaw, xyv_list, yaw_list, v_list,
                 lookahead_dist, lookahead_points, lookbehind_points, slope):
    """Calculate target point and heading error using adaptive pure pursuit logic."""
    waypoints_xy = xyv_list[:, :2]
    dists = np.sqrt(np.sum((waypoints_xy - curr_pos)**2, axis=1))
    closest_idx = np.argmin(dists)
    
    target_idx = max(0, closest_idx + lookahead_points - lookbehind_points)
    target_idx = min(target_idx, len(waypoints_xy)-1)
    
    path_yaw = yaw_list[target_idx]
    yaw_diff = min(abs(path_yaw - curr_yaw), math.pi)
    L_mod = max(0.5, lookahead_dist * (1.0 - slope*(yaw_diff/math.pi)))
    
    while target_idx < len(waypoints_xy)-1 and \
          np.linalg.norm(waypoints_xy[target_idx] - curr_pos) < L_mod:
        target_idx += 1
    
    target_point = waypoints_xy[target_idx]
    dx, dy = target_point - curr_pos
    heading_to_point = math.atan2(dy, dx)
    error = (heading_to_point - curr_yaw + math.pi) % (2*math.pi) - math.pi
    return error, v_list[target_idx], target_point, target_idx

# ------------------------------------------------------------------------
# Main Node with Robust Initialization
# ------------------------------------------------------------------------

class PurePursuit(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')
        
        # Initialize all required attributes first
        self.target_point = np.array([0.0, 0.0])  # Critical initialization
        self.curr_target_idx = 0
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_steer = 0.0
        self.min_front_distance = float('inf')
        self.speed = 0.0

        # Declare parameters
        self.declare_parameters(namespace='',
            parameters=[
                ('lookahead_distance', 1.2),
                ('lookahead_points', 8),
                ('lookbehind_points', 2),
                ('L_slope_atten', 0.7),
                ('kp', 0.6), ('ki', 0.0), ('kd', 0.005),
                ('steer_alpha', 1.0), ('max_control', 0.48),
                ('base_speed', 4.5), ('box_speed', 2.0),
                ('real_world', False)
            ])

        # Load waypoints
        waypoints = np.loadtxt('./mpc_levine_1000.csv', delimiter=',')
        self.x_list, self.y_list = waypoints[:, 0], waypoints[:, 1]
        base_speed = self.get_parameter('base_speed').value
        self.v_list = compute_speed_list(self.x_list, self.y_list, base_speed)
        self.yaw_list = compute_yaw_list(self.x_list, self.y_list)
        self.xyv_list = np.column_stack((self.x_list, self.y_list, self.v_list))

        # Initialize ROS components
        self.initialize_ros_components()

        # Bounding box configuration
        self.box_x_min, self.box_x_max = -21.0432, -15.8303
        self.box_y_min, self.box_y_max = -4.71817, 8.26059

        self.get_logger().info("Pure Pursuit node initialized")

    def initialize_ros_components(self):
        """Set up ROS subscribers and publishers."""
        if self.get_parameter('real_world').value:
            self.odom_sub_ = self.create_subscription(
                PoseStamped, '/pf/viz/inferred_pose', self.pose_callback, 10)
        else:
            self.odom_sub_ = self.create_subscription(
                Odometry, '/ego_racecar/odom', self.pose_callback, 10)
        
        self.scan_sub_ = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        
        self.drive_pub_ = self.create_publisher(
            AckermannDriveStamped, '/drive', 10)
        self.waypoint_pub_ = self.create_publisher(Marker, '/waypoint', 10)
        self.waypoint_path_pub_ = self.create_publisher(
            Marker, '/waypoint_path', 10)

    def scan_callback(self, scan_msg: LaserScan):
        """Process LIDAR data with obstacle detection."""
        num_readings = 150  # ~±30 degree FOV
        center_idx = len(scan_msg.ranges) // 2
        lower_idx = center_idx - (num_readings // 2)
        upper_idx = center_idx + (num_readings // 2)
        
        ranges = np.array(scan_msg.ranges[lower_idx:upper_idx])
        valid_mask = (ranges >= scan_msg.range_min) & (ranges <= scan_msg.range_max)
        valid_ranges = ranges[valid_mask]
        
        self.min_front_distance = np.min(valid_ranges) if valid_ranges.size > 0 else float('inf')

    def pose_callback(self, msg):
        """Main control loop with safety checks."""
        # Pose processing
        if self.get_parameter('real_world').value:
            pos = msg.pose.position
            curr_quat = msg.pose.orientation
        else:
            pos = msg.pose.pose.position
            curr_quat = msg.pose.pose.orientation
            self.speed = msg.twist.twist.linear.x

        curr_pos = np.array([pos.x, pos.y])
        curr_yaw = math.atan2(
            2*(curr_quat.w * curr_quat.z + curr_quat.x * curr_quat.y),
            1 - 2*(curr_quat.y**2 + curr_quat.z**2)
        )

        # Calculate target point
        params = self.get_parameters([
            'lookahead_distance', 'lookahead_points',
            'lookbehind_points', 'L_slope_atten'
        ])
        error, target_v, target_point, _ = get_lookahead(
            curr_pos, curr_yaw, self.xyv_list, self.yaw_list, self.v_list,
            params[0].value, params[1].value, params[2].value, params[3].value
        )
        self.target_point = target_point  # Ensure this is always set

        # Obstacle response
        if self.min_front_distance < 0.5:
            target_v = 0.0
        elif self.min_front_distance < 1.0:
            target_v = min(target_v, 1.0)
        elif self.min_front_distance < 2.0:
            target_v = min(target_v, 2.0)

        # Bounding box speed limit
        if self.is_in_bounding_box(curr_pos[0], curr_pos[1]):
            target_v = min(target_v, self.get_parameter('box_speed').value)

        # Calculate steering
        steer = self.calculate_steering(error)
        
        # Publish commands
        cmd = AckermannDriveStamped()
        cmd.drive.speed = float(target_v)
        cmd.drive.steering_angle = float(steer)
        self.drive_pub_.publish(cmd)
        
        # Update visualization
        self.visualize_waypoints()

    def calculate_steering(self, error):
        """PID steering calculation with smoothing."""
        params = self.get_parameters(['kp', 'ki', 'kd', 'steer_alpha', 'max_control'])
        kp, ki, kd, alpha, max_steer = [p.value for p in params]
        
        d_error = error - self.prev_error
        self.integral += error
        self.prev_error = error
        
        raw_steer = kp*error + ki*self.integral + kd*d_error
        raw_steer = np.clip(raw_steer, -max_steer, max_steer)
        return alpha*raw_steer + (1-alpha)*self.prev_steer

    def is_in_bounding_box(self, x, y):
        """Check if current position is within predefined bounding box."""
        return (self.box_x_min <= x <= self.box_x_max) and \
               (self.box_y_min <= y <= self.box_y_max)

    def visualize_waypoints(self):
        """Visualize path and target point in RViz."""
        # Path visualization
        path_marker = Marker()
        path_marker.header.frame_id = 'map'
        path_marker.type = Marker.LINE_STRIP
        path_marker.scale.x = 0.05
        path_marker.points = [Point(x=float(x), y=float(y)) 
                            for x, y in zip(self.x_list, self.y_list)]
        self.waypoint_path_pub_.publish(path_marker)

        # Target point visualization
        target_marker = Marker()
        target_marker.header.frame_id = 'map'
        target_marker.type = Marker.SPHERE
        target_marker.scale.x = target_marker.scale.y = 0.3
        target_marker.pose.position.x = float(self.target_point[0])
        target_marker.pose.position.y = float(self.target_point[1])
        self.waypoint_pub_.publish(target_marker)

def main(args=None):
    rclpy.init(args=args)
    controller = PurePursuit()
    controller.get_logger().info("Pure Pursuit node started")
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
