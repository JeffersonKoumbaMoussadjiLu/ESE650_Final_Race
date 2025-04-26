#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from nav_msgs.msg import Odometry
from scipy.spatial import KDTree, transform
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped
from shapely.geometry import Point, Polygon

class PurePursuit(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')
        self.sim = True

        self.obstacle_detected = False
        self.obstacle_dist = np.inf
        self.overtake_side = 'left'
        self.safe_to_overtake = False

        # ROS Subscribers and Publishers
        if self.sim:
            odom_topic = "/ego_racecar/odom"
            self.create_subscription(Odometry, odom_topic, self.pose_callback, 10)
        else:
            odom_topic = "/pf/viz/inferred_pose"
            self.create_subscription(PoseStamped, odom_topic, self.pose_callback, 10)

        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)

        self.L = 1.72
        self.P = 0.435

        csv_data = np.loadtxt("/home/amy/lab_ws/src/lab5/pure_pursuit/scripts/clicked_points.csv", delimiter=",", skiprows=0)
        self.waypoints = csv_data[:, 0:2]

        self.bounding_box = {
            "x_min": min(1.3489, 0.578487, -4.12148, -5.58624),
            "x_max": max(1.3489, 0.578487, -4.12148, -5.58624),
            "y_min": min(1.87811, 2.5409, -6.66662, -5.56692),
            "y_max": max(1.87811, 2.5409, -6.66662, -5.56692)
        }

        self.region_polygon = Polygon([
            (-5.0035, -7.5792), 
            (-6.0583, -6.5964), 
            (1.5986, 2.2037), 
            (0.9630, 2.7362)
        ])

        self.marker_pub = self.create_publisher(MarkerArray, '/waypoints_markers', 10)
        self.timer = self.create_timer(1.0, self.publish_waypoints_markers)

        self.polygon_pub = self.create_publisher(Marker, "/polygon_marker", 10)
        self.timer = self.create_timer(1.0, self.publish_polygon_markers)  

    def lidar_callback(self, scan_msg):
        angle_min = scan_msg.angle_min
        angle_increment = scan_msg.angle_increment
        ranges = np.array(scan_msg.ranges)

        # Define sector (-15° to +15° front)
        front_angle = np.pi / 6
        n = len(ranges)
        angles = angle_min + np.arange(n) * angle_increment

        front_indices = np.where((angles >= -front_angle) & (angles <= front_angle))[0]
        front_ranges = ranges[front_indices]
        valid = np.isfinite(front_ranges)
        front_ranges = front_ranges[valid]

        if len(front_ranges) == 0:
            self.obstacle_detected = False
            self.obstacle_dist = np.inf
            self.safe_to_overtake = False
            return

        min_dist = np.min(front_ranges)
        self.obstacle_dist = min_dist
        OBSTACLE_THRESHOLD = 0.5  # meters

        self.obstacle_detected = (min_dist < OBSTACLE_THRESHOLD)

        # Sector logic for safe overtake (±25° window)
        side_angle = np.pi / 7.2  # ~25°
        left_indices = np.where((angles > 0.0) & (angles <= side_angle))[0]
        right_indices = np.where((angles < 0.0) & (angles >= -side_angle))[0]

        left_mean = np.nanmean(ranges[left_indices]) if len(left_indices) > 0 else np.inf
        right_mean = np.nanmean(ranges[right_indices]) if len(right_indices) > 0 else np.inf

        SAFE_GAP = 1.3 # meters needed to shift and pass

        if self.obstacle_detected:
            # Choose side with bigger gap, and only shift if it's big enough
            if left_mean > SAFE_GAP and left_mean > right_mean:
                self.overtake_side = 'left'
                self.safe_to_overtake = True
            # elif right_mean > SAFE_GAP:
            #     self.overtake_side = 'right'
            #     self.safe_to_overtake = True
            else:
                self.safe_to_overtake = False
        else:
            self.safe_to_overtake = False

    def publish_waypoints_markers(self):
        marker_array = MarkerArray()
        for i, wp in enumerate(self.waypoints):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "waypoints"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = wp[0]
            marker.pose.position.y = wp[1]
            marker.pose.position.z = 0.1
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0 
            marker.color.r = 1.0 
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker_array.markers.append(marker)
        self.marker_pub.publish(marker_array) 

    def publish_polygon_markers(self):
        marker_array = MarkerArray()
        polygon_corners = [
            (-5.0035, -7.5792), 
            (-6.0583, -6.5964), 
            (1.5986, 2.2037), 
            (0.9630, 2.7362)
        ]
        for i, (x, y) in enumerate(polygon_corners):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "polygon_corners"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.1
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker_array.markers.append(marker)
        self.marker_pub.publish(marker_array)

    def pose_callback(self, pose_msg):
        if self.sim:
            car_x = pose_msg.pose.pose.position.x
            car_y = pose_msg.pose.pose.position.y
            quat = pose_msg.pose.pose.orientation
        else:
            car_x = pose_msg.pose.position.x
            car_y = pose_msg.pose.position.y
            quat = pose_msg.pose.orientation

        quat = [quat.x, quat.y, quat.z, quat.w]
        R = transform.Rotation.from_quat(quat)
        self.rot = R.as_matrix()

        car_position = Point(car_x, car_y)
        in_region = self.region_polygon.contains(car_position)

        self.kd_tree = KDTree(self.waypoints)
        _, idx = self.kd_tree.query([car_x, car_y])
        for i in range(idx, len(self.waypoints)):
            dist = np.linalg.norm(self.waypoints[i] - np.array([car_x, car_y]))
            if dist >= self.L:
                goal_x, goal_y = self.waypoints[i]
                break
        else:
            return

        # Local goal and OVERTAKE/SAFE-TO-PASS LOGIC
        local_goal = self.translatePoint(np.array([car_x, car_y]), np.array([goal_x, goal_y]))

        if self.obstacle_detected and self.obstacle_dist < 3.0:
            if self.safe_to_overtake:
                offset = 0.65
                if self.overtake_side == 'left':
                    local_goal[1] += offset
                else:
                    local_goal[1] -= offset
                print("Overtaking on", self.overtake_side)
                speed_override = None  # Use normal speed logic
            else:
                print("Obstacle ahead, but not enough space: slowing down.")
                speed_override = 1.5  # meters/sec, or tune lower as needed
        else:
            speed_override = None

        goal_y_vehicle = local_goal[1]
        curvature = 2 * goal_y_vehicle / (self.L ** 2)
        steering_angle = self.P * curvature

        drive_msg = AckermannDriveStamped()
        max_steering = 0.4
        max_speed = 4.5
        min_speed = 1.5
        abs_sa = abs(steering_angle)
        steer_fraction = abs_sa / max_steering
        speed = max_speed - (max_speed - min_speed) * steer_fraction

        # Apply speed override if required
        if speed_override is not None:
            speed = speed_override

        if speed <= 1.5:
            speed = 2.0
        elif speed >= 4.3:
            speed = 5.0

        drive_msg.drive.speed = speed
        drive_msg.drive.steering_angle = steering_angle
        self.drive_pub.publish(drive_msg)

    def get_yaw_from_pose(self, pose_msg):
        q = pose_msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y ** 2 + q.z ** 2)
        return np.arctan2(siny_cosp, cosy_cosp)
    
    def translatePoint(self, currPoint, targetPoint):
        H = np.zeros((4, 4))
        H[0:3, 0:3] = np.linalg.inv(self.rot)
        H[0, 3] = currPoint[0]
        H[1, 3] = currPoint[1]
        H[3, 3] = 1.0
        dir = targetPoint - currPoint
        translated_point = (H @ np.array((dir[0], dir[1], 0, 0))).reshape((4))
        return translated_point

def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit Initialized")
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)
    pure_pursuit_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
