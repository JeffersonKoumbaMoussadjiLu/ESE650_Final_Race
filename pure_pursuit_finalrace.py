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

class PurePursuit(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')
        self.sim = False

        # Subscribe to odometry
        if self.sim:
            odom_topic = "/ego_racecar/odom"
            self.create_subscription(Odometry, odom_topic, self.pose_callback, 10)
        else:
            odom_topic = "/pf/viz/inferred_pose"
            self.create_subscription(PoseStamped, odom_topic, self.pose_callback, 10)

        # Publish drive commands
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        # Subscribe to Lidar scan
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.latest_scan = None

        # Load waypoints
        self.L = 2.3
        self.P = 0.435
        csv_data = np.loadtxt("/home/nvidia/f1tenth_ws/src/slam-and-pure-pursuit-team10/pure_pursuit/scripts/jeff.csv", delimiter=",", skiprows=0)
        self.waypoints = csv_data[:, 0:2]
        self.kd_tree = KDTree(self.waypoints)

        # Waypoint visualization
        self.marker_pub = self.create_publisher(MarkerArray, '/waypoints_markers', 10)
        self.timer = self.create_timer(1.0, self.publish_waypoints_markers)

    def scan_callback(self, scan_msg):
        """Save the latest Lidar scan."""
        self.latest_scan = scan_msg

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

        # Find closest waypoint
        _, idx = self.kd_tree.query([car_x, car_y])

        # Find goal point
        for i in range(idx, len(self.waypoints)):
            dist = np.linalg.norm(self.waypoints[i] - np.array([car_x, car_y]))
            if dist >= self.L:
                goal_x, goal_y = self.waypoints[i]
                break
        else:
            for i in range(0, idx):
                dist = np.linalg.norm(self.waypoints[i] - np.array([car_x, car_y]))
                if dist >= self.L:
                    goal_x, goal_y = self.waypoints[i]
                    break
            else:
                return

        # Transform goal point to vehicle frame
        goal_y_vehicle = self.translatePoint(np.array([car_x, car_y]), np.array([goal_x, goal_y]))[1]

        # Calculate curvature and steering
        curvature = 2 * goal_y_vehicle / (self.L ** 2)
        steering_angle = self.P * curvature

        # Base speed control
        drive_msg = AckermannDriveStamped()
        max_steering = 0.4
        max_speed = 4.5
        min_speed = 1.5
        abs_sa = abs(steering_angle)
        steer_fraction = abs_sa / max_steering
        speed = max_speed - (max_speed - min_speed) * steer_fraction

        # Lidar-based braking
        if self.latest_scan is not None:
            ranges = np.array(self.latest_scan.ranges)
            num_points = len(ranges)
            if num_points > 0:
                center_idx = num_points // 2
                window_size = 75  # 150 total
                start_idx = max(0, center_idx - window_size)
                end_idx = min(num_points, center_idx + window_size)

                forward_ranges = ranges[start_idx:end_idx]
                valid_ranges = forward_ranges[forward_ranges > 0.08]

                if valid_ranges.size > 0:
                    min_forward_dist = np.min(valid_ranges)

                    if min_forward_dist < 0.3:
                        speed = 0.0
                        print("min distance less than 0.3")
                    elif min_forward_dist < 1.0:
                        speed = min(speed, 1.0)
                        print("min distance less than 1 meter")
                    elif min_forward_dist < 2.0:
                        speed = min(speed, 2.0)
                        print("min distnce 2 meters less than")

        if speed > 4.3:
            speed = 5.0
        if speed < 2.0 and speed != 0.0:
            speed = 2.0

        drive_msg.drive.speed = speed
        drive_msg.drive.steering_angle = steering_angle
        self.drive_pub.publish(drive_msg)

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
