'''
This code does a lap in 10-11 sec
This code use the jeff.csv and use Amy's code  with a dynamic allocation of the speed:
steer_fraction = abs_sa / max_steering  # range [0..1]
        speed = max_speed - (max_speed - min_speed)*steer_fraction
This code is really depend on L (Look ahead distance) and the waypoints 
A bad tuning of L or inconsistent waypoints will make really drift (really sensitive).
Make sure that particler filter is well aligned. Do 1-2 lap in case to adjust it .
'''

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
# TODO CHECK: include needed ROS msg type headers and libraries
from nav_msgs.msg import Odometry
from scipy.spatial import KDTree, transform
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped
from shapely.geometry import Point, Polygon

class PurePursuit(Node):
    """ 
    Implement Pure Pursuit on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('pure_pursuit_node_opp')
        self.sim = True

        # TODO: create ROS subscribers and publishers
        if self.sim:
            odom_topic = "/opp_racecar/odom"
            self.create_subscription(Odometry, odom_topic, self.pose_callback, 10)
        else:
            odom_topic = "/pf/viz/inferred_pose"
            self.create_subscription(PoseStamped, odom_topic, self.pose_callback, 10)

        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/opp_drive', 10)

        self.L = 1.72  #1.5 (1.7 very good) => (1.72 excellent) => (1.7175 testing)
        self.P = 0.435 #0.435 (Good)

        csv_data = np.loadtxt("/home/amy/lab_ws/src/lab5/pure_pursuit/scripts/clicked_points.csv", delimiter=",", skiprows=0)
        self.waypoints = csv_data[:, 0:2]
        # self.kd_tree = KDTree(self.waypoints)
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
        
        # Define the 4 corners of the rectangle (polygon)
        polygon_corners = [
            (-5.0035, -7.5792),  # Corner 1
            (-6.0583, -6.5964),  # Corner 2
            (1.5986, 2.2037),    # Corner 3
            (0.9630, 2.7362)     # Corner 4
        ]
        
        # Loop over the corners and create a marker for each
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
            marker.pose.position.z = 0.1  # Slight elevation to make it visible
            marker.scale.x = 0.2  # Size of the sphere
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0  # Full opacity
            marker.color.r = 0.0  # Red
            marker.color.g = 1.0
            marker.color.b = 0.0

            # Add marker to the marker array
            marker_array.markers.append(marker)

        # Publish the marker array
        self.marker_pub.publish(marker_array)
    
    def pose_callback(self, pose_msg):
        # TODO: find the current waypoint to track using methods mentioned in lecture
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

        #self.L = max(0.1, 0.5 * pose_msg.twist.twist.linear.x)
        car_position = Point(car_x, car_y)
        in_region = self.region_polygon.contains(car_position)

        # Use limited waypoints if inside the region, otherwise use all
        # if in_region:
        #     self.waypoints = np.vstack((self.full_waypoints[1:60], self.full_waypoints[175:201]))
        # else:
        #     self.waypoints = self.full_waypoints

        self.kd_tree = KDTree(self.waypoints)

        _, idx = self.kd_tree.query([car_x, car_y])
        for i in range(idx, len(self.waypoints)):
            dist = np.linalg.norm(self.waypoints[i] - np.array([car_x, car_y]))
            if dist >= self.L:
                goal_x, goal_y = self.waypoints[i]
                break
        else:
            # self.get_logger().warn("No valid lookahead point found.")
            return
        
        # print(goal_x, goal_y)

        # TODO: transform goal point to vehicle frame of reference
        goal_y_vehicle = self.translatePoint(np.array([car_x, car_y]), np.array([goal_x, goal_y]))[1]

        # TODO: calculate curvature/steering angle
        curvature = 2 * goal_y_vehicle / (self.L ** 2)
        steering_angle = self.P * curvature

        # TODO: publish drive message, don't forget to limit the steering angle.

        '''
        steering_angle = np.clip(steering_angle, -0.4, 0.4)

        drive_msg = AckermannDriveStamped()
        # if np.abs(steering_angle) > 0.2:
        #     drive_msg.drive.speed = 1.5
        if np.abs(steering_angle) > 0.12:
            drive_msg.drive.speed = 2.0

        elif np.abs(steering_angle) > 0.12:
            drive_msg.drive.speed = 1.5

        elif np.abs(steering_angle) > 0.24:
            drive_msg.drive.speed = 1.0
        else:
            drive_msg.drive.speed = 4.0

        print(steering_angle, "almost")
        '''
        #  Compute speed based on steering angle 
        # Example: linear interpolation from 4.5 m/s (straight) to 1.5 m/s (max steer)
        drive_msg = AckermannDriveStamped()
        max_steering = 0.4
        max_speed = 3.0
        min_speed = 1.5
        abs_sa = abs(steering_angle)
        steer_fraction = abs_sa / max_steering  # range [0..1]
        speed = max_speed - (max_speed - min_speed)*steer_fraction

        if speed <= 1.5:
            speed = 2.0
        elif speed >= 4.3:
            speed = 5.0

        drive_msg.drive.speed = speed

        print(speed, "eh eh I am very fast boi")

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