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
    """ 
    Implement Pure Pursuit + rectangular speed & lookahead zones + Lidar-based braking.
    Also visualizes:
      - Waypoints and box outlines
      - The lookahead circle around the car
    """
    def __init__(self):
        super().__init__('pure_pursuit_node')
        self.sim = True

        # Subscribe to odometry or pose
        if self.sim:
            odom_topic = "/ego_racecar/odom"
            self.create_subscription(Odometry, odom_topic, self.pose_callback, 10)
        else:
            odom_topic = "/pf/viz/inferred_pose"
            self.create_subscription(PoseStamped, odom_topic, self.pose_callback, 10)

        # Publisher for drive commands
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        # Subscribe to Lidar scan for dynamic braking
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.latest_scan = None

        # Default Pure Pursuit parameters
        self.default_L = 1.5  # default lookahead
        self.P = 0.435
        self.default_speed = 3.5

        # Load waypoints and build KD-tree
        csv_data = np.loadtxt(
            "./jeff.csv",
            delimiter=",",
            skiprows=0
        )
        self.waypoints = csv_data[:, 0:2]
        self.kd_tree = KDTree(self.waypoints)

        # -- NEW BOX DEFINITIONS (6 boxes) --
        # The 'speed' and 'lookahead' here are placeholders; tune them as needed.
        self.speed_zones = [
            {
                "name": "Box1",
                "corners": [
                    (0.198457, 0.190445),
                    (-0.87294, 1.16791),                    
                    (-4.93407, -4.26399),
                    (-3.553, -5.21938),
                ],
                "speed": 5.0,
                "lookahead": 3.0
            },
            {
                "name": "Box2",
                "corners": [
                    (-3.553, -5.21938),
                    (-4.93407, -4.26399),
                    (-6.67631, -6.6357),
                    (-5.27002, -7.77882)
                ],
                "speed": 3.0,
                "lookahead": 2.0
            },
            {
                "name": "Box3",
                "corners": [
                    (-3.89093, -6.58539),
                    (-5.00228, -7.98292),
                    (-1.28036, -10.838),
                    (-0.0925005, -9.16933),
                ],
                "speed": 3.0,
                "lookahead": 2.0
            },
            {
                "name": "Box4",
                "corners": [
                    (-0.0509792, -9.08308),
                    (1.43881, -7.23712),
                    (-2.68007, -4.196),
                    (-4.02462, -6.48412)
                ],
                "speed": 3.0,
                "lookahead": 1.5
            },
            {
                "name": "Box5",
                "corners": [
                    (-1.25988, -5.06604),
                    (-2.40547, -3.95976),
                    (0.456809, 0.0404336),
                    (1.73969, -1.02733)
                ],
                "speed": 4.5,
                "lookahead": 3.0
            },
            {
                "name": "Box6",
                "corners": [
                    (1.94017, -0.763478),
                    (-0.779704, 1.25985),
                    (0.816008, 3.10399),
                    (3.68153, 1.31497)
                ],
                "speed": 2.5,
                "lookahead": 2.0
            },
        ]

        # Create marker publishers
        self.marker_pub = self.create_publisher(MarkerArray, '/waypoints_markers', 10)
        self.lookahead_marker_pub = self.create_publisher(Marker, '/lookahead_marker', 10)

        # Timer to periodically publish markers (waypoints + boxes)
        self.timer = self.create_timer(1.0, self.publish_markers)

    def scan_callback(self, scan_msg):
        """Save the latest Lidar scan for dynamic braking logic."""
        self.latest_scan = scan_msg

    def publish_markers(self):
        """
        Publish a MarkerArray containing:
          1) A small sphere for each waypoint.
          2) A line strip for each box corner set.
        """
        marker_array = MarkerArray()
        marker_id = 0

        # --- Waypoint Markers ---
        for i, wp in enumerate(self.waypoints):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "waypoints"
            marker.id = marker_id
            marker_id += 1

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

        # --- Box Markers (Line Strips) ---
        for box in self.speed_zones:
            corners = box["corners"]
            line_strip = Marker()
            line_strip.header.frame_id = "map"
            line_strip.header.stamp = self.get_clock().now().to_msg()
            line_strip.ns = "boxes"
            line_strip.id = marker_id
            marker_id += 1

            line_strip.type = Marker.LINE_STRIP
            line_strip.action = Marker.ADD
            line_strip.scale.x = 0.05  # thickness of lines
            line_strip.color.a = 1.0
            line_strip.color.r = 0.0
            line_strip.color.g = 1.0  # green lines
            line_strip.color.b = 0.0

            # Close the loop by repeating the first corner at the end
            for corner in corners:
                pt = self.create_point(corner[0], corner[1], 0.0)
                line_strip.points.append(pt)
            first_corner = corners[0]
            line_strip.points.append(self.create_point(first_corner[0], first_corner[1], 0.0))

            marker_array.markers.append(line_strip)

        # Publish all markers
        self.marker_pub.publish(marker_array)

    def pose_callback(self, pose_msg):
        # 1) Extract car position/orientation
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

        # 2) Determine if we are inside any zone => override speed & lookahead
        speed = self.default_speed
        L = self.default_L
        for box in self.speed_zones:
            if self.is_point_in_box(car_x, car_y, box["corners"]):
                speed = box["speed"]
                L = box["lookahead"]
                break

        # 3) Find the lookahead waypoint
        goal_x, goal_y = self.get_goal_waypoint(car_x, car_y, L)
        if goal_x is None or goal_y is None:
            return  # no valid waypoint found

        # 4) Transform the goal point into vehicle frame
        goal_y_vehicle = self.translate_point(
            np.array([car_x, car_y]), 
            np.array([goal_x, goal_y])
        )[1]

        # 5) Compute steering from curvature
        curvature = 2.0 * goal_y_vehicle / (L ** 2)
        steering_angle = self.P * curvature

        # 6) Lidar-based braking logic
        if self.latest_scan is not None:
            ranges = np.array(self.latest_scan.ranges)
            num_points = len(ranges)
            if num_points > 0:
                center_idx = num_points // 2
                window_size = 75  # look +/- 75 samples around the center
                start_idx = max(0, center_idx - window_size)
                end_idx = min(num_points, center_idx + window_size)

                forward_ranges = ranges[start_idx:end_idx]
                valid_ranges = forward_ranges[forward_ranges > 0.08]  # ignore <0.08

                if valid_ranges.size > 0:
                    min_forward_dist = np.min(valid_ranges)
                    # Example braking logic (commented out):
                    # if min_forward_dist < 0.3:
                    #     speed = 0.0
                    # elif min_forward_dist < 1.0:
                    #     speed = min(speed, 1.0)
                    # elif min_forward_dist < 2.0:
                    #     speed = min(speed, 2.0)

        # 7) Publish the drive command
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = speed
        drive_msg.drive.steering_angle = steering_angle
        self.drive_pub.publish(drive_msg)

        # 8) Publish a visualization marker showing the current lookahead radius
        self.publish_lookahead_marker(car_x, car_y, L)

    def publish_lookahead_marker(self, car_x, car_y, L):
        """
        Publish a translucent sphere to represent the lookahead distance around the car.
        """
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "lookahead_circle"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # Center the circle at the car's position
        marker.pose.position.x = car_x
        marker.pose.position.y = car_y
        marker.pose.position.z = 0.0

        # Sphere scale.x/scale.y = diameter => 2 * radius
        marker.scale.x = 2.0 * L
        marker.scale.y = 2.0 * L
        marker.scale.z = 0.01  # keep it thin

        # Slightly transparent color (cyan)
        marker.color.a = 0.3
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0

        self.lookahead_marker_pub.publish(marker)

    def get_goal_waypoint(self, car_x, car_y, look_ahead):
        """
        Find a waypoint at least 'look_ahead' meters from (car_x, car_y).
        Returns (goal_x, goal_y) or (None, None) if not found.
        """
        _, idx = self.kd_tree.query([car_x, car_y])
        # Search forward from idx
        for i in range(idx, len(self.waypoints)):
            dist = np.linalg.norm(self.waypoints[i] - np.array([car_x, car_y]))
            if dist >= look_ahead:
                return self.waypoints[i][0], self.waypoints[i][1]

        # Wrap around if necessary
        for i in range(idx):
            dist = np.linalg.norm(self.waypoints[i] - np.array([car_x, car_y]))
            if dist >= look_ahead:
                return self.waypoints[i][0], self.waypoints[i][1]

        return None, None

    def is_point_in_box(self, x, y, corners):
        """
        corners = [pt1, pt2, pt3, pt4] in any order, but we'll do a simple bounding-box check.
        """
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        if (min(xs) <= x <= max(xs)) and (min(ys) <= y <= max(ys)):
            return True
        return False

    def create_point(self, x, y, z):
        from geometry_msgs.msg import Point
        p = Point()
        p.x = x
        p.y = y
        p.z = z
        return p

    def translate_point(self, currPoint, targetPoint):
        """
        Transform 'targetPoint' from the map frame to the local vehicle frame,
        using the vehicle's rotation at 'currPoint'.
        """
        H = np.zeros((4, 4))
        # rotation world->vehicle
        H[0:3, 0:3] = np.linalg.inv(self.rot)
        H[0, 3] = currPoint[0]
        H[1, 3] = currPoint[1]
        H[3, 3] = 1.0

        dir_vec = targetPoint - currPoint
        # We multiply by [dx, dy, 0, 0] to effectively do R^-1*(target - current)
        translated_point = (H @ np.array([dir_vec[0], dir_vec[1], 0, 0])).reshape((4))
        return translated_point

def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit with NEW Speed & Lookahead Zones + Lidar-based braking + Visualization")
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)

    pure_pursuit_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
