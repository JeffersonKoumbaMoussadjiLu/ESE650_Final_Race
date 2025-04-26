#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker

class ClickedPointVisualizer(Node):
    def __init__(self):
        super().__init__('clicked_point_visualizer')
        self.subscription = self.create_subscription(
            PointStamped,
            '/clicked_point',
            self.clicked_point_callback,
            10)
        self.marker_pub = self.create_publisher(Marker, '/visualization_marker', 10)
        self.point_count = 0  # To give unique marker IDs

    def clicked_point_callback(self, msg):
        marker = Marker()
        marker.header.frame_id = "map"  # Adjust this if necessary
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "clicked_points"
        marker.id = self.point_count
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = msg.point.x
        marker.pose.position.y = msg.point.y
        marker.pose.position.z = msg.point.z
        
        marker.scale.x = 0.2  # Adjust the size as needed
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        
        marker.color.a = 1.0  # Fully visible
        marker.color.r = 1.0  # Red color
        marker.color.g = 0.0
        marker.color.b = 0.0
        
        self.marker_pub.publish(marker)
        self.get_logger().info(f"Published Marker at: {msg.point.x}, {msg.point.y}, {msg.point.z}")
        
        self.point_count += 1  # Increment ID for unique markers


def main(args=None):
    rclpy.init(args=args)
    node = ClickedPointVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()