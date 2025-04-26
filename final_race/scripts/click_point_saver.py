#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import csv
import os
from geometry_msgs.msg import PointStamped

class ClickedPointSaver(Node):
    def __init__(self):
        super().__init__('clicked_point_saver')
        self.subscription = self.create_subscription(
            PointStamped,
            '/clicked_point',
            self.clicked_point_callback,
            10)
        self.subscription  # Prevent unused variable warning
        
        # Define CSV file path
        self.csv_filename = os.path.join(os.path.expanduser("~"), 'clicked_points.csv')
        
        # Write CSV header
        with open(self.csv_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["#x", "y"])

        self.get_logger().info(f"Saving clicked points to {self.csv_filename}")

    def clicked_point_callback(self, msg):
        x, y = msg.point.x, msg.point.y
        
        with open(self.csv_filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([x, y])
        
        self.get_logger().info(f"Saved point: {x}, {y}")


def main(args=None):
    rclpy.init(args=args)
    node = ClickedPointSaver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()