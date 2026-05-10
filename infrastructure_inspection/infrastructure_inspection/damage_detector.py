import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import csv
import math
from datetime import datetime
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from visualization_msgs.msg import Marker

class DamageDetector(Node):
    def __init__(self):
        super().__init__('damage_detector')
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        
        self.bridge = CvBridge()
        self.last_alert_time = self.get_clock().now()
        self.current_dist = 0.0
        
        # RViz Marker Publisher
        self.marker_pub = self.create_publisher(Marker, 'damage_marker', 10)
        self.marker_id = 0
        
        self.get_logger().info("Inspection Node Started - Calculating Object Coordinates")
        self.report_dir = 'inspection_results'
        if not os.path.exists(self.report_dir):
            os.makedirs(self.report_dir)
            os.makedirs(os.path.join(self.report_dir, 'snapshots'))
            
        self.csv_path = os.path.join(self.report_dir, 'report.csv')
        
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Robot_X', 'Robot_Y', 'Object_X', 'Object_Y', 'Distance', 'Image_Path'])

    def scan_callback(self, msg):
        front_ranges = msg.ranges[0:10] + msg.ranges[-10:]
        valid_ranges = [r for r in front_ranges if not math.isinf(r) and r > 0.1]
        if valid_ranges:
            self.current_dist = min(valid_ranges)
        else:
            self.current_dist = 0.0

    def get_robot_pose(self):
        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform('map', 'base_footprint', now)
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            q = trans.transform.rotation
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            return x, y, yaw
        except TransformException:
            return None, None, None

    def save_detection(self, rx, ry, ox, oy, dist, image):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        img_name = f"damage_{timestamp}.jpg"
        img_path = os.path.join(self.report_dir, 'snapshots', img_name)
        cv2.imwrite(img_path, image)
        with open(self.csv_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, f"{rx:.2f}", f"{ry:.2f}", f"{ox:.2f}", f"{oy:.2f}", f"{dist:.2f}", img_name])
        self.get_logger().info(f"Saved report entry and snapshot: {img_name}")

    def publish_marker(self, x, y):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "damage"
        marker.id = self.marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.1
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0 
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        self.marker_pub.publish(marker)
        self.marker_id += 1

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        lower_red1, upper_red1 = np.array([0, 50, 50]), np.array([10, 255, 255])
        lower_red2, upper_red2 = np.array([160, 50, 50]), np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv_image, lower_red1, upper_red1) + cv2.inRange(hsv_image, lower_red2, upper_red2)
        contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            if cv2.contourArea(cnt) > 100:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        max_area = max([cv2.contourArea(cnt) for cnt in contours]) if contours else 0

        if max_area > 150:
            now = self.get_clock().now()
            if (now - self.last_alert_time).nanoseconds / 1e9 > 2.0:
                rx, ry, ryaw = self.get_robot_pose()
                if rx is not None and self.current_dist > 0:
                    obj_x = rx + (self.current_dist * math.cos(ryaw))
                    obj_y = ry + (self.current_dist * math.sin(ryaw))
                    self.get_logger().warn(f"--- DAMAGE DETECTED ---")
                    self.save_detection(rx, ry, obj_x, obj_y, self.current_dist, cv_image)
                    self.publish_marker(obj_x, obj_y)
                    self.last_alert_time = now

        cv2.imshow("Inspection Camera", cv_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = DamageDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
