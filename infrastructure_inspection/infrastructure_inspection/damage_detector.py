import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
import cv2
import numpy as np
import math

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

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
        
        self.get_logger().info("Inspection Node Started - Calculating Object Coordinates")

    def scan_callback(self, msg):
        # Tökum lítið úrval af geislum beint fram (t.d. 10 gráður)
        # og finnum minnstu fjarlægðina sem er ekki 'inf'
        front_ranges = msg.ranges[0:5] + msg.ranges[-5:]
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
            
            # Breyta Quaternion yfir í Yaw (snúning á Z ás)
            q = trans.transform.rotation
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            
            return x, y, yaw
        except TransformException:
            return None, None, None

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        lower_red1, upper_red1 = np.array([0, 50, 50]), np.array([10, 255, 255])
        lower_red2, upper_red2 = np.array([160, 50, 50]), np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv_image, lower_red1, upper_red1) + cv2.inRange(hsv_image, lower_red2, upper_red2)
        contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        damage_found = any(cv2.contourArea(cnt) > 150 for cnt in contours)

        if damage_found:
            now = self.get_clock().now()
            if (now - self.last_alert_time).nanoseconds / 1e9 > 2.0:
                rx, ry, ryaw = self.get_robot_pose()
                
                if rx is not None and self.current_dist > 0:
                    # REIKNA HNIT HLUTARINS:
                    obj_x = rx + (self.current_dist * math.cos(ryaw))
                    obj_y = ry + (self.current_dist * math.sin(ryaw))
                    
                    self.get_logger().warn(f"--- DAMAGE DETECTED ---")
                    self.get_logger().info(f"Robot at: ({rx:.2f}, {ry:.2f})")
                    self.get_logger().warn(f"OBJECT AT: X={obj_x:.2f}, Y={obj_y:.2f}")
                    self.get_logger().info(f"Distance: {self.current_dist:.2f}m")
                    
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
