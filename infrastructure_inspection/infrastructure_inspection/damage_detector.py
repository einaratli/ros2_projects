import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class DamageDetector(Node):
    def __init__(self):
        super().__init__('damage_detector')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)
        self.bridge = CvBridge()
        self.get_logger().info("Inspection Node Started - Looking for red damage...")

    def image_callback(self, msg):
        # 1. Breyta yfir í OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # 2. Breyta í HSV (betra fyrir litagreiningu)
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # 3. Skilgreina rauðan lit (þarf tvö svið því rauður er á báðum endum HSV)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        red_mask = mask1 + mask2
        
        # 4. Finna útlínur (contours) á rauða litnum
        contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:  # Bara sýna ef hluturinn er nógu stór
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(cv_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(cv_image, "DAMAGE DETECTED", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                self.get_logger().warn("!!! DAMAGE DETECTED AT EQUIPMENT !!!")

        # 5. Sýna bæði myndina og síuna (mask)
        cv2.imshow("Inspection Camera", cv_image)
        cv2.imshow("Red Color Mask", red_mask)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = DamageDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
