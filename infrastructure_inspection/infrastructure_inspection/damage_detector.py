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
        self.latest_scan = None
        
        self.marker_pub = self.create_publisher(Marker, 'damage_marker', 10)
        
        # NÝTT: Geymsla fyrir flöggin okkar til að fylgjast með sögunni
        # Uppbygging: { marker_id: {'x': x, 'y': y, 'count': staðfestingar, 'timestamp': tími} }
        self.tracked_markers = {}
        self.next_marker_id = 0
        
        self.get_logger().info("Inspection Node Started - Outlier Removal System Active")
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
        self.latest_scan = msg

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

    def publish_marker(self, marker_id, x, y, action=Marker.ADD):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "damage"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = action  # Getur verið Marker.ADD eða Marker.DELETE
        
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

    def process_new_measurement(self, ox, oy, rx, ry, target_dist, cv_image):
        match_found = False
        threshold_dist = 0.4  # Ef ný mæling er innan við 40cm frá gamalli, þá er þetta sama dósin
        
        for m_id, data in self.tracked_markers.items():
            dist_between = math.sqrt((ox - data['x'])**2 + (oy - data['y'])**2)
            
            if dist_between < threshold_dist:
                # Við fundum samsvörun! Uppfærð hnitin yfir í nýju (nákvæmari) mælinguna
                data['x'] = ox
                data['y'] = oy
                data['count'] += 1
                match_found = True
                
                # Uppfærum staðsetninguna á flagginu í RViz
                self.publish_marker(m_id, ox, oy, Marker.ADD)
                
                # Ef þetta er í þriðja skiptið sem við staðfestum dósina, vistum við hana í CSV skýrsluna
                if data['count'] == 3:
                    self.save_detection(rx, ry, ox, oy, target_dist, cv_image)
                    self.get_logger().warn(f"--- DAMAGE CONFIRMED AND SAVED AT {target_dist:.2f}m ---")
                break
                
        if not match_found:
            # Þetta er alveg ný mæling (gæti verið fyrsta flugtakið eða suð)
            m_id = self.next_marker_id
            self.tracked_markers[m_id] = {
                'x': ox,
                'y': oy,
                'count': 1,
                'timestamp': self.get_clock().now()
            }
            self.next_marker_id += 1
            
            # Setjum bráðabirgðaflagg í RViz
            self.publish_marker(m_id, ox, oy, Marker.ADD)
            
        # Hreinsikerfi (Outlier Removal): 
        # Ef eitthvað flagg hefur verið inni í meira en 5 sekúndur en fékk aldrei fleiri en 2 staðfestingar,
        # þá var það örugglega slæm fyrsta mæling. Eyðum því!
        current_time = self.get_clock().now()
        ids_to_delete = []
        
        for m_id, data in self.tracked_markers.items():
            age = (current_time - data['timestamp']).nanoseconds / 1e9
            if age > 5.0 and data['count'] < 3:
                ids_to_delete.append(m_id)
                
        for m_id in ids_to_delete:
            self.get_logger().info(f"Fjarlægi rammvitlaust fyrsta flagg (ID: {m_id})")
            # Sendum skipun á RViz um að eyða punktinum
            self.publish_marker(m_id, self.tracked_markers[m_id]['x'], self.tracked_markers[m_id]['y'], Marker.DELETE)
            # Eyðum úr minninu okkar
            del self.tracked_markers[m_id]

    def image_callback(self, msg):
        if self.latest_scan is None:
            return

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        lower_red1, upper_red1 = np.array([0, 50, 50]), np.array([10, 255, 255])
        lower_red2, upper_red2 = np.array([160, 50, 50]), np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv_image, lower_red1, upper_red1) + cv2.inRange(hsv_image, lower_red2, upper_red2)
        contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_boxes = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 100:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                detected_boxes.append((x, y, w, h, cv2.contourArea(cnt)))

        if detected_boxes:
            best_box = max(detected_boxes, key=lambda b: b[4])
            bx, by, bw, bh, b_area = best_box
            
            now = self.get_clock().now()
            if (now - self.last_alert_time).nanoseconds / 1e9 > 0.5:  # Hraðari mælingar (0.5s) til að safna punktum hraðar
                rx, ry, ryaw = self.get_robot_pose()
                
                if rx is not None:
                    image_width = cv_image.shape[1]
                    cx = bx + (bw / 2.0)
                    
                    relative_pos = (cx - (image_width / 2.0)) / image_width
                    fov_rad = math.radians(62)
                    angle_offset = -relative_pos * fov_rad
                    angle_deg = math.degrees(angle_offset)
                    
                    camera_estimated_dist = 45.0 / float(bw) if bw > 0 else 2.5
                    
                    laser_idx = int(angle_deg) % 360
                    lidar_dist = self.latest_scan.ranges[laser_idx]
                    
                    if not math.isinf(lidar_dist) and lidar_dist > 0.3 and abs(lidar_dist - camera_estimated_dist) < 0.4:
                        target_dist = lidar_dist
                    else:
                        target_dist = camera_estimated_dist
                    
                    if 0.3 < target_dist < 3.0:
                        actual_angle = ryaw + angle_offset
                        camera_offset = 0.07
                        
                        obj_x = rx + (camera_offset * math.cos(ryaw)) + (target_dist * math.cos(actual_angle))
                        obj_y = ry + (camera_offset * math.sin(ryaw)) + (target_dist * math.sin(actual_angle))
                        
                        # Sendum hnitið í vinnslu og síun í stað þess að vista beint
                        self.process_new_measurement(obj_x, obj_y, rx, ry, target_dist, cv_image)
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