#!/usr/bin/env python3
"""
yolo_fuzzy_speed.py
Human detection and fuzzy speed control for the delivery robot:
  1. YOLO11n detects people in the colour camera image
  2. The depth camera measures the distance to the nearest person
  3. Fuzzy logic converts distance into a speed factor (near = slow, far = fast)
  4. The planner speed (/nav_vel) is multiplied by this factor and sent to the wheels (/cmd_vel)
Safety:
  - Full stop when a person is within 0.4 m
  - Voice alert when a person is within 3 m (only during delivery)
  - Sends zero speed when this node closes
Inputs : /camera/color/image_raw, /camera/depth/image_raw, /nav_vel, /delivery_active
Outputs: /cmd_vel, /person_detected, camera window
Run: cd ~/catkin_ws && rosrun fyp_cz yolo_fuzzy_speed.py
"""
import rospy, cv2, numpy as np, subprocess, time, torch
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from ultralytics import YOLO
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, Bool
from cv_bridge import CvBridge

# ---- CPU limits (leave CPU for navigation) ----
torch.set_num_threads(2)
DETECT_HZ = 5             # YOLO runs 5 times per second

# ---- Distance settings ----
PX_HEIGHT_AT_1M = 940.0   # backup only (box-height method)
MAX_DIST = 5.0
FRAME_H = 480             # camera image height (pixels)
CLOSE_DIST = 0.4          # anything under this = "close"
STOP_DIST = 0.4           # safety stop distance (same as CLOSE_DIST)

# ---- Fuzzy logic controller ----
distance = ctrl.Antecedent(np.arange(0, 5.01, 0.05), 'distance')
speed = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'speed')
distance['near']   = fuzz.trimf(distance.universe, [0, 0, 1.5])
distance['medium'] = fuzz.trimf(distance.universe, [1.0, 2.25, 3.5])
distance['far']    = fuzz.trimf(distance.universe, [3.0, 5.0, 5.0])
speed['slow']     = fuzz.trimf(speed.universe, [0, 0, 0.2])
speed['moderate'] = fuzz.trimf(speed.universe, [0.3, 0.5, 0.7])
speed['fast']     = fuzz.trimf(speed.universe, [0.8, 1.0, 1.0])
rules = [ctrl.Rule(distance['near'], speed['slow']),
         ctrl.Rule(distance['medium'], speed['moderate']),
         ctrl.Rule(distance['far'], speed['fast'])]
sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))

def required_speed(dist_m):
    sim.input['distance'] = float(np.clip(dist_m, 0, MAX_DIST))
    sim.compute()
    return sim.output['speed']

def dist_text(dist, how):
    """'<0.40 m' when too close to measure, else e.g. '1.25 m'."""
    return f"<{CLOSE_DIST:.2f} m" if how == 'close' else f"{dist:.2f} m"

# ---- ROS setup ----
rospy.init_node('yolo_fuzzy_speed')
bridge = CvBridge()
model = YOLO('yolo11n.pt')
latest = {'frame': None, 'depth': None}
factor = {'v': 1.0}            # 0..1 speed multiplier
mode = {'delivering': False}   # set by delivery.py

# ---- Voice alert (robot speaker, only during delivery) ----
ALERT_DIST = 3.0   # speak if a person is closer than 3 m
COOLDOWN = 5.0     # wait 5 s before speaking again
last_alert = {'t': 0.0}

def voice_alert():
    if time.time() - last_alert['t'] > COOLDOWN:
        subprocess.Popen(['espeak', '-v', 'en-gb+f4', '-s', '145', '-g', '5', 'Excuse me, delivery robot passing'])
        last_alert['t'] = time.time()

def on_image(msg):                        # colour camera
    latest['frame'] = bridge.imgmsg_to_cv2(msg, 'bgr8')

def on_depth(msg):                        # depth camera (millimetres)
    latest['depth'] = bridge.imgmsg_to_cv2(msg, 'passthrough')

def on_delivery(msg):                     # True while a delivery is running
    mode['delivering'] = msg.data

def person_distance(depth, x1, y1, x2, y2):
    """Depth first; if depth fails, use box size as backup."""
    if depth is not None:
        w, h = x2 - x1, y2 - y1
        xa, xb = max(0, x1 + w // 4), min(depth.shape[1], x2 - w // 4)
        ya, yb = max(0, y1 + h // 4), min(depth.shape[0], y2 - h // 4)
        roi = depth[ya:yb, xa:xb]
        valid = roi[roi > 0]
        if valid.size > 50:
            d = float(np.median(valid)) / 1000.0
            if d < CLOSE_DIST:                 # anything under 0.4 m = close
                return CLOSE_DIST, 'close'
            return min(d, MAX_DIST), 'depth'
    box_h = y2 - y1
    if box_h > 0.7 * FRAME_H:                  # person fills the screen = close
        return CLOSE_DIST, 'close'
    return min(PX_HEIGHT_AT_1M / box_h, MAX_DIST), 'box'

cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
rospy.on_shutdown(lambda: cmd_pub.publish(Twist()))   # stop wheels when YOLO closes
person_pub = rospy.Publisher('/person_detected', Float32, queue_size=1)

def on_nav_vel(msg):
    """Planner speed -> scaled -> wheels. Scale BOTH so the robot keeps the planned path."""
    out = Twist()
    out.linear.x = msg.linear.x * factor['v']
    out.angular.z = msg.angular.z * factor['v']
    cmd_pub.publish(out)

rospy.Subscriber('/camera/color/image_raw', Image, on_image, queue_size=1, buff_size=2**24)
rospy.Subscriber('/camera/depth/image_raw', Image, on_depth, queue_size=1, buff_size=2**24)
rospy.Subscriber('/nav_vel', Twist, on_nav_vel, queue_size=1)
rospy.Subscriber('/delivery_active', Bool, on_delivery, queue_size=1)

# ---- Detection loop ----
rate = rospy.Rate(DETECT_HZ)
while not rospy.is_shutdown():
    frame = latest['frame']
    if frame is None:
        rate.sleep(); continue
    frame = frame.copy()
    depth = latest['depth']
    results = model(frame, classes=[0], imgsz=320, verbose=False)
    nearest_dist, src = None, ''
    for b in results[0].boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        if y2 - y1 <= 0 or x2 - x1 <= 0: continue
        dist, how = person_distance(depth, x1, y1, x2, y2)
        if nearest_dist is None or dist < nearest_dist:
            nearest_dist, src = dist, how
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"person {float(b.conf[0]):.2f}  {dist_text(dist, how)}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    factor['v'] = required_speed(nearest_dist) if nearest_dist is not None else 1.0
    if nearest_dist is not None and nearest_dist <= STOP_DIST:  # safety stop
        factor['v'] = 0.0
    if mode['delivering'] and nearest_dist is not None and nearest_dist < ALERT_DIST:
        voice_alert()
    person_pub.publish(nearest_dist if nearest_dist is not None else -1.0)   # for dashboard

    label = f"Speed factor: {factor['v']:.2f}"
    if nearest_dist is not None:
        label += f"  (nearest {dist_text(nearest_dist, src)}, {src})"
    label += "  | Voice ON" if mode['delivering'] else "  | Voice OFF"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("YOLO11 + Fuzzy Speed Control", frame)
    if cv2.waitKey(1) == ord('q'):
        break
    rate.sleep()
cv2.destroyAllWindows()