# Autonomous Indoor Logistics Robot for Laboratory Materials Delivery

Final-year project (Academic Project), Faculty of Computer Science and Information Technology, Universiti Malaya.
Stakeholder: Department of Parasitology, Faculty of Medicine, Universiti Malaya.

A ROS-based delivery robot (Jupiter2) that carries laboratory materials between rooms along the department corridor.
Staff send a delivery from a web dashboard; the robot navigates autonomously, slows down near people using
YOLO11 and fuzzy logic, and announces its arrival.

## Features
- **SLAM navigation:** GMapping map, AMCL localisation, A* global path and DWA local planner (move_base)
- **Obstacle avoidance:** keeps at least 15 cm from obstacles (footprint padding), live obstacles from the LiDAR
- **Human detection:** YOLO11n detects people; the depth camera measures their distance
- **Variable speed control:** Mamdani fuzzy logic scales the robot speed by distance; full stop within 0.4 m
- **Voice alerts:** "Excuse me, delivery robot passing" (person within 3 m) and an arrival announcement
- **Web dashboard:** live map, robot position, origin/destination, route, ETA, person alert, PIN-protected dispatch
- **Delivery log:** every delivery saved to a CSV file

## System Overview
```
move_base (A* + DWA) --/nav_vel--> yolo_fuzzy_speed (speed x fuzzy factor) --/cmd_vel--> robot base
                                          ^
                          RGB-D camera (colour + depth)
delivery.py (Flask dashboard) --> move_base goal, /delivery_active
```

## Hardware and Software
| Item | Details |
|---|---|
| Robot | Jupiter2 (RPLIDAR, RGB-D camera, wheel encoders + IMU, touchscreen, speaker) |
| OS / ROS | Ubuntu 20.04, ROS Noetic |
| Language | Python 3.8, HTML/CSS/JavaScript |
| Libraries | GMapping, ROS Navigation stack, actionlib, tf2, OpenCV, cv_bridge, Ultralytics YOLO11, PyTorch (CPU), scikit-fuzzy, Flask, espeak |

## Package Structure
```
fyp_cz/
├── launch/
│   └── fyp_navigation.launch
├── scripts/
│   ├── yolo_fuzzy_speed.py
│   ├── delivery.py
│   ├── plot_map.py
│   └── extract_landmarks.py
├── web/
│   ├── delivery.html
│   └── icon.png
├── maps/
│   ├── my_map.pgm / my_map.yaml
│   ├── waypoints.png / waypoints.yaml
│   └── landmarks.png
├── delivery_logs/
├── LICENSE
├── rosgraph.png
├── package.xml
└── CMakeLists.txt
```

## Installation
```bash
cd ~/catkin_ws/src
git clone https://github.com/chizhenn/Autonomous-Indoor-Logistics-Robot-for-Laboratory-Materials-Delivery.git fyp_cz
cd ~/catkin_ws && catkin_make && source devel/setup.bash
chmod +x src/fyp_cz/scripts/*.py
pip3 install ultralytics scikit-fuzzy flask
sudo apt install espeak
```
Requires the Jupiter2 packages (`jupiterobot2_bringup`, `jupiterobot2_navigation`, `jupiterobot2_teleop`).

## Usage
**1. Mapping (once)**
```bash
roslaunch jupiterobot2_bringup jupiterobot2_bringup.launch
roslaunch jupiterobot2_navigation jupiterobot2_slam.launch
roslaunch jupiterobot2_teleop joystick_teleop.launch
rosrun map_server map_saver -f ~/catkin_ws/src/fyp_cz/maps/my_map
```

**2. Record waypoints:** drive to each room door, face the room, then run
`rostopic echo -n1 /amcl_pose` and add x, y, z, w to `maps/waypoints.yaml`.

**3. Delivery system (one terminal each)**
```bash
roslaunch jupiterobot2_bringup jupiterobot2_bringup.launch   # 1. robot base
# 2. start the cameras (multi_camera.launch)
roslaunch fyp_cz fyp_navigation.launch                       # 3. then set 2D Pose Estimate in RViz
# 4. teleop (optional, to help localisation)
cd ~/catkin_ws && rosrun fyp_cz yolo_fuzzy_speed.py          # 5. human detection + fuzzy speed
rosrun fyp_cz delivery.py                                    # 6. dashboard (opens Firefox)
```
Dashboard: `http://localhost:5000` on the robot, or `http://<robot-ip>:5000` on the same Wi-Fi.

## Key Settings
| Setting | Value |
|---|---|
| Obstacle clearance | 0.15 m footprint padding |
| Max speed | 0.26 m/s |
| Detection rate | 5 Hz (CPU, YOLO11n, 320 px input) |
| Safety stop | person within 0.4 m |
| Voice alert | person within 3 m, during delivery only |

## Author
Ooi Chien Zhen

Supervisor: Dr. Zati Hakim Azizul Hasan

## License
MIT License
