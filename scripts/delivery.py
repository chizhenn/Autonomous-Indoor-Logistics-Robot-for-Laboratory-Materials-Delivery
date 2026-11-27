#!/usr/bin/env python3
"""
delivery.py
Web delivery dashboard for the delivery robot (Flask + ROS):
  1. Serves delivery.html at http://<robot-ip>:5000 (opens Firefox on the robot automatically)
  2. User picks a destination + PIN -> sends the goal to move_base
  3. Shows live status: robot position, route, live obstacles, speed, ETA, person alert
  4. Logs the result (SUCCESSFUL / FAILED / CANCELLED) and announces arrival on the robot speaker
  5. "Close Delivery Service" (PIN) stops the delivery and shuts down this node
  6. Saves every delivery record to delivery_logs/delivery_log_<date>_<time>.csv
  7. After STOP, the user can mark the delivery as arrived (robot stuck at the destination)
Safety: PIN required (5 wrong tries = 60 s lock), one delivery at a time, STOP needs no PIN
Inputs : /amcl_pose, /odom, /move_base/NavfnROS/plan, /move_base/local_costmap/costmap, /person_detected
Outputs: move_base goal, /delivery_active
Files  : maps/waypoints.yaml, maps/my_map.yaml, web/delivery.html, web/icon.png
Logs   : delivery_logs/ (one CSV per service session)
Run: rosrun fyp_cz delivery.py
"""
import os, math, time, threading, logging, subprocess, signal, csv, datetime, socket, yaml, cv2, numpy as np
import rospy, rospkg, actionlib, tf2_ros
from flask import Flask, jsonify, request, Response, send_file
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, Path, OccupancyGrid
from std_msgs.msg import Float32, Bool

# ---------------- Settings ----------------
PIN = "8977"            # <-- change this
PORT = 5000
DEFAULT_SPEED = 0.15    # m/s, ETA guess before robot moves
MAX_PIN_TRIES = 5       # lock for 60 s after 5 wrong PINs
OPEN_BROWSER = True     # open Firefox on the robot at start

# ---------------- Load map + waypoints ----------------
pkg = rospkg.RosPack().get_path('fyp_cz')
HTML_FILE = os.path.join(pkg, 'web', 'delivery.html')

# ---------------- Delivery log (CSV) ----------------
LOG_DIR = os.path.join(pkg, 'delivery_logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, datetime.datetime.now().strftime('delivery_log_%Y-%m-%d_%H-%M-%S.csv'))
LOG_HEADER = ['No.', 'Date', 'Start Time', 'End Time', 'Origin', 'Destination', 'Result',
              'Duration (s)', 'Planned Route (m)', 'Distance Travelled (m)',
              'Average Speed (m/s)', 'Closest Person (m)', 'Safety Stops']
log_count = {'n': 0}

wp = yaml.safe_load(open(os.path.join(pkg, 'maps', 'waypoints.yaml')))
NAMES = {k.lower(): k for k in wp}          # case-insensitive lookup
mp = yaml.safe_load(open(os.path.join(pkg, 'maps', 'my_map.yaml')))
img = mp['image'] if os.path.isabs(mp['image']) else os.path.join(pkg, 'maps', mp['image'])
raw = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
RES = mp['resolution']; OX, OY = mp['origin'][0], mp['origin'][1]
H = raw.shape[0]

# crop away the empty (unknown) border
known = np.argwhere(raw != 205)
y0, x0 = map(int, known.min(0)); y1, x1 = map(int, known.max(0))
y0, x0 = max(0, y0 - 20), max(0, x0 - 20)
y1, x1 = min(raw.shape[0] - 1, y1 + 20), min(raw.shape[1] - 1, x1 + 20)
crop = raw[y0:y1 + 1, x0:x1 + 1]

# light-theme map image (BGR)
col = np.full((crop.shape[0], crop.shape[1], 3), (243, 240, 238), np.uint8)  # unknown
col[crop >= 250] = (255, 255, 255)                                           # free space
col[crop <= 50] = (81, 65, 55)                                               # walls
MAP_PNG = cv2.imencode('.png', col)[1].tobytes()

def to_px(x, y):
    """map metres -> cropped image pixels"""
    return [round((x - OX) / RES - x0, 1), round(H - (y - OY) / RES - y0, 1)]

def on_saved_wall(mx, my):
    """True if (mx, my) is already a wall on the saved map (within 2 pixels)."""
    px, py = int((mx - OX) / RES), int(H - (my - OY) / RES)
    if 0 <= py < raw.shape[0] and 0 <= px < raw.shape[1]:
        return bool((raw[max(0, py - 2):py + 3, max(0, px - 2):px + 3] <= 50).any())
    return False

# ---------------- Shared state ----------------
lock = threading.Lock()
S = {'pose': None, 'speed': 0.0, 'path': [], 'status': 'Idle',
     'origin': None, 'origin_xy': None, 'dest': None, 'start_t': 0.0,
     'avg': DEFAULT_SPEED, 'still_since': None, 'person': None, 'person_t': 0.0,
     'obstacles': [], 'obst_t': 0.0,
     'start_dt': None, 'travel': 0.0, 'odom_t': None, 'plan_len': None,
     'min_person': None, 'stops': 0, 'in_stop': False, 'log_no': None}
fails = {'n': 0, 'until': 0.0}
client = None
active_pub = None
tf_buf = None

def announce(text):
    """Speak on the robot's own speaker."""
    try:
        subprocess.Popen(['espeak', '-v', 'en-gb+f4', '-s', '145', '-g', '5', text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        rospy.logwarn('espeak not installed: announcement skipped.')

def on_pose(m):
    p = m.pose.pose
    S['pose'] = (p.position.x, p.position.y, 2 * math.atan2(p.orientation.z, p.orientation.w))

def on_odom(m):
    v = abs(m.twist.twist.linear.x)
    S['speed'] = v
    now = time.time()
    if S['status'] == 'Delivering' and S['odom_t'] is not None:
        S['travel'] += v * min(now - S['odom_t'], 1.0)      # distance travelled (m)
    S['odom_t'] = now
    if S['status'] == 'Delivering':
        if v > 0.02:
            S['avg'] = 0.98 * S['avg'] + 0.02 * v     # smooth average speed
            S['still_since'] = None
        elif S['still_since'] is None:
            S['still_since'] = time.time()

def on_plan(m):
    S['path'] = [(p.pose.position.x, p.pose.position.y) for p in m.poses]
    if S['status'] == 'Delivering' and S['plan_len'] is None and len(S['path']) > 1:
        S['plan_len'] = sum(math.hypot(b[0] - a[0], b[1] - a[1])
                            for a, b in zip(S['path'], S['path'][1:]))   # first planned route (m)

def on_person(m):
    S['person'], S['person_t'] = m.data, time.time()
    if S['status'] == 'Delivering':
        d = m.data
        if d >= 0 and (S['min_person'] is None or d < S['min_person']):
            S['min_person'] = d                              # closest person this delivery
        stopped = 0 <= d <= 0.4
        if stopped and not S['in_stop']:
            S['stops'] += 1                                  # count each safety stop once
        S['in_stop'] = stopped

def on_costmap(g):
    """Live obstacles = lethal cells in the local costmap that are NOT on the saved map."""
    try:
        t = tf_buf.lookup_transform('map', g.header.frame_id, rospy.Time(0), rospy.Duration(0.2))
    except Exception:
        return
    tx, ty = t.transform.translation.x, t.transform.translation.y
    q = t.transform.rotation
    yaw = 2 * math.atan2(q.z, q.w)
    cy, sy = math.cos(yaw), math.sin(yaw)
    res, w = g.info.resolution, g.info.width
    gx, gy = g.info.origin.position.x, g.info.origin.position.y
    data = np.asarray(g.data, dtype=np.int16).reshape(g.info.height, w)
    rows, cols = np.nonzero(data == 100)                 # 100 = lethal obstacle
    pts = []
    for r, cc in zip(rows, cols):
        lx, ly = gx + (cc + 0.5) * res, gy + (r + 0.5) * res
        mx, my = tx + cy * lx - sy * ly, ty + sy * lx + cy * ly
        if not on_saved_wall(mx, my):
            pts.append(to_px(mx, my))
    S['obstacles'], S['obst_t'] = pts[:3000], time.time()

def done_cb(goal_state, result):
    with lock:
        if goal_state == GoalStatus.SUCCEEDED:
            S['status'], res = 'Arrived', 'SUCCESSFUL'
            announce(f"Your delivery for {S['dest']} is here. Please collect your items.")
        elif goal_state in (GoalStatus.PREEMPTED, GoalStatus.RECALLED):
            S['status'], res = 'Cancelled', 'CANCELLED (stopped by user)'
        else:
            S['status'], res = 'Failed', 'FAILED'
        S['path'] = []
        dur = time.time() - S['start_t']
        write_log(res.split(' ')[0], dur)                   # save to CSV log
    active_pub.publish(False)
    rospy.loginfo(f"[DELIVERY] Result : {res} | {S['origin']} -> {S['dest']} | {dur:.0f} s")

def write_log(result, dur):
    """Append one delivery record to the CSV log."""
    log_count['n'] += 1
    S['log_no'] = log_count['n']                             # remember this record (for "mark as arrived")
    end = datetime.datetime.now()
    start = S['start_dt'] or end
    f = lambda v: '' if v is None else f'{v:.2f}'
    row = [log_count['n'], start.strftime('%Y-%m-%d'), start.strftime('%H:%M:%S'), end.strftime('%H:%M:%S'),
           S['origin'], S['dest'], result, f'{dur:.0f}', f(S['plan_len']), f(S['travel']),
           f(S['travel'] / dur if dur > 0 else 0), f(S['min_person']), S['stops']]
    try:
        new = not os.path.exists(LOG_FILE)
        with open(LOG_FILE, 'a', newline='') as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(LOG_HEADER)
            w.writerow(row)
    except Exception as e:
        rospy.logwarn(f'Could not write delivery log: {e}')

def set_log_result(no, result):
    """Change the Result of record number <no> in the CSV log."""
    try:
        with open(LOG_FILE, newline='') as fh:
            rows = list(csv.reader(fh))
        for r in rows[1:]:
            if r and r[0] == str(no):
                r[6] = result
        with open(LOG_FILE, 'w', newline='') as fh:
            csv.writer(fh).writerows(rows)
        return True
    except Exception as e:
        rospy.logwarn(f'Could not update delivery log: {e}')
        return False

def origin_name(x, y):
    best = min(wp, key=lambda k: math.hypot(wp[k]['x'] - x, wp[k]['y'] - y))
    return best if math.hypot(wp[best]['x'] - x, wp[best]['y'] - y) < 0.5 else 'Start'

def remaining(path, x, y):
    if len(path) < 2:
        return None
    i = min(range(len(path)), key=lambda k: (path[k][0] - x) ** 2 + (path[k][1] - y) ** 2)
    d = math.hypot(path[i][0] - x, path[i][1] - y)
    for a, b in zip(path[i:], path[i + 1:]):
        d += math.hypot(b[0] - a[0], b[1] - a[1])
    return d

def open_browser():
    env = dict(os.environ); env.setdefault('DISPLAY', ':0')
    env['GTK_IM_MODULE'] = 'ibus'      # use IBus so the screen keyboard pops up
    env['GNOME_ACCESSIBILITY'] = '1'   # Firefox tells Ubuntu when a text box is tapped
    env['MOZ_USE_XINPUT2'] = '1'       # proper touchscreen input
    try:
        subprocess.Popen(['firefox', '--kiosk', f'http://localhost:{PORT}'], env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        rospy.logwarn(f'Firefox not found. Open http://localhost:{PORT} manually.')

def check_pin(pin):
    """None if the PIN is correct, otherwise an error response."""
    if time.time() < fails['until']:
        return jsonify(ok=False, msg='Too many incorrect attempts. Please wait 60 seconds.'), 429
    if str(pin or '').strip() != PIN:
        fails['n'] += 1
        if fails['n'] >= MAX_PIN_TRIES:
            fails.update(n=0, until=time.time() + 60)
        return jsonify(ok=False, msg='Incorrect PIN. Please try again.'), 403
    fails['n'] = 0
    return None

def terminal_shell():
    """Find the shell of the terminal tab that started this script, so the tab can be closed."""
    def info(pid):
        try:
            name = open(f'/proc/{pid}/comm').read().strip()
            ppid = int(open(f'/proc/{pid}/stat').read().rsplit(')', 1)[1].split()[1])
            return name, ppid
        except Exception:
            return None, None
    pid = os.getppid()
    for _ in range(4):
        name, ppid = info(pid)
        if name is None:
            return None
        parent, _ = info(ppid)
        if name in ('bash', 'zsh', 'sh') and parent and 'terminal' in parent:
            return pid
        pid = ppid
    return None

def close_service():
    rospy.loginfo('[DELIVERY] Delivery service closed from the dashboard.')
    shell = terminal_shell()
    if shell:
        os.kill(shell, signal.SIGHUP)      # closes the delivery terminal tab
    os._exit(0)

# ---------------- Web server ----------------
app = Flask(__name__)
logging.getLogger('werkzeug').setLevel(logging.ERROR)   # keep terminal quiet

@app.route('/')
def index():
    r = send_file(HTML_FILE)
    r.headers['Cache-Control'] = 'no-store'
    return r

@app.route('/map.png')
def map_png():
    return Response(MAP_PNG, mimetype='image/png')

@app.route('/icon.png')
def icon_png():
    return send_file(os.path.join(pkg, 'web', 'icon.png'), mimetype='image/png')

@app.route('/manifest.json')
def manifest():
    return jsonify(name='Laboratory Delivery Robot', short_name='Lab Delivery',
                   start_url='/', display='standalone',
                   background_color='#f5f7fa', theme_color='#1e3a8a',
                   icons=[{'src': '/icon.png', 'sizes': '512x512', 'type': 'image/png'}])

@app.route('/api/info')
def info():
    return jsonify(waypoints=[{'name': k, 'px': to_px(v['x'], v['y'])}
                              for k, v in sorted(wp.items())])

@app.route('/api/state')
def state():
    pose = S['pose']
    out = {'status': S['status'], 'speed': round(S['speed'], 2),
           'robot': (to_px(pose[0], pose[1]) + [pose[2]]) if pose else None,
           'path': [to_px(x, y) for x, y in S['path'][::5]],
           'obstacles': S['obstacles'] if time.time() - S['obst_t'] < 3.0 else [],
           'origin': None, 'dest': None, 'eta': None, 'person': None}
    if S['dest']:
        out['origin'] = {'name': S['origin'], 'px': to_px(*S['origin_xy'])}
        out['dest'] = {'name': S['dest'], 'px': to_px(wp[S['dest']]['x'], wp[S['dest']]['y'])}
    if S['status'] == 'Delivering' and pose:
        rem = remaining(S['path'], pose[0], pose[1])
        if S['still_since'] and time.time() - S['still_since'] > 1.5:
            out['eta'] = 'paused'
        elif rem is None:
            out['eta'] = 'calculating'
        else:
            out['eta'] = int(rem / max(S['avg'], 0.05))
    if S['person'] is not None and S['person'] >= 0 and time.time() - S['person_t'] < 1.0:
        out['person'] = round(S['person'], 2)
    return jsonify(out)

@app.route('/api/go', methods=['POST'])
def go():
    d = request.get_json(silent=True) or {}
    bad = check_pin(d.get('pin'))
    if bad:
        return bad
    name = NAMES.get(str(d.get('dest', '')).strip().lower())
    if name is None:
        return jsonify(ok=False, msg='Please select a valid destination.'), 400
    if S['status'] == 'Delivering':
        return jsonify(ok=False, msg='The robot is on a delivery. Stop it before starting a new one.'), 409
    if S['pose'] is None:
        return jsonify(ok=False, msg='Robot location unknown. Set the initial pose in RViz first.'), 409
    if not client.wait_for_server(rospy.Duration(1.0)):
        return jsonify(ok=False, msg='Navigation system is not running.'), 503
    p = wp[name]
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = 'map'
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position.x = p['x']
    goal.target_pose.pose.position.y = p['y']
    goal.target_pose.pose.orientation.z = p['z']
    goal.target_pose.pose.orientation.w = p['w']
    with lock:
        S.update(status='Delivering', origin=origin_name(*S['pose'][:2]),
                 origin_xy=S['pose'][:2], dest=name, path=[], still_since=None,
                 start_t=time.time(), start_dt=datetime.datetime.now(), travel=0.0,
                 plan_len=None, min_person=None, stops=0, in_stop=False, log_no=None)
    active_pub.publish(True)
    client.send_goal(goal, done_cb=done_cb)
    rospy.loginfo(f"[DELIVERY] Started: {S['origin']} -> {name}")
    return jsonify(ok=True, msg=f'Delivery to {name} has started.')

@app.route('/api/cancel', methods=['POST'])
def cancel():
    client.cancel_all_goals()          # STOP never needs a PIN
    return jsonify(ok=True, msg='Stop command sent. The robot is stopping.')

@app.route('/api/arrived', methods=['POST'])
def arrived():
    """After STOP: the user confirms the robot had already reached the destination."""
    for _ in range(30):                               # wait up to 3 s for the stop to be logged
        if S['status'] != 'Delivering' and S['log_no']:
            break
        time.sleep(0.1)
    with lock:
        if S['status'] == 'Arrived':
            return jsonify(ok=True, msg=f"Delivery to {S['dest']} completed successfully.")
        if S['status'] not in ('Cancelled', 'Failed') or not S['log_no']:
            return jsonify(ok=False, msg='No stopped delivery to update.'), 409
        set_log_result(S['log_no'], 'SUCCESSFUL (marked arrived by user)')
        S['status'] = 'Arrived'
    announce(f"Your delivery for {S['dest']} is here. Please collect your items.")
    rospy.loginfo(f"[DELIVERY] Result : SUCCESSFUL (marked arrived by user) | {S['origin']} -> {S['dest']}")
    return jsonify(ok=True, msg=f"Delivery to {S['dest']} marked as arrived.")

@app.route('/api/shutdown', methods=['POST'])
def shutdown():
    d = request.get_json(silent=True) or {}
    bad = check_pin(d.get('pin'))
    if bad:
        return bad
    client.cancel_all_goals()          # stop any active delivery
    active_pub.publish(False)          # voice alerts off
    threading.Timer(1.0, close_service).start()   # reply to the page first, then exit
    return jsonify(ok=True, msg='Delivery Service Closed')

# ---------------- Main ----------------
if __name__ == '__main__':
    rospy.init_node('delivery', disable_signals=True)
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    active_pub = rospy.Publisher('/delivery_active', Bool, queue_size=1, latch=True)
    active_pub.publish(False)
    tf_buf = tf2_ros.Buffer()
    tf2_ros.TransformListener(tf_buf)
    rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, on_pose)
    rospy.Subscriber('/odom', Odometry, on_odom)
    rospy.Subscriber('/move_base/NavfnROS/plan', Path, on_plan)
    rospy.Subscriber('/person_detected', Float32, on_person)
    rospy.Subscriber('/move_base/local_costmap/costmap', OccupancyGrid, on_costmap,
                     queue_size=1, buff_size=2**22)

    print(f'Delivery log file: {LOG_FILE}')
    print('Delivery dashboard running:')
    print(f'  On the robot : http://localhost:{PORT}')
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))          # sends nothing; only finds the Wi-Fi address in use
        print(f'  Other devices: http://{s.getsockname()[0]}:{PORT}')
        s.close()
    except Exception:
        print('  Other devices: not connected to Wi-Fi')
    if OPEN_BROWSER:
        threading.Timer(2.0, open_browser).start()
    app.run(host='0.0.0.0', port=PORT, threaded=True)
