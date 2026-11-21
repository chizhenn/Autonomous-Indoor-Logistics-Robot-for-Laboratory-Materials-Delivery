#!/usr/bin/env python3
"""
extract_landmarks.py
Extracts walls (straight line segments) from the saved map using the
Probabilistic Hough Transform (OpenCV HoughLinesP):
  1. Loads the map image (my_map.yaml -> my_map.pgm)
  2. Keeps only wall pixels, removes noise, thins walls to 1-pixel centre lines
  3. HoughLinesP finds straight wall segments (repeatable, gaps > 25 cm kept open)
  4. Removes duplicate lines, then plots the original map next to the
     extracted walls (red) and endpoints (blue)
Output: maps/landmarks.png
Not a ROS node: offline tool, no topics.
Run: rosrun fyp_cz extract_landmarks.py
"""
import os, yaml, rospkg
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')           # save image without opening a window
import matplotlib.pyplot as plt

# ---- Settings (map resolution 0.05 m/pixel) ----
MIN_WALL_M = 0.5    # shortest wall to keep (m)
MAX_GAP_M  = 0.25   # join breaks smaller than this (m); doorways stay open

# load map from package
pkg = rospkg.RosPack().get_path('fyp_cz')
with open(os.path.join(pkg, 'maps', 'my_map.yaml')) as f:
    m = yaml.safe_load(f)
img_path = m['image'] if os.path.isabs(m['image']) else os.path.join(pkg, 'maps', m['image'])
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
res = m['resolution']

# wall pixels -> white on black; remove isolated noise pixels
walls = (img < 50).astype(np.uint8) * 255
walls = cv2.morphologyEx(walls, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

# thin walls to 1-pixel centre lines (gives one clean line per wall)
def skeleton(b):
    k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    sk = np.zeros_like(b)
    while cv2.countNonZero(b):
        er = cv2.erode(b, k)
        sk |= cv2.subtract(b, cv2.dilate(er, k))
        b = er
    return sk
walls = skeleton(walls)

# Probabilistic Hough Transform
segs = cv2.HoughLinesP(walls, rho=1, theta=np.pi / 180, threshold=15,
                       minLineLength=int(MIN_WALL_M / res),
                       maxLineGap=int(MAX_GAP_M / res))
segs = [] if segs is None else segs.reshape(-1, 4).tolist()

# remove duplicates (overlapping lines on the same wall): keep the longest
def pt_seg_dist(p, a, b):
    p, a, b = map(np.array, (p, a, b))
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1)
    return np.linalg.norm(p - (a + t * ab))

segs.sort(key=lambda s: -np.hypot(s[2] - s[0], s[3] - s[1]))
lines = []
for x1, y1, x2, y2 in segs:
    dup = any(pt_seg_dist((x1, y1), k[:2], k[2:]) < 4 and pt_seg_dist((x2, y2), k[:2], k[2:]) < 4
              for k in lines)
    if not dup:
        lines.append((x1, y1, x2, y2))
print("Landmarks (walls) extracted:", len(lines))

# plot result
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(img, cmap='gray'); ax[0].set_title('Original Map'); ax[0].axis('off')
ax[1].imshow(img, cmap='gray', alpha=0.3)
for x1, y1, x2, y2 in lines:
    ax[1].plot([x1, x2], [y1, y2], 'r-', lw=2)
    ax[1].plot([x1, x2], [y1, y2], 'bo', ms=3)
ax[1].set_title('Landmark Extraction (Hough Transform)\nred = walls, blue = endpoints')
ax[1].axis('off')
plt.tight_layout()

out = os.path.join(pkg, 'maps', 'landmarks.png')
plt.savefig(out, dpi=110, bbox_inches='tight')
print("Saved:", out)