#!/usr/bin/env python3
"""
plot_map.py
Plots the saved map with all delivery waypoints:
  1. Loads the map (my_map.yaml -> my_map.pgm) and the waypoints (waypoints.yaml)
  2. Crops away the empty grey border so the corridor is bigger
  3. Marks each waypoint with a number (left to right along the corridor)
  4. Lists the numbers with their room names beside the map (no overlapping labels)
Output: maps/waypoints_map.png
Not a ROS node: offline tool, no topics.
Run: rosrun fyp_cz plot_map.py
"""
import os, yaml, rospkg
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

pkg = rospkg.RosPack().get_path('fyp_cz')

# load map
m = yaml.safe_load(open(os.path.join(pkg, 'maps', 'my_map.yaml')))
img_path = m['image'] if os.path.isabs(m['image']) else os.path.join(pkg, 'maps', m['image'])
img = np.array(Image.open(img_path))
res, ox, oy = m['resolution'], m['origin'][0], m['origin'][1]

# load waypoints, numbered left to right along the corridor
wp = yaml.safe_load(open(os.path.join(pkg, 'maps', 'waypoints.yaml')))
items = sorted(wp.items(), key=lambda kv: kv[1]['x'])

# crop away the unknown (grey) border
known = np.argwhere(img != 205)
y0, x0 = known.min(0) - 20
y1, x1 = known.max(0) + 20
y0, x0 = max(0, y0), max(0, x0)

fig, ax = plt.subplots(figsize=(14, 6))
ax.imshow(img[y0:y1, x0:x1], cmap='gray', vmin=0, vmax=255)
legend = []
for i, (name, p) in enumerate(items, 1):
    px = (p['x'] - ox) / res - x0
    py = img.shape[0] - (p['y'] - oy) / res - y0
    dy = -14 if p['z'] >= 0 else 14             # number above or below, by corridor side
    ax.plot(px, py, 'o', color='red', ms=6)
    ax.annotate(str(i), (px, py), (px, py + dy), ha='center', va='center',
                fontsize=8, fontweight='bold', color='white',
                bbox=dict(boxstyle='circle,pad=0.25', fc='red', ec='none'))
    legend.append(f'{i:>2}. {name}')
ax.set_title('Delivery Waypoints - Department of Parasitology')
ax.axis('off')

# numbered room list beside the map
fig.text(0.80, 0.5, '\n'.join(legend), va='center', fontsize=9, family='monospace')
plt.subplots_adjust(left=0.02, right=0.78)

out = os.path.join(pkg, 'maps', 'waypoints_map.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print("Saved:", out)
plt.show()