"""
Save a cropped + zoomed view of just the specimen area for easy review.
"""
import cv2
import numpy as np

DIAG = r"C:\Users\smada\VisualAnalyser\VisualAnalyser\diag_1_120.png"
OUT  = r"C:\Users\smada\VisualAnalyser\VisualAnalyser\diag_1_120_zoom.png"

vis = cv2.imread(DIAG)
h, w = vis.shape[:2]

# Crop: show from ~70% of height downward, full width
# (this covers the specimen + brackets above it)
y0 = int(h * 0.68)
y1 = h
crop = vis[y0:y1, :]

# Scale up 2× so it's easier to read
zoomed = cv2.resize(crop, (crop.shape[1], crop.shape[0] * 2), interpolation=cv2.INTER_LINEAR)

cv2.imwrite(OUT, zoomed)
print(f"Saved zoomed crop → {OUT}  size: {zoomed.shape[1]}x{zoomed.shape[0]}")
