"""
One-shot diagnostic: auto-detect parameters on sample 1_120.png
and save an annotated PNG showing ROI, x_left, x_right, x_tip,
W_full bracket (blue) and W_lig bracket (orange).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np
from pathlib import Path
from crack_analyser import (
    load_image, compute_otsu_threshold,
    detect_sample_row_range, detect_sample_edges, detect_crack_tip
)

SCALE = 0.01587          # mm/px (same as before — adjust if different)
IMG_PATH = r"C:\Users\smada\OneDrive\שולחן העבודה\trial image processing for paper 2\pics for pyt\sample 1_120.png"
OUT_PATH = r"C:\Users\smada\VisualAnalyser\VisualAnalyser\diag_1_120.png"

img = load_image(IMG_PATH)
h, w = img.shape
print(f"Image size: {w} x {h} px")

thresh = compute_otsu_threshold(img)
print(f"Otsu threshold: {thresh}")

row_top, row_bottom = detect_sample_row_range(img)
# Apply 98% floor on ROI bottom (same as process_all_frames)
row_bottom = max(row_bottom, int(h * 0.98))
print(f"ROI  → row_top={row_top} ({row_top/h*100:.1f}%)  row_bottom={row_bottom} ({row_bottom/h*100:.1f}%)")

x_left, x_right = detect_sample_edges(img, row_top, row_bottom, thresh)
W_full_px = x_right - x_left
W_full_mm = W_full_px * SCALE
print(f"Edges → x_left={x_left}, x_right={x_right}  W_full = {W_full_mm:.3f} mm")

x_tip, conf = detect_crack_tip(img, x_left, x_right, row_top, row_bottom, 'left', thresh)
W_lig_px = x_right - x_tip
W_lig_mm = W_lig_px * SCALE
print(f"Tip   → x_tip={x_tip}  conf={conf:.1f}  W_lig = {W_lig_mm:.3f} mm")

# ── Build colour overlay ───────────────────────────────────────────────────────
vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Red dashed ROI rectangle
def draw_dashed_hline(im, y, x0, x1, color, dash=20, gap=10):
    x = x0
    draw = True
    while x < x1:
        xe = min(x + (dash if draw else gap), x1)
        if draw:
            cv2.line(im, (x, y), (xe, y), color, 2)
        x = xe
        draw = not draw

def draw_dashed_vline(im, x, y0, y1, color, dash=20, gap=10):
    y = y0
    draw = True
    while y < y1:
        ye = min(y + (dash if draw else gap), y1)
        if draw:
            cv2.line(im, (x, y), (x, ye), color, 2)
        y = ye
        draw = not draw

RED  = (0, 0, 255)
draw_dashed_hline(vis, row_top,    x_left, x_right, RED)
draw_dashed_hline(vis, row_bottom, x_left, x_right, RED)
draw_dashed_vline(vis, x_left,  row_top, row_bottom, RED)
draw_dashed_vline(vis, x_right, row_top, row_bottom, RED)

# Blue vertical lines for x_left / x_right (sample edges)
BLUE = (255, 100, 0)
cv2.line(vis, (x_left,  0), (x_left,  h-1), BLUE, 3)
cv2.line(vis, (x_right, 0), (x_right, h-1), BLUE, 3)

# Cyan vertical line for crack tip
CYAN = (255, 200, 0)
cv2.line(vis, (x_tip, row_top), (x_tip, row_bottom), CYAN, 3)

# ── Measurement brackets ───────────────────────────────────────────────────────
brac_y_full = row_top - 40
brac_y_lig  = row_top - 80
thick = 3; tip_h = 20

# W_full bracket (blue) ← x_left to x_right
cv2.line(vis, (x_left,  brac_y_full - tip_h//2), (x_left,  brac_y_full + tip_h//2), BLUE, thick)
cv2.line(vis, (x_right, brac_y_full - tip_h//2), (x_right, brac_y_full + tip_h//2), BLUE, thick)
cv2.line(vis, (x_left,  brac_y_full), (x_right, brac_y_full), BLUE, thick)
label_W = f"W_full={W_full_mm:.2f}mm"
cv2.putText(vis, label_W, ((x_left+x_right)//2 - 120, brac_y_full - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, BLUE, 2)

# W_lig bracket (orange) ← x_tip to x_right
ORANGE = (0, 165, 255)
cv2.line(vis, (x_tip,   brac_y_lig - tip_h//2), (x_tip,   brac_y_lig + tip_h//2), ORANGE, thick)
cv2.line(vis, (x_right, brac_y_lig - tip_h//2), (x_right, brac_y_lig + tip_h//2), ORANGE, thick)
cv2.line(vis, (x_tip,   brac_y_lig), (x_right, brac_y_lig), ORANGE, thick)
label_L = f"W_lig={W_lig_mm:.2f}mm"
cv2.putText(vis, label_L, ((x_tip+x_right)//2 - 100, brac_y_lig - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, ORANGE, 2)

# Legend top-left
cv2.putText(vis, "Red dashed = ROI", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, RED, 2)
cv2.putText(vis, "Blue lines = sample edges", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, BLUE, 2)
cv2.putText(vis, "Cyan line = crack tip", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, CYAN, 2)

cv2.imwrite(OUT_PATH, vis)
print(f"\nAnnotated image saved → {OUT_PATH}")
