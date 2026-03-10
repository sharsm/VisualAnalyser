"""
diagnose_edges.py
-----------------
Shows auto-detected x_left, x_right (full width) and x_tip (ligament)
for a grid of frames, with the session offset correction applied.
Saves a PNG grid so the user can inspect markings without the GUI.
"""

import cv2
import numpy as np
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
import sys

sys.path.insert(0, str(Path(__file__).parent))
from crack_analyser import (
    load_image, compute_otsu_threshold,
    detect_sample_row_range, detect_sample_edges, detect_crack_tip
)

# ── Config ────────────────────────────────────────────────────────────────────
FOLDER      = r"C:\Users\smada\OneDrive\שולחן העבודה\trial image processing for paper 2\pics for pyt\sample_1_11_1_26"
SCALE       = 0.01587   # mm/px
NOTCH_SIDE  = 'left'
SHOW_FRAMES = list(range(0, 50, 5))   # frames 0,5,10,15,…,45  (10 frames)

# ── Natural sort ──────────────────────────────────────────────────────────────
def _natural_key(p):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'(\d+)', p.stem)]

SUPPORTED_EXT = {'.png', '.tif', '.tiff', '.bmp', '.jpg', '.jpeg'}
folder = Path(FOLDER)
all_paths = sorted(
    [p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_EXT],
    key=_natural_key
)
print(f'Total frames in folder: {len(all_paths)}')

# ── Load session offsets ──────────────────────────────────────────────────────
session_path = folder / '_va_session.json'
offset_xl = offset_xr = 0
session_tips = {}
if session_path.exists():
    data = json.load(open(session_path))
    if '_x_left' in data or '_x_right' in data:
        # We'll compute offsets after first-frame auto-detection
        session_xl_corrected = data.get('_x_left', None)
        session_xr_corrected = data.get('_x_right', None)
    else:
        session_xl_corrected = session_xr_corrected = None
    for k, v in data.items():
        if not k.startswith('_'):
            session_tips[int(k)] = int(v)

# ── Detect from first frame ───────────────────────────────────────────────────
first_img  = load_image(all_paths[0])
threshold  = compute_otsu_threshold(first_img)
row_top, row_bot = detect_sample_row_range(first_img)
h0 = first_img.shape[0]
rt0 = min(row_top,  h0 - 1)
rb0 = max(min(row_bot, h0 - 1), int(h0 * 0.98))

xl0_auto, xr0_auto = detect_sample_edges(first_img, rt0, rb0, threshold)

# Compute offsets from session
if session_xl_corrected is not None:
    offset_xl = session_xl_corrected - xl0_auto
if session_xr_corrected is not None:
    offset_xr = session_xr_corrected - xr0_auto

print(f'Otsu threshold : {threshold}')
print(f'Auto x_left(0) : {xl0_auto}   offset = {offset_xl:+d}  → corrected = {xl0_auto + offset_xl}')
print(f'Auto x_right(0): {xr0_auto}   offset = {offset_xr:+d}  → corrected = {xr0_auto + offset_xr}')

# ── Build figure ──────────────────────────────────────────────────────────────
n_show  = len(SHOW_FRAMES)
n_cols  = 5
n_rows  = (n_show + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 5))
axes = np.array(axes).flatten()

for ax_i, fi in enumerate(SHOW_FRAMES):
    ax = axes[ax_i]

    if fi >= len(all_paths):
        ax.axis('off')
        continue

    img = load_image(all_paths[fi])
    h, w = img.shape

    rt = min(rt0, h - 1)
    rb = max(min(rb0, h - 1), int(h * 0.98))

    # Per-frame edge detection + offset correction
    xl_auto, xr_auto = detect_sample_edges(img, rt, rb, threshold)
    xl = int(np.clip(xl_auto + offset_xl, 0, w - 1))
    xr = int(np.clip(xr_auto + offset_xr, 0, w - 1))

    # Crack tip: session correction if available, else auto
    if fi in session_tips:
        xt = session_tips[fi]
        tip_src = 'session'
    else:
        xt, _ = detect_crack_tip(img, xl, xr, rt, rb, NOTCH_SIDE, threshold)
        tip_src = 'auto'

    # Crop to ROI for display
    pad = 30
    crop = img[max(0, rt - pad): min(h, rb + pad), :]
    y_offset = max(0, rt - pad)

    ax.imshow(crop, cmap='gray', aspect='auto', vmin=0, vmax=255)

    # Blue lines: x_left, x_right
    ax.axvline(xl, color='deepskyblue', lw=1.5, label='x_left')
    ax.axvline(xr, color='deepskyblue', lw=1.5, label='x_right')

    # Red line: x_tip
    ax.axvline(xt, color='red', lw=1.8,
               label=f'x_tip ({tip_src})')

    # ROI lines
    ax.axhline(rt - y_offset, color='orange', lw=1, ls='--', alpha=0.7)
    ax.axhline(rb - y_offset, color='orange', lw=1, ls='--', alpha=0.7)

    W_full = (xr - xl) * SCALE
    W_lig  = (xr - xt) * SCALE
    ax.set_title(
        f'Frame {fi}\n'
        f'W_full={W_full:.2f} mm  W_lig={W_lig:.2f} mm\n'
        f'xl={xl}  xr={xr}  xt={xt}',
        fontsize=7
    )
    ax.axis('off')

# Hide unused axes
for ax_i in range(len(SHOW_FRAMES), len(axes)):
    axes[ax_i].axis('off')

# Legend
patches = [
    mpatches.Patch(color='deepskyblue', label='x_left / x_right (blue)'),
    mpatches.Patch(color='red',         label='x_tip / crack tip (red)'),
    mpatches.Patch(color='orange',      label='ROI top/bottom (orange)'),
]
fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=9,
           bbox_to_anchor=(0.5, 0.01))

fig.suptitle(
    f'Auto-detection diagnostic — frames {SHOW_FRAMES[0]}–{SHOW_FRAMES[-1]}\n'
    f'x_left offset={offset_xl:+d} px  |  x_right offset={offset_xr:+d} px  |  '
    f'Threshold={threshold}',
    fontsize=10
)
fig.tight_layout(rect=[0, 0.05, 1, 0.97])

out = Path(__file__).parent / 'diag_edges.png'
fig.savefig(str(out), dpi=130)
print(f'\nSaved → {out}')
