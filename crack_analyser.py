"""
VisualAnalyser – Crack Propagation Onset Detection
====================================================
Detects crack tip in sequential CCD images of tensile tests, computes
    a(λ)  = (1 - W_lig(λ) / W_full(λ)) · W_full,0
    Δa(λ) = a(λ) - a₀
and identifies the onset frame of crack propagation.

Expected image setup
--------------------
- Bright/grey sample on dark background
- Horizontal crack growing left→right (notch always on the LEFT)
- W_full,0 is auto-detected from the first frame
- Image threshold is auto-computed via Otsu's method
- Fixed parameters: fps = 5 Hz, onset threshold = 0.1 mm
- Only user input required at startup: image folder + mm/pixel scale factor
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import json

SUPPORTED_EXT = {'.png', '.tif', '.tiff', '.bmp', '.jpg', '.jpeg'}

# ── Image loading ─────────────────────────────────────────────────────────────

def load_image_paths(folder: str) -> list:
    """Return naturally-sorted list of image paths in folder.

    Uses numeric sort so that sample_1_2.png comes before sample_1_10.png,
    not after it (which is what plain alphabetical sort would give).

    If a _va_skip.txt file exists next to the images, any file whose stem
    appears in that file (one stem per line, # = comment) is excluded.
    """
    import re
    folder = Path(folder)
    paths = [p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_EXT]

    # Load skip list if present
    skip_file = folder / '_va_skip.txt'
    skip_stems = set()
    if skip_file.exists():
        for line in skip_file.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line and not line.startswith('#'):
                skip_stems.add(line.lower())
        if skip_stems:
            before = len(paths)
            paths = [p for p in paths if p.stem.lower() not in skip_stems]
            print(f'Skipping {before - len(paths)} frame(s) listed in _va_skip.txt.')

    def _natural_key(p):
        return [int(t) if t.isdigit() else t.lower()
                for t in re.split(r'(\d+)', p.stem)]

    return sorted(paths, key=_natural_key)


def load_image(path) -> np.ndarray:
    """Load image as 8-bit grayscale."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img


def compute_otsu_threshold(image: np.ndarray) -> int:
    """Return the optimal binary threshold via Otsu's method."""
    thresh_val, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(thresh_val)


# ── Sample geometry detection ─────────────────────────────────────────────────

def detect_sample_row_range(image: np.ndarray,
                             margin_frac: float = 0.05,
                             search_top_frac: float = 0.78) -> tuple:
    """
    Find the vertical extent of the specimen body.

    Setup: the specimen (hydrogel) is at the VERY BOTTOM of the image.
    The metal grip hardware occupies the upper ~78 % of the image.
    We therefore search only below `search_top_frac` of the image height,
    which skips both the upper grip and the large metal lower-grip plate.

    Within that search zone we find the largest contiguous band of rows
    whose mean intensity exceeds 60 % of the local maximum.  This threshold
    is high enough to separate the hydrogel specimen from the dark background
    below it while still capturing the full specimen height.

    Returns (row_top, row_bottom) with a small inward margin.
    """
    h = image.shape[0]
    search_start = int(h * search_top_frac)

    sub = image[search_start:, :]
    row_means = sub.mean(axis=1)
    local_max = row_means.max()

    if local_max == 0:
        return int(h * 0.80), int(h * 0.95)

    thresh = local_max * 0.60          # separates hydrogel from dark bottom
    bright = (row_means > thresh).astype(int)

    diff   = np.diff(bright, prepend=0, append=0)
    starts = np.where(diff ==  1)[0]
    ends   = np.where(diff == -1)[0]

    if len(starts) == 0:
        return int(h * 0.80), int(h * 0.95)

    lengths = ends - starts
    best    = int(np.argmax(lengths))
    top = starts[best] + search_start
    bot = ends[best]   + search_start

    margin = max(int((bot - top) * margin_frac), 2)
    return int(top + margin), int(bot - margin)


def detect_sample_edges(image: np.ndarray, row_top: int, row_bottom: int,
                         threshold: int = 60) -> tuple:
    """
    Find x_left and x_right of the specimen by scanning horizontal intensity
    profiles across the ROI rows (every ~40th row for speed).

    Returns (x_left, x_right) as the median of per-row first/last bright pixel.
    """
    x_lefts, x_rights = [], []
    step = max(1, (row_bottom - row_top) // 40)

    for r in range(row_top, row_bottom, step):
        row = image[r, :]
        bright = np.where(row > threshold)[0]
        if len(bright) >= 2:
            x_lefts.append(int(bright[0]))
            x_rights.append(int(bright[-1]))

    if not x_lefts:
        return 0, image.shape[1] - 1

    # Use the 5th-percentile for x_left (true specimen edge is the leftmost
    # bright pixel; the notch makes most rows appear to start at a larger x).
    # Use the 95th-percentile for x_right (symmetric reason: crack opening
    # on the right side would push the apparent right edge left).
    x_left  = int(np.percentile(x_lefts,  5))
    x_right = int(np.percentile(x_rights, 95))
    return x_left, x_right


def detect_crack_tip(image: np.ndarray, x_left: int, x_right: int,
                     row_top: int, row_bottom: int,
                     notch_side: str = 'left',
                     threshold: int = 60) -> tuple:
    """
    Detect the crack tip x-coordinate using a column-wise dark-fraction scan,
    followed by a row-wise confidence estimate.

    For notch_side='right': the notch is on the right, crack grows right→left.
      The crack tip is the leftmost extent of the contiguous dark region that
      starts at the right edge.
    For notch_side='left': the notch is on the left, crack grows left→right.
      The crack tip is the rightmost extent of the contiguous dark region that
      starts at the left edge.

    Returns
    -------
    x_tip       : int   – pixel x-coordinate of crack tip
    confidence  : float – std-dev of per-row estimates (px); lower = better
    """
    if x_right <= x_left:
        return (x_right if notch_side == 'right' else x_left), 9999.0

    sample = image[row_top:row_bottom, x_left:x_right]  # (n_rows, width)
    n_rows, width = sample.shape

    if width == 0 or n_rows == 0:
        return (x_right if notch_side == 'right' else x_left), 9999.0

    # ── Step 1: column-wise dark fraction ─────────────────────────────────
    dark_frac = (sample < threshold).mean(axis=0)   # shape: (width,)

    # Threshold: 2 % of ROI rows dark per column is sufficient to mark a
    # crack column.  The crack in a tall ROI occupies only a narrow band of
    # rows, so the per-column dark fraction is small even at the open crack.
    # Gap tolerance of 15 columns bridges the narrow crack tip where the
    # opening is only a few pixels tall.
    DARK_THRESH = 0.02
    GAP_TOL     = 15   # consecutive below-threshold columns still bridged

    if notch_side == 'right':
        # Scan from RIGHT to LEFT; find leftmost column of contiguous dark
        # region that is connected to the right edge.
        last_dark_col = width - 1   # stays at right if no crack found
        found = False
        for xi in range(width - 1, -1, -1):
            if dark_frac[xi] >= DARK_THRESH:
                found = True
                last_dark_col = xi
            elif found:
                look_left = dark_frac[max(0, xi - GAP_TOL): xi]
                if not (look_left >= DARK_THRESH).any():
                    break   # sustained brightness → crack tip found
        x_tip = x_left + last_dark_col

    else:  # notch_side == 'left'
        # Scan LEFT → RIGHT; find rightmost column of contiguous dark region.
        last_dark_col = 0
        found = False
        for xi in range(width):
            if dark_frac[xi] >= DARK_THRESH:
                found = True
                last_dark_col = xi
            elif found:
                look_right = dark_frac[xi + 1: xi + 1 + GAP_TOL]
                if not (look_right >= DARK_THRESH).any():
                    break
        x_tip = x_left + last_dark_col

    # ── Step 2: row-wise confidence estimate ──────────────────────────────
    row_tips = []
    step = max(1, n_rows // 40)

    for ri in range(0, n_rows, step):
        row = sample[ri, :]
        bright = row > threshold
        dark_in_row = (~bright).sum()

        # Skip rows with no meaningful dark pixels inside sample
        if dark_in_row < max(2, width * 0.03):
            continue

        if notch_side == 'right':
            # Find the leftmost bright pixel scanning from right
            for xi in range(width - 1, -1, -1):
                if bright[xi]:
                    end = min(xi + 4, width)
                    if bright[max(0, xi - 3): xi + 1].mean() > 0.5:
                        row_tips.append(xi + x_left)
                        break
        else:
            # Find rightmost bright pixel scanning from left
            for xi in range(width):
                if bright[xi]:
                    if bright[xi: min(xi + 4, width)].mean() > 0.5:
                        row_tips.append(xi + x_left)
                        break

    confidence = float(np.std(row_tips)) if len(row_tips) >= 3 else 9999.0
    return int(x_tip), confidence


# ── Batch processing ──────────────────────────────────────────────────────────

def process_all_frames(image_paths: list, notch_side: str = 'left',
                        threshold: int = None) -> pd.DataFrame:
    """
    Auto-detect crack tip, x_left and x_right for every frame.

    x_left and x_right are detected **per frame** so that W_full(λ) can vary
    with Poisson lateral contraction as required by the formula:
        a(λ) = (1 - W_lig(λ) / W_full(λ)) · W_full,0

    The vertical ROI (row_top, row_bottom) is detected from the first frame
    and kept fixed — the specimen does not move vertically between frames.

    If threshold is None, Otsu's method is applied to the first frame.

    Returns a DataFrame with one row per frame.
    """
    records = []
    row_top = row_bottom = None
    _threshold = threshold

    import re as _re
    def _frame_num(p):
        nums = _re.findall(r'\d+', Path(p).stem)
        return int(nums[-1]) if nums else pos

    for pos, path in enumerate(tqdm(image_paths, desc='Detecting crack tips', unit='frame')):
        i = _frame_num(path)
        img = load_image(path)

        if row_top is None:
            row_top, row_bottom = detect_sample_row_range(img)
            if _threshold is None:
                _threshold = compute_otsu_threshold(img)

        # Clamp ROI to current image height (frames may differ in size).
        # Always extend the bottom to at least 98 % of frame height so that
        # crack openings that grow downward in later frames are never cut off.
        h  = img.shape[0]
        rt = min(row_top,    h - 1)
        rb = max(min(row_bottom, h - 1), int(h * 0.98))

        # Detect x_left and x_right per frame (captures Poisson contraction).
        _x_left, _x_right = detect_sample_edges(img, rt, rb, _threshold)

        x_tip, conf = detect_crack_tip(img, _x_left, _x_right, rt, rb,
                                        notch_side, _threshold)

        records.append({
            'frame'            : i,
            'path'             : str(path),
            'x_left'           : _x_left,
            'x_right'          : _x_right,
            'x_left_raw'       : _x_left,    # never modified; used for rolling-median fallback
            'x_right_raw'      : _x_right,   # never modified; used for rolling-median fallback
            'x_tip'            : x_tip,
            'confidence'       : conf,
            'row_top'          : rt,
            'row_bottom'       : rb,
            'corrected'        : False,
            'x_left_corrected' : False,
            'x_right_corrected': False,
        })

    df = pd.DataFrame(records)
    df.index = df['frame']   # integer index == frame number
    return df


# ── Measurement calculation ───────────────────────────────────────────────────

def compute_measurements(df: pd.DataFrame, scale_mm_per_pixel: float,
                          W_full_0_mm: float, notch_side: str = 'left',
                          fps: float = 5.0) -> pd.DataFrame:
    """
    Add physical measurement columns to the DataFrame in-place (on a copy).

    Columns added: W_full_px, W_lig_px, W_full_mm, W_lig_mm, a_mm,
                   delta_a_mm, time_s
    """
    df = df.copy()

    df['W_full_px'] = (df['x_right'] - df['x_left']).clip(lower=1)

    if notch_side == 'right':
        # Ligament = from crack tip leftward to the free (left) edge
        df['W_lig_px'] = (df['x_tip'] - df['x_left']).clip(lower=0)
    else:
        # Ligament = from crack tip rightward to the free (right) edge
        df['W_lig_px'] = (df['x_right'] - df['x_tip']).clip(lower=0)

    df['W_full_mm'] = df['W_full_px'] * scale_mm_per_pixel
    df['W_lig_mm']  = df['W_lig_px']  * scale_mm_per_pixel

    df['a_mm'] = (
        (1.0 - df['W_lig_mm'] / df['W_full_mm'].clip(lower=1e-6)) * W_full_0_mm
    )

    a_0 = float(df['a_mm'].iloc[0])
    df['delta_a_mm'] = (df['a_mm'] - a_0).clip(lower=0)
    df['time_s']     = df['frame'] / fps

    return df


# ── Temporal smoothing ────────────────────────────────────────────────────────

def smooth_sample_edges(df: pd.DataFrame, window: int = 11) -> pd.DataFrame:
    """
    Smooth per-frame x_left and x_right with a rolling median.

    Edge detection is noisier than crack-tip detection, so a wider window
    (11 frames) is used.  Smoothing preserves the low-frequency Poisson
    contraction trend while suppressing frame-to-frame noise.
    """
    df = df.copy()
    for col in ('x_left', 'x_right'):
        df[col] = (pd.Series(df[col].values.astype(float))
                   .rolling(window=window, center=True, min_periods=1)
                   .median()
                   .round()
                   .astype(int)
                   .values)
    return df


def smooth_sample_edges_with_anchors(df: pd.DataFrame, window: int = 11) -> pd.DataFrame:
    """
    Smooth per-frame x_left and x_right using corrected frames as hard anchors.

    Strategy (mirrors smooth_crack_tips):
    1. Corrected frames (x_right_corrected / x_left_corrected = True) are kept exactly.
    2. Uncorrected frames *between* two anchors → linear interpolation between anchors.
    3. Uncorrected frames *outside* the anchor range → rolling median on the raw
       auto-detected values (x_right_raw / x_left_raw), which preserves the
       Poisson contraction trend while suppressing frame-to-frame noise.

    If the corrected-flag columns are missing (old DataFrame), falls back to a
    plain rolling median (same as the old smooth_sample_edges).
    """
    df = df.copy()

    for col, corr_col, raw_col in [
        ('x_right', 'x_right_corrected', 'x_right_raw'),
        ('x_left',  'x_left_corrected',  'x_left_raw'),
    ]:
        # ── Fallback: no anchor columns → plain rolling median ────────────
        if corr_col not in df.columns:
            df[col] = (pd.Series(df[col].values.astype(float))
                       .rolling(window=window, center=True, min_periods=1)
                       .median().round().astype(int).values)
            continue

        frames    = df.index.tolist()
        corrected = df[corr_col].values
        vals      = df[col].values.astype(float)
        n         = len(frames)

        anchors = [(pos, fi, int(vals[pos]))
                   for pos, fi in enumerate(frames) if corrected[pos]]

        # ── Step 1: linear interpolation between anchors ──────────────────
        if len(anchors) >= 2:
            for k in range(len(anchors) - 1):
                p0, f0, v0 = anchors[k]
                p1, f1, v1 = anchors[k + 1]
                for pos in range(p0 + 1, p1):
                    if not corrected[pos]:
                        frac = (frames[pos] - f0) / max(f1 - f0, 1)
                        vals[pos] = round(v0 + frac * (v1 - v0))

        # ── Step 2: rolling median on raw for frames outside anchor range ─
        raw_vals = (df[raw_col].values.astype(float)
                    if raw_col in df.columns else df[col].values.astype(float))
        smoothed = (pd.Series(raw_vals)
                    .rolling(window=window, center=True, min_periods=1)
                    .median().round().values)

        first_pos = anchors[0][0] if anchors else n
        last_pos  = anchors[-1][0] if anchors else -1

        for pos in range(n):
            if corrected[pos]:
                continue
            if pos < first_pos or pos > last_pos:
                vals[pos] = smoothed[pos]

        # ── Apply smoothed values ─────────────────────────────────────────
        for pos, fi in enumerate(frames):
            if not corrected[pos]:
                df.loc[fi, col] = int(vals[pos])

    return df


def smooth_crack_tips(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """
    Smooth x_tip using corrected frames as hard anchors.

    Strategy
    --------
    1. Frames already marked 'corrected' are never changed.
    2. Uncorrected frames that fall *between* two corrected frames are
       linearly interpolated from those anchors.  This handles long runs
       of bad auto-detections (e.g. 10 consecutive frames) that a rolling
       median cannot fix.
    3. Uncorrected frames that fall *outside* the corrected range (before
       the first anchor or after the last anchor) are smoothed with a
       rolling median (window=7) as a fallback.

    The raw (pre-smoothing) value is preserved in x_tip_raw.
    """
    df = df.copy()
    df['x_tip_raw'] = df['x_tip'].copy()

    frames    = df.index.tolist()          # integer frame numbers
    corrected = df['corrected'].values
    tips      = df['x_tip'].values.astype(float)
    n         = len(frames)

    # ── Step 1: anchor-based linear interpolation ─────────────────────────
    # Build list of (position, frame_index, x_tip) for corrected frames
    anchors = [(pos, fi, int(tips[pos]))
               for pos, fi in enumerate(frames) if corrected[pos]]

    if len(anchors) >= 2:
        for k in range(len(anchors) - 1):
            p0, f0, v0 = anchors[k]
            p1, f1, v1 = anchors[k + 1]
            # Fill uncorrected positions between the two anchors
            for pos in range(p0 + 1, p1):
                if not corrected[pos]:
                    # linear interpolation in frame-index space
                    fi   = frames[pos]
                    frac = (fi - f0) / max(f1 - f0, 1)
                    tips[pos] = round(v0 + frac * (v1 - v0))

    # ── Step 2: frames outside the corrected range ────────────────────────
    if anchors:
        first_anchor_pos  = anchors[0][0]
        first_anchor_val  = anchors[0][2]
        last_anchor_pos   = anchors[-1][0]
        last_anchor_val   = anchors[-1][2]
    else:
        first_anchor_pos  = n
        first_anchor_val  = None
        last_anchor_pos   = -1
        last_anchor_val   = None

    # Frames BEFORE the first anchor: rolling median on raw values
    raw = df['x_tip_raw'].values.astype(float)
    smoothed_raw = (pd.Series(raw)
                    .rolling(window=window, center=True, min_periods=1)
                    .median()
                    .round()
                    .values)

    for pos in range(n):
        if corrected[pos]:
            continue
        if pos < first_anchor_pos:
            # Pre-anchor region: rolling median is best we can do
            tips[pos] = smoothed_raw[pos]
        elif pos > last_anchor_pos:
            # Post-anchor region: hold the last corrected value constant.
            # This is correct for pre-onset frames (notch tip doesn't move)
            # and avoids the rolling-median artifact from bad raw detections.
            tips[pos] = last_anchor_val

    # ── Apply smoothed values ─────────────────────────────────────────────
    for pos, fi in enumerate(frames):
        if not corrected[pos]:
            df.loc[fi, 'x_tip'] = int(tips[pos])

    return df


# ── Flagging ──────────────────────────────────────────────────────────────────

def flag_uncertain_frames(df: pd.DataFrame,
                          deviation_thresh: int = 15,
                          jump_thresh: int = 10,
                          calibration_step: int = 50) -> list:
    """
    Return sorted list of frame indices that should be reviewed by the user.

    Rules
    -----
    - Frame 0 is always flagged (confirm initial notch tip = a₀).
    - Every calibration_step-th frame is flagged until manually corrected
      (periodic anchors so the user can calibrate x_tip and edges throughout).
    - Raw detection deviated > deviation_thresh px from smoothed value.
    - Smoothed x_tip jumps > jump_thresh px vs previous frame.
    """
    flagged = {0}

    if len(df) < 2:
        return sorted(flagged)

    raw_col     = 'x_tip_raw' if 'x_tip_raw' in df.columns else 'x_tip'
    tips_raw    = df[raw_col].values
    tips_smooth = df['x_tip'].values
    corrected   = df['corrected'].values
    frames      = df['frame'].values

    # Periodic calibration frames — flag until manually confirmed
    # Dense (every 20) for frames 0-400, coarse (every 50) for 401-1500
    calib_positions = [pos for pos, fi in enumerate(frames)
                       if (fi <= 400 and fi % 20 == 0) or
                          (400 < fi <= 1500 and fi % calibration_step == 0)]
    uncalibrated = [pos for pos in calib_positions if not corrected[pos]]
    for pos in uncalibrated:
        flagged.add(pos)

    # Uncertainty-based flagging only after all calibration frames are confirmed
    if uncalibrated:
        return sorted(flagged)

    for i in range(len(df)):
        if corrected[i]:
            continue   # user already reviewed this frame — skip
        # Raw detector was far from smoothed consensus
        if abs(int(tips_raw[i]) - int(tips_smooth[i])) > deviation_thresh:
            flagged.add(i)

    delta_a = df['delta_a_mm'].values if 'delta_a_mm' in df.columns else None

    for i in range(1, len(df)):
        if corrected[i]:
            continue
        # Smoothed tip jumped too fast (real event or residual error)
        if abs(int(tips_smooth[i]) - int(tips_smooth[i - 1])) > jump_thresh:
            flagged.add(i)
            flagged.add(i - 1)
        # Δa decreased vs previous frame (physically impossible — crack can't close)
        if delta_a is not None and delta_a[i] < delta_a[i - 1]:
            flagged.add(i)

    return sorted(flagged)


# ── Onset detection ───────────────────────────────────────────────────────────

def find_onset_frame(df: pd.DataFrame, threshold_mm: float = 0.01,
                     persistence: int = 5) -> int:
    """
    Return the frame index where Δa first persistently exceeds threshold_mm.

    A 5-frame moving median is applied first to suppress noise spikes.
    Returns -1 if onset is not detected.
    """
    try:
        from scipy.signal import medfilt
        kernel = min(5, len(df))
        kernel = kernel if kernel % 2 == 1 else kernel - 1   # must be odd
        smoothed = medfilt(df['delta_a_mm'].values, kernel_size=max(1, kernel))
    except ImportError:
        smoothed = df['delta_a_mm'].values

    n = len(smoothed)
    for i in range(n - persistence + 1):
        window = smoothed[i: i + persistence]
        if (window >= threshold_mm).all():
            return int(df['frame'].iloc[i])

    return -1


# ── Export ────────────────────────────────────────────────────────────────────

def export_csv(df: pd.DataFrame, output_path: str, onset_frame: int = -1):
    """Write the results CSV and print a summary to console."""
    out_cols = ['frame', 'time_s', 'W_full_mm', 'W_lig_mm',
                'a_mm', 'delta_a_mm', 'corrected']
    for col in out_cols:
        if col not in df.columns:
            df[col] = None

    df[out_cols].to_csv(output_path, index=False, float_format='%.6f')
    print(f'\nResults saved → {output_path}')


def export_annotated_tif(df: pd.DataFrame, paths: list, frame_indices: list,
                         output_dir: str, onset_frame: int = -1):
    """Save annotated TIF images for the requested frame indices."""
    from PIL import Image, ImageDraw, ImageFont
    import io

    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)

    # Build frame→path lookup
    frame_to_path = {int(df.loc[fi, 'frame']): paths[pos]
                     for pos, fi in enumerate(df.index)}

    try:
        font = ImageFont.truetype("arial.ttf", 36)
        font_sm = ImageFont.truetype("arial.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
        font_sm = font

    for fi in sorted(frame_indices):
        if fi not in df.index:
            print(f'  Frame {fi}: not in dataset, skipping.')
            continue
        row   = df.loc[fi]
        fnum  = int(row['frame'])
        path  = frame_to_path.get(fnum)
        if path is None:
            print(f'  Frame {fnum}: image path not found, skipping.')
            continue

        img = load_image(str(path))            # H×W uint8 grayscale
        h, w = img.shape

        pil = Image.fromarray(img).convert('RGB')
        draw = ImageDraw.Draw(pil)

        x_left   = int(row['x_left'])
        x_right  = int(row['x_right'])
        x_tip    = int(row['x_tip'])
        row_top  = int(row['row_top'])  if 'row_top'  in df.columns else 0
        row_bot  = int(row['row_bottom']) if 'row_bottom' in df.columns else h - 1

        # ROI top/bottom (orange dashed)
        for x in range(0, w, 20):
            draw.line([(x, row_top), (x + 10, row_top)], fill=(255, 165, 0), width=3)
            draw.line([(x, row_bot), (x + 10, row_bot)], fill=(255, 165, 0), width=3)

        # x_left / x_right (blue)
        draw.line([(x_left,  0), (x_left,  h - 1)], fill=(0, 100, 255), width=3)
        draw.line([(x_right, 0), (x_right, h - 1)], fill=(0, 100, 255), width=3)

        # x_tip (red)
        draw.line([(x_tip, 0), (x_tip, h - 1)], fill=(255, 30, 30), width=3)

        # Text overlay
        da   = float(row.get('delta_a_mm', 0))
        wf   = float(row.get('W_full_mm',  0))
        wl   = float(row.get('W_lig_mm',   0))
        t_s  = float(row.get('time_s',     0))
        onset_tag = '  ★ ONSET' if fnum == onset_frame else ''
        lines = [
            f'Frame {fnum}  t={t_s:.1f}s{onset_tag}',
            f'Δa = {da:.3f} mm',
            f'W_full = {wf:.3f} mm   W_lig = {wl:.3f} mm',
        ]
        y_txt = 10
        for line in lines:
            draw.text((10, y_txt), line, fill=(255, 255, 0), font=font)
            y_txt += 42

        out_path = out_dir / f'frame_{fnum:04d}_annotated.tif'
        pil.save(str(out_path), format='TIFF')
        print(f'  Saved: {out_path.name}')


def ask_frames_to_export(df: pd.DataFrame, onset_frame: int) -> list:
    """Show a dialog asking which frames to export as annotated TIF."""
    import tkinter as tk
    from tkinter import simpledialog

    n_frames   = len(df)
    last_frame = int(df['frame'].max())
    suggestion = '0'
    if onset_frame >= 0:
        suggestion += f', {onset_frame}'
        suggestion += f', {min(onset_frame + 50, last_frame)}'
        suggestion += f', {min(onset_frame + 100, last_frame)}'
    suggestion += f', {last_frame}'

    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    answer = simpledialog.askstring(
        'Export annotated TIFs',
        f'Enter frame numbers to export as annotated TIF\n'
        f'(comma-separated, range e.g. 0-10, or "none").\n'
        f'Total frames: {n_frames}  |  Onset: {"frame " + str(onset_frame) if onset_frame >= 0 else "not detected"}\n\n'
        f'Suggestion: {suggestion}',
        initialvalue=suggestion,
        parent=root,
    )
    root.destroy()

    if not answer or answer.strip().lower() == 'none':
        return []

    result = []
    for part in answer.split(','):
        part = part.strip()
        if '-' in part:
            try:
                a, b = part.split('-', 1)
                result.extend(range(int(a), int(b) + 1))
            except ValueError:
                pass
        else:
            try:
                result.append(int(part))
            except ValueError:
                pass
    return sorted(set(result))


def export_excel(df: pd.DataFrame, W_full_0_mm: float, output_path: str):
    """Write an Excel file with per-frame measurements.
    delta_a_mm and a_lambda_mm are forced monotonically non-decreasing
    (cumulative max) and clipped to >= 0, since crack advance is irreversible.
    """
    a_0 = round(float(df['a_mm'].iloc[0]), 6)
    delta_a_mono = df['delta_a_mm'].clip(lower=0).cummax().round(6)
    a_lambda_mono = (a_0 + delta_a_mono).round(6)
    out = pd.DataFrame({
        'frame'        : df['frame'],
        'a_0_mm'       : a_0,
        'a_lambda_mm'  : a_lambda_mono,
        'delta_a_mm'   : delta_a_mono,
        'W_full_0_mm'  : round(W_full_0_mm, 6),
        'W_full_mm'    : df['W_full_mm'].round(6),
        'W_lig_mm'     : df['W_lig_mm'].round(6),
    })
    out.to_excel(output_path, index=False)
    print(f'Excel saved   → {output_path}')


# ── Session persistence ───────────────────────────────────────────────────────

def save_session(df: pd.DataFrame, folder: str):
    """
    Save corrected x_tip values and per-frame x_right / x_left corrections.

    Session file format
    -------------------
    Integer keys              → x_tip corrections  (frame_index: x_tip_px)
    "_xright_corrections"     → dict of frame_index: x_right_px (per-frame)
    "_xleft_corrections"      → dict of frame_index: x_left_px  (per-frame)
    "_x_right" / "_x_left"   → frame-0 values (backward-compat for old code)
    """
    # ── x_tip corrections ─────────────────────────────────────────────────────
    corrected = df[df['corrected'] == True]
    data = {str(int(row['frame'])): int(row['x_tip'])
            for _, row in corrected.iterrows()}

    # ── per-frame x_right corrections ─────────────────────────────────────────
    if 'x_right_corrected' in df.columns:
        xr_corr = {str(int(fi)): int(df.loc[fi, 'x_right'])
                   for fi in df.index if bool(df.loc[fi, 'x_right_corrected'])}
        if xr_corr:
            data['_xright_corrections'] = xr_corr

    # ── per-frame x_left corrections ──────────────────────────────────────────
    if 'x_left_corrected' in df.columns:
        xl_corr = {str(int(fi)): int(df.loc[fi, 'x_left'])
                   for fi in df.index if bool(df.loc[fi, 'x_left_corrected'])}
        if xl_corr:
            data['_xleft_corrections'] = xl_corr

    # ── backward-compat: frame-0 values as single keys ────────────────────────
    fi0 = df.index[0] if len(df) > 0 else None
    if fi0 is not None:
        data['_x_left']  = int(df.loc[fi0, 'x_left'])
        data['_x_right'] = int(df.loc[fi0, 'x_right'])

    session_path = Path(folder) / '_va_session.json'
    with open(session_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_session(df: pd.DataFrame, folder: str) -> pd.DataFrame:
    """
    Load and apply corrections from a previous session file.

    Supports both new format (_xright_corrections / _xleft_corrections dicts)
    and legacy format (_x_right / _x_left single-value offset keys).

    The 'corrected' flag columns must already exist in df (added by
    process_all_frames).  If they are absent they are created here.
    """
    session_path = Path(folder) / '_va_session.json'
    if not session_path.exists():
        return df

    with open(session_path) as f:
        data = json.load(f)

    # Ensure corrected-flag columns exist
    for col in ('x_right_corrected', 'x_left_corrected'):
        if col not in df.columns:
            df[col] = False

    # ── x_right corrections ───────────────────────────────────────────────────
    if '_xright_corrections' in data:
        # New per-frame format
        count = 0
        for frame_str, xr in data['_xright_corrections'].items():
            fi = int(frame_str)
            if fi in df.index:
                df.loc[fi, 'x_right']           = int(xr)
                df.loc[fi, 'x_right_corrected'] = True
                count += 1
        print(f'  Restored {count} x_right anchor(s) from session.')
    elif '_x_right' in data:
        # Legacy: single offset — apply to all frames + mark frame 0 as anchor
        xr0_auto      = int(df['x_right'].iloc[0])
        xr0_corrected = int(data['_x_right'])
        offset_xr     = xr0_corrected - xr0_auto
        df['x_right'] = (df['x_right'] + offset_xr)
        if 'x_right_raw' in df.columns:
            df['x_right_raw'] = df['x_right_raw'] + offset_xr  # keep raw consistent
        fi0 = df.index[0]
        df.loc[fi0, 'x_right_corrected'] = True
        print(f'  Restored x_right offset = {offset_xr:+d} px  '
              f'(frame 0: {xr0_corrected} px)  [legacy format]')

    # ── x_left corrections ────────────────────────────────────────────────────
    if '_xleft_corrections' in data:
        count = 0
        for frame_str, xl in data['_xleft_corrections'].items():
            fi = int(frame_str)
            if fi in df.index:
                df.loc[fi, 'x_left']           = int(xl)
                df.loc[fi, 'x_left_corrected'] = True
                count += 1
        print(f'  Restored {count} x_left anchor(s) from session.')
    elif '_x_left' in data:
        xl0_auto      = int(df['x_left'].iloc[0])
        xl0_corrected = int(data['_x_left'])
        offset_xl     = xl0_corrected - xl0_auto
        df['x_left']  = (df['x_left'] + offset_xl).clip(lower=0)
        if 'x_left_raw' in df.columns:
            df['x_left_raw'] = (df['x_left_raw'] + offset_xl).clip(lower=0)
        fi0 = df.index[0]
        df.loc[fi0, 'x_left_corrected'] = True
        print(f'  Restored x_left offset = {offset_xl:+d} px  '
              f'(frame 0: {xl0_corrected} px)  [legacy format]')

    # ── x_tip corrections ─────────────────────────────────────────────────────
    count = 0
    for frame_str, x_tip in data.items():
        if frame_str.startswith('_'):
            continue
        fi = int(frame_str)
        if fi in df.index:
            df.loc[fi, 'x_tip']     = int(x_tip)
            df.loc[fi, 'corrected'] = True
            count += 1

    print(f'Loaded {count} x_tip correction(s) from session ({session_path.name}).')
    return df


# ── Plot export ───────────────────────────────────────────────────────────────

def export_plot(df: pd.DataFrame, output_path: str, onset_frame: int = -1):
    """Save the Δa vs frame chart as a PNG (backend-independent)."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure(figsize=(10, 5))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    ax.plot(df['frame'], df['delta_a_mm'], 'b-', lw=1.5, label='Δa(λ)')

    if onset_frame >= 0:
        mask = df['frame'] == onset_frame
        t_vals = df.loc[mask, 'time_s'].values
        t_str  = f'  (t = {t_vals[0]:.3f} s)' if len(t_vals) else ''
        ax.axvline(onset_frame, color='red', ls='--', lw=1.5,
                   label=f'Onset frame {onset_frame}{t_str}')

    ax.set_xlabel('Frame', fontsize=11)
    ax.set_ylabel('Δa (mm)', fontsize=11)
    ax.set_title('Crack propagation  Δa vs frame', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f'Plot saved   → {output_path}')


# ── Parameter dialogs ─────────────────────────────────────────────────────────

def ask_parameters() -> dict:
    """
    Collect run parameters.
    If called as:  python crack_analyser.py <folder> [scale]
    uses command-line arguments directly (no dialog).
    Otherwise opens tkinter dialogs.
    """
    import sys

    # ── Command-line mode ──────────────────────────────────────────────────────
    if len(sys.argv) >= 2:
        folder = sys.argv[1].strip('"').strip("'")
        scale_s = sys.argv[2] if len(sys.argv) >= 3 else '0.01587'
        try:
            return {
                'folder'            : folder,
                'scale_mm_per_pixel': float(scale_s),
            }
        except ValueError as exc:
            print(f'Parameter error: {exc}')
            return {}

    # ── Interactive dialog mode ────────────────────────────────────────────────
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    folder = filedialog.askdirectory(title='Select folder with image frames')
    if not folder:
        root.destroy()
        return {}

    scale_s = simpledialog.askstring(
        'Scale',
        'Scale factor (mm per pixel).\n'
        'Example: if 63 px = 1 mm  →  enter  0.01587',
        initialvalue='0.01587',
        parent=root,
    )
    if scale_s is None:
        root.destroy()
        return {}

    root.destroy()

    try:
        return {
            'folder'            : folder,
            'scale_mm_per_pixel': float(scale_s),
        }
    except ValueError as exc:
        print(f'Parameter error: {exc}')
        return {}


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print('VisualAnalyser – Crack Propagation Onset Detection')
    print('=' * 50)

    params = ask_parameters()
    if not params:
        print('Aborted.')
        return

    paths = load_image_paths(params['folder'])
    if not paths:
        print('No supported image files found in the selected folder.')
        return

    print(f'Found {len(paths)} image(s) in:  {params["folder"]}')

    # ── Fixed parameters ──────────────────────────────────────────────────────
    NOTCH_SIDE          = 'left'
    FPS                 = 5.0
    ONSET_THRESHOLD_MM  = 0.1

    # ── Auto-detect threshold from first frame (Otsu) ─────────────────────────
    first_img = load_image(paths[0])
    threshold = compute_otsu_threshold(first_img)
    print(f'Otsu threshold (first frame): {threshold}')

    # Store fixed parameters so the reviewer can access them.
    # W_full_0_mm is computed below after session load + edge smoothing.
    params.update({
        'notch_side'        : NOTCH_SIDE,
        'onset_threshold_mm': ONSET_THRESHOLD_MM,
        'img_threshold'     : threshold,
        'fps'               : FPS,
        'W_full_0_mm'       : 0.0,   # placeholder — updated after session load
    })

    # ── Batch auto-detection (x_left, x_right, x_tip per frame) ──────────────
    # x_left and x_right are detected per frame so W_full(λ) can reflect
    # Poisson lateral contraction throughout the test.
    df_raw = process_all_frames(paths, NOTCH_SIDE, threshold)

    # ── Restore previous session (corrections become anchors) ─────────────────
    # Load BEFORE smoothing so corrected frames act as hard anchors for
    # smooth_sample_edges_with_anchors (linear interpolation between anchors).
    df_raw = load_session(df_raw, params['folder'])

    # ── Smooth sample edges with anchor-based interpolation ───────────────────
    # Corrected frames are kept exactly; uncorrected frames between two anchors
    # get linear interpolation; frames outside the anchor range get a rolling
    # median on the raw auto-detected values.
    df_raw = smooth_sample_edges_with_anchors(df_raw, window=11)

    # ── W_full,0 = corrected frame-0 width ────────────────────────────────────
    x_left_used  = int(df_raw['x_left'].iloc[0])
    x_right_used = int(df_raw['x_right'].iloc[0])
    W_full_0_mm  = (x_right_used - x_left_used) * params['scale_mm_per_pixel']
    params['W_full_0_mm'] = W_full_0_mm
    print(f'W_full,0 = {W_full_0_mm:.3f} mm  '
          f'(x_left={x_left_used}, x_right={x_right_used})  |  '
          f'Notch: {NOTCH_SIDE}  |  fps: {FPS}  |  '
          f'Onset thresh: {ONSET_THRESHOLD_MM} mm')

    # ── Smooth x_tip with anchor-based interpolation + rolling median ─────────
    df_raw = smooth_crack_tips(df_raw, window=7)
    print(f'Smoothing applied (window=7).  '
          f'Max raw deviation: '
          f'{(df_raw["x_tip_raw"] - df_raw["x_tip"]).abs().max():.0f} px')

    df = compute_measurements(df_raw, params['scale_mm_per_pixel'],
                              W_full_0_mm, NOTCH_SIDE, FPS)

    # ── Interactive review ────────────────────────────────────────────────────
    flagged = flag_uncertain_frames(df)
    print(f'Flagged {len(flagged)} frame(s) for review.')

    from interactive_reviewer import InteractiveReviewer
    reviewer = InteractiveReviewer(
        df=df, image_paths=paths, flagged_indices=flagged, params=params
    )
    reviewer.run()
    df = reviewer.df   # DataFrame updated with user corrections

    # ── Final recompute + onset ───────────────────────────────────────────────
    # Use params['W_full_0_mm'] — may have been updated by the user in the reviewer
    # (by dragging the blue sample-edge lines to correct the specimen width).
    W_full_0_mm = params['W_full_0_mm']
    df = compute_measurements(df, params['scale_mm_per_pixel'],
                              W_full_0_mm, NOTCH_SIDE, FPS)

    onset = find_onset_frame(df, ONSET_THRESHOLD_MM)

    # ── Export ────────────────────────────────────────────────────────────────
    # Outputs go into a _results sub-folder so they are never picked up as
    # input images on the next run.
    out_dir = Path(params['folder']) / '_results'
    out_dir.mkdir(exist_ok=True)
    export_csv(df, str(out_dir / 'crack_measurements.csv'), onset)
    export_excel(df, W_full_0_mm, str(out_dir / 'crack_measurements.xlsx'))
    export_plot(df, str(out_dir / 'crack_measurements.png'), onset)
    save_session(df, params['folder'])   # session file stays next to the images

    # ── Annotated TIF export ──────────────────────────────────────────────────
    EXPORT_FRAMES = [0, 96, 192, 288, 384, 479, 575, 671, 767,
                     863, 959, 1055, 1151, 1246, 1342, 1438, 1534]
    tif_frames = [f for f in EXPORT_FRAMES if f in df['frame'].values]
    print(f'\nExporting {len(tif_frames)} annotated TIF(s)...')
    export_annotated_tif(df, paths, tif_frames, str(out_dir), onset)

    # ── Crack advance report ──────────────────────────────────────────────────
    max_da   = float(df['delta_a_mm'].max())
    max_fr   = int(df.loc[df['delta_a_mm'] == df['delta_a_mm'].max(), 'frame'].iloc[0])
    print()
    print('=' * 50)
    if onset >= 0:
        t_onset = float(df.loc[df['frame'] == onset, 'time_s'].values[0])
        print(f'  ✔  התקדמות קרע זוהתה!')
        print(f'     Onset: פריים {onset}  (t = {t_onset:.1f} s)')
        print(f'     מקסימום Δa = {max_da:.3f} מ"מ')
    else:
        print(f'  ✘  אין התקדמות קרע')
        print(f'     מקסימום Δa = {max_da:.3f} מ"מ  (בפריים {max_fr})')
        print(f'     סף ה-onset = {ONSET_THRESHOLD_MM:.2f} מ"מ  —  נדרשים עוד פריימים')
    print('=' * 50)


if __name__ == '__main__':
    main()
