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
    """
    import re
    folder = Path(folder)
    paths = [p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_EXT]

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
                        threshold: int = None,
                        x_left_fixed: int = None,
                        x_right_fixed: int = None) -> pd.DataFrame:
    """
    Auto-detect crack tip for every frame.

    Geometry (row_top, row_bottom, x_left, x_right) is detected from the
    first frame and kept FIXED for all subsequent frames.  This is physically
    correct because the specimen does not move laterally, and avoids the
    problem where open-crack rows give a wrong x_left estimate in later frames.

    If threshold is None, Otsu's method is applied to the first frame.

    Returns a DataFrame with one row per frame.
    """
    records = []
    row_top = row_bottom = None
    _threshold = threshold
    _x_left  = x_left_fixed
    _x_right = x_right_fixed

    for i, path in enumerate(tqdm(image_paths, desc='Detecting crack tips', unit='frame')):
        img = load_image(path)

        if row_top is None:
            row_top, row_bottom = detect_sample_row_range(img)
            if _threshold is None:
                _threshold = compute_otsu_threshold(img)
            if _x_left is None or _x_right is None:
                _x_left, _x_right = detect_sample_edges(
                    img, row_top, row_bottom, _threshold
                )

        # Clamp ROI to current image height (frames may differ in size).
        # Always extend the bottom to at least 98 % of frame height so that
        # crack openings that grow downward in later frames are never cut off.
        h  = img.shape[0]
        rt = min(row_top,    h - 1)
        rb = max(min(row_bottom, h - 1), int(h * 0.98))

        x_tip, conf = detect_crack_tip(img, _x_left, _x_right, rt, rb,
                                        notch_side, _threshold)

        records.append({
            'frame'     : i,
            'path'      : str(path),
            'x_left'    : _x_left,
            'x_right'   : _x_right,
            'x_tip'     : x_tip,
            'confidence': conf,
            'row_top'   : rt,
            'row_bottom': rb,
            'corrected' : False,
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
    df['delta_a_mm'] = df['a_mm'] - a_0
    df['time_s']     = df['frame'] / fps

    return df


# ── Temporal smoothing ────────────────────────────────────────────────────────

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

    # ── Step 2: rolling median for frames outside the corrected range ─────
    if anchors:
        first_anchor_pos = anchors[0][0]
        last_anchor_pos  = anchors[-1][0]
    else:
        first_anchor_pos = n
        last_anchor_pos  = -1

    # Positions outside anchor range (use rolling median on raw values)
    raw = df['x_tip_raw'].values.astype(float)
    smoothed_raw = (pd.Series(raw)
                    .rolling(window=window, center=True, min_periods=1)
                    .median()
                    .round()
                    .values)

    for pos in range(n):
        if corrected[pos]:
            continue
        if pos < first_anchor_pos or pos > last_anchor_pos:
            tips[pos] = smoothed_raw[pos]

    # ── Apply smoothed values ─────────────────────────────────────────────
    for pos, fi in enumerate(frames):
        if not corrected[pos]:
            df.loc[fi, 'x_tip'] = int(tips[pos])

    return df


# ── Flagging ──────────────────────────────────────────────────────────────────

def flag_uncertain_frames(df: pd.DataFrame,
                          deviation_thresh: int = 15,
                          jump_thresh: int = 10) -> list:
    """
    Return sorted list of frame indices that should be reviewed by the user.

    Rules
    -----
    - Frame 0 is always flagged (confirm initial notch tip = a₀).
    - Raw detection deviated > deviation_thresh px from smoothed value
      (indicates the raw detector was confused on that frame).
    - Smoothed x_tip jumps > jump_thresh px vs previous frame
      (real crack advance or lingering detection error).
    """
    flagged = {0}

    if len(df) < 2:
        return sorted(flagged)

    raw_col   = 'x_tip_raw' if 'x_tip_raw' in df.columns else 'x_tip'
    tips_raw    = df[raw_col].values
    tips_smooth = df['x_tip'].values
    corrected   = df['corrected'].values

    for i in range(len(df)):
        if corrected[i]:
            continue   # user already reviewed this frame — skip
        # Raw detector was far from smoothed consensus
        if abs(int(tips_raw[i]) - int(tips_smooth[i])) > deviation_thresh:
            flagged.add(i)

    for i in range(1, len(df)):
        if corrected[i]:
            continue
        # Smoothed tip jumped too fast (real event or residual error)
        if abs(int(tips_smooth[i]) - int(tips_smooth[i - 1])) > jump_thresh:
            flagged.add(i)
            flagged.add(i - 1)

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

    if onset_frame >= 0:
        mask = df['frame'] == onset_frame
        t = df.loc[mask, 'time_s'].values
        time_str = f'  (t = {t[0]:.3f} s)' if len(t) else ''
        print(f'Crack propagation onset: frame {onset_frame}{time_str}')
    else:
        print('Crack propagation onset: not detected (Δa never exceeded threshold).')


# ── Session persistence ───────────────────────────────────────────────────────

def save_session(df: pd.DataFrame, folder: str):
    """Save corrected x_tip values to a JSON file in the image folder."""
    corrected = df[df['corrected'] == True]
    data = {str(int(row['frame'])): int(row['x_tip'])
            for _, row in corrected.iterrows()}
    session_path = Path(folder) / '_va_session.json'
    with open(session_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_session(df: pd.DataFrame, folder: str) -> pd.DataFrame:
    """
    Load and apply crack-tip corrections from a previous session file.
    Returns the (possibly updated) DataFrame.
    """
    session_path = Path(folder) / '_va_session.json'
    if not session_path.exists():
        return df

    with open(session_path) as f:
        data = json.load(f)

    count = 0
    for frame_str, x_tip in data.items():
        fi = int(frame_str)
        if fi in df.index:
            df.loc[fi, 'x_tip']    = int(x_tip)
            df.loc[fi, 'corrected'] = True
            count += 1

    print(f'Loaded {count} correction(s) from previous session ({session_path.name}).')
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
    Collect run parameters interactively via tkinter dialogs.
    Returns a dict with folder and scale_mm_per_pixel, or an empty dict if
    cancelled.  All other parameters (notch side, fps, thresholds, W_full_0)
    are determined automatically inside main().
    """
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

    # ── Auto-detect W_full,0 from first frame ─────────────────────────────────
    row_top_0, row_bottom_0 = detect_sample_row_range(first_img)
    x_left_0, x_right_0    = detect_sample_edges(first_img, row_top_0, row_bottom_0, threshold)
    W_full_0_px = x_right_0 - x_left_0
    W_full_0_mm = W_full_0_px * params['scale_mm_per_pixel']
    print(f'W_full,0 = {W_full_0_mm:.3f} mm  ({W_full_0_px} px)  |  '
          f'Notch: {NOTCH_SIDE}  |  fps: {FPS}  |  '
          f'Onset thresh: {ONSET_THRESHOLD_MM} mm')

    # Store auto-detected values so the reviewer can access them
    params.update({
        'W_full_0_mm'       : W_full_0_mm,
        'notch_side'        : NOTCH_SIDE,
        'onset_threshold_mm': ONSET_THRESHOLD_MM,
        'img_threshold'     : threshold,
        'fps'               : FPS,
    })

    # ── Batch auto-detection ──────────────────────────────────────────────────
    df_raw = process_all_frames(paths, NOTCH_SIDE, threshold,
                                x_left_fixed=x_left_0,
                                x_right_fixed=x_right_0)

    # ── Restore previous session if available ─────────────────────────────────
    df_raw = load_session(df_raw, params['folder'])

    # ── Smooth x_tip with rolling median (suppresses ±2-3 px noise) ──────────
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
    export_plot(df, str(out_dir / 'crack_measurements.png'), onset)
    save_session(df, params['folder'])   # session file stays next to the images


if __name__ == '__main__':
    main()
