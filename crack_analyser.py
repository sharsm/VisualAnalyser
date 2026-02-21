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
- Horizontal crack (grows right→left by default; change notch_side if needed)
- Pre-cut notch on the right edge of the specimen
- Scale: 63 px = 1 mm  (can be changed at runtime)
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

SUPPORTED_EXT = {'.png', '.tif', '.tiff', '.bmp', '.jpg', '.jpeg'}

# ── Image loading ─────────────────────────────────────────────────────────────

def load_image_paths(folder: str) -> list:
    """Return sorted list of image paths in folder (by filename)."""
    folder = Path(folder)
    paths = sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_EXT]
    )
    return paths


def load_image(path) -> np.ndarray:
    """Load image as 8-bit grayscale."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img


# ── Sample geometry detection ─────────────────────────────────────────────────

def detect_sample_row_range(image: np.ndarray, margin_frac: float = 0.20) -> tuple:
    """
    Find the vertical extent of the specimen body by looking for the largest
    contiguous band of bright rows (row-mean intensity > 60 % of max).

    60 % is chosen to exclude dark grip-gap rows (typically < 50 % of max)
    while keeping the bright specimen body rows (typically > 70 % of max).

    Returns (row_top, row_bottom) using the middle (1-2·margin_frac) of the
    bright band, which avoids grip-edge noise.
    """
    row_means = image.mean(axis=1)
    thresh = row_means.max() * 0.60
    bright = (row_means > thresh).astype(int)

    diff = np.diff(bright, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    if len(starts) == 0:
        h = image.shape[0]
        return int(h * margin_frac), int(h * (1 - margin_frac))

    lengths = ends - starts
    best = int(np.argmax(lengths))
    top, bot = starts[best], ends[best]

    margin = int((bot - top) * margin_frac)
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

    return int(np.median(x_lefts)), int(np.median(x_rights))


def detect_crack_tip(image: np.ndarray, x_left: int, x_right: int,
                     row_top: int, row_bottom: int,
                     notch_side: str = 'right',
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
    DARK_THRESH = 0.08                               # ≥8 % dark → crack column

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
                # Allow up to 5 consecutive bright columns (artifact/gap)
                look_left = dark_frac[max(0, xi - 4): xi]
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
                look_right = dark_frac[xi + 1: xi + 5]
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

def process_all_frames(image_paths: list, notch_side: str = 'right',
                        threshold: int = 60) -> pd.DataFrame:
    """
    Auto-detect sample edges and crack tip for every frame.

    The ROI (row_top, row_bottom) is detected from the first frame and reused
    for all subsequent frames to keep measurements consistent.

    Returns a DataFrame with one row per frame.
    """
    records = []
    row_top = row_bottom = None

    for i, path in enumerate(tqdm(image_paths, desc='Detecting crack tips', unit='frame')):
        img = load_image(path)

        if row_top is None:
            row_top, row_bottom = detect_sample_row_range(img)

        x_left, x_right = detect_sample_edges(img, row_top, row_bottom, threshold)
        x_tip, conf = detect_crack_tip(img, x_left, x_right, row_top, row_bottom,
                                        notch_side, threshold)

        records.append({
            'frame'     : i,
            'path'      : str(path),
            'x_left'    : x_left,
            'x_right'   : x_right,
            'x_tip'     : x_tip,
            'confidence': conf,
            'row_top'   : row_top,
            'row_bottom': row_bottom,
            'corrected' : False,
        })

    df = pd.DataFrame(records)
    df.index = df['frame']   # integer index == frame number
    return df


# ── Measurement calculation ───────────────────────────────────────────────────

def compute_measurements(df: pd.DataFrame, scale_mm_per_pixel: float,
                          W_full_0_mm: float, notch_side: str = 'right',
                          fps: float = 4.0) -> pd.DataFrame:
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


# ── Flagging ──────────────────────────────────────────────────────────────────

def flag_uncertain_frames(df: pd.DataFrame) -> list:
    """
    Return sorted list of frame indices that should be reviewed by the user.

    Rules
    -----
    - Frame 0 is always flagged (user must confirm the initial notch tip = a₀).
    - Abrupt crack-tip jump (>5 px between consecutive frames).
    - High per-frame confidence score (>3× median AND >2 px).
    - Sudden change in sample width (>3 %, indicating possible sample slip).
    """
    flagged = {0}

    if len(df) < 2:
        return sorted(flagged)

    med_conf = df['confidence'].median()
    tips = df['x_tip'].values

    for i in range(1, len(df)):
        jump = abs(int(tips[i]) - int(tips[i - 1]))
        if jump > 5:
            flagged.add(i)
            flagged.add(i - 1)

        if df['confidence'].iloc[i] > med_conf * 3 and df['confidence'].iloc[i] > 2.0:
            flagged.add(i)

        w_prev = df['x_right'].iloc[i - 1] - df['x_left'].iloc[i - 1]
        w_curr = df['x_right'].iloc[i]     - df['x_left'].iloc[i]
        if w_prev > 0 and abs(w_curr - w_prev) / w_prev > 0.03:
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

    if onset_frame >= 0:
        mask = df['frame'] == onset_frame
        t = df.loc[mask, 'time_s'].values
        time_str = f'  (t = {t[0]:.3f} s)' if len(t) else ''
        print(f'Crack propagation onset: frame {onset_frame}{time_str}')
    else:
        print('Crack propagation onset: not detected (Δa never exceeded threshold).')


# ── Parameter dialogs ─────────────────────────────────────────────────────────

def ask_parameters() -> dict:
    """
    Collect run parameters interactively via tkinter dialogs.
    Returns a dict with all parameters, or an empty dict if cancelled.
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    folder = filedialog.askdirectory(title='Select folder with image frames')
    if not folder:
        root.destroy()
        return {}

    def ask(title, prompt, default):
        return simpledialog.askstring(
            title, prompt, initialvalue=str(default), parent=root
        )

    scale_s = ask(
        'Scale',
        'Scale factor (mm per pixel).\n'
        'Example: 1/63 ≈ 0.01587  (63 px = 1 mm)',
        '0.01587'
    )
    if scale_s is None:
        root.destroy(); return {}

    wfull_s = ask(
        'Initial Width',
        'Initial full sample width  W_full,0  (mm):',
        '10.0'
    )
    if wfull_s is None:
        root.destroy(); return {}

    notch_s = ask(
        'Notch Side',
        "Which side holds the notch?\n"
        "Enter  'left'  or  'right'  (default: right):",
        'right'
    )
    if notch_s is None:
        root.destroy(); return {}
    notch_s = notch_s.strip().lower()
    if notch_s not in ('left', 'right'):
        messagebox.showerror('Error', "Notch side must be 'left' or 'right'.")
        root.destroy(); return {}

    onset_s = ask(
        'Onset Threshold',
        'Onset detection: Δa threshold (mm).\n'
        'Crack propagation starts when Δa persistently exceeds this value.',
        '0.05'
    )
    if onset_s is None:
        root.destroy(); return {}

    thresh_s = ask(
        'Image Threshold',
        'Pixel intensity threshold to separate sample (bright) from\n'
        'background/crack (dark).  Range 0–255:',
        '60'
    )
    if thresh_s is None:
        root.destroy(); return {}

    root.destroy()

    try:
        return {
            'folder'             : folder,
            'scale_mm_per_pixel' : float(scale_s),
            'W_full_0_mm'        : float(wfull_s),
            'notch_side'         : notch_s,
            'onset_threshold_mm' : float(onset_s),
            'img_threshold'      : int(thresh_s),
            'fps'                : 4.0,      # 4 frames per second
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
    print(f'Scale: {params["scale_mm_per_pixel"]:.5f} mm/px  |  '
          f'W_full,0 = {params["W_full_0_mm"]} mm  |  '
          f'Notch side: {params["notch_side"]}')

    # ── Batch auto-detection ──
    df_raw = process_all_frames(
        paths, params['notch_side'], params['img_threshold']
    )
    df = compute_measurements(
        df_raw,
        params['scale_mm_per_pixel'],
        params['W_full_0_mm'],
        params['notch_side'],
        params['fps'],
    )

    # ── Interactive review ──
    flagged = flag_uncertain_frames(df)
    print(f'Flagged {len(flagged)} frame(s) for review.')

    from interactive_reviewer import InteractiveReviewer
    reviewer = InteractiveReviewer(
        df=df, image_paths=paths, flagged_indices=flagged, params=params
    )
    reviewer.run()
    df = reviewer.df   # DataFrame updated with user corrections

    # ── Final recompute + onset ──
    df = compute_measurements(
        df,
        params['scale_mm_per_pixel'],
        params['W_full_0_mm'],
        params['notch_side'],
        params['fps'],
    )

    onset = find_onset_frame(df, params['onset_threshold_mm'])

    # ── Export ──
    output_path = str(Path(params['folder']) / 'crack_measurements.csv')
    export_csv(df, output_path, onset)


if __name__ == '__main__':
    main()
