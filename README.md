# VisualAnalyser – Crack Propagation Onset Detection

Detects the crack tip in sequential CCD images from tensile tests, computes crack length and growth, and identifies the frame at which crack propagation begins.

---

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: `opencv-python`, `numpy`, `pandas`, `matplotlib`, `scipy`, `tqdm`

---

## Usage

```bash
python crack_analyser.py
```

A series of dialogs will guide you through:

| Dialog | Description |
|---|---|
| Folder | Select the folder containing the image sequence |
| Scale | mm per pixel (e.g. `0.01587` for 63 px = 1 mm) |
| Initial Width | `W_full,0` — nominal specimen width in mm |
| Notch Side | `left` or `right` — which edge holds the pre-cut notch |
| Onset Threshold | Δa (mm) that must be sustained to declare propagation onset |
| Image Threshold | Pixel intensity (0–255) separating bright specimen from dark background/crack |
| Frame Rate | Camera fps (used only for the `time_s` column in the CSV) |

---

## Expected image setup

- Bright / grey specimen on a dark background
- Horizontal crack
- Pre-cut notch on one edge (specify `notch_side`)
- Consistent framing across all frames

---

## How it works

1. **Auto-detect** sample boundaries and crack tip in every frame via column-wise dark-fraction scanning.
2. **Compute** physical measurements per frame:
   - `W_full` — full specimen width (mm)
   - `W_lig` — remaining ligament width (mm)
   - `a(λ)` — crack length (mm)
   - `Δa(λ)` — crack growth since frame 0 (mm)
3. **Flag** uncertain frames (large jumps, low confidence, width changes).
4. **Interactive review** — GUI shows each flagged frame; click on the image to correct the crack-tip position. Corrections are auto-saved after every click.
5. **Detect onset** — first frame where Δa persistently exceeds the threshold.
6. **Export** results as CSV and a PNG plot.

---

## Outputs

All files are written to the selected image folder:

| File | Description |
|---|---|
| `crack_measurements.csv` | Per-frame results: `frame`, `time_s`, `W_full_mm`, `W_lig_mm`, `a_mm`, `delta_a_mm`, `corrected` |
| `crack_measurements.png` | Δa vs frame chart with onset marker |
| `_va_session.json` | Auto-saved corrections (loaded automatically on the next run) |

---

## Interactive reviewer controls

| Action | Effect |
|---|---|
| Click on image | Move crack-tip line to clicked position |
| `←` / `p` | Previous flagged frame |
| `→` / `n` | Next flagged frame |
| `a` | Accept and go to next |
| `q` | Close and export |
| Accept All button | Accept auto-detections for all remaining frames and close |
| Export & Close button | Close and export |
