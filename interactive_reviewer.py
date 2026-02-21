"""
VisualAnalyser – Interactive Reviewer
======================================
Matplotlib GUI for reviewing and correcting auto-detected crack tips.

Layout
------
Left panel  : current frame image with overlays
Right panel : Δa vs frame plot
Bottom row  : navigation / accept / export buttons

Controls
--------
Click on image        → move red crack-tip line to clicked x-position
← / p                 → previous flagged frame
→ / n                 → next flagged frame
a                     → accept current and go to next
q                     → close & export
"""

import matplotlib
matplotlib.use('TkAgg')          # interactive backend on Windows

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from pathlib import Path
import numpy as np
import json

from crack_analyser import load_image, save_session


class InteractiveReviewer:
    """
    Shows flagged frames one at a time and lets the user correct the crack tip
    by clicking on the image.  All corrections are stored in self.df and the
    measurements are recomputed live.
    """

    def __init__(self, df, image_paths, flagged_indices, params):
        """
        Parameters
        ----------
        df              : pd.DataFrame  – output of compute_measurements()
        image_paths     : list of Path  – sequential image file paths
        flagged_indices : list of int   – frame indices to review
        params          : dict          – run parameters (scale, notch_side, …)
        """
        self.df          = df.copy()
        self.image_paths = list(image_paths)
        self.flagged     = sorted(set(flagged_indices))
        self.params      = params
        self.cur         = 0              # position within self.flagged
        self.session_path = (
            Path(self.image_paths[0]).parent / '_va_session.json'
            if self.image_paths else None
        )

        if not self.flagged:
            print('No frames to review.')
            return

        self._build_figure()
        self._update_display()

    # ── Figure construction ───────────────────────────────────────────────────

    def _build_figure(self):
        self.fig = plt.figure(figsize=(17, 8))
        try:
            self.fig.canvas.manager.set_window_title('VisualAnalyser – Crack Tip Review')
        except AttributeError:
            pass

        # Main axes
        self.ax_img  = self.fig.add_axes([0.01, 0.14, 0.56, 0.84])
        self.ax_plot = self.fig.add_axes([0.63, 0.14, 0.35, 0.84])

        # Buttons  (x, y, w, h)
        bh, by = 0.07, 0.02
        self.btn_prev   = Button(self.fig.add_axes([0.01, by, 0.10, bh]), '← Prev')
        self.btn_next   = Button(self.fig.add_axes([0.13, by, 0.10, bh]), 'Next →')
        self.btn_accept = Button(self.fig.add_axes([0.27, by, 0.13, bh]), 'Accept')
        self.btn_all    = Button(self.fig.add_axes([0.43, by, 0.14, bh]), 'Accept All')
        self.btn_export = Button(self.fig.add_axes([0.77, by, 0.21, bh]), 'Export & Close')

        self.btn_prev  .on_clicked(lambda _: self._go(-1))
        self.btn_next  .on_clicked(lambda _: self._go(+1))
        self.btn_accept.on_clicked(lambda _: self._go(+1))
        self.btn_all   .on_clicked(lambda _: plt.close(self.fig))
        self.btn_export.on_clicked(lambda _: plt.close(self.fig))

        # Keyboard shortcut hint
        self.fig.text(0.01, 0.005,
                      'Keys:  ← / p  prev    → / n  next    a  accept    q  quit',
                      fontsize=7, color='gray')

        # Events
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event',    self._on_key)

    # ── Navigation ────────────────────────────────────────────────────────────

    def _go(self, delta: int):
        new = self.cur + delta
        if 0 <= new < len(self.flagged):
            self.cur = new
            self._update_display()

    @property
    def _fi(self) -> int:
        """Frame index of the currently shown flagged frame."""
        return self.flagged[self.cur]

    # ── Display ───────────────────────────────────────────────────────────────

    def _update_display(self):
        fi  = self._fi
        img = load_image(Path(self.image_paths[fi]))

        # Fetch row from df  (index == frame number)
        row     = self.df.loc[fi]
        x_left  = int(row['x_left'])
        x_right = int(row['x_right'])
        x_tip   = int(row['x_tip'])
        row_top = int(row.get('row_top', 0))
        row_bot = int(row.get('row_bottom', img.shape[0]))
        a_mm    = float(row['a_mm'])
        da_mm   = float(row['delta_a_mm'])
        conf    = float(row['confidence'])
        corr    = bool(row['corrected'])

        # ── Image panel ──────────────────────────────────────────────────
        self.ax_img.clear()
        self.ax_img.imshow(img, cmap='gray', aspect='auto', vmin=0, vmax=255)

        # Sample edges (blue dashed)
        self.ax_img.axvline(x_left,  color='deepskyblue', ls='--', lw=1.5,
                            alpha=0.85, label='Sample edges')
        self.ax_img.axvline(x_right, color='deepskyblue', ls='--', lw=1.5, alpha=0.85)

        # Crack tip (red solid)
        self.ax_img.axvline(x_tip, color='red', ls='-', lw=2.2, label='Crack tip')

        # ROI rectangle (red)
        from matplotlib.patches import Rectangle
        roi_rect = Rectangle(
            (x_left, row_top),
            x_right - x_left, row_bot - row_top,
            linewidth=1.2, edgecolor='red', facecolor='none',
            linestyle='--', alpha=0.7
        )
        self.ax_img.add_patch(roi_rect)

        arm_a = dict(arrowstyle='<->', color='orange', lw=1.8)
        arm_b = dict(arrowstyle='<->', color='deepskyblue', lw=1.8)

        # W_lig bracket  (orange – ligament from crack tip to free edge)
        y_lig  = row_top + (row_bot - row_top) * 0.08
        # W_full bracket (blue   – full specimen width)
        y_full = row_top + (row_bot - row_top) * 0.20

        ns = self.params['notch_side']
        if ns == 'right':
            self.ax_img.annotate('', xy=(x_left, y_lig), xytext=(x_tip, y_lig),
                                 arrowprops=arm_a)
            self.ax_img.text((x_left + x_tip) / 2, y_lig - 7,
                             'W_lig', ha='center', color='orange', fontsize=8,
                             fontweight='bold')
        else:
            self.ax_img.annotate('', xy=(x_tip, y_lig), xytext=(x_right, y_lig),
                                 arrowprops=arm_a)
            self.ax_img.text((x_tip + x_right) / 2, y_lig - 7,
                             'W_lig', ha='center', color='orange', fontsize=8,
                             fontweight='bold')

        # W_full bracket (always x_left → x_right)
        self.ax_img.annotate('', xy=(x_left, y_full), xytext=(x_right, y_full),
                             arrowprops=arm_b)
        W_full_mm = float(row['W_full_mm'])
        self.ax_img.text((x_left + x_right) / 2, y_full - 7,
                         f'W_full = {W_full_mm:.2f} mm',
                         ha='center', color='deepskyblue', fontsize=8,
                         fontweight='bold')

        status = '  ✓ CORRECTED' if corr else ''
        flag_str = f'{self.cur + 1} / {len(self.flagged)}'
        conf_str = f'{conf:.1f} px' if conf < 9000 else 'low confidence'
        self.ax_img.set_title(
            f'Frame {fi}   [{flag_str} flagged]{status}\n'
            f'a = {a_mm:.3f} mm     Δa = {da_mm:.4f} mm     conf = {conf_str}\n'
            'Click on image to correct the crack tip (red line)',
            fontsize=9
        )
        self.ax_img.set_xlim(0, img.shape[1])
        self.ax_img.set_ylim(img.shape[0], 0)
        self.ax_img.axis('off')
        self.ax_img.legend(loc='lower right', fontsize=7, framealpha=0.6)

        # ── Plot panel ───────────────────────────────────────────────────
        self.ax_plot.clear()
        frames  = self.df['frame'].values
        delta_a = self.df['delta_a_mm'].values

        self.ax_plot.plot(frames, delta_a, 'b-', lw=1.2, label='Δa(λ)')
        self.ax_plot.axvline(fi, color='red', lw=2.0, alpha=0.75,
                             label=f'Frame {fi}')
        thresh = self.params['onset_threshold_mm']
        self.ax_plot.axhline(thresh, color='limegreen', ls='--', lw=1.2,
                             alpha=0.85, label=f'Onset thresh {thresh} mm')

        self.ax_plot.set_xlabel('Frame', fontsize=9)
        self.ax_plot.set_ylabel('Δa (mm)', fontsize=9)
        self.ax_plot.set_title('Crack propagation  Δa vs frame', fontsize=9)
        self.ax_plot.legend(fontsize=7)
        self.ax_plot.grid(True, alpha=0.30)

        self.fig.canvas.draw_idle()

    # ── Interaction ───────────────────────────────────────────────────────────

    def _on_click(self, event):
        """Left-click on image → move crack tip to that x-coordinate."""
        if event.inaxes is not self.ax_img:
            return
        if event.button != 1 or event.xdata is None:
            return

        fi     = self._fi
        x_left  = int(self.df.loc[fi, 'x_left'])
        x_right = int(self.df.loc[fi, 'x_right'])
        x_new   = int(round(np.clip(event.xdata, x_left, x_right)))

        self.df.loc[fi, 'x_tip']    = x_new
        self.df.loc[fi, 'corrected'] = True

        self._recompute_all()
        self._save_session()
        self._update_display()

    def _on_key(self, event):
        mapping = {'right': +1, 'n': +1, 'left': -1, 'p': -1, 'a': +1}
        if event.key in mapping:
            self._go(mapping[event.key])
        elif event.key == 'q':
            plt.close(self.fig)

    # ── Recompute measurements after a correction ─────────────────────────────

    def _recompute_all(self):
        """
        Recompute a_mm and delta_a_mm for every frame using the current x_tip
        values (some may have been corrected by the user).
        Called after every click so the plot updates in real time.
        """
        s  = self.params['scale_mm_per_pixel']
        W0 = self.params['W_full_0_mm']
        ns = self.params['notch_side']

        for i in self.df.index:
            xl = int(self.df.at[i, 'x_left'])
            xr = int(self.df.at[i, 'x_right'])
            xt = int(self.df.at[i, 'x_tip'])

            wf = max(xr - xl, 1)
            wl = max((xt - xl) if ns == 'right' else (xr - xt), 0)

            wf_mm = wf * s
            wl_mm = wl * s

            self.df.at[i, 'W_full_px'] = wf
            self.df.at[i, 'W_lig_px']  = wl
            self.df.at[i, 'W_full_mm'] = wf_mm
            self.df.at[i, 'W_lig_mm']  = wl_mm
            self.df.at[i, 'a_mm']      = (1.0 - wl_mm / max(wf_mm, 1e-6)) * W0

        a_0 = float(self.df['a_mm'].iloc[0])
        self.df['delta_a_mm'] = self.df['a_mm'] - a_0

    def _save_session(self):
        """Persist all corrected x_tip values to disk after every click."""
        if self.session_path is None:
            return
        save_session(self.df, str(self.session_path.parent))

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self):
        """Show the GUI and block until the window is closed."""
        if not self.flagged:
            return
        plt.show()
