"""
VisualAnalyser – Interactive Reviewer
======================================
Matplotlib GUI for reviewing and correcting auto-detected crack tips.

Layout
------
Left panel  : current frame image with overlays
Right panel : Δa vs frame plot
Bottom row  : navigation / accept / re-detect / zoom / export buttons

Controls
--------
Drag red   ROI top/bot  ▲▼  → adjust measurement zone vertically
Drag blue  sample edges ◄►  → adjust W_full / sample boundaries
Click on image (elsewhere)  → move crack-tip line to clicked x
Scroll wheel                → zoom in / out (centred on cursor)
← / p                       → previous flagged frame
→ / n                       → next flagged frame
a                           → accept current and go to next
r                           → reset zoom to full view
q                           → close & export
"""

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from pathlib import Path
import numpy as np

from crack_analyser import (load_image, save_session, detect_crack_tip,
                            smooth_sample_edges_with_anchors,
                            smooth_crack_tips)


class InteractiveReviewer:
    """
    Shows flagged frames one at a time and lets the user:
      • drag the red ROI top/bottom lines  → adjust vertical measurement zone
      • drag the blue left/right edge lines → adjust W_full / sample boundaries
      • click anywhere else                 → correct the crack-tip x position
      • scroll wheel                        → zoom in / out
    All corrections update self.df and recompute measurements live.
    """

    _GRAB_PX = 10   # screen-pixel radius for grabbing any draggable line

    def __init__(self, df, image_paths, flagged_indices, params):
        self.df          = df.copy()
        self.image_paths = list(image_paths)
        self.flagged     = sorted(set(flagged_indices))
        self.params      = params
        self.cur         = 0
        self.session_path = (
            Path(self.image_paths[0]).parent / '_va_session.json'
            if self.image_paths else None
        )

        # Drag state:  'top' | 'bottom' | 'xleft' | 'xright' | None
        self._drag_roi   = None
        self._top_line   = None   # ROI top edge   (red, horizontal, draggable)
        self._bot_line   = None   # ROI bottom edge (red, horizontal, draggable)
        self._left_line  = None   # sample x_left   (blue, vertical, draggable)
        self._right_line = None   # sample x_right  (blue, vertical, draggable)

        self._img_cache    = {}     # fi → ndarray
        self._ax_ready     = False  # False until first _update_display
        self._all_frames   = list(self.df.index)   # full ordered list of frame indices
        self._orig_flagged = set(flagged_indices)  # original auto-flagged set (for title marker)
        self._browse_mode  = False  # True = show every frame, not just flagged

        if not self.flagged:
            print('No frames to review.')
            return

        self._build_figure()
        self._update_display()

    # ── Figure ───────────────────────────────────────────────────────────────

    def _build_figure(self):
        self.fig = plt.figure(figsize=(17, 8))
        try:
            self.fig.canvas.manager.set_window_title('VisualAnalyser – Crack Tip Review')
        except AttributeError:
            pass

        self.ax_img  = self.fig.add_axes([0.01, 0.14, 0.56, 0.84])
        self.ax_plot = self.fig.add_axes([0.63, 0.14, 0.35, 0.84])

        bh, by = 0.07, 0.02
        self.btn_prev       = Button(self.fig.add_axes([0.01, by, 0.07, bh]), '← Prev')
        self.btn_next       = Button(self.fig.add_axes([0.09, by, 0.07, bh]), 'Next →')
        self.btn_accept     = Button(self.fig.add_axes([0.17, by, 0.09, bh]), 'Accept')
        self.btn_all        = Button(self.fig.add_axes([0.27, by, 0.09, bh]), 'Accept All')
        self.btn_browse     = Button(self.fig.add_axes([0.37, by, 0.10, bh]), 'Browse All')
        self.btn_redetect   = Button(self.fig.add_axes([0.48, by, 0.12, bh]), 'Re-detect All')
        self.btn_reset_zoom = Button(self.fig.add_axes([0.61, by, 0.09, bh]), 'Reset Zoom')
        self.btn_export     = Button(self.fig.add_axes([0.77, by, 0.21, bh]), 'Export & Close')

        self.btn_prev      .on_clicked(lambda _: self._go(-1))
        self.btn_next      .on_clicked(lambda _: self._go(+1))
        self.btn_accept    .on_clicked(lambda _: self._go(+1))
        self.btn_all       .on_clicked(lambda _: plt.close(self.fig))
        self.btn_browse    .on_clicked(lambda _: self._toggle_browse())
        self.btn_redetect  .on_clicked(lambda _: self._redetect_all())
        self.btn_reset_zoom.on_clicked(lambda _: self._reset_zoom())
        self.btn_export    .on_clicked(lambda _: plt.close(self.fig))

        self.fig.text(
            0.01, 0.005,
            'Drag ▲▼ red ROI edges  |  Drag ◄► blue sample edges (W_full)'
            '  |  Click image → crack tip  |  Scroll → zoom  |  r reset  |  q quit',
            fontsize=7, color='gray'
        )

        self.fig.canvas.mpl_connect('button_press_event',   self._on_click)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.fig.canvas.mpl_connect('motion_notify_event',  self._on_motion)
        self.fig.canvas.mpl_connect('scroll_event',         self._on_scroll)
        self.fig.canvas.mpl_connect('key_press_event',      self._on_key)

    # ── Navigation ────────────────────────────────────────────────────────────

    def _go(self, delta: int):
        new = self.cur + delta
        if 0 <= new < len(self.flagged):
            self.cur = new
            self._update_display()

    @property
    def _fi(self) -> int:
        return self.flagged[self.cur]

    # ── Display ───────────────────────────────────────────────────────────────

    def _update_display(self):
        fi  = self._fi
        img = self._load_cached(fi)

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
        h_img   = img.shape[0]
        w_img   = img.shape[1]
        y_mid   = (row_top + row_bot) // 2   # mid-ROI for drag-hint labels

        # Preserve zoom across refreshes
        if self._ax_ready:
            saved_xlim = self.ax_img.get_xlim()
            saved_ylim = self.ax_img.get_ylim()
        else:
            saved_xlim = (0, w_img)
            saved_ylim = (h_img, 0)
            self._ax_ready = True

        # ── Image panel ──────────────────────────────────────────────────
        self.ax_img.clear()
        self.ax_img.imshow(img, cmap='gray', aspect='auto', vmin=0, vmax=255)

        # ── Blue draggable sample edges (W_full) ─────────────────────────
        self._left_line, = self.ax_img.plot(
            [x_left, x_left], [0, h_img],
            color='deepskyblue', ls='--', lw=2.5, alpha=0.90, zorder=5,
            label='Sample edges ◄► drag'
        )
        self.ax_img.text(x_left + 6, y_mid, '◄ drag',
                         color='deepskyblue', fontsize=7, alpha=0.80,
                         rotation=90, va='center', zorder=6)

        self._right_line, = self.ax_img.plot(
            [x_right, x_right], [0, h_img],
            color='deepskyblue', ls='--', lw=2.5, alpha=0.90, zorder=5
        )
        self.ax_img.text(x_right - 18, y_mid, '► drag',
                         color='deepskyblue', fontsize=7, alpha=0.80,
                         rotation=90, va='center', zorder=6)

        # ── Red crack-tip line ───────────────────────────────────────────
        self.ax_img.axvline(x_tip, color='red', ls='-', lw=2.2,
                            label='Crack tip')

        # ── Red ROI left / right sides (thin, not draggable) ─────────────
        self.ax_img.plot([x_left,  x_left],  [row_top, row_bot],
                         color='red', ls='--', lw=1.2, alpha=0.55)
        self.ax_img.plot([x_right, x_right], [row_top, row_bot],
                         color='red', ls='--', lw=1.2, alpha=0.55)

        # ── Red ROI top edge  – DRAGGABLE ────────────────────────────────
        self._top_line, = self.ax_img.plot(
            [x_left, x_right], [row_top, row_top],
            color='red', ls='--', lw=2.8, alpha=0.95, zorder=5,
            label='ROI ▲▼ drag'
        )
        self.ax_img.text(x_left + 6, row_top - 10, '▲ drag ROI',
                         color='red', fontsize=7, alpha=0.80, zorder=6)

        # ── Red ROI bottom edge – DRAGGABLE ──────────────────────────────
        self._bot_line, = self.ax_img.plot(
            [x_left, x_right], [row_bot, row_bot],
            color='red', ls='--', lw=2.8, alpha=0.95, zorder=5
        )
        self.ax_img.text(x_left + 6, row_bot + 14, '▼ drag ROI',
                         color='red', fontsize=7, alpha=0.80, zorder=6)

        # ── W_lig bracket (orange) ───────────────────────────────────────
        arm_a = dict(arrowstyle='<->', color='orange', lw=1.8)
        arm_b = dict(arrowstyle='<->', color='deepskyblue', lw=1.8)
        y_lig  = row_top + (row_bot - row_top) * 0.08
        y_full = row_top + (row_bot - row_top) * 0.22

        ns = self.params['notch_side']
        if ns == 'right':
            self.ax_img.annotate('', xy=(x_left, y_lig), xytext=(x_tip, y_lig),
                                 arrowprops=arm_a)
            self.ax_img.text((x_left + x_tip) / 2, y_lig - 7,
                             'W_lig', ha='center', color='orange',
                             fontsize=8, fontweight='bold')
        else:
            self.ax_img.annotate('', xy=(x_tip, y_lig), xytext=(x_right, y_lig),
                                 arrowprops=arm_a)
            self.ax_img.text((x_tip + x_right) / 2, y_lig - 7,
                             'W_lig', ha='center', color='orange',
                             fontsize=8, fontweight='bold')

        # ── W_full bracket (blue) ────────────────────────────────────────
        self.ax_img.annotate('', xy=(x_left, y_full), xytext=(x_right, y_full),
                             arrowprops=arm_b)
        W_full_mm = float(row['W_full_mm'])
        self.ax_img.text((x_left + x_right) / 2, y_full - 7,
                         f'W_full = {W_full_mm:.2f} mm',
                         ha='center', color='deepskyblue',
                         fontsize=8, fontweight='bold')

        status    = '  ✓ CORRECTED' if corr else ''
        flag_mark = '⚑ ' if fi in self._orig_flagged else ''
        mode_str  = f'{"ALL" if self._browse_mode else "FLAGGED"} {self.cur + 1}/{len(self.flagged)}'
        conf_str  = f'{conf:.1f} px' if conf < 9000 else 'low confidence'
        fname     = Path(self.image_paths[fi]).name
        self.ax_img.set_title(
            f'{fname}   Frame {fi}   [{mode_str}]  {flag_mark}{status}\n'
            f'a = {a_mm:.3f} mm     Δa = {da_mm:.4f} mm     conf = {conf_str}\n'
            'Click → crack tip   |   Drag ▲▼ red ROI   |   Drag ◄► blue W_full   |   Scroll → zoom',
            fontsize=9
        )

        self.ax_img.set_xlim(saved_xlim)
        self.ax_img.set_ylim(saved_ylim)
        self.ax_img.axis('off')
        self.ax_img.legend(loc='lower right', fontsize=7, framealpha=0.6)

        # ── Plot panel ───────────────────────────────────────────────────
        self.ax_plot.clear()
        self.ax_plot.plot(self.df['frame'].values, self.df['delta_a_mm'].values,
                          'b-', lw=1.2, label='Δa(λ)')
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

    # ── Mouse: click ──────────────────────────────────────────────────────────

    def _on_click(self, event):
        """
        Priority order (all comparisons in screen pixels, zoom-independent):
          1. Near blue x_left  line → start xleft drag
          2. Near blue x_right line → start xright drag
          3. Near red  ROI top  line → start top drag
          4. Near red  ROI bot  line → start bottom drag
          5. Anywhere else → move crack tip to clicked x
        """
        if event.inaxes is not self.ax_img:
            return
        if event.button != 1 or event.xdata is None or event.ydata is None:
            return

        fi      = self._fi
        row     = self.df.loc[fi]
        x_left  = int(row['x_left'])
        x_right = int(row['x_right'])
        row_top = int(row['row_top'])
        row_bot = int(row['row_bottom'])

        trans = self.ax_img.transData

        # Screen positions of the four draggable lines
        scr_xl, _ = trans.transform([x_left,  0])
        scr_xr, _ = trans.transform([x_right, 0])
        _, scr_rt = trans.transform([0, row_top])
        _, scr_rb = trans.transform([0, row_bot])

        cx, cy = event.x, event.y   # screen pixel coordinates of click

        if abs(cx - scr_xl) <= self._GRAB_PX:
            self._drag_roi = 'xleft'
            return
        if abs(cx - scr_xr) <= self._GRAB_PX:
            self._drag_roi = 'xright'
            return
        if abs(cy - scr_rt) <= self._GRAB_PX:
            self._drag_roi = 'top'
            return
        if abs(cy - scr_rb) <= self._GRAB_PX:
            self._drag_roi = 'bottom'
            return

        # Default: move crack tip
        self._drag_roi = None
        x_new = int(round(np.clip(event.xdata, x_left, x_right)))
        self.df.loc[fi, 'x_tip']    = x_new
        self.df.loc[fi, 'corrected'] = True
        self._reinterpolate_tips()
        self._recompute_all()
        self._save_session()
        self._update_display()

    # ── Mouse: motion (live drag feedback) ───────────────────────────────────

    def _on_motion(self, event):
        if self._drag_roi is None:
            return
        if event.inaxes is not self.ax_img or event.xdata is None or event.ydata is None:
            return

        fi      = self._fi
        row     = self.df.loc[fi]
        x_left  = int(row['x_left'])
        x_right = int(row['x_right'])
        row_top = int(row['row_top'])
        row_bot = int(row['row_bottom'])
        img     = self._load_cached(fi)
        h, w    = img.shape[:2]

        if self._drag_roi == 'xleft' and self._left_line is not None:
            x = int(round(np.clip(event.xdata, 0, x_right - 10)))
            self._left_line.set_xdata([x, x])

        elif self._drag_roi == 'xright' and self._right_line is not None:
            x = int(round(np.clip(event.xdata, x_left + 10, w - 1)))
            self._right_line.set_xdata([x, x])

        elif self._drag_roi == 'top' and self._top_line is not None:
            y = int(round(np.clip(event.ydata, 0, row_bot - 10)))
            self._top_line.set_ydata([y, y])

        elif self._drag_roi == 'bottom' and self._bot_line is not None:
            y = int(round(np.clip(event.ydata, row_top + 10, h - 1)))
            self._bot_line.set_ydata([y, y])

        self.fig.canvas.draw_idle()

    # ── Mouse: release (commit drag) ──────────────────────────────────────────

    def _on_release(self, event):
        if self._drag_roi is None or event.button != 1:
            self._drag_roi = None
            return

        fi  = self._fi
        img = self._load_cached(fi)
        h, w = img.shape[:2]

        row     = self.df.loc[fi]
        x_left  = int(row['x_left'])
        x_right = int(row['x_right'])
        row_top = int(row['row_top'])
        row_bot = int(row['row_bottom'])

        # Read final position from the line artist (reliable even if cursor left axes)
        try:
            if self._drag_roi == 'xleft':
                new_xl = int(round(self._left_line.get_xdata()[0]))
                new_xl = int(np.clip(new_xl, 0, x_right - 10))
                new_xr = x_right
            elif self._drag_roi == 'xright':
                new_xr = int(round(self._right_line.get_xdata()[0]))
                new_xr = int(np.clip(new_xr, x_left + 10, w - 1))
                new_xl = x_left
            elif self._drag_roi == 'top':
                new_top = int(round(self._top_line.get_ydata()[0]))
                new_top = int(np.clip(new_top, 0, row_bot - 10))
                new_bot = row_bot
            elif self._drag_roi == 'bottom':
                new_bot = int(round(self._bot_line.get_ydata()[0]))
                new_bot = int(np.clip(new_bot, row_top + 10, h - 1))
                new_top = row_top
        except (TypeError, AttributeError):
            self._drag_roi = None
            return

        scale = self.params['scale_mm_per_pixel']

        if self._drag_roi in ('xleft', 'xright'):
            # ── Per-frame correction: only the current frame is updated ────
            # All other frames will be re-interpolated between corrected anchors.
            if self._drag_roi == 'xright':
                self.df.loc[fi, 'x_right'] = new_xr
                if 'x_right_corrected' in self.df.columns:
                    self.df.loc[fi, 'x_right_corrected'] = True
                print(f'Frame {fi}: x_right corrected → {new_xr} px')
            else:
                self.df.loc[fi, 'x_left'] = new_xl
                if 'x_left_corrected' in self.df.columns:
                    self.df.loc[fi, 'x_left_corrected'] = True
                print(f'Frame {fi}: x_left corrected → {new_xl} px')
            # Re-interpolate all frames between anchors, then recompute W_full_0
            self._reinterpolate_edges()
            # Do NOT re-detect x_tip — user's manual corrections must be preserved.

        else:  # top / bottom ROI
            print(f'ROI → row_top={new_top} ({new_top/h*100:.1f}%)  '
                  f'row_bottom={new_bot} ({new_bot/h*100:.1f}%)')
            self.df['row_top']    = new_top
            self.df['row_bottom'] = new_bot
            self._redetect_frame(fi)

        self._drag_roi = None
        self._recompute_all()
        self._save_session()
        self._update_display()

    # ── Scroll zoom ───────────────────────────────────────────────────────────

    def _on_scroll(self, event):
        if event.inaxes is not self.ax_img:
            return
        factor = 0.80 if event.button == 'up' else (1.0 / 0.80)
        xlim = list(self.ax_img.get_xlim())
        ylim = list(self.ax_img.get_ylim())
        xc, yc = event.xdata, event.ydata
        self.ax_img.set_xlim([xc + (x - xc) * factor for x in xlim])
        self.ax_img.set_ylim([yc + (y - yc) * factor for y in ylim])
        self.fig.canvas.draw_idle()

    # ── Keyboard ──────────────────────────────────────────────────────────────

    def _on_key(self, event):
        mapping = {'right': +1, 'n': +1, 'left': -1, 'p': -1, 'a': +1}
        if event.key in mapping:
            self._go(mapping[event.key])
        elif event.key == 'q':
            plt.close(self.fig)
        elif event.key == 'r':
            self._reset_zoom()

    def _reset_zoom(self):
        fi  = self._fi
        img = self._load_cached(fi)
        self.ax_img.set_xlim(0, img.shape[1])
        self.ax_img.set_ylim(img.shape[0], 0)
        self.fig.canvas.draw_idle()

    # ── Re-detection ──────────────────────────────────────────────────────────

    def _load_cached(self, fi: int) -> np.ndarray:
        if fi not in self._img_cache:
            self._img_cache[fi] = load_image(Path(self.image_paths[fi]))
        return self._img_cache[fi]

    def _redetect_frame(self, fi: int):
        img     = self._load_cached(fi)
        x_left  = int(self.df.loc[fi, 'x_left'])
        x_right = int(self.df.loc[fi, 'x_right'])
        row_top = int(self.df.loc[fi, 'row_top'])
        row_bot = int(self.df.loc[fi, 'row_bottom'])
        thresh  = int(self.params.get('img_threshold', 60))
        ns      = self.params['notch_side']
        x_tip, conf = detect_crack_tip(img, x_left, x_right, row_top, row_bot,
                                        ns, thresh)
        self.df.loc[fi, 'x_tip']      = x_tip
        self.df.loc[fi, 'confidence'] = conf

    def _redetect_all(self):
        n = len(self.df)
        print(f'Re-detecting {n} frames …')
        for fi in self.df.index:
            self._redetect_frame(fi)
        self._recompute_all()
        self._save_session()
        self._update_display()
        print('Re-detection complete.')

    # ── Edge re-interpolation ─────────────────────────────────────────────────

    def _reinterpolate_edges(self):
        """
        Re-interpolate x_right / x_left for all uncorrected frames after a
        per-frame correction has been applied to the current frame.

        Uses linear interpolation between corrected anchor frames and a rolling
        median fallback outside the anchor range (same logic as
        smooth_sample_edges_with_anchors in crack_analyser).

        Also updates params['W_full_0_mm'] from the (possibly changed) frame-0
        x_right and x_left values.
        """
        self.df = smooth_sample_edges_with_anchors(self.df, window=11)

        # Update W_full,0 from frame 0 (constant baseline)
        fi0 = self.df.index[0]
        xr0 = int(self.df.loc[fi0, 'x_right'])
        xl0 = int(self.df.loc[fi0, 'x_left'])
        self.params['W_full_0_mm'] = (xr0 - xl0) * self.params['scale_mm_per_pixel']

    def _reinterpolate_tips(self):
        """
        Re-interpolate x_tip for all uncorrected frames after a correction.
        Uses the same anchor-based logic as smooth_crack_tips in crack_analyser:
        linear interpolation between corrected anchors, rolling median outside.
        """
        self.df = smooth_crack_tips(self.df, window=7)

    # ── Recompute measurements ────────────────────────────────────────────────

    def _recompute_all(self):
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
        self.df['delta_a_mm'] = (self.df['a_mm'] - a_0).clip(lower=0)

    def _save_session(self):
        if self.session_path is None:
            return
        save_session(self.df, str(self.session_path.parent))

    # ── Browse-all toggle ─────────────────────────────────────────────────────

    def _toggle_browse(self):
        """
        Switch between 'flagged frames only' and 'all frames' mode.
        The current frame index is preserved as closely as possible.
        """
        current_fi = self._fi   # save the frame we're looking at

        self._browse_mode = not self._browse_mode

        if self._browse_mode:
            # Switch to all frames; find position of current frame in full list
            self.flagged = self._all_frames[:]
            self.btn_browse.label.set_text('Flagged Only')
        else:
            # Switch back to flagged-only list
            from crack_analyser import flag_uncertain_frames
            self.flagged = sorted(set(flag_uncertain_frames(self.df)))
            if not self.flagged:
                self.flagged = [self._all_frames[0]]
            self.btn_browse.label.set_text('Browse All')

        # Try to stay on the same frame
        if current_fi in self.flagged:
            self.cur = self.flagged.index(current_fi)
        else:
            self.cur = 0

        self._update_display()

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self):
        if not self.flagged:
            return
        plt.show()
