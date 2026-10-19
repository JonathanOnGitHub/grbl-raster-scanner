#!/usr/bin/env python3
"""Live annotated view of /dev/video2: detects tube + droplet, controls syringe via GRBL.

Uses PyQt5 for display (this OpenCV was built GUI-less, so cv2.imshow doesn't work).
"""
import argparse
import math
import subprocess
import sys
import time

import cv2
import numpy as np

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import QApplication, QLabel, QWidget


def find_tube_and_droplet(gray, frame_buf, anchor=None):
    """Return (droplet_box, tube_box) or (None, None).

    anchor: optional (x, y) in full-frame pixel coords marking the
    upper tip of the tube (the point the droplet comes out of). When
    supplied, the search bands are constrained around it for robustness.

    Motion-based detector:
      - droplet: pixels whose intensity varied across the rolling buffer
        (refractive highlights twinkle; static parts do not).
      - tube: pixels consistently different from the smoothed background
        (the plastic tube shape, which is static but off-background).
    """
    if frame_buf is None or len(frame_buf) < 5:
        return None, None

    h, w = gray.shape
    arr = np.stack(frame_buf, axis=0)  # (N, h, w)
    span = (arr.max(axis=0).astype(np.int16) - arr.min(axis=0).astype(np.int16))
    bg = np.median(arr, axis=0).astype(np.uint8)
    bg_smooth = cv2.GaussianBlur(bg, (31, 31), 0)
    bg_dev = cv2.absdiff(bg, bg_smooth)

    # If we have an anchor, restrict to ROI bands around it; otherwise
    # fall back to a wide central strip.
    if anchor is not None:
        ax, ay = anchor
        # Horizontal band around the click: +/- 80 px.
        x0 = max(0, ax - 80)
        x1 = min(w, ax + 80)
        # Vertical ROI for tube: everything BELOW the click (anchor is the
        # upper tip of the tube).
        tube_y_top = min(h - 1, max(0, ay))
        tube_y_bot = h
        # Vertical ROI for droplet: everything ABOVE the click and within
        # ~250 px upward.
        droplet_y_top = max(0, ay - 250)
        droplet_y_bot = ay + 5
    else:
        x0, x1 = int(w * 0.2), int(w * 0.8)
        tube_y_top = 0
        tube_y_bot = h
        droplet_y_top = 0
        droplet_y_bot = int(h * 0.6)

    span_roi = span[:, x0:x1]
    bg_dev_roi = bg_dev[:, x0:x1]
    roi_w = x1 - x0

    # --- Droplet mask: pixels with significant variation across the buffer.
    motion = (span_roi > 14).astype(np.uint8) * 255
    motion = cv2.morphologyEx(motion, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # Restrict to the droplet vertical ROI.
    drop_band = np.zeros_like(motion)
    drop_band[droplet_y_top:droplet_y_bot] = motion[droplet_y_top:droplet_y_bot]

    num, _, stats, _ = cv2.connectedComponentsWithStats(drop_band, connectivity=8)
    droplet = None
    best_area = 0
    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        if area < 300 or ww < 20 or hh < 20:
            continue
        aspect = hh / max(1.0, ww)
        if aspect > 3.5 or aspect < 0.3:
            continue
        if area > best_area:
            best_area = area
            droplet = (x, y, ww, hh)

    if droplet is None:
        return None, None
    dx, dy, dw, dh = droplet
    droplet_full = (dx + x0, dy, dw, dh)
    droplet_bottom = dy + dh

    # --- Tube mask: pixels consistently off the local smoothed background.
    tube_band_full = (bg_dev_roi > 4).astype(np.uint8) * 255
    tube_band_full = cv2.morphologyEx(
        tube_band_full, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 25)))

    # If anchor is set, find the longest column-run starting just below
    # the anchor. Otherwise fall back to the previous "longest column in
    # droplet vicinity" logic.
    if anchor is not None:
        # Convert anchor.x to ROI coords.
        ax_roi = ax - x0
        col_win = max(0, ax_roi - 40)
        col_hi = min(roi_w, ax_roi + 40)
        # Restrict vertical scan to below the anchor.
        scan_top = tube_y_top - 0  # already in frame coords; convert to ROI
        scan_bot = tube_y_bot
        best_col = -1
        best_run = 0
        best_run_top = best_run_bot = 0
        for c in range(col_win, col_hi):
            col = tube_band_full[scan_top:scan_bot, c]
            if not col.any():
                continue
            ys = np.where(col > 0)[0]
            if len(ys) < 30:
                continue
            gaps = np.where(np.diff(ys) > 1)[0]
            if len(gaps) == 0:
                run_top, run_bot = ys[0] + scan_top, ys[-1] + scan_top
            else:
                seg = max(np.split(ys, gaps + 1), key=len)
                run_top, run_bot = seg[0] + scan_top, seg[-1] + scan_top
            run = run_bot - run_top
            if run > best_run:
                best_run = run
                best_col = c
                best_run_top = run_top
                best_run_bot = run_bot
    else:
        dx_c = dx + dw // 2
        col_win = max(0, dx_c - 60)
        col_hi = min(roi_w, dx_c + 60)
        if col_hi <= col_win:
            return droplet_full, None
        best_col = -1
        best_run = 0
        best_run_top = best_run_bot = 0
        for c in range(col_win, col_hi):
            col = tube_band_full[:, c]
            if not col.any():
                continue
            ys = np.where(col > 0)[0]
            if len(ys) < 30:
                continue
            gaps = np.where(np.diff(ys) > 1)[0]
            if len(gaps) == 0:
                run_top, run_bot = ys[0], ys[-1]
            else:
                seg = max(np.split(ys, gaps + 1), key=len)
                run_top, run_bot = seg[0], seg[-1]
            run = run_bot - run_top
            if run > best_run:
                best_run = run
                best_col = c
                best_run_top = run_top
                best_run_bot = run_bot

    if best_col < 0 or best_run < 40:
        return droplet_full, None

    # Tube bbox: same horizontal extent as droplet (or click anchor), and
    # vertical extent from just above the droplet bottom down to the run bottom.
    if anchor is not None:
        pad = 30
        tube_x0 = max(0, ax - pad)
        tube_x1 = min(w, ax + pad)
    else:
        tube_x0 = max(0, dx + x0 - 10)
        tube_x1 = min(w, dx + x0 + dw + 10)
    tube_y0 = droplet_bottom
    tube_y1 = best_run_bot
    tube = (tube_x0, tube_y0, tube_x1 - tube_x0, tube_y1 - tube_y0)
    return droplet_full, tube


def droplet_volume_mm3(droplet_h_px, droplet_w_px, mm_per_px, tube_radius_mm):
    """Estimate hanging-droplet volume from bbox height/width and tube radius.

    Two regimes:
      - Small droplet (h <= 2r): spherical cap on the tube tip.
            V = (pi h^2 / 3) (3r - h)
      - Large droplet (h > 2r, the droplet bulges past a hemisphere):
            prolate spheroid V = (4/3) pi a^2 b with a = w/2, b = h/2.

    Returns (volume_mm3, model) or (None, None) on invalid input.
    """
    if mm_per_px is None or tube_radius_mm is None or tube_radius_mm <= 0:
        return None, None
    if droplet_h_px is None or droplet_w_px is None:
        return None, None
    h_mm = droplet_h_px * mm_per_px
    w_mm = droplet_w_px * mm_per_px
    if h_mm <= 0 or w_mm <= 0:
        return None, None
    r_mm = tube_radius_mm
    if h_mm <= 2 * r_mm:
        v = (math.pi * h_mm * h_mm / 3.0) * (3.0 * r_mm - h_mm)
        return v, "cap"
    # Prolate spheroid: short axis = horizontal half-width, long axis = vertical half-height.
    a = w_mm / 2.0
    b = h_mm / 2.0
    v = (4.0 / 3.0) * math.pi * a * a * b
    return v, "spheroid"


def annotate(frame, droplet, tube, tube_diameter_mm=None, mm_per_px=None, anchor=None,
              pending_click=None):
    out = frame.copy()

    # Draw a pending first click as a small marker so the user sees it.
    if pending_click is not None:
        px, py = pending_click
        cv2.circle(out, (px, py), 5, (255, 255, 0), 2)

    if anchor is not None:
        ax, ay = anchor
        cv2.line(out, (ax - 18, ay), (ax + 18, ay), (255, 255, 0), 2)
        cv2.line(out, (ax, ay - 18), (ax, ay + 18), (255, 255, 0), 2)
        cv2.circle(out, (ax, ay), 4, (255, 255, 0), -1)

    if tube is not None:
        x, y, w, h = tube
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if droplet is not None:
        x, y, w, h = droplet
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cx = x + w // 2
        cy = y + h // 2
        cv2.circle(out, (cx, cy), 4, (0, 0, 255), -1)

    return out


def metrics_lines(droplet, tube, tube_diameter_mm, mm_per_px, target_nL=None):
    """Return a list of text lines for the top-left overlay panel."""
    lines = []
    if target_nL is not None:
        lines.append(f"target : {target_nL:.0f} nL")
    if droplet is None and tube is None:
        lines.append("droplet: --")
        lines.append("tube   : --")
        return lines

    if tube is not None:
        x, y, w, h = tube
        if mm_per_px is not None:
            lines.append(f"tube   : {w * mm_per_px:.2f} x {h * mm_per_px:.2f} mm  ({w}x{h} px)")
        else:
            lines.append(f"tube   : {w} x {h} px")

    if droplet is not None:
        x, y, w, h = droplet
        if mm_per_px is not None:
            lines.append(f"droplet: {w * mm_per_px:.2f} x {h * mm_per_px:.2f} mm  ({w}x{h} px)")
        else:
            lines.append(f"droplet: {w} x {h} px")

        if mm_per_px is not None and tube_diameter_mm is not None:
            vol_mm3, model = droplet_volume_mm3(
                h, w, mm_per_px, tube_diameter_mm / 2.0)
            if vol_mm3 is not None:
                nL = vol_mm3 * 1e3
                if target_nL is not None:
                    err_pct = 100.0 * (nL - target_nL) / max(target_nL, 1e-6)
                    lines.append(f"volume : {nL:.1f} nL  ({err_pct:+.1f}% vs target)")
                else:
                    lines.append(f"volume : {nL:.1f} nL  (model: {model})")

        if tube is not None and mm_per_px is not None:
            tx, ty, tw, th = tube
            gap_px = ty - (y + h)
            lines.append(f"gap    : {gap_px * mm_per_px:.3f} mm  ({gap_px} px)")

    if mm_per_px is not None:
        lines.append(f"scale  : {mm_per_px:.4f} mm/px  ({1 / mm_per_px:.1f} px/mm)")

    return lines


def draw_metrics_panel(frame, lines, origin=(20, 20), line_height=22, font_scale=0.55):
    """Draw a black panel with the given lines onto `frame` (in place).

    Returns the frame.
    """
    if not lines:
        return frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    pad = 12
    # Measure widest line.
    widths = [cv2.getTextSize(ln, font, font_scale, thickness)[0][0] for ln in lines]
    panel_w = max(widths) + 2 * pad
    panel_h = line_height * len(lines) + 2 * pad
    x0, y0 = origin
    # Background.
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    # Cyan border accent.
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (255, 255, 0), 1)
    # Text.
    for i, ln in enumerate(lines):
        ty = y0 + pad + line_height * (i + 1) - 6
        cv2.putText(frame, ln, (x0 + pad, ty), font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)
    return frame


def push_syringe(port, delta):
    cmd = ["python3", "grbl_x_move.py", port, f"{delta}"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return r.returncode == 0, (r.stdout + r.stderr).strip()
    except Exception as e:
        return False, str(e)


class PumpModel:
    """Piecewise-linear pump model for a peristaltic pump with a dead zone.

    ΔV_nL = max(0, slope * (|grbl_delta| - dead_zone) * 1000) * sign(grbl_delta)

    Attributes:
        slope: mm^3 of water per grbl unit ABOVE the dead zone.
        dead_zone: grbl units of "take-up" before water starts moving.
        learn_rate: EMA factor [0,1] for adapting from new observations.
    """

    def __init__(self, slope=0.5, dead_zone=0.0, learn_rate=0.5,
                 min_slope=0.05, max_dead_zone=0.4):
        self.slope = slope
        self.dead_zone = dead_zone
        self.learn_rate = learn_rate
        self.min_slope = min_slope
        self.max_dead_zone = max_dead_zone
        self.history = []  # list of (grbl_delta, observed_dV_nL)

    def predict_nL(self, grbl_delta):
        """Return expected volume change in nL for a given grbl step."""
        eff = abs(grbl_delta) - self.dead_zone
        if eff <= 0:
            return 0.0
        return eff * self.slope * 1000.0 * (1 if grbl_delta > 0 else -1)

    def grbl_for_nL(self, target_nL):
        """Inverse: how many grbl units to achieve target_nL volume change."""
        sign = 1 if target_nL >= 0 else -1
        abs_nL = abs(target_nL)
        if self.slope <= 0:
            return sign * (self.dead_zone + 0.5)  # arbitrary
        return sign * (self.dead_zone + abs_nL / (self.slope * 1000.0))

    def update(self, grbl_delta, observed_dV_nL):
        """Adapt the model from a new observation (EMA)."""
        self.history.append((grbl_delta, observed_dV_nL))
        a = self.learn_rate
        if abs(grbl_delta) <= 1e-9:
            return
        # If we observed near-zero volume change, the step was inside the
        # dead zone (or the pump didn't move). Bump up dead_zone.
        if abs(observed_dV_nL) < 1.0:  # less than 1 nL
            new_dz = min(self.dead_zone + 0.5 * abs(grbl_delta),
                         self.max_dead_zone)
            self.dead_zone = (1 - a) * self.dead_zone + a * new_dz
            return
        # Otherwise, the pump moved. If our dead zone is bigger than this
        # step, reduce it slightly (the dead zone must be smaller than the
        # smallest step that produced water).
        if abs(grbl_delta) <= self.dead_zone:
            self.dead_zone = (1 - a) * self.dead_zone + a * 0.7 * abs(grbl_delta)
            return
        # Above the dead zone: fit slope.
        eff = abs(grbl_delta) - self.dead_zone
        implied_slope = abs(observed_dV_nL) / (eff * 1000.0)
        self.slope = max(self.min_slope,
                         (1 - a) * self.slope + a * implied_slope)

    def calibrate(self, port, grbl_scale, measure_nL, settle_s=1.0,
                  kick_size=0.1, n_kicks=2, dead_zone_probe=0.2):
        """Run a small calibration sequence: kick +kick_size *n_kicks,
        then back. Returns a short status string.

        measure_nL: callable returning current volume in nL.
        dead_zone_probe: extra small forward + back probes to estimate
            the dead zone. Skipped if 0.
        """
        before = measure_nL()
        if before is None:
            return "calibrate: no initial measurement"
        # Optional small-kick probe to estimate dead zone.
        dV_probe_fwd = 0.0
        if dead_zone_probe > 0:
            ok, _ = push_syringe(port, dead_zone_probe * grbl_scale)
            if ok:
                time.sleep(settle_s)
                after_probe = measure_nL()
                if after_probe is not None:
                    dV_probe_fwd = after_probe - before
                ok, _ = push_syringe(port, -dead_zone_probe * grbl_scale)
                if ok:
                    time.sleep(settle_s)
                    settled = measure_nL()
                    if settled is not None:
                        # Reset drift to original.
                        pass
        # Main forward kicks.
        for _ in range(n_kicks):
            ok, _ = push_syringe(port, kick_size * grbl_scale)
            if not ok:
                return "calibrate: push failed"
            time.sleep(settle_s)
        peak = measure_nL()
        if peak is None:
            return "calibrate: no measurement after forward kick"
        dV_fwd = peak - before
        # Push back the same total.
        for _ in range(n_kicks):
            ok, _ = push_syringe(port, -kick_size * grbl_scale)
            if not ok:
                return "calibrate: return push failed"
            time.sleep(settle_s)
        after = measure_nL()
        if after is None:
            return "calibrate: no measurement after return"
        dV_back = after - peak
        # Fit: forward grbl = +kick_size*n_kicks, observed dV_fwd nL.
        total_grbl = kick_size * n_kicks
        sample = max(abs(dV_fwd), abs(dV_back))
        # Update dead zone estimate from the small-kick probe.
        if dead_zone_probe > 0 and abs(dV_probe_fwd) < 1.0:
            self.dead_zone = min(self.max_dead_zone, dead_zone_probe)
        # Update slope from the main kick (assumes pump operates above DZ).
        if sample < 1.0:
            self.dead_zone = min(self.max_dead_zone, total_grbl * 0.8)
            return (f"calibrate: weak response (dV={sample:.1f}nL for "
                    f"{total_grbl}grbl), set dead_zone={self.dead_zone:.3f}grbl")
        eff = max(total_grbl - self.dead_zone, 0.01)
        implied_slope = sample / (eff * 1000.0)
        self.slope = max(self.min_slope, implied_slope)
        return (f"calibrate: fwd dV={dV_fwd:.0f}nL back dV={dV_back:.0f}nL "
                f"slope={self.slope:.3f}mm^3/grbl dead_zone={self.dead_zone:.3f}grbl")


def volume_feedback(port, grbl_scale, target_nL, get_volume_nL,
                    pump_model, tolerance_pct=10.0, max_iters=12,
                    settle_s=1.0, max_step=2.0, on_iter=None):
    """Adaptive volume-feedback loop using a PumpModel.

    Each iteration:
      1. measure current volume
      2. compute required dV_nL
      3. ask the pump model for the grbl step that would produce dV_nL
      4. push
      5. wait, re-measure
      6. update the model from (grbl_delta, observed dV_nL)
      7. check tolerance, repeat
    """
    history = []
    last_grbl = None
    if on_iter:
        try:
            on_iter(0, None, target_nL, None)
        except Exception:
            pass
    for it in range(1, max_iters + 1):
        cur = get_volume_nL()
        if cur is None:
            return {"reached": False, "iters": it, "final_nL": None,
                    "target_nL": target_nL, "history": history,
                    "reason": "no measurement"}
        if on_iter:
            try:
                on_iter(it, cur, target_nL, None)
            except Exception:
                pass
        err_nL = target_nL - cur
        if target_nL > 0 and abs(err_nL) / target_nL <= tolerance_pct / 100.0:
            return {"reached": True, "iters": it, "final_nL": cur,
                    "target_nL": target_nL, "history": history,
                    "reason": "in tolerance"}
        grbl_delta = pump_model.grbl_for_nL(err_nL)
        # Cap step size.
        if abs(grbl_delta) > max_step:
            grbl_delta = math.copysign(max_step, grbl_delta)
        # If dead zone dominates (err small, model says step is mostly DZ),
        # try a single effective step just above the dead zone.
        if (pump_model.dead_zone > 0
                and abs(grbl_delta) <= pump_model.dead_zone * 1.05
                and abs(err_nL) > 0):
            grbl_delta = math.copysign(
                pump_model.dead_zone + 0.5 * max_step, grbl_delta)
        if abs(grbl_delta) < 1e-6:
            return {"reached": True, "iters": it, "final_nL": cur,
                    "target_nL": target_nL, "history": history,
                    "reason": "delta=0"}
        logical_delta = grbl_delta * grbl_scale
        ok, _msg = push_syringe(port, logical_delta)
        history.append((grbl_delta, cur))
        if not ok:
            return {"reached": False, "iters": it, "final_nL": cur,
                    "target_nL": target_nL, "history": history,
                    "reason": "push failed"}
        time.sleep(settle_s)
        cur2 = get_volume_nL()
        if cur2 is None:
            return {"reached": False, "iters": it, "final_nL": None,
                    "target_nL": target_nL, "history": history,
                    "reason": "no measurement after push"}
        history.append((grbl_delta, cur2))
        dV_nL = cur2 - cur
        pump_model.update(grbl_delta, dV_nL)
        last_grbl = grbl_delta
    cur = get_volume_nL()
    return {"reached": False, "iters": max_iters, "final_nL": cur,
            "target_nL": target_nL, "history": history,
            "reason": "max iters", "pump_model": pump_model}


class Viewer(QWidget):
    def __init__(self, cap, port, step, buf_len=30, grbl_scale=10.0, grbl_unit="mm",
                 tube_diameter_mm=1.0, frame_size=(640, 480),
                 mm3_per_grbl_unit=0.5, vol_tolerance_pct=10.0,
                 vol_max_iters=12, vol_settle_s=1.0):
        super().__init__()
        self.cap = cap
        self.port = port
        self.step = step
        self.grbl_scale = grbl_scale
        self.grbl_unit = grbl_unit
        self.tube_diameter_mm = tube_diameter_mm
        self.frame_w, self.frame_h = frame_size
        self.mm3_per_grbl_unit = mm3_per_grbl_unit
        self.vol_tolerance_pct = vol_tolerance_pct
        self.vol_max_iters = vol_max_iters
        self.vol_settle_s = vol_settle_s
        self.mm_per_px = None        # set when two nozzle clicks are paired
        self.anchor = None           # (x, y) midpoint + click row of the nozzle
        self._pending_click = None   # first click of a pair, awaiting second
        self._multiplier_buf = ""
        self._last_msg = ""
        self.buf_len = buf_len
        self._volume_target_nL = None  # set when feedback run is in progress
        self._volume_busy = False
        # Adaptive pump model: a peristaltic pump has a dead zone and
        # approximately linear flow above it. The model self-calibrates
        # from observations during each feedback run.
        self.pump_model = PumpModel(slope=0.5, dead_zone=0.0)
        self._pump_calibrated = False
        self.frame_buf = []  # rolling buffer of grayscale frames
        self.setWindowTitle(f"tube+droplet  ({port}  step={step})")
        self.setStyleSheet("QWidget { background: black; }")

        self.label = QLabel(self)  # child of the window so it paints
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMinimumSize(1, 1)
        self.label.setGeometry(0, 0, self.width(), self.height())

        self.status = QLabel(self.label)  # overlay on the video
        self.status.setStyleSheet(
            "QLabel { background-color: rgba(0,0,0,160); color: white; "
            "padding: 8px 14px; font-size: 16px; }"
        )
        self.status.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.status.setMinimumHeight(36)
        self.status.setMaximumHeight(48)

        self._update_status()

        screen = QApplication.primaryScreen().geometry()
        self.resize(screen.width(), screen.height())
        self.showFullScreen()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)  # ~30 fps

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.label.setGeometry(0, 0, self.width(), self.height())
        margin = 16
        self.status.adjustSize()
        sh = self.status.sizeHint()
        self.status.setGeometry(margin,
                                self.height() - sh.height() - margin,
                                min(self.width() - 2 * margin, sh.width() + 24),
                                sh.height())
        self.status.raise_()

    def _current_multiplier(self):
        if not self._multiplier_buf:
            return 1
        try:
            return int(self._multiplier_buf)
        except ValueError:
            return 1

    def _update_status(self, last_msg=""):
        m = self._multiplier_buf or "1"
        n = len(self.frame_buf)
        warm = "" if n >= 5 else f"  warming up ({n}/5)"
        unit = self.grbl_unit
        scale = ""
        if self.mm_per_px is not None:
            scale = f"  scale={self.mm_per_px:.4f}mm/px"
        if self._pending_click is not None:
            anch = "  anchor=click other edge"
        elif self.anchor is not None:
            anch = "  anchor=set"
        else:
            anch = ""
        self.status.setText(
            f"step={self.step}{unit}  x{m}{scale}{anch}    "
            f"L-click 2x=nozzle  R=cal pump  n=clear  z/x=push  #V=target nL  q=quit  Esc=exit{warm}    {last_msg}"
        )
        self.status.adjustSize()
        # Re-pin to the bottom-left.
        margin = 16
        sh = self.status.sizeHint()
        self.status.setGeometry(margin,
                                self.height() - sh.height() - margin,
                                min(self.width() - 2 * margin, sh.width() + 24),
                                sh.height())
        self.status.raise_()

    def _current_volume_nL(self):
        """Measure the current droplet volume in nL, or None if not measurable."""
        if not self.frame_buf or self.mm_per_px is None:
            return None
        gray = self.frame_buf[-1]
        d, t = find_tube_and_droplet(gray, self.frame_buf, anchor=self.anchor)
        if d is None:
            return None
        vol_mm3, _model = droplet_volume_mm3(
            d[3], d[2], self.mm_per_px, self.tube_diameter_mm / 2.0)
        if vol_mm3 is None:
            return None
        return vol_mm3 * 1e3

    def _set_volume_target(self, target_nL):
        """Run a volume-feedback loop on a background thread."""
        import threading
        if self.mm_per_px is None:
            self._last_msg = "[vol] no scale - click nozzle first"
            print(self._last_msg, file=sys.stderr)
            self._update_status(self._last_msg)
            return
        if self.tube_diameter_mm is None:
            self._last_msg = "[vol] no tube diameter"
            print(self._last_msg, file=sys.stderr)
            self._update_status(self._last_msg)
            return

        def worker():
            def get_vol():
                return self._current_volume_nL()

            self._volume_busy = True
            QTimer.singleShot(0, lambda: self._set_target(target_nL))
            try:
                if not self._pump_calibrated:
                    cal_msg = self.pump_model.calibrate(
                        self.port, self.grbl_scale, get_vol,
                        settle_s=self.vol_settle_s)
                    print(cal_msg, file=sys.stderr)
                    QTimer.singleShot(0, lambda m=cal_msg: self._set_msg(m))
                    self._pump_calibrated = True
                _run()
            finally:
                self._volume_busy = False
                QTimer.singleShot(0, lambda: self._set_target(None))

        def _run():
            def report(it, cur, tgt, delta):
                msg = (f"[vol] iter {it}: target={tgt:.0f}nL "
                       f"current={'?' if cur is None else f'{cur:.0f}'}nL")
                if delta is not None:
                    msg += (f" step={delta * self.grbl_scale:+.3f}{self.grbl_unit} "
                            f"({delta:+.3f}grbl)")
                print(msg, file=sys.stderr)
                QTimer.singleShot(0, lambda m=msg: self._set_msg(m))

            res = volume_feedback(
                self.port, self.grbl_scale,
                target_nL, get_vol, self.pump_model,
                tolerance_pct=self.vol_tolerance_pct,
                max_iters=self.vol_max_iters,
                settle_s=self.vol_settle_s,
                on_iter=report,
            )
            tag = "OK" if res["reached"] else "FAIL"
            final = res["final_nL"]
            final_s = f"{final:.0f}" if final is not None else "?"
            model = self.pump_model
            line = (f"[vol {tag}] target={res['target_nL']:.0f}nL "
                    f"final={final_s}nL iters={res['iters']} "
                    f"reason={res.get('reason', '?')} "
                    f"slope={model.slope:.3f}mm^3/grbl "
                    f"DZ={model.dead_zone:.3f}grbl")
            print(line, file=sys.stderr)
            QTimer.singleShot(0, lambda l=line: self._set_msg(l))

        threading.Thread(target=worker, daemon=True).start()

    def _recalibrate_pump(self):
        """Run pump calibration on a background thread; updates self.pump_model."""
        import threading
        if self._volume_busy:
            self._last_msg = "[cal] busy, please wait"
            print(self._last_msg, file=sys.stderr)
            self._update_status(self._last_msg)
            return
        if self.mm_per_px is None:
            self._last_msg = "[cal] no scale - click nozzle first"
            print(self._last_msg, file=sys.stderr)
            self._update_status(self._last_msg)
            return

        def worker():
            def get_vol():
                return self._current_volume_nL()

            self._volume_busy = True
            try:
                msg = self.pump_model.calibrate(
                    self.port, self.grbl_scale, get_vol,
                    settle_s=self.vol_settle_s)
                print(msg, file=sys.stderr)
                QTimer.singleShot(0, lambda m=msg: self._set_msg(m))
                self._pump_calibrated = True
            finally:
                self._volume_busy = False

        threading.Thread(target=worker, daemon=True).start()

    def _set_msg(self, msg):
        self._last_msg = msg
        self._update_status(msg)

    def _set_target(self, nL):
        self._volume_target_nL = nL
        self._update_status(self._last_msg)

    def _push(self, sign):
        mult = self._current_multiplier()
        delta = sign * self.step * mult / self.grbl_scale
        ok, msg = push_syringe(self.port, delta)
        unit = self.grbl_unit
        tag = "OK" if ok else "ERR"
        line = (f"[{tag}] push {delta * self.grbl_scale:+.2f}{unit} "
                f"({delta:+.4f} grbl) {msg}").strip()
        print(line, file=sys.stderr)
        self._last_msg = line
        self._update_status(line)
        self._multiplier_buf = ""

    def _scaled_image_rect(self):
        """Where the video frame sits inside the label, after KeepAspectRatio."""
        lbl_w, lbl_h = self.label.width(), self.label.height()
        if self.frame_w <= 0 or self.frame_h <= 0:
            return None
        scale = min(lbl_w / self.frame_w, lbl_h / self.frame_h)
        draw_w = self.frame_w * scale
        draw_h = self.frame_h * scale
        x0 = (lbl_w - draw_w) / 2.0
        y0 = (lbl_h - draw_h) / 2.0
        return x0, y0, draw_w, draw_h, scale

    def _widget_to_frame(self, wx, wy):
        r = self._scaled_image_rect()
        if r is None:
            return None
        x0, y0, draw_w, draw_h, scale = r
        if wx < x0 or wy < y0 or wx > x0 + draw_w or wy > y0 + draw_h:
            return None
        fx = (wx - x0) / scale
        fy = (wy - y0) / scale
        return int(round(fx)), int(round(fy))

    def mousePressEvent(self, e):
        btn = e.button()
        pt = self._widget_to_frame(e.x(), e.y())
        if pt is None:
            return
        # Right-click cancels any pending pair.
        if btn in (Qt.RightButton, Qt.MiddleButton):
            if self._pending_click is not None:
                self._pending_click = None
                self._last_msg = "[anchor] pending click cleared"
                print(self._last_msg, file=sys.stderr)
            else:
                self._last_msg = "[anchor] nothing to cancel"
            self._update_status(self._last_msg)
            return
        if btn != Qt.LeftButton:
            return
        if self._pending_click is None:
            # First click of a pair.
            self._pending_click = pt
            self._last_msg = (f"[anchor] 1st edge at ({pt[0]}, {pt[1]}) - "
                              f"click the other edge of the nozzle")
            print(self._last_msg, file=sys.stderr)
            self._update_status(self._last_msg)
            return
        # Second click of a pair.
        p1 = self._pending_click
        p2 = pt
        self._pending_click = None
        if abs(p1[1] - p2[1]) > 30:
            self._last_msg = (f"[anchor] clicks too far apart vertically "
                              f"({abs(p1[1]-p2[1])}px) - try again, right-click to cancel")
            print(self._last_msg, file=sys.stderr)
            self._update_status(self._last_msg)
            return
        x_left, x_right = sorted([p1[0], p2[0]])
        width_px = x_right - x_left
        if width_px < 5:
            self._last_msg = "[anchor] clicks too close (<5px) - try again"
            print(self._last_msg, file=sys.stderr)
            self._update_status(self._last_msg)
            return
        mid_x = (x_left + x_right) // 2
        mid_y = (p1[1] + p2[1]) // 2
        self.anchor = (mid_x, mid_y)
        self.mm_per_px = self.tube_diameter_mm / width_px
        self._last_msg = (f"[anchor] nozzle: width={width_px}px -> "
                          f"{self.mm_per_px:.4f} mm/px, "
                          f"tip=({mid_x}, {mid_y})")
        print(self._last_msg, file=sys.stderr)
        self._update_status(self._last_msg)

    def showHelp(self):
        print(
            "Click the LEFT edge of the nozzle, then the RIGHT edge.\n"
            "This sets the tube diameter (1 mm default) and anchors detection.\n"
            "Right-click cancels a pending first click.\n"
            "Press 'n' to clear the anchor and scale (back to auto-detect).",
            file=sys.stderr,
        )

    def tick(self):
        ok, frame = self.cap.read()
        if not ok:
            return
        # Camera is mounted 180° from reality: rotate once here so the
        # detector, annotations, anchor clicks and metrics all work in the
        # same coordinates the user sees.
        frame = cv2.flip(frame, -1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame_buf.append(gray)
        if len(self.frame_buf) > self.buf_len:
            self.frame_buf.pop(0)
        droplet, tube = find_tube_and_droplet(gray, self.frame_buf, anchor=self.anchor)
        self._update_status(self._last_msg)
        vis = annotate(frame, droplet, tube,
                       tube_diameter_mm=self.tube_diameter_mm,
                       mm_per_px=self.mm_per_px,
                       anchor=self.anchor,
                       pending_click=self._pending_click)
        lines = metrics_lines(droplet, tube,
                              self.tube_diameter_mm, self.mm_per_px,
                              target_nL=self._volume_target_nL)
        draw_metrics_panel(vis, lines)
        vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        h, w, ch = vis.shape
        img = QImage(vis.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        self.label.setPixmap(pix.scaled(self.size(),
                                        Qt.KeepAspectRatio,
                                        Qt.SmoothTransformation))
        self.status.raise_()

    def keyPressEvent(self, e):
        k = e.key()
        if k == Qt.Key_Escape:
            self.close()
            return
        if k in (Qt.Key_Q,):
            self.close()
            return
        text = e.text()
        if text and text.isdigit() and len(self._multiplier_buf) < 4:
            self._multiplier_buf += text
            self._update_status()
            return
        if k in (Qt.Key_Plus, Qt.Key_Equal, Qt.Key_Z):
            self._push(+1)
            return
        if k in (Qt.Key_Minus, Qt.Key_X):
            self._push(-1)
            return
        if k in (Qt.Key_V,):
            if self._volume_busy:
                self._last_msg = "[vol] busy, please wait"
                print(self._last_msg, file=sys.stderr)
                self._update_status(self._last_msg)
                return
            try:
                tgt = int(self._multiplier_buf) if self._multiplier_buf else None
            except ValueError:
                tgt = None
            self._multiplier_buf = ""
            if tgt is None or tgt <= 0:
                self._last_msg = "[vol] need a positive number before V (e.g. 50V)"
                print(self._last_msg, file=sys.stderr)
                self._update_status(self._last_msg)
                return
            self._set_volume_target(tgt)
            return
        if k in (Qt.Key_Backspace,):
            self._multiplier_buf = self._multiplier_buf[:-1]
            self._update_status()
            return
        if k in (Qt.Key_N,):
            self.anchor = None
            self._pending_click = None
            self.mm_per_px = None
            self._last_msg = "[anchor] cleared (auto-detect, no scale)"
            print(self._last_msg, file=sys.stderr)
            self._update_status(self._last_msg)
            return
        if k in (Qt.Key_R,):
            self._recalibrate_pump()
            return
        super().keyPressEvent(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="/dev/video2")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--step", type=float, default=0.1,
                    help="logical step per keypress (default 0.1 mm)")
    ap.add_argument("--grbl-scale", type=float, default=10.0,
                    help="multiplier to convert logical step -> grbl argument "
                         "(default 10.0: 0.1 mm -> 0.01 grbl units)")
    ap.add_argument("--grbl-unit", default="mm",
                    help="label for the logical step shown in the status bar")
    ap.add_argument("--tube-diameter-mm", type=float, default=1.0,
                    help="tube outer diameter in mm (used for nozzle-click "
                         "calibration; e.g. 0.2 for a 0.2 mm OD nozzle)")
    ap.add_argument("--mm3-per-grbl", type=float, default=0.5,
                    help="mm^3 of water dispensed per +1.0 grbl x step "
                         "(used to convert volume error -> grbl move). "
                         "Calibrate by measuring how much volume a known "
                         "push adds.")
    ap.add_argument("--vol-tolerance-pct", type=float, default=10.0,
                    help="stop volume feedback when |V-Vtarget|/Vtarget <= this")
    ap.add_argument("--vol-max-iters", type=int, default=12)
    ap.add_argument("--vol-settle-s", type=float, default=1.0,
                    help="seconds to wait after each push before re-measuring")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"failed to open {args.video}", file=sys.stderr)
        return 2

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    print("Keys: q=quit  +/- (or z/x)=push +/-step", file=sys.stderr)
    print(f"Video: {args.video}  GRBL port: {args.port}  step: {args.step}{args.grbl_unit}  "
          f"grbl-scale: {args.grbl_scale}  tube: {args.tube_diameter_mm}mm", file=sys.stderr)

    app = QApplication(sys.argv)
    win = Viewer(cap, args.port, args.step,
                 grbl_scale=args.grbl_scale, grbl_unit=args.grbl_unit,
                 tube_diameter_mm=args.tube_diameter_mm,
                 frame_size=(frame_w, frame_h),
                 mm3_per_grbl_unit=args.mm3_per_grbl,
                 vol_tolerance_pct=args.vol_tolerance_pct,
                 vol_max_iters=args.vol_max_iters,
                 vol_settle_s=args.vol_settle_s)
    win.show()
    win.showHelp()
    rc = app.exec_()
    cap.release()
    return rc


if __name__ == "__main__":
    sys.exit(main())