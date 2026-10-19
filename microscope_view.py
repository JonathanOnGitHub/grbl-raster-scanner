#!/usr/bin/env python3
"""Live viewer for a USB microscope (V4L2).

Features:
  - Auto-detects /dev/videoX devices and picks one that returns frames
  - Live view in a resizable PyQt5 window
  - Resolution selector (populated from the device's supported formats)
  - Basic V4L2 controls (brightness, contrast, saturation, hue, gamma,
    exposure, focus, white balance, zoom) with sliders, when available
  - Crosshair / reticle overlay + optional calibrated scale bar
  - Snapshot to disk (PNG, sequentially numbered)
  - FITS the frame to the window while preserving aspect ratio
"""

import os
import sys
import glob
import time
import argparse
from datetime import datetime

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets


PREFERRED_FOURCC = cv2.VideoWriter_fourcc(*"MJPG")
FALLBACK_FOURCC = cv2.VideoWriter_fourcc(*"YUYV")


def _ffmpeg_max_resolution(dev):
    """Use `ffmpeg -list_formats all` to find the largest MJPEG resolution
    the device supports without actually opening it. Returns (0, 0) if the
    tool isn't available or no usable format exists."""
    import subprocess
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-f", "v4l2",
             "-list_formats", "all", "-i", dev],
            capture_output=True, text=True, timeout=3)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 0, 0
    best = (0, 0)
    for line in out.stderr.splitlines():
        line = line.strip()
        if "mjpeg" not in line.lower():
            continue
        # lines look like: "  Compressed: mjpeg : Motion-JPEG : 1280x720 1920x1080 ..."
        tokens = line.split(":")
        if len(tokens) < 4:
            continue
        for tok in tokens[3].split():
            if "x" in tok:
                parts = tok.split("x")
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    w, h = int(parts[0]), int(parts[1])
                    if w * h > best[0] * best[1]:
                        best = (w, h)
    return best


def _udev_info(dev):
    """Read udev properties for a /dev/videoX node. Returns a dict, or None."""
    import subprocess
    try:
        out = subprocess.run(
            ["udevadm", "info", "--query=property", f"--name={dev}"],
            capture_output=True, text=True, timeout=2)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    info = {}
    for line in out.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k] = v
    return info


def list_video_devices():
    """Return [(path, w, h, score, label), ...] for usable V4L2 capture devices.

    Skips metadata-only nodes (VIDIOC_G_INPUT returns ENOTTY). Higher score
    means a more likely USB microscope / external camera — used to pick a
    sensible default when --dev isn't given.
    """
    out = []
    for d in sorted(glob.glob("/dev/video*")):
        cap = cv2.VideoCapture(d)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FOURCC, PREFERRED_FOURCC)
        try:
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 500)
        except Exception:
            pass
        ok, frame = cap.read()
        if not (ok and frame is not None and frame.size > 0):
            cap.release()
            continue
        default_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        default_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        info = _udev_info(d) or {}
        vendor = (info.get("ID_USB_VENDOR") or info.get("ID_VENDOR")
                  or "").strip()
        model = (info.get("ID_USB_MODEL") or info.get("ID_MODEL")
                 or "").strip()
        label = f"{vendor} {model}".strip() or os.path.basename(d)

        score = 0
        model_l = model.lower()
        vendor_l = vendor.lower()
        idx = int(os.path.basename(d)[5:])

        # Discriminator — built-in laptop webcams identify with a real
        # vendor and a model like "Integrated Camera". Cheap USB microscopes
        # almost always identify with vendor=Generic.
        is_likely_webcam = (
            "integrated" in model_l or
            ("generic" not in vendor_l and vendor not in ("", "Generic")
             and model not in ("", "HD_camera", "USB Camera", "USB2.0 Camera"))
        )
        if is_likely_webcam:
            score -= 1_000_000
        else:
            score += 1000

        # Larger native MJPEG resolution is a (weak) microscope signal.
        max_w, max_h = _ffmpeg_max_resolution(d)
        score += max_w * max_h // 1000
        score += idx
        out.append((d, default_w, default_h, score, label))
    return out


def pick_default_device(devs):
    """Pick the most microscope-like device from the candidates."""
    if not devs:
        return None
    if len(devs) == 1:
        return devs[0][0]
    return max(devs, key=lambda t: t[3])[0]


def device_label(dev):
    """Human-readable label for a /dev/videoX node, for the UI dropdown."""
    info = _udev_info(dev) or {}
    vendor = (info.get("ID_USB_VENDOR") or info.get("ID_VENDOR")
              or "").strip()
    model = (info.get("ID_USB_MODEL") or info.get("ID_MODEL")
             or "").strip()
    if vendor or model:
        return f"{vendor} {model}".strip()
    return dev


def probe_resolutions(dev):
    """List supported resolutions for the device, without opening it.

    Uses ffmpeg's V4L2 format enumeration so we don't fight the streaming
    thread for the device. Returns [(w, h, fourcc_str), ...].
    """
    import subprocess
    res = []
    seen = set()
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-f", "v4l2",
             "-list_formats", "all", "-i", dev],
            capture_output=True, text=True, timeout=3)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return res
    for line in out.stderr.splitlines():
        line = line.strip()
        lower = line.lower()
        fourcc = None
        if "mjpeg" in lower:
            fourcc = "MJPG"
        elif "yuyv" in lower:
            fourcc = "YUYV"
        if fourcc is None or ":" not in line:
            continue
        for tok in line.split(":")[-1].split():
            if "x" in tok:
                parts = tok.split("x")
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    w, h = int(parts[0]), int(parts[1])
                    key = (w, h, fourcc)
                    if key not in seen:
                        seen.add(key)
                        res.append((w, h, fourcc))
    res.sort(key=lambda t: -t[0] * t[1])
    return res


def fourcc_to_str(fc):
    s = "".join(chr((fc >> (8 * i)) & 0xFF) for i in range(4))
    return s.strip() or "YUYV"


def detect_tube(bgr, threshold=240, invert=False, min_width_px=20,
                auto_darkest=False):
    """Find the tube in `bgr`.

    Two modes:

    * **Manual threshold** (default; `auto_darkest=False`): any pixel
      darker than `threshold` is kept (or brighter if `invert=True`).
    * **Auto-darkest** (`auto_darkest=True`): an adaptive algorithm that
      finds the darkest contiguous region in the image, without any
      threshold tuning. Suitable for a transparent tube whose interior
      is the darkest part of an otherwise bright image — the algorithm
      starts at the image's darkest pixel and grows a connected region
      outward until the brightness rises sharply.

    Returns (contour, mean_width_px, bbox) or (None, 0.0, None) if nothing
    matches the min_width_px filter.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    if auto_darkest:
        core = _darkest_blob_mask(gray, min_width_px)
        mask = _grow_to_tube_outline(gray, core, min_width_px)
    else:
        gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
        flag = cv2.THRESH_BINARY_INV if not invert else cv2.THRESH_BINARY
        _, mask = cv2.threshold(gray_blur, int(threshold), 255, flag)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0, None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < max(500, min_width_px * min_width_px):
        return None, 0.0, None
    x, y, w, h = cv2.boundingRect(contour)
    bbox = (x, y, w, h)
    width_sum = 0
    row_count = 0
    for yy in range(y, y + h):
        xs = np.where(mask[yy] > 0)[0]
        if xs.size >= 2:
            in_range = xs[(xs >= x) & (xs < x + w)]
            if in_range.size >= 2:
                width_sum += int(in_range[-1] - in_range[0] + 1)
                row_count += 1
    mean_width = width_sum / row_count if row_count else 0.0
    return contour, mean_width, bbox


def _skeletonize(mask):
    """Zhang-Suen-style morphological skeletonization.

    Reduces a binary mask to a 1-pixel-wide skeleton while preserving the
    overall shape. Useful for extracting the centreline of a thick blob.
    Capped iterations for safety on pathological inputs.
    """
    img = (mask > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    work = img.copy()
    max_iters = max(img.shape) + 1
    for _ in range(max_iters):
        opened = cv2.morphologyEx(work, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(work, opened)
        eroded = cv2.erode(work, element)
        skel = cv2.bitwise_or(skel, temp)
        work = eroded
        if cv2.countNonZero(work) == 0:
            break
    return skel


def _darkest_blob_mask(gray, min_width_px=20):
    """Return a binary mask covering the darkest contiguous region.

    Strategy:
      1. Find the image's darkest pixels as seeds.
      2. Region-grow from them, accepting neighbouring pixels while their
         brightness stays within a small delta of the running mean — the
         growth stops naturally when the bright background is reached.
      3. The grown blob is the centreline / core of the tube.
    """
    h, w = gray.shape
    min_val = int(gray.min())
    seed_mask = (gray <= min_val + 5).astype(np.uint8) * 255
    if not seed_mask.any():
        return np.zeros_like(gray)

    accepted = seed_mask.copy()
    running_max = float(min_val)
    for _ in range(int(min_width_px * 3)):
        dil = cv2.dilate(accepted, np.ones((3, 3), np.uint8))
        new_pixels = (dil > 0) & (accepted == 0)
        if not np.any(new_pixels):
            break
        candidates = gray[new_pixels]
        delta = candidates.astype(np.int32) - int(running_max)
        keep = delta <= 12
        ys, xs = np.where(new_pixels)
        accepted[ys[keep], xs[keep]] = 255
        if keep.any():
            running_max = float(max(running_max, candidates[keep].max()))
        ys_acc, xs_acc = np.where(accepted > 0)
        if xs_acc.size:
            cur_w = xs_acc.max() - xs_acc.min() + 1
            if cur_w > min_width_px * 4:
                break

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    accepted = cv2.morphologyEx(accepted, cv2.MORPH_OPEN, kernel, iterations=1)
    accepted = cv2.morphologyEx(accepted, cv2.MORPH_CLOSE, kernel, iterations=2)
    return accepted


def _grow_to_tube_outline(gray, core_mask, min_width_px=20):
    """Given a binary mask of the tube's dark region, expand outward to the
    actual tube outline by looking for the largest radial brightness jump.

    Returns a new mask covering the full tube body.
    """
    h, w = gray.shape
    ys, xs = np.where(core_mask > 0)
    if ys.size == 0:
        return core_mask

    # Reduce the core to its skeleton so we walk outward from the centre of
    # the tube, not the centre of mass. Skeleton points are still dark seeds
    # but they trace the centreline.
    skel = _skeletonize(core_mask)
    sk_ys, sk_xs = np.where(skel > 0)
    if sk_ys.size < 5:
        # Fall back to core points if skeleton is empty.
        sk_ys, sk_xs = ys, xs

    # Estimate the typical tube radius from the centreline spread in x,
    # not from the area of the core (the core can be a thick blob that
    # includes the whole tube — using it would over-estimate the radius
    # and cause max_probe to balloon).
    core_radius = max(1.0, (xs.max() - xs.min() + 1) / 2.0)

    # Probe radius — generous enough to reach the wall but capped so we
    # don't scan halfway across the image on a thin tube.
    max_probe = int(max(min_width_px * 2,
                        min(core_radius * 3,
                            min(h, w) // 4)))

    # Build a background-brightness estimate from the four image corners.
    corner_pixels = np.concatenate([
        gray[0, :].ravel(), gray[-1, :].ravel(),
        gray[:, 0].ravel(), gray[:, -1].ravel()])
    bg_brightness = float(np.median(corner_pixels))

    # Collect left/right edge samples along the tube's centreline.
    samples = []
    by, bx = sk_ys.min(), sk_xs.min()
    ey, ex = sk_ys.max(), sk_xs.max()
    step = max(1, (ey - by) // 80)  # ~80 samples along the tube length

    for yy in range(by, ey + 1, step):
        row_xs = sk_xs[sk_ys == yy]
        if row_xs.size == 0:
            continue
        centre_x = float(row_xs.mean())
        # Walk outward. The wall edge is where the smoothed brightness
        # first reaches 95% of the background brightness.
        def find_edge(direction):
            xs_walked = []
            vs_walked = []
            for r in range(1, max_probe):
                xx = int(centre_x + direction * r)
                if xx < 0 or xx >= w:
                    break
                xs_walked.append(xx)
                vs_walked.append(float(gray[yy, xx]))
            if len(xs_walked) < 4:
                return None
            vs = np.asarray(vs_walked)
            # Light smoothing — just enough to suppress single-pixel noise.
            k = 5 if len(vs) >= 5 else max(3, len(vs) // 2)
            if k % 2 == 0:
                k += 1
            vs_s = cv2.GaussianBlur(vs.reshape(1, -1), (k, 1), 0).ravel()
            target = 0.95 * bg_brightness
            above = np.where(vs_s >= target)[0]
            if above.size == 0:
                return xs_walked[-1]
            return xs_walked[int(above[0])]

        left_edge = find_edge(-1)
        right_edge = find_edge(+1)
        if left_edge is not None and right_edge is not None:
            samples.append((yy, left_edge, right_edge))

    if len(samples) < 5:
        return core_mask  # not enough evidence; fall back to core

    # Median tube radius across all samples, then build a thick stroke
    # along the centreline at that radius. Use a small vertical closing to
    # fill gaps between samples without spilling sideways.
    widths = [r - l for (_, l, r) in samples]
    median_width = int(max(min_width_px, np.median(widths)))
    half = max(2, median_width // 2)

    mask = np.zeros_like(gray)
    for yy, l_e, r_e in samples:
        cv2.line(mask, (int(l_e), yy), (int(r_e), yy), 255, 1)
    # Vertical close large enough to fuse sample rows into one contour,
    # horizontal close kept small so the tube width isn't inflated.
    kernel_h = max(3, step + 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, kernel_h))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def _wire_collapsible(groupbox):
    """Make a checkable QGroupBox actually hide its body when unchecked.

    The body is captured lazily after the caller has populated the layout
    (QtCore.QTimer.singleShot(0, ...)), since this helper is typically called
    before the layout's children exist.
    """
    def _capture_and_apply():
        layout = groupbox.layout()
        if layout is None:
            return
        children = []
        for i in range(layout.count()):
            item = layout.itemAt(i)
            w = item.widget()
            if w is not None:
                children.append(w)
        groupbox._collapsible_children = children

        def _apply(checked):
            for w in groupbox._collapsible_children:
                w.setVisible(checked)

        try:
            groupbox.toggled.disconnect(_apply)
        except TypeError:
            pass
        groupbox.toggled.connect(_apply)
        _apply(groupbox.isChecked())

    QtCore.QTimer.singleShot(0, _capture_and_apply)


CONTROL_MAP = [
    ("Brightness",     cv2.CAP_PROP_BRIGHTNESS),
    ("Contrast",       cv2.CAP_PROP_CONTRAST),
    ("Saturation",     cv2.CAP_PROP_SATURATION),
    ("Hue",            cv2.CAP_PROP_HUE),
    ("Gamma",          cv2.CAP_PROP_GAMMA),
    ("Exposure",       cv2.CAP_PROP_EXPOSURE),
    ("Focus",          cv2.CAP_PROP_FOCUS),
    ("Zoom",           cv2.CAP_PROP_ZOOM),
    ("White Balance",  cv2.CAP_PROP_WHITE_BALANCE_BLUE_U),
    ("Gain",           cv2.CAP_PROP_GAIN),
    ("Backlight",      cv2.CAP_PROP_BACKLIGHT),
    ("Sharpness",      cv2.CAP_PROP_SHARPNESS),
]


class VideoThread(QtCore.QThread):
    frame_signal = QtCore.pyqtSignal(np.ndarray)
    error_signal = QtCore.pyqtSignal(str)

    def __init__(self, dev, width, height, fourcc, parent=None):
        super().__init__(parent)
        self.dev = dev
        self.width = width
        self.height = height
        self.fourcc = fourcc
        self._stop = False
        # Pending V4L2 control updates (prop, value) tuples, applied by the
        # thread itself so we never fight it for the open file handle.
        self._pending_controls = []
        self._controls_lock = QtCore.QMutex()

    def run(self):
        cap = cv2.VideoCapture(self.dev, cv2.CAP_V4L2)
        if not cap.isOpened():
            self.error_signal.emit(f"Failed to open {self.dev}")
            return
        cap.set(cv2.CAP_PROP_FOURCC, self.fourcc)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Short read timeout so cap.read() can't block forever if the camera
        # is wedged (the symptom otherwise is a stuck "Starting…" screen).
        try:
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)
        except Exception:
            pass
        bad_reads = 0
        while not self._stop:
            # Drain any pending control updates.
            self._controls_lock.lock()
            pending = self._pending_controls
            self._pending_controls = []
            self._controls_lock.unlock()
            for prop, value in pending:
                cap.set(prop, value)

            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                bad_reads += 1
                if bad_reads == 1 or bad_reads % 20 == 0:
                    self.error_signal.emit(
                        f"No frame from {self.dev} (try {bad_reads})")
                time.sleep(0.1)
                continue
            if bad_reads:
                self.error_signal.emit(f"Recovered after {bad_reads} failed reads")
                bad_reads = 0
            # Camera is mounted 180° from reality: rotate here so everything
            # downstream (manual flip toggles, tube detection, overlays,
            # snapshots) operates on the correct orientation.
            frame = cv2.flip(frame, -1)
            self.frame_signal.emit(frame)
        cap.release()

    def stop(self):
        self._stop = True
        self.wait(2000)

    def set_control(self, prop, value):
        self._controls_lock.lock()
        self._pending_controls.append((prop, value))
        self._controls_lock.unlock()


class TubeDetectThread(QtCore.QThread):
    """Off-thread tube detector so the GUI never freezes.

    Takes the latest frame plus current detection parameters and runs
    detect_tube. Results are emitted via the `result` signal (always —
    even on failure, which carries None). Drops intermediate frames so
    slow detection doesn't queue up work.
    """
    result = QtCore.pyqtSignal(object, float, object)  # contour, mean_w, bbox

    def __init__(self, parent=None):
        super().__init__(parent)
        self._latest_frame = None
        self._latest_kwargs = None
        self._lock = QtCore.QMutex()
        self._has_work = QtCore.QWaitCondition()
        self._stop = False

    def submit(self, frame, **kwargs):
        self._lock.lock()
        self._latest_frame = frame
        self._latest_kwargs = kwargs
        self._has_work.wakeAll()
        self._lock.unlock()

    def run(self):
        while not self._stop:
            self._lock.lock()
            while (self._latest_frame is None or
                   self._latest_kwargs is None) and not self._stop:
                self._has_work.wait(self._lock)
            if self._stop:
                self._lock.unlock()
                break
            frame = self._latest_frame
            kwargs = self._latest_kwargs
            self._latest_frame = None
            self._latest_kwargs = None
            self._lock.unlock()
            try:
                c, w, bb = detect_tube(frame, **kwargs)
            except Exception:
                c, w, bb = None, 0.0, None
            self.result.emit(c, w, bb)

    def stop(self):
        self._stop = True
        self._has_work.wakeAll()
        self.wait(2000)


class MicroscopeViewer(QtWidgets.QMainWindow):
    def __init__(self, dev, width, height, fourcc, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Microscope — {dev} @ {width}x{height}")
        self.resize(1280, 800)

        self.dev = dev
        self.fourcc = fourcc
        self.snapshot_dir = os.path.join(os.getcwd(), "snapshots")
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.snapshot_count = 0

        self.show_crosshair = True
        self.show_scale = False
        self.scale_um_per_px = 1.0
        self.flip_v = False
        self.flip_h = False
        self.detect_tube = False
        self.tube_min_width_px = 20
        self.tube_threshold = 240           # manual grayscale threshold
        self.tube_invert = False            # True → "lighter than" instead of "darker than"
        self.tube_auto_darkest = True       # use flood-fill-from-darkest mode (default ON)
        self._last_detect_contour = None
        self._last_detect_w = 0.0
        self._last_detect_bbox = None

        self._latest_frame = None

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        self.video_label = QtWidgets.QLabel("Starting…\n(no frame yet)")
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setStyleSheet("background:#000; color:#888;")
        self.video_label.setMinimumSize(640, 480)
        root.addWidget(self.video_label, 1)
        self._first_frame_received = False
        self._no_frame_watchdog = QtCore.QTimer(self)
        self._no_frame_watchdog.setSingleShot(True)
        self._no_frame_watchdog.timeout.connect(self._announce_no_frame)

        self.panel_toggle_action = QtWidgets.QAction("Hide Panel", self)
        self.panel_toggle_action.setShortcut(QtGui.QKeySequence("H"))
        self.panel_toggle_action.setCheckable(True)
        self.panel_toggle_action.toggled.connect(self._on_panel_toggle)
        self.addAction(self.panel_toggle_action)

        tb = QtWidgets.QToolButton(self)
        tb.setDefaultAction(self.panel_toggle_action)
        tb.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        corner = QtWidgets.QWidget()
        corner_layout = QtWidgets.QHBoxLayout(corner)
        corner_layout.setContentsMargins(0, 0, 4, 0)
        corner_layout.addWidget(tb)
        corner_layout.addStretch(1)

        side_container = QtWidgets.QWidget()
        side_container.setMinimumWidth(280)
        side_container.setMaximumWidth(420)
        side = QtWidgets.QVBoxLayout(side_container)
        side.setContentsMargins(6, 6, 6, 6)

        self.dock = QtWidgets.QDockWidget("Controls", self)
        self.dock.setObjectName("controlsDock")
        self.dock.setWidget(side_container)
        self.dock.setAllowedAreas(QtCore.Qt.RightDockWidgetArea | QtCore.Qt.LeftDockWidgetArea)
        self.dock.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable |
                              QtWidgets.QDockWidget.DockWidgetFloatable |
                              QtWidgets.QDockWidget.DockWidgetClosable)
        self.dock.setTitleBarWidget(corner)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.dock)
        self.dock.visibilityChanged.connect(self._on_dock_visibility)

        gb_dev = QtWidgets.QGroupBox("Device")
        gb_dev.setCheckable(True)
        _wire_collapsible(gb_dev)
        vl = QtWidgets.QFormLayout(gb_dev)
        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        self._populate_devices(self.dev)
        vl.addRow("Camera:", self.device_combo)
        side.addWidget(gb_dev)

        gb_res = QtWidgets.QGroupBox("Resolution")
        gb_res.setCheckable(True)
        _wire_collapsible(gb_res)
        vl = QtWidgets.QFormLayout(gb_res)
        self.res_combo = QtWidgets.QComboBox()
        self._populate_resolutions(width, height)
        vl.addRow("Size:", self.res_combo)
        self.apply_res_btn = QtWidgets.QPushButton("Apply")
        self.apply_res_btn.clicked.connect(self.apply_resolution)
        vl.addRow("", self.apply_res_btn)
        side.addWidget(gb_res)

        gb_overlay = QtWidgets.QGroupBox("Overlay")
        gb_overlay.setCheckable(True)
        _wire_collapsible(gb_overlay)
        vl = QtWidgets.QFormLayout(gb_overlay)
        self.crosshair_cb = QtWidgets.QCheckBox("Show crosshair")
        self.crosshair_cb.setChecked(True)
        self.crosshair_cb.toggled.connect(lambda v: setattr(self, "show_crosshair", v))
        vl.addRow(self.crosshair_cb)
        self.scale_cb = QtWidgets.QCheckBox("Show scale bar")
        self.scale_cb.toggled.connect(lambda v: setattr(self, "show_scale", v))
        vl.addRow(self.scale_cb)
        self.flip_v_cb = QtWidgets.QCheckBox("Flip vertically (mirror top↔bottom)")
        self.flip_v_cb.toggled.connect(lambda v: setattr(self, "flip_v", v))
        vl.addRow(self.flip_v_cb)
        self.flip_h_cb = QtWidgets.QCheckBox("Flip horizontally (mirror left↔right)")
        self.flip_h_cb.toggled.connect(lambda v: setattr(self, "flip_h", v))
        vl.addRow(self.flip_h_cb)
        self.scale_edit = QtWidgets.QLineEdit("1.0")
        self.scale_edit.setValidator(QtGui.QDoubleValidator(0.000001, 1e9, 6))
        self.scale_edit.textChanged.connect(self._update_scale)
        vl.addRow("µm / pixel:", self.scale_edit)
        side.addWidget(gb_overlay)

        gb_tube = QtWidgets.QGroupBox("Tube detection")
        gb_tube.setCheckable(True)
        _wire_collapsible(gb_tube)
        vl = QtWidgets.QFormLayout(gb_tube)
        self.detect_tube_cb = QtWidgets.QCheckBox("Enable detection")
        self.detect_tube_cb.toggled.connect(lambda v: setattr(self, "detect_tube", v))
        vl.addRow(self.detect_tube_cb)
        self.tube_auto_cb = QtWidgets.QCheckBox(
            "Auto: detect darkest contiguous region")
        self.tube_auto_cb.setChecked(True)
        self.tube_auto_cb.toggled.connect(
            lambda v: setattr(self, "tube_auto_darkest", v))
        vl.addRow(self.tube_auto_cb)
        self.tube_threshold_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.tube_threshold_slider.setRange(0, 255)
        self.tube_threshold_slider.setValue(self.tube_threshold)
        self.tube_threshold_slider.setToolTip(
            "Grayscale threshold (0–255). Pixels darker than this are kept "
            "(or lighter, if 'Invert' is on). For a front-lit transparent "
            "tube on a white background, try values close to white (220–250)."
        )
        self.tube_threshold_slider.valueChanged.connect(self._on_threshold_change)
        self.tube_threshold_label = QtWidgets.QLabel(f"{self.tube_threshold}")
        thresh_row = QtWidgets.QWidget()
        thresh_layout = QtWidgets.QHBoxLayout(thresh_row)
        thresh_layout.setContentsMargins(0, 0, 0, 0)
        thresh_layout.addWidget(self.tube_threshold_slider, 1)
        thresh_layout.addWidget(self.tube_threshold_label, 0)
        vl.addRow("Threshold:", thresh_row)
        self.tube_invert_cb = QtWidgets.QCheckBox(
            "Invert (keep brighter than threshold)")
        self.tube_invert_cb.toggled.connect(
            lambda v: setattr(self, "tube_invert", v))
        vl.addRow(self.tube_invert_cb)
        self.tube_min_width_spin = QtWidgets.QSpinBox()
        self.tube_min_width_spin.setRange(1, 10000)
        self.tube_min_width_spin.setValue(self.tube_min_width_px)
        self.tube_min_width_spin.setSuffix(" px")
        self.tube_min_width_spin.valueChanged.connect(
            lambda v: setattr(self, "tube_min_width_px", v))
        vl.addRow("Min width:", self.tube_min_width_spin)
        self.tube_width_label = QtWidgets.QLabel("— px")
        vl.addRow("Mean width:", self.tube_width_label)
        self.tube_bbox_label = QtWidgets.QLabel("—")
        vl.addRow("Bounding box:", self.tube_bbox_label)
        side.addWidget(gb_tube)

        gb_ctrl = QtWidgets.QGroupBox("Controls")
        gb_ctrl.setCheckable(True)
        _wire_collapsible(gb_ctrl)
        self.controls_layout = QtWidgets.QFormLayout(gb_ctrl)
        self._control_widgets = {}
        for name, prop in CONTROL_MAP:
            row = QtWidgets.QHBoxLayout()
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setEnabled(False)
            slider.valueChanged.connect(lambda v, p=prop: self._set_prop(p, v))
            label = QtWidgets.QLabel("—")
            slider.user_prop = prop
            slider.user_label = label
            row.addWidget(slider, 1)
            row.addWidget(label, 0)
            wrap = QtWidgets.QWidget()
            wrap.setLayout(row)
            self.controls_layout.addRow(f"{name}:", wrap)
            self._control_widgets[prop] = (slider, label)
        side.addWidget(gb_ctrl, 1)

        gb_act = QtWidgets.QGroupBox("Actions")
        gb_act.setCheckable(True)
        _wire_collapsible(gb_act)
        vl = QtWidgets.QFormLayout(gb_act)
        self.snap_btn = QtWidgets.QPushButton("Snapshot (S)")
        self.snap_btn.clicked.connect(self.snapshot)
        vl.addRow(self.snap_btn)
        self.retry_btn = QtWidgets.QPushButton("Retry (R)")
        self.retry_btn.clicked.connect(self.retry_stream)
        vl.addRow(self.retry_btn)
        self.fps_label = QtWidgets.QLabel("— fps")
        vl.addRow(self.fps_label)
        side.addWidget(gb_act)

        side.addStretch(1)

        self.statusBar().showMessage(f"Device: {dev}")

        self.thread = VideoThread(self.dev, width, height, fourcc)
        self.thread.frame_signal.connect(self.on_frame)
        self.thread.error_signal.connect(self.on_error)
        self.thread.start()

        # Tube detection runs off-thread so a slow detection can't freeze
        # the GUI. Latest-frame-only semantics: new frames overwrite the
        # pending one, so the detector always works on the freshest image.
        self.detect_thread = TubeDetectThread(self)
        self.detect_thread.result.connect(self._on_detect_result)
        self.detect_thread.start()

        self._no_frame_watchdog.start(5000)

        self._fps_t0 = time.time()
        self._fps_n = 0

        QtWidgets.QShortcut(QtGui.QKeySequence("S"), self, activated=self.snapshot)
        QtWidgets.QShortcut(QtGui.QKeySequence("R"), self, activated=self.retry_stream)
        QtWidgets.QShortcut(QtGui.QKeySequence("Q"), self, activated=self.close)
        QtWidgets.QShortcut(QtGui.QKeySequence("Esc"), self, activated=self.close)
        QtWidgets.QShortcut(QtGui.QKeySequence("F"), self, activated=self.toggle_fullscreen)

        self._probe_controls()

    def _on_panel_toggle(self, checked):
        if checked:
            self.dock.hide()
        else:
            self.dock.show()

    def _on_dock_visibility(self, visible):
        self.panel_toggle_action.blockSignals(True)
        self.panel_toggle_action.setChecked(not visible)
        self.panel_toggle_action.setText("Show Panel" if not visible else "Hide Panel")
        self.panel_toggle_action.blockSignals(False)

    def _on_threshold_change(self, v):
        self.tube_threshold = v
        self.tube_threshold_label.setText(f"{v}")

    def _on_device_changed(self, idx):
        data = self.device_combo.itemData(idx)
        if not data or data == self.dev:
            return
        self.dev = data
        res = self.res_combo.currentData() or (1280, 720)
        self._populate_resolutions(*res)
        self.restart_stream(res[0], res[1], self.fourcc)
        self._probe_controls()
        self.panel_toggle_action.blockSignals(True)
        self.panel_toggle_action.setChecked(not visible)
        self.panel_toggle_action.setText("Show Panel" if not visible else "Hide Panel")
        self.panel_toggle_action.blockSignals(False)

    def _populate_devices(self, current_dev):
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        devs = list_video_devices()
        if not devs:
            self.device_combo.addItem("(no cameras found)", "")
            return
        for d, w, h, _score, label in devs:
            self.device_combo.addItem(f"{label}  [{d}]  ({w}x{h})", d)
        idx = max(0, next((i for i, (d, *_ ) in enumerate(devs) if d == current_dev), 0))
        self.device_combo.setCurrentIndex(idx)
        self.device_combo.blockSignals(False)

    def _populate_resolutions(self, current_w, current_h):
        self.res_combo.blockSignals(True)
        self.res_combo.clear()
        ress = probe_resolutions(self.dev)
        if not ress:
            self.res_combo.addItem(f"{current_w}x{current_h}", (current_w, current_h))
        else:
            cur = None
            for i, (w, h, fc) in enumerate(ress):
                label = f"{w} x {h}  [{fc}]"
                self.res_combo.addItem(label, (w, h))
                if w == current_w and h == current_h:
                    cur = i
            if cur is not None:
                self.res_combo.setCurrentIndex(cur)
        self.res_combo.blockSignals(False)

    def _probe_controls(self):
        """Disable all sliders until the streaming thread reports ranges."""
        for prop, (slider, label) in self._control_widgets.items():
            slider.setEnabled(False)
            label.setText("…")
        self._controls_pending = True
        # The streaming thread will fill in the ranges once a frame is live
        # (see _on_frame_after_probe below). If the thread is still alive we
        # just wait; if it isn't, we run a one-shot best-effort probe via
        # v4l2-ctl-style introspection. For now, fall back to a single open
        # with CAP_ANY so we don't fight the streaming thread for V4L2.
        QtCore.QTimer.singleShot(1500, self._probe_controls_via_secondary)

    def _probe_controls_via_secondary(self):
        """Best-effort: try one extra open with a short timeout. If it fails
        because the streaming thread has the device, we just leave sliders
        disabled — they're still wired to set_control."""
        if not any(s.isEnabled() for s, _ in self._control_widgets.values()):
            cap = cv2.VideoCapture(self.dev)
            if not cap.isOpened():
                cap.release()
                return
            try:
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 500)
            except Exception:
                pass
            cap.set(cv2.CAP_PROP_FOURCC, self.fourcc)
            for prop, (slider, label) in self._control_widgets.items():
                try:
                    cap.set(prop, cap.get(prop))
                    cap.set(prop, cap.get(prop) + 1)
                    new = cap.get(prop)
                    cap.set(prop, cap.get(prop) - 2)
                    lo = cap.get(prop)
                    cap.set(prop, cap.get(prop) + 2)
                    hi = cap.get(prop)
                    cap.set(prop, lo)
                    if hi <= lo:
                        slider.setEnabled(False)
                        label.setText("n/a")
                        continue
                    slider.blockSignals(True)
                    slider.setRange(int(lo), int(hi))
                    cur = int(cap.get(prop))
                    cur = max(int(lo), min(int(hi), cur))
                    slider.setValue(cur)
                    slider.setEnabled(True)
                    slider.blockSignals(False)
                    label.setText(f"{cur}  ({int(lo)}–{int(hi)})")
                    slider.user_lo = int(lo)
                    slider.user_hi = int(hi)
                except Exception:
                    slider.setEnabled(False)
                    label.setText("n/a")
            cap.release()

    def _set_prop(self, prop, v):
        # Hand off to the streaming thread so we never open the device twice.
        self.thread.set_control(prop, v)
        slider, label = self._control_widgets[prop]
        lo = getattr(slider, "user_lo", v)
        hi = getattr(slider, "user_hi", v)
        label.setText(f"{v}  ({lo}–{hi})")

    def _update_scale(self, txt):
        try:
            self.scale_um_per_px = float(txt)
        except ValueError:
            self.scale_um_per_px = 0.0

    def apply_resolution(self):
        data = self.res_combo.currentData()
        if not data:
            return
        w, h = data
        self.restart_stream(w, h, self.fourcc)

    def restart_stream(self, w, h, fourcc):
        self.thread.stop()
        self.thread = VideoThread(self.dev, w, h, fourcc)
        self.thread.frame_signal.connect(self.on_frame)
        self.thread.error_signal.connect(self.on_error)
        self.thread.start()
        self._first_frame_received = False
        self.video_label.setText(f"Connecting to {self.dev}…\n(no frame yet)")
        self._no_frame_watchdog.start(5000)
        self.setWindowTitle(f"Microscope — {self.dev} @ {w}x{h}")
        self.statusBar().showMessage(f"Device: {self.dev}  Resolution: {w}x{h}")

    def on_frame(self, frame):
        # Apply flips first so overlays and snapshots show the same image
        # the user is looking at.
        bgr = frame
        if self.flip_v or self.flip_h:
            bgr = cv2.flip(bgr, -1 if (self.flip_v and self.flip_h)
                           else (0 if self.flip_v else 1))
        self._latest_frame = bgr
        if not self._first_frame_received:
            self._first_frame_received = True
            self._no_frame_watchdog.stop()
            self.statusBar().showMessage(
                f"Streaming from {self.dev} @ "
                f"{bgr.shape[1]}x{bgr.shape[0]}", 3000)
        if self.show_crosshair or self.show_scale or self.detect_tube:
            bgr = bgr.copy()
            if self.show_crosshair:
                self._draw_crosshair(bgr)
            if self.show_scale and self.scale_um_per_px > 0:
                self._draw_scale(bgr)
            if self.detect_tube:
                # Submit to the worker thread; result arrives via
                # _on_detect_result and is drawn on the next frame.
                self.detect_thread.submit(
                    bgr.copy(),
                    threshold=self.tube_threshold,
                    invert=self.tube_invert,
                    min_width_px=self.tube_min_width_px,
                    auto_darkest=self.tube_auto_darkest)
                # If we have a stale successful result, draw it now so
                # the user sees feedback even while a new one is in flight.
                if self._last_detect_contour is not None:
                    contour = self._last_detect_contour
                    mean_w = self._last_detect_w
                    bbox = self._last_detect_bbox
                    if mean_w >= self.tube_min_width_px:
                        cv2.drawContours(bgr, [contour], -1, (0, 0, 255), 2)
                        x, y, w, h = bbox
                        label = f"{mean_w:.1f} px"
                        cv2.putText(bgr, label, (x, max(0, y - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 0, 255), 1, cv2.LINE_AA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        target = self.video_label.size()
        sw, sh = target.width(), target.height()
        if sw <= 0 or sh <= 0:
            return
        scale = min(sw / w, sh / h)
        nw, nh = int(w * scale), int(h * scale)
        if nw < 1 or nh < 1:
            return
        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        qimg = QtGui.QImage(resized.data, nw, nh, 3 * nw,
                            QtGui.QImage.Format_RGB888).copy()
        pix = QtGui.QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pix)
        self._fps_n += 1
        dt = time.time() - self._fps_t0
        if dt >= 1.0:
            self.fps_label.setText(f"{self._fps_n / dt:.1f} fps")
            self._fps_t0 = time.time()
            self._fps_n = 0

    def on_error(self, msg):
        self.statusBar().showMessage(msg, 3000)

    def _on_detect_result(self, contour, mean_w, bbox):
        """Receive a tube-detection result from the worker thread."""
        self._last_detect_contour = contour
        self._last_detect_w = mean_w
        self._last_detect_bbox = bbox
        if contour is not None and mean_w >= self.tube_min_width_px:
            self.tube_width_label.setText(f"{mean_w:.1f} px")
            x, y, w, h = bbox
            self.tube_bbox_label.setText(f"x={x} y={y} w={w} h={h}")
        else:
            self.tube_width_label.setText("not detected")
            self.tube_bbox_label.setText("—")

    def _announce_no_frame(self):
        self.video_label.setText(
            f"No frame from {self.dev}\n\n"
            f"• Is another app using it? (browser, cheese, Zoom)\n"
            f"• Try selecting another camera in the Device panel\n"
            f"• Press R to retry"
        )
        self.statusBar().showMessage(
            f"No frame received from {self.dev}", 0)

    def retry_stream(self):
        self.statusBar().showMessage(f"Retrying {self.dev}…", 2000)
        res = self.res_combo.currentData() or (1280, 720)
        self.restart_stream(res[0], res[1], self.fourcc)
        self._probe_controls()

    def _draw_crosshair(self, bgr):
        h, w = bgr.shape[:2]
        color = (0, 255, 0)
        cv2.line(bgr, (w // 2, 0), (w // 2, h), color, 1)
        cv2.line(bgr, (0, h // 2), (w, h // 2), color, 1)
        s = 18
        cx, cy = w // 2, h // 2
        cv2.circle(bgr, (cx, cy), 4, color, 1)
        cv2.line(bgr, (cx - s, cy), (cx - 4, cy), color, 1)
        cv2.line(bgr, (cx + 4, cy), (cx + s, cy), color, 1)
        cv2.line(bgr, (cx, cy - s), (cx, cy - 4), color, 1)
        cv2.line(bgr, (cx, cy + 4), (cx, cy + s), color, 1)

    def _draw_scale(self, bgr):
        h, w = bgr.shape[:2]
        target_um = max(50.0, w * self.scale_um_per_px / 6.0)
        candidates = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
        best = min(candidates, key=lambda c: abs(c - target_um))
        bar_px = max(1, int(round(best / self.scale_um_per_px)))
        if bar_px > w // 2:
            best = max(10, best // 10)
            bar_px = max(1, int(round(best / self.scale_um_per_px)))
        x0, y0 = 20, h - 30
        x1 = x0 + bar_px
        color = (255, 255, 255)
        cv2.line(bgr, (x0, y0), (x1, y0), color, 2)
        cv2.line(bgr, (x0, y0 - 6), (x0, y0 + 6), color, 2)
        cv2.line(bgr, (x1, y0 - 6), (x1, y0 + 6), color, 2)
        label = f"{best:g} µm"
        cv2.putText(bgr, label, (x0, y0 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def snapshot(self):
        if self._latest_frame is None:
            self.statusBar().showMessage("No frame yet", 2000)
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.snapshot_count += 1
        path = os.path.join(self.snapshot_dir,
                            f"microscope_{ts}_{self.snapshot_count:03d}.png")
        cv2.imwrite(path, self._latest_frame)
        self.statusBar().showMessage(f"Saved {path}", 3000)

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def closeEvent(self, ev):
        self.thread.stop()
        if hasattr(self, "detect_thread"):
            self.detect_thread.stop()
        super().closeEvent(ev)


def parse_args():
    p = argparse.ArgumentParser(description="Live viewer for a USB microscope.")
    p.add_argument("--dev", help="/dev/videoX to use (default: auto-detect)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    return p.parse_args()


def main():
    args = parse_args()
    dev = args.dev
    width, height = args.width, args.height

    devs = list_video_devices()
    if not devs:
        print("No usable /dev/videoX devices found.", file=sys.stderr)
        sys.exit(1)

    if dev is None:
        dev = pick_default_device(devs)
        chosen = next((t for t in devs if t[0] == dev), devs[0])
        print(f"Auto-selected {dev} ({chosen[1]}x{chosen[2]})  "
              f"({chosen[4]})")
        print(f"Available: " + ", ".join(
            f"{t[0]} ({t[4]})" for t in devs))
    else:
        if not any(t[0] == dev for t in devs):
            print(f"{dev} did not return frames. Available: "
                  f"{[t[0] for t in devs]}", file=sys.stderr)
            sys.exit(1)

    fourcc = PREFERRED_FOURCC
    app_qt = QtWidgets.QApplication(sys.argv)
    win = MicroscopeViewer(dev, width, height, fourcc)
    win.show()
    sys.exit(app_qt.exec_())


if __name__ == "__main__":
    main()