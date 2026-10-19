"""Tube + droplet detection for the back-lit USB microscope on /dev/video2.

All measurements are reported in units of the tube's perpendicular width
(W_tube). That gives us a calibration-free scale that survives refocus,
zoom changes, and tube-to-tube swaps.

The detector is intentionally classical (OpenCV). It works on
back-lit translucent tubing: the tube shows as a slightly-different-intensity
column against a near-uniform background, and a water droplet shows as a
darker, roughly round protrusion at the tube tip.

Pipeline (see `analyze` for the full thing):
  1. Grab a frame (BGR -> grayscale).
  2. Background-flatten with a large Gaussian blur subtracted from the
     original (this kills the global back-light gradient).
  3. Contrast-stretch + light blur to clean sensor noise.
  4. Find the tube axis: the dominant near-vertical bright/dark line in
     the lower half of the image. We use a Hough line search restricted
     to near-vertical angles, then keep the line whose neighbourhood is
     the strongest edge.
  5. Extract a strip perpendicular to the axis, get the tube's perpendicular
     width W_tube by measuring where the local contrast dips back to the
     background (the "valley" between the two bright tube walls).
  6. Above the tube tip, look for a darker blob: threshold a small ROI
     above the tip, find the largest contour, fit a min-enclosing circle.
     Report its diameter in W_tube units.

If anything fails the detector returns a result with `ok=False` and a
`reason` string, rather than throwing. The live viewer uses that to show
"no tube seen" / "no droplet" overlays.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np


# ---- result types ----------------------------------------------------------


@dataclass
class TubeMeasurement:
    """Where the tube is and how wide it is, in pixels."""
    ok: bool = False
    reason: str = ""
    # Tube axis as a line through the image: (x0, y0) and unit direction.
    p0: Tuple[float, float] = (0.0, 0.0)
    p1: Tuple[float, float] = (0.0, 0.0)
    # The tip of the tube (the open end where droplets form) in pixel coords.
    tip: Tuple[float, float] = (0.0, 0.0)
    # Unit vector pointing FROM the tip INTO the body of the tube.
    # The droplet, if present, sits on the opposite side (-body).
    body_dir: Tuple[float, float] = (0.0, 1.0)
    # Perpendicular width in pixels.
    width_px: float = 0.0
    # The bounding box used for tube analysis (helpful for overlays).
    roi: Tuple[int, int, int, int] = (0, 0, 0, 0)


@dataclass
class DropletMeasurement:
    """The droplet above the tube tip, measured in W_tube units."""
    ok: bool = False
    reason: str = ""
    center: Tuple[float, float] = (0.0, 0.0)
    radius_px: float = 0.0
    # Diameter expressed as a multiple of the tube width.
    diameter_in_tube_widths: float = 0.0
    area_px: float = 0.0
    # Vertical extent of the droplet from the tube tip (in tube-widths).
    # Positive = above the tip; negative = inside the tube.
    vertical_extent_in_tube_widths: float = 0.0


@dataclass
class AnalysisResult:
    width: int = 0
    height: int = 0
    tube: TubeMeasurement = field(default_factory=TubeMeasurement)
    droplet: DropletMeasurement = field(default_factory=DropletMeasurement)
    # A copy of the frame with all overlays drawn, suitable for display.
    debug: Optional[np.ndarray] = None


# ---- helpers ---------------------------------------------------------------


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _bg_subtract(gray: np.ndarray, ksize: int = 51) -> np.ndarray:
    """Subtract a large Gaussian blur from the image. The result is
    approximately zero-mean outside regions that differ from the local
    background, which is exactly what we want for back-lit scenes."""
    blur = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    diff = cv2.subtract(gray, blur)
    return diff


def _stretch(gray: np.ndarray, lo: int = 1, hi: int = 99) -> np.ndarray:
    """Contrast-stretch to the [lo, hi] percentiles, then rescale to 0..255."""
    g = gray.astype(np.float32)
    a, b = np.percentile(g, lo), np.percentile(g, hi)
    if b - a < 1e-3:
        return np.zeros_like(gray)
    out = (g - a) * (255.0 / (b - a))
    return np.clip(out, 0, 255).astype(np.uint8)


def _line_angle_deg(p0, p1) -> float:
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    return math.degrees(math.atan2(dy, dx))


# ---- tube axis detection ---------------------------------------------------


def _find_tube_axis_bright_col(
    gray: np.ndarray,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], float]]:
    """Return the tube's centre axis as ((x0, y0), (x1, y1), score), or
    None if no credible tube is found.

    Approach (bright-column COM):
      1. Subtract a wide horizontal blur to remove the row background.
      2. Threshold positive deviations to get a bright-pixel mask.
      3. For each row, compute the centre-of-mass (x) of bright pixels.
         Smooth vertically and find the longest contiguous run of
         rows where the COM is well-defined and stable.
      4. The average COM across that run is the tube's x-coordinate.
         A small linear fit gives the tilt. p0 is the centre at the
         smallest y, p1 is the centre at the largest y.

    This works robustly on real back-lit scenes and on synthetic
    renders that have a clear bright tube body.
    """
    h, w = gray.shape
    bg_row = cv2.GaussianBlur(gray.astype(np.float32), (101, 1), 0)
    dev = gray.astype(np.float32) - bg_row
    bright = (dev > 5).astype(np.float32)

    com_x = np.full(h, -1.0, dtype=np.float32)
    for y in range(h):
        cols = np.where(bright[y] > 0)[0]
        if cols.size >= 2:
            com_x[y] = float(np.mean(cols))
    good = com_x >= 0
    com_smooth = np.where(good, com_x, 0.0)
    com_smooth = cv2.GaussianBlur(
        com_smooth.reshape(-1, 1).astype(np.float32), (1, 21), 0,
    ).flatten()
    good_smooth = cv2.GaussianBlur(
        good.astype(np.float32).reshape(-1, 1), (1, 21), 0,
    ).flatten() > 0.5
    com_smooth[~good_smooth] = -1

    if not good_smooth.any():
        return None
    runs = []
    i = 0
    while i < h:
        if good_smooth[i]:
            j = i
            while j < h and good_smooth[j]:
                j += 1
            runs.append((i, j, j - i, float(np.mean(com_smooth[i:j]))))
            i = j
        else:
            i += 1
    if not runs:
        return None
    s, e, length, avg_x = max(runs, key=lambda r: r[2])
    if length < 30:
        return None
    ys = np.arange(s, e)
    xs = com_smooth[s:e]
    valid = xs >= 0
    if valid.sum() < 2:
        slope = 0.0
        intercept = avg_x
    else:
        slope, intercept = np.polyfit(ys[valid], xs[valid], 1)
    y_top = float(s)
    y_bot = float(e - 1)
    x_top = float(slope * y_top + intercept)
    x_bot = float(slope * y_bot + intercept)
    return (x_top, y_top), (x_bot, y_bot), float(length)


def _find_tube_axis(
    gray: np.ndarray,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], float]]:
    """Hough + snap-to-centre axis finder (fallback).

    Used when the bright-column approach fails (e.g. on scenes where
    the tube body has very little contrast against the background,
    or the bright column is broken by a droplet).
    """
    h, w = gray.shape
    edges = cv2.Canny(gray, 10, 30, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 360, threshold=30,
        minLineLength=max(40, h // 6), maxLineGap=20,
    )
    if lines is None:
        return None

    best = None
    best_score = -1.0
    cx_img = w / 2.0
    angle_tol_deg = 45.0
    line_arr = lines[:, 0, :] if lines.ndim == 3 else lines
    for ln in line_arr:
        x1, y1, x2, y2 = (float(v) for v in ln)
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 30:
            continue
        ang = abs(_line_angle_deg((x1, y1), (x2, y2)))
        vert_score = 1.0 - min(1.0, abs(90 - ang) / angle_tol_deg)
        if vert_score <= 0:
            continue
        mx = (x1 + x2) / 2.0
        cx_score = 1.0 - min(1.0, abs(mx - cx_img) / (w / 2.0))
        score = length * (0.7 * vert_score + 0.3 * cx_score)
        if score > best_score:
            best_score = score
            best = (x1, y1, x2, y2)
    if best is None:
        return None
    (x1, y1, x2, y2) = best
    if y1 > y2:
        x1, y1, x2, y2 = x2, y2, x1, y1
    dx = x2 - x1
    dy = y2 - y1
    L = math.hypot(dx, dy)
    if L < 1e-3:
        return None
    ux, uy = dx / L, dy / L
    nx, ny = -uy, ux

    y_mid = 0.5 * (y1 + y2)
    t_mid = (y_mid - y1) / dy if abs(dy) > 1e-3 else 0.5
    cx_mid = x1 + t_mid * dx
    cy_mid = y1 + t_mid * dy
    half = int(min(w, h) * 0.4)
    samples = np.arange(-half, half + 1, dtype=np.float32)
    map_x = (cx_mid + samples * nx).astype(np.float32).reshape(1, -1)
    map_y = (cy_mid + samples * ny).astype(np.float32).reshape(1, -1)
    prof = cv2.remap(gray, map_x, map_y, cv2.INTER_LINEAR)[0].astype(np.float32)
    prof_s = cv2.GaussianBlur(prof.reshape(1, -1), (1, 21), 0).flatten()
    margin = max(20, len(prof_s) // 5)
    bg_region = np.concatenate([prof_s[:margin], prof_s[-margin:]])
    bg = float(np.median(bg_region))
    sigma = max(2.0, 1.4826 * float(np.median(np.abs(bg_region - bg))) + 1e-3)
    thresh = max(3.0, 2.5 * sigma)
    dev = np.abs(prof_s - bg)
    above = dev > thresh
    if not above.any():
        return None
    runs = []
    i = 0
    while i < len(above):
        if above[i]:
            j = i
            while j < len(above) and above[j]:
                j += 1
            runs.append((i, j, j - i, float(dev[i:j].sum())))
            i = j
        else:
            i += 1
    if not runs:
        return None
    peak_idx = int(np.argmax(dev))
    chosen = None
    for (s, e, length, score) in runs:
        if s <= peak_idx < e:
            chosen = (s, e, length)
            break
    if chosen is None:
        chosen = max(runs, key=lambda r: r[2])[:3]
    s, e, _ = chosen
    centre_idx = int(0.5 * (s + e - 1))
    snap = float(centre_idx - half)
    cx_c = cx_mid + snap * nx
    cy_c = cy_mid + snap * ny
    y_top = 0
    y_bot = h - 1
    def x_at_y(yy):
        if abs(uy) < 1e-6:
            return cx_c, yy
        u = (yy - cy_c) / uy
        return cx_c + u * ux, yy
    x_top, _ = x_at_y(y_top)
    x_bot, _ = x_at_y(y_bot)
    return (x_top, y_top), (x_bot, y_bot), snap


# ---- perpendicular width measurement --------------------------------------


def _measure_tube_width(
    gray: np.ndarray,
    axis_p0: Tuple[float, float],
    axis_p1: Tuple[float, float],
    sample_y_ratio: float = 0.75,
) -> Tuple[float, Tuple[int, int, int, int]]:
    """Walk a line perpendicular to the tube axis at y = sample_y_ratio
    (fraction from top), and find the tube's *bright* (or dark) core
    width.

    The detector uses the *intensity* profile. We measure the local
    background as the median of the outer 20% of the profile, then find
    the contiguous run of pixels around the axis whose intensity is
    significantly *brighter* than that background. That is the
    back-lit water column. We take its full extent as W_tube. This
    gives a stable, well-defined unit that matches the bore of the
    tube (and therefore the droplet, which sits in/above the bore).

    Returns (width_px, roi_bbox).
    """
    h, w = gray.shape
    y_sample = _clamp(sample_y_ratio, 0.0, 1.0) * h
    t = (y_sample - axis_p0[1]) / max(1e-3, axis_p1[1] - axis_p0[1])
    t = _clamp(t, 0.0, 1.0)
    cx = axis_p0[0] + t * (axis_p1[0] - axis_p0[0])
    cy = axis_p0[1] + t * (axis_p1[1] - axis_p0[1])
    dx = axis_p1[0] - axis_p0[0]
    dy = axis_p1[1] - axis_p0[1]
    L = math.hypot(dx, dy)
    if L < 1e-3:
        return 0.0, (0, 0, 0, 0)
    nx, ny = -dy / L, dx / L  # perpendicular (left of axis when going down)

    half = int(min(w, h) * 0.4)
    samples = np.arange(-half, half + 1, dtype=np.float32)
    xs = cx + samples * nx
    ys = cy + samples * ny
    map_x = xs.astype(np.float32).reshape(1, -1)
    map_y = ys.astype(np.float32).reshape(1, -1)
    prof = cv2.remap(gray, map_x, map_y, cv2.INTER_LINEAR)[0].astype(np.float32)
    # Light denoise
    prof_s = cv2.GaussianBlur(prof.reshape(1, -1), (1, 5), 0).flatten()

    # Local background: median of the outer 20% of the profile.
    margin = max(20, len(prof_s) // 5)
    bg_region = np.concatenate([prof_s[:margin], prof_s[-margin:]])
    bg = float(np.median(bg_region))
    sigma = max(1.0, 1.4826 * float(np.median(np.abs(bg_region - bg))) + 1e-3)
    thresh = max(3.0, 2.5 * sigma)
    bright = prof_s > bg + thresh

    if not bright.any():
        return 0.0, (0, 0, 0, 0)
    # The longest contiguous run of "brighter than bg" pixels.
    runs = []
    i = 0
    while i < len(bright):
        if bright[i]:
            j = i
            while j < len(bright) and bright[j]:
                j += 1
            runs.append((i, j, j - i))
            i = j
        else:
            i += 1
    if not runs:
        return 0.0, (0, 0, 0, 0)
    s, e, _ = max(runs, key=lambda r: r[2])
    if e - s < 4:
        return 0.0, (0, 0, 0, 0)
    width_px = float(e - s)

    x0 = int(_clamp(cx + (s - half) * nx, 0, w - 1))
    y0 = int(_clamp(cy + (s - half) * ny, 0, h - 1))
    x1 = int(_clamp(cx + (e - 1 - half) * nx, 0, w - 1))
    y1 = int(_clamp(cy + (e - 1 - half) * ny, 0, h - 1))
    bx0, by0 = min(x0, x1), min(y0, y1)
    bx1, by1 = max(x0, x1), max(y0, y1)
    return width_px, (bx0, by0, bx1 - bx0 + 1, by1 - by0 + 1)


# ---- tip detection ---------------------------------------------------------


def _find_tube_tip(
    gray: np.ndarray,
    axis_p0: Tuple[float, float],
    axis_p1: Tuple[float, float],
    droplet_xy: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float, float, float]:
    """Find the tube's tip and the body direction.

    The "tip" is the open end of the tube — the end OPPOSITE the body
    of the tube (and the end where a droplet, if any, sits).

    Strategy:
      1. If a droplet was already found (caller passes its position),
         the tip is the end of the tube axis that is *closer* to the
         droplet.
      2. Otherwise, walk outward from the centre along the axis in both
         directions and pick the end with the *smaller* extent (i.e.
         the tube just terminates there) as the tip. The body
         direction is then the opposite.

    Returns (tip_x, tip_y, body_dx, body_dy).
    """
    h, w = gray.shape
    dx = axis_p1[0] - axis_p0[0]
    dy = axis_p1[1] - axis_p0[1]
    L = math.hypot(dx, dy)
    if L < 1e-3:
        return axis_p0[0], axis_p0[1], 0.0, 1.0
    ux, uy = dx / L, dy / L
    nx, ny = -uy, ux
    axis_mid = ((axis_p0[0] + axis_p1[0]) / 2.0, (axis_p0[1] + axis_p1[1]) / 2.0)

    if droplet_xy is not None:
        # Project droplet onto the axis; the tip is the end nearest
        # the droplet.
        dxc = droplet_xy[0] - axis_mid[0]
        dyc = droplet_xy[1] - axis_mid[1]
        proj = dxc * ux + dyc * uy
        if proj > 0:
            tip_s = L / 2  # at axis_p1
            body_dx, body_dy = -ux, -uy
        else:
            tip_s = -L / 2  # at axis_p0
            body_dx, body_dy = ux, uy
        tip_x = axis_mid[0] + tip_s * ux
        tip_y = axis_mid[1] + tip_s * uy
        return float(tip_x), float(tip_y), float(body_dx), float(body_dy)

    # No droplet: walk outward from the centre and find where the tube
    # contrast terminates. Tip = end with the smaller extent.
    bg_y = int(h * 0.05) if h > 20 else 0
    bg_strip = gray[max(0, bg_y - 2): bg_y + 3, :]
    bg_std = float(np.std(bg_strip)) + 1e-3
    threshold = 2.0 * bg_std

    def end_extent(s_direction):
        step = 2
        s = 0.0
        last_contrast = float("inf")
        end = s_direction * L / 2
        while abs(s) < L / 2:
            cx = axis_mid[0] + s * ux
            cy = axis_mid[1] + s * uy
            half = 25
            xs = cx + np.arange(-half, half + 1) * nx
            ys = cy + np.arange(-half, half + 1) * ny
            map_x = xs.astype(np.float32).reshape(1, -1)
            map_y = ys.astype(np.float32).reshape(1, -1)
            strip = cv2.remap(gray, map_x, map_y, cv2.INTER_LINEAR)[0]
            contrast = float(np.std(strip))
            if contrast < threshold:
                end = s
                break
            last_contrast = contrast
            s += step * s_direction
        return end

    end_pos = end_extent(+1)
    end_neg = end_extent(-1)
    if abs(end_pos) < abs(end_neg):
        tip_s = end_pos
        body_dx, body_dy = -ux, -uy
    else:
        tip_s = end_neg
        body_dx, body_dy = ux, uy
    tip_x = axis_mid[0] + tip_s * ux
    tip_y = axis_mid[1] + tip_s * uy
    return float(tip_x), float(tip_y), float(body_dx), float(body_dy)


# ---- droplet detection -----------------------------------------------------


def _find_droplet(
    gray: np.ndarray,
    axis_p0: Tuple[float, float],
    axis_p1: Tuple[float, float],
    tip: Tuple[float, float],
    body_dx: float,
    body_dy: float,
    width_px: float,
) -> DropletMeasurement:
    """Find a droplet at the tube tip — the side OPPOSITE the body
    direction (i.e. the open end of the tube where water exits).

    A water droplet on a back-lit tube reads as a darker, roughly round
    region. We:
      1. Take a ROI on the "outside" side of the tip. The ROI is a
         few W_tube in each direction (along the body axis and
         perpendicular).
      2. The "outside" direction is the opposite of the body direction
         (i.e. away from the body of the tube).
      3. Threshold for "darker than local background" within the ROI.
      4. Largest connected component -> min enclosing circle.

    If the droplet is *merged* with the body (so the bright column
    extends past the tip), the ROI may not include the droplet. In
    that case the caller can extend the column to terminate at the
    visible body's edge. We work around this by searching a
    generous ROI on the outside of the tip.
    """
    h, w = gray.shape
    if width_px < 6:
        return DropletMeasurement(ok=False, reason="tube too narrow to measure droplet")

    # Axis unit vectors.
    dx = axis_p1[0] - axis_p0[0]
    dy = axis_p1[1] - axis_p0[1]
    L = math.hypot(dx, dy)
    if L < 1e-3:
        return DropletMeasurement(ok=False, reason="axis degenerate")
    ux, uy = dx / L, dy / L
    nx, ny = -uy, ux

    # The "tip-out" direction is the opposite of the body direction.
    out_dx, out_dy = -body_dx, -body_dy

    look_ahead = 4.0
    look_side = 2.5
    half_w = int(width_px * look_side)
    ahead = int(width_px * look_ahead)
    p_out_x = tip[0] + out_dx * ahead
    p_out_y = tip[1] + out_dy * ahead
    a = (tip[0] + nx * half_w, tip[1] + ny * half_w)
    b = (tip[0] - nx * half_w, tip[1] - ny * half_w)
    c = (p_out_x - nx * half_w, p_out_y - ny * half_w)
    d = (p_out_x + nx * half_w, p_out_y + ny * half_w)
    poly = np.array([a, b, c, d], dtype=np.float32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return DropletMeasurement(ok=False, reason="ROI empty")
    bx0, by0, bx1, by1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    roi_mask = mask[by0:by1 + 1, bx0:bx1 + 1]
    roi_gray = gray[by0:by1 + 1, bx0:bx1 + 1]

    erode_px = max(3, int(width_px * 0.6))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px, erode_px))
    bg_mask = cv2.erode(roi_mask, kernel, iterations=1)
    bg_pixels = roi_gray[bg_mask > 0]
    if bg_pixels.size < 20:
        # The ROI is too small to erode — fall back to a global
        # background estimate.
        bg_mean = float(np.median(roi_gray[roi_mask > 0]))
    else:
        bg_mean = float(np.mean(bg_pixels))
    delta = 8
    fg_mask = ((roi_gray < (bg_mean - delta)) & (roi_mask > 0)).astype(np.uint8) * 255
    if fg_mask.sum() < 5:
        return DropletMeasurement(ok=False, reason="no darker blob above tip")
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
    if n <= 1:
        return DropletMeasurement(ok=False, reason="no connected component")
    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = 1 + int(np.argmax(areas))
    area_px = float(stats[biggest, cv2.CC_STAT_AREA])
    if area_px < max(8.0, width_px * 0.5):
        return DropletMeasurement(ok=False, reason="blob too small")
    ys2, xs2 = np.where(labels == biggest)
    pts = np.column_stack((xs2, ys2)).astype(np.float32)
    (mx, my), radius = cv2.minEnclosingCircle(pts)
    if radius < width_px * 0.10:
        return DropletMeasurement(ok=False, reason="circle too small")
    cx_img = float(mx + bx0)
    cy_img = float(my + by0)
    dxc = cx_img - tip[0]
    dyc = cy_img - tip[1]
    proj = dxc * out_dx + dyc * out_dy
    return DropletMeasurement(
        ok=True,
        center=(cx_img, cy_img),
        radius_px=float(radius),
        diameter_in_tube_widths=float(2 * radius) / width_px,
        area_px=area_px,
        vertical_extent_in_tube_widths=float(proj) / width_px,
    )


def _extend_to(px, py, anchor, distance):
    """Push (px, py) away from `anchor` by `distance` pixels along the line
    joining them."""
    dx = px - anchor[0]
    dy = py - anchor[1]
    n = math.hypot(dx, dy)
    if n < 1e-6:
        return px, py
    s = (n + distance) / n
    return anchor[0] + dx * s, anchor[1] + dy * s


# ---- top-level -------------------------------------------------------------


def analyze(bgr: np.ndarray) -> AnalysisResult:
    """Run the full tube + droplet analysis on a BGR frame."""
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_s = cv2.GaussianBlur(gray, (3, 3), 0)
    flat = _bg_subtract(gray, ksize=max(31, (min(h, w) // 8) | 1))
    flat = _stretch(flat, lo=2, hi=98)
    flat = cv2.GaussianBlur(flat, (5, 5), 0)

    res = AnalysisResult(width=w, height=h)

    # Try the column-COM axis finder first (more robust on real
    # scenes). Fall back to Hough if it fails to find a credible
    # column.
    axis = _find_tube_axis_bright_col(gray_s)
    if axis is None:
        axis = _find_tube_axis(gray_s)
    if axis is None:
        res.tube.reason = "no tube-like line found"
        res.debug = _make_debug(bgr, res)
        return res
    p0, p1, _snap = axis

    # Sample a few rows to get a robust width estimate. Use the lightly-
    # smoothed grayscale for the intensity-profile width measurement.
    widths = []
    for ratio in (0.55, 0.65, 0.75, 0.85):
        w_px, _roi = _measure_tube_width(gray_s, p0, p1, sample_y_ratio=ratio)
        if w_px > 4:
            widths.append(w_px)
    if not widths:
        res.tube = TubeMeasurement(
            ok=False, reason="could not measure tube width", p0=p0, p1=p1,
        )
        res.debug = _make_debug(bgr, res)
        return res
    width_px = float(np.median(widths))

    L_axis = math.hypot(p1[0] - p0[0], p1[1] - p0[1]) + 1e-9
    ux_axis = (p1[0] - p0[0]) / L_axis
    uy_axis = (p1[1] - p0[1]) / L_axis

    droplet_p0, droplet_p1 = _find_droplet_both_sides(
        gray, p0, p1, width_px,
    )
    if droplet_p0.ok and (not droplet_p1.ok or
                          droplet_p0.area_px >= droplet_p1.area_px):
        droplet = droplet_p0
        tip_x, tip_y = p0
        body_dx, body_dy = ux_axis, uy_axis
    elif droplet_p1.ok:
        droplet = droplet_p1
        tip_x, tip_y = p1
        body_dx, body_dy = -ux_axis, -uy_axis
    else:
        # No droplet found on either end of the axis. The droplet
        # might be merged with the bright column at the top of the
        # body (typical for back-lit scenes). Try a merged search.
        droplet_merged = _find_droplet_merged(
            gray, p0, p1, width_px, ux_axis, uy_axis,
        )
        if droplet_merged.ok:
            droplet = droplet_merged
            tip_x, tip_y = p0
            body_dx, body_dy = ux_axis, uy_axis
        else:
            droplet = droplet_p0  # no droplet anywhere
            tip_x, tip_y = p0
            body_dx, body_dy = ux_axis, uy_axis

    res.tube = TubeMeasurement(
        ok=True, p0=p0, p1=p1, tip=(tip_x, tip_y),
        body_dir=(body_dx, body_dy),
        width_px=width_px,
    )
    res.droplet = droplet
    res.debug = _make_debug(bgr, res)
    return res


def _find_droplet_merged(
    gray: np.ndarray,
    axis_p0: Tuple[float, float],
    axis_p1: Tuple[float, float],
    width_px: float,
    ux_axis: float,
    uy_axis: float,
) -> DropletMeasurement:
    """Search for a droplet that is *merged* with the top of the bright
    column. The column may extend past the actual tip into the
    droplet's bright cap. Look for a dark region *near* p0 (the top
    end of the column), in the *body* direction (down into the
    column).

    We restrict the search to the top portion of the column (the
    first ~2 W_tube below p0) so we don't accidentally pick up the
    tube's own darker interior (bore, etc.).
    """
    h, w = gray.shape
    if width_px < 6:
        return DropletMeasurement(ok=False, reason="tube too narrow")
    nx, ny = -uy_axis, ux_axis

    look_ahead = 2.0
    look_side = 1.6
    half_w = int(width_px * look_side)
    ahead = int(width_px * look_ahead)
    # Search region: from p0, extending into the body (down) by `ahead`,
    # and perpendicular by ±half_w.
    p_out_x = axis_p0[0] + ux_axis * ahead
    p_out_y = axis_p0[1] + uy_axis * ahead
    a = (axis_p0[0] + nx * half_w, axis_p0[1] + ny * half_w)
    b = (axis_p0[0] - nx * half_w, axis_p0[1] - ny * half_w)
    c = (p_out_x - nx * half_w, p_out_y - ny * half_w)
    d = (p_out_x + nx * half_w, p_out_y + ny * half_w)
    poly = np.array([a, b, c, d], dtype=np.float32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return DropletMeasurement(ok=False, reason="ROI empty")
    bx0, by0, bx1, by1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    roi_mask = mask[by0:by1 + 1, bx0:bx1 + 1]
    roi_gray = gray[by0:by1 + 1, bx0:bx1 + 1]

    # The local background: use the median of pixels near the
    # *perpendicular edges* of the ROI (which are along the tube
    # wall, not on the tube).
    margin = max(3, int(0.2 * width_px))
    edge_pixels = np.concatenate([
        roi_gray[:margin, :].flatten(),
        roi_gray[-margin:, :].flatten(),
        roi_gray[:, :margin].flatten(),
        roi_gray[:, -margin:].flatten(),
    ])
    if edge_pixels.size < 20:
        return DropletMeasurement(ok=False, reason="edge sample too small")
    bg_mean = float(np.median(edge_pixels))
    delta = 8
    fg_mask = ((roi_gray < (bg_mean - delta)) & (roi_mask > 0)).astype(np.uint8) * 255
    if fg_mask.sum() < 5:
        return DropletMeasurement(ok=False, reason="no darker blob merged")
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
    if n <= 1:
        return DropletMeasurement(ok=False, reason="no connected component")
    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = 1 + int(np.argmax(areas))
    area_px = float(stats[biggest, cv2.CC_STAT_AREA])
    if area_px < max(8.0, width_px * 0.5):
        return DropletMeasurement(ok=False, reason="blob too small")
    ys2, xs2 = np.where(labels == biggest)
    pts = np.column_stack((xs2, ys2)).astype(np.float32)
    (mx, my), radius = cv2.minEnclosingCircle(pts)
    if radius < width_px * 0.10:
        return DropletMeasurement(ok=False, reason="circle too small")
    cx_img = float(mx + bx0)
    cy_img = float(my + by0)
    dxc = cx_img - axis_p0[0]
    dyc = cy_img - axis_p0[1]
    proj = dxc * ux_axis + dyc * uy_axis
    return DropletMeasurement(
        ok=True,
        center=(cx_img, cy_img),
        radius_px=float(radius),
        diameter_in_tube_widths=float(2 * radius) / width_px,
        area_px=area_px,
        vertical_extent_in_tube_widths=float(proj) / width_px,
    )


def _find_droplet_both_sides(
    gray: np.ndarray,
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    width_px: float,
) -> Tuple[DropletMeasurement, DropletMeasurement]:
    """Try the droplet detector on each end of the axis, treating that
    end as the tip. Returns (droplet_if_p0_is_tip, droplet_if_p1_is_tip).
    """
    L = math.hypot(p1[0] - p0[0], p1[1] - p0[1]) + 1e-9
    ux0 = (p1[0] - p0[0]) / L
    uy0 = (p1[1] - p0[1]) / L
    # Tip at p0, body direction = p0 -> p1.
    d0 = _find_droplet(gray, p0, p1, p0, ux0, uy0, width_px)
    # Tip at p1, body direction = p1 -> p0.
    d1 = _find_droplet(gray, p1, p0, p1, -ux0, -uy0, width_px)
    return d0, d1


def _find_droplet_loose(
    gray: np.ndarray,
    axis_p0: Tuple[float, float],
    axis_p1: Tuple[float, float],
    width_px: float,
) -> DropletMeasurement:
    """Loose droplet search that doesn't assume a particular tip end.
    Sample perpendicular strips every 0.2 W_tube along the axis (and
    just past each end). For each strip, look for a contiguous dark
    run. The most credible (largest, roundest) dark run along the
    strip is the droplet. The tip is the end of the axis nearest
    the dark blob.
    """
    h, w = gray.shape
    if width_px < 6:
        return DropletMeasurement(ok=False, reason="tube too narrow")
    dx = axis_p1[0] - axis_p0[0]
    dy = axis_p1[1] - axis_p0[1]
    L = math.hypot(dx, dy)
    if L < 1e-3:
        return DropletMeasurement(ok=False, reason="axis degenerate")
    ux, uy = dx / L, dy / L
    nx, ny = -uy, ux

    look = max(60, int(2.5 * width_px))
    step = max(2, int(0.2 * width_px))
    candidates = []
    # Sample perpendicular strips at evenly-spaced points along the
    # axis and just past each end.
    positions = list(range(-int(L / 2) - look, int(L / 2) + look + 1, step))
    # Also include the actual endpoints + a margin in body direction.
    for s in positions:
        cx = (axis_p0[0] + axis_p1[0]) / 2.0 + s * ux
        cy = (axis_p0[1] + axis_p1[1]) / 2.0 + s * uy
        if not (0 <= cx < w and 0 <= cy < h):
            continue
        xs = cx + np.arange(-look, look + 1) * nx
        ys = cy + np.arange(-look, look + 1) * ny
        map_x = xs.astype(np.float32).reshape(1, -1)
        map_y = ys.astype(np.float32).reshape(1, -1)
        strip = cv2.remap(gray, map_x, map_y, cv2.INTER_LINEAR)[0]
        med = float(np.median(strip))
        dark = strip < (med - 8)
        if dark.sum() < max(15.0, width_px * 0.3):
            continue
        idx = np.where(dark)[0]
        runs = []
        ii = 0
        while ii < len(idx):
            jj = ii
            while jj + 1 < len(idx) and idx[jj + 1] == idx[jj] + 1:
                jj += 1
            runs.append(idx[ii:jj + 1])
            ii = jj + 1
        if not runs:
            continue
        # Pick the longest, but require it to be reasonably round:
        # a single dark run that occupies a small fraction of the
        # strip width is a droplet; a long run is the column itself.
        best = max(runs, key=lambda r: r.size)
        run_len = best.size
        if run_len < max(15.0, width_px * 0.3):
            continue
        if run_len > 1.5 * width_px:
            # Too long — probably the column itself, not a droplet.
            continue
        centre_offset = 0.5 * (best[0] + best[-1]) - look
        # Offset from axis centre: if the dark run is centred on
        # the axis (offset near 0), it's the column, not a droplet.
        # A droplet is offset by ~half a W_tube.
        if abs(centre_offset) < 0.2 * width_px:
            continue
        blob_cx = cx + centre_offset * nx
        blob_cy = cy + centre_offset * ny
        # Project onto the axis to get axial position.
        dxc = blob_cx - (axis_p0[0] + axis_p1[0]) / 2.0
        dyc = blob_cy - (axis_p0[1] + axis_p1[1]) / 2.0
        s_blob = dxc * ux + dyc * uy
        # Score: bigger is better.
        score = float(run_len)
        candidates.append((score, s_blob, blob_cx, blob_cy, float(2 * run_len / 2) / width_px))
    if not candidates:
        return DropletMeasurement(ok=False, reason="no dark blob found")
    candidates.sort(reverse=True)
    score, s_blob, bcx, bcy, D_W = candidates[0]
    return DropletMeasurement(
        ok=True,
        center=(bcx, bcy),
        radius_px=D_W * width_px / 2.0,
        diameter_in_tube_widths=D_W,
        area_px=float("nan"),  # not computed in loose search
        vertical_extent_in_tube_widths=float(s_blob) / width_px,
    )


# ---- visualisation ---------------------------------------------------------


def _make_debug(bgr: np.ndarray, res: AnalysisResult) -> np.ndarray:
    out = bgr.copy()
    h, w = out.shape[:2]
    if res.tube.p0 != (0.0, 0.0) or res.tube.p1 != (0.0, 0.0):
        p0 = tuple(int(v) for v in res.tube.p0)
        p1 = tuple(int(v) for v in res.tube.p1)
        cv2.line(out, p0, p1, (0, 255, 255), 1, cv2.LINE_AA)
    if res.tube.ok:
        tip = tuple(int(v) for v in res.tube.tip)
        cv2.circle(out, tip, 4, (0, 255, 0), 2, cv2.LINE_AA)
        # Width annotation
        cv2.putText(
            out,
            f"tube W = {res.tube.width_px:.1f} px (unit)",
            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA,
        )
    else:
        cv2.putText(
            out, f"tube: {res.tube.reason}",
            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA,
        )
    if res.droplet.ok:
        c = tuple(int(v) for v in res.droplet.center)
        r = int(round(res.droplet.radius_px))
        cv2.circle(out, c, max(2, r), (255, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(
            out,
            f"drop D = {res.droplet.diameter_in_tube_widths:.2f} W_tube",
            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 1, cv2.LINE_AA,
        )
        cv2.putText(
            out,
            f"drop height = {res.droplet.vertical_extent_in_tube_widths:.2f} W_tube",
            (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 1, cv2.LINE_AA,
        )
    else:
        cv2.putText(
            out, f"drop: {res.droplet.reason}",
            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA,
        )
    cv2.putText(
        out, f"{w}x{h}",
        (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA,
    )
    return out
