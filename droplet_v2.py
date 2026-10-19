"""Drop detection v2 — orientation-agnostic tube + droplet detector.

Built on top of the live-tool's `droplet_cv.analyze` (which works on
synthetic tip-up scenes) but adds a robust, orientation-agnostic tip +
droplet search that handles:

  * the camera being physically mounted upside-down (so the tube tip
    can be at either end of the detected axis);
  * droplets that bulge out from the bright tube body (rather than
    being darker-than-background blobs);
  * droplets that are still merging into the tube end.

The detector returns the same `AnalysisResult` shape as `droplet_cv`
so the existing live/measure/loop tools can use it with no changes —
just point `PYTHONPATH` here and call `droplet_v2.analyze`.

Pipeline:

  1. Find the tube axis (bright-column COM, with a HoughLinesP
     fallback that mirrors `droplet_cv`).
  2. For each end of the axis, sample perpendicular profiles along a
     few W_tube's worth of column. Find the end where the bright
     "column" actually terminates (the tip), versus the end where the
     tube runs off the image or keeps going (the body). We then look
     for an adjacent *darker* blob on the *outside* of the tip, OR
     for the column to thicken/bulge into a round cap (a droplet
     that's still attached).
  3. Compute W_tube from a representative perpendicular profile taken
     ~3 W_tube into the body from the tip (i.e. always well-defined,
     never near the meniscus).
  4. The droplet's diameter is the width of the cap/bulge in the
     axial profile, measured in W_tube units. We also fit a
     min-enclosing circle to the droplet mask for a diameter estimate
     that's robust to non-spherical drops.

Usage:

    import droplet_v2 as dv2
    res = dv2.analyze(bgr_frame)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

# Reuse the well-validated helpers from droplet_cv (axis finder, BG
# subtract, etc.). Keeping the dependency means the v2 changes only
# touch the broken parts.
import droplet_cv as dc


# --- result types -----------------------------------------------------------


@dataclass
class V2TubeMeasurement(dc.TubeMeasurement):
    """Tube info from v2 (extends the original)."""

    # Which end of the axis is the tip: -1 if we chose p0, +1 if p1.
    tip_end: int = 0
    # Body direction: unit vector pointing INTO the body (away from tip).
    body_dir: Tuple[float, float] = (0.0, 0.0)
    # Tip pixel.
    tip: Tuple[float, float] = (0.0, 0.0)


@dataclass
class V2DropletMeasurement(dc.DropletMeasurement):
    """Droplet info from v2 (extends the original)."""

    # How the diameter was measured: "circle" (min-enclosing circle)
    # or "column-bulge" (width of the column as it widens past the
    # tip). Useful for debugging.
    diameter_source: str = ""
    # The lowest point of the tube's body direction (where the column
    # actually terminates). In pixels.
    column_end_x: float = 0.0
    column_end_y: float = 0.0


@dataclass
class AnalysisResult:
    width: int = 0
    height: int = 0
    tube: V2TubeMeasurement = field(default_factory=V2TubeMeasurement)
    droplet: V2DropletMeasurement = field(default_factory=V2DropletMeasurement)
    debug: Optional[np.ndarray] = None


# --- helpers ----------------------------------------------------------------


def _perp_unit(ux: float, uy: float) -> Tuple[float, float]:
    return -uy, ux


def _sample_perp_profile(
    gray: np.ndarray,
    cx: float,
    cy: float,
    ux: float,
    uy: float,
    half: int,
) -> np.ndarray:
    nx, ny = _perp_unit(ux, uy)
    samples = np.arange(-half, half + 1, dtype=np.float32)
    xs = cx + samples * nx
    ys = cy + samples * ny
    map_x = xs.astype(np.float32).reshape(1, -1)
    map_y = ys.astype(np.float32).reshape(1, -1)
    return cv2.remap(gray, map_x, map_y, cv2.INTER_LINEAR)[0]


def _find_axis(gray: np.ndarray):
    """Same strategy as `droplet_cv`: bright-column COM with a
    HoughLinesP fallback. Returns ((p0, p1), length) where p0 has
    smaller y (i.e. is the "upper" end in image coordinates)."""
    axis = dc._find_tube_axis_bright_col(gray)
    if axis is not None:
        p0, p1, length = axis
        # The existing helper returns (x_top, y_top), (x_bot, y_bot).
        return (p0, p1), length
    axis = dc._find_tube_axis(gray)
    if axis is None:
        return None, 0.0
    p0, p1, score = axis
    return (p0, p1), score


def _axis_unit(p0: Tuple[float, float], p1: Tuple[float, float]):
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    L = math.hypot(dx, dy) + 1e-9
    return dx / L, dy / L, L


def _column_extent_at(
    gray: np.ndarray,
    p: Tuple[float, float],
    ux: float,
    uy: float,
    width_px: float,
):
    """Sample a perpendicular strip at point `p`. Return (width_px,
    bg_intensity, dev_intensity) where `width_px` is the contiguous
    bright run around the axis and `dev_intensity` is how much it
    deviates from local background.

    This is the same logic as `_measure_tube_width` in droplet_cv
    but for an arbitrary sample point along the axis.
    """
    half = max(20, int(width_px * 1.5))
    prof = _sample_perp_profile(gray, p[0], p[1], ux, uy, half)
    prof = cv2.GaussianBlur(prof.reshape(1, -1), (1, 5), 0).flatten()
    margin = max(15, len(prof) // 5)
    bg = float(np.median(np.concatenate([prof[:margin], prof[-margin:]])))
    sigma = max(1.0, 1.4826 * float(np.median(
        np.abs(np.concatenate([prof[:margin], prof[-margin:]]) - bg)
    )) + 1e-3)
    thresh = max(3.0, 2.0 * sigma)
    above = prof > bg + thresh
    if not above.any():
        return 0.0, bg, 0.0
    # find the bright run containing the axis centre (index = half)
    centre = half
    if not above[centre]:
        # find nearest run to centre
        runs = []
        i = 0
        while i < len(above):
            if above[i]:
                j = i
                while j < len(above) and above[j]:
                    j += 1
                runs.append((i, j))
                i = j
            else:
                i += 1
        if not runs:
            return 0.0, bg, 0.0
        s, e = min(runs, key=lambda r: min(abs(r[0] - centre), abs(r[1] - 1 - centre)))
    else:
        # expand the run containing centre
        s = e = centre
        while s > 0 and above[s - 1]:
            s -= 1
        while e < len(above) - 1 and above[e + 1]:
            e += 1
    width_px = float(e - s + 1)
    dev = float(np.max(prof[s:e + 1]) - bg)
    return width_px, bg, dev


def _measure_w_tube(
    gray: np.ndarray,
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    tip: Tuple[float, float],
    body_dx: float,
    body_dy: float,
) -> float:
    """Measure the tube's perpendicular width by sampling perpendicular
    intensity profiles at several positions *into the body* from the
    tip (further than ~1.5 W_tube so we don't pick up the droplet).
    Returns the median (most representative value).
    """
    h, w = gray.shape
    ux, uy, L = _axis_unit(p0, p1)
    nx, ny = _perp_unit(ux, uy)
    tip_x, tip_y = tip
    # Use dc._measure_tube_width directly: it returns the longest bright
    # run around the axis for a given Y ratio. We convert "Y ratio from
    # top" to a sample point on the body half.
    # Map: tip -> s_body=0, body-end -> s_body=L (in body_dir).
    # Y ratio in dc._measure_tube_width is fraction of *image height*
    # from top, not axis position. Instead, sample at fixed body-axial
    # offsets and use our own profile sampler.
    # Use original dc._measure_tube_width at the image-y that
    # corresponds to several axial samples.
    widths = []
    half_w = int(0.3 * min(h, w))  # wide enough
    for k_body in (0.20, 0.30, 0.40, 0.50):
        bx = tip_x + k_body * L * body_dx
        by = tip_y + k_body * L * body_dy
        if not (0 <= bx < w and 0 <= by < h):
            continue
        prof = _sample_perp_profile(gray, bx, by, body_dx, body_dy, half_w)
        prof = cv2.GaussianBlur(prof.reshape(1, -1), (1, 5), 0).flatten()
        margin = max(15, half_w // 5)
        bg_region = np.concatenate([prof[:margin], prof[-margin:]])
        bg = float(np.median(bg_region))
        sigma = max(1.0, 1.4826 * float(np.median(np.abs(bg_region - bg))) + 1e-3)
        thresh = max(3.0, 2.0 * sigma)
        bright = prof > bg + thresh
        if not bright.any():
            continue
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
            continue
        s, e, length = max(runs, key=lambda r: r[2])
        if length > 3 * (half_w // 5):  # exclude runs spanning full profile
            continue
        if length >= 4:
            widths.append(float(length))
    if not widths:
        return 0.0
    return float(np.median(widths))


def _find_dark_blob_outside_tip(
    gray: np.ndarray,
    tip: Tuple[float, float],
    out_dx: float,
    out_dy: float,
    nx: float,
    ny: float,
    width_px: float,
):
    """Look for a darker-than-background blob on the *outside* of the
    tube tip (i.e. away from the body). Returns the centre and
    radius (in pixels) of the largest connected dark blob inside a
    generous ROI, or (None, None, None) if nothing credible.
    """
    h, w = gray.shape
    look_ahead = 4.0 * width_px
    look_side = 3.0 * width_px
    half_w = int(look_side / 2)
    ahead = int(look_ahead)
    p_out_x = tip[0] + out_dx * ahead
    p_out_y = tip[1] + out_dy * ahead
    poly = np.array([
        [tip[0] + nx * half_w, tip[1] + ny * half_w],
        [tip[0] - nx * half_w, tip[1] - ny * half_w],
        [p_out_x - nx * half_w, p_out_y - ny * half_w],
        [p_out_x + nx * half_w, p_out_y + ny * half_w],
    ], dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None, None, None
    bx0, by0, bx1, by1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    roi_mask = mask[by0:by1 + 1, bx0:bx1 + 1]
    roi_gray = gray[by0:by1 + 1, bx0:bx1 + 1]
    # background: erode the ROI and take the median of the rim.
    erode_px = max(5, int(width_px * 0.6))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (erode_px, erode_px)
    )
    bg_mask = cv2.erode(roi_mask, kernel, iterations=1)
    bg_pixels = roi_gray[bg_mask > 0]
    if bg_pixels.size < 30:
        bg = float(np.median(roi_gray[roi_mask > 0]))
    else:
        bg = float(np.mean(bg_pixels))
    delta = 6
    fg = ((roi_gray < (bg - delta)) & (roi_mask > 0)).astype(np.uint8) * 255
    if fg.sum() < max(20, width_px):
        return None, None, None
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n <= 1:
        return None, None, None
    # Largest component that's roughly within a couple of W_tube of
    # the tip — ignore small noise blobs and huge background regions.
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_area = 0.0
    best_label = -1
    for i, a in enumerate(areas, start=1):
        if a < 0.05 * (width_px * width_px):
            continue
        bx, by, bw, bh, _ = stats[i]
        # centroid
        cx_local = bx + bw / 2.0
        cy_local = by + bh / 2.0
        # distance from tip (in pixels, but project onto out-axis)
        cxp = cx_local + bx0
        cyp = cy_local + by0
        ax = cxp - tip[0]
        ay = cyp - tip[1]
        proj_outside = ax * out_dx + ay * out_dy
        if proj_outside < 0.2 * width_px:
            continue  # too close to the tip (probably the meniscus itself)
        if a > best_area:
            best_area = a
            best_label = i
    if best_label < 0:
        return None, None, None
    ys2, xs2 = np.where(labels == best_label)
    pts = np.column_stack((xs2, ys2)).astype(np.float32)
    (mx, my), radius = cv2.minEnclosingCircle(pts)
    return float(mx + bx0), float(my + by0), float(radius)


def _column_axial_profile(
    gray: np.ndarray,
    tip: Tuple[float, float],
    body_dx: float,
    body_dy: float,
    nx: float,
    ny: float,
    width_px: float,
    extent_units: float = 5.0,
):
    """Walk along the tube's axis from the tip *into the body*, and at
    each step record the width of the *bright* perpendicular strip.
    The strip is wide near the meniscus (the droplet cap on the
    outside, or the bore widening into the next section of tubing),
    and narrows to W_tube further along. The first sample that drops
    to ~W_tube is the end of the cap / droplet bulge; the diameter is
    the *axial* extent of the cap region.

    Returns the per-step (s_units, width_px) arrays where
    s_units is in units of W_tube measured from the tip into the
    body. Always includes negative s_units (i.e. outside the tip) so
    the caller can see the cap geometry on both sides.
    """
    step = max(1.0, 0.15 * width_px)
    half = max(20, int(1.6 * width_px))
    s_max = extent_units * width_px
    s_min = -0.5 * extent_units * width_px
    s_arr = []
    w_arr = []
    s = s_min
    while s <= s_max:
        # absolute point = tip + s * body_dir  (s>0 is inside body, s<0 outside)
        cx = tip[0] + s * body_dx
        cy = tip[1] + s * body_dy
        prof = _sample_perp_profile(gray, cx, cy, body_dx, body_dy, half)
        prof = cv2.GaussianBlur(prof.reshape(1, -1), (1, 5), 0).flatten()
        margin = max(10, half // 3)
        bg = float(np.median(np.concatenate([prof[:margin], prof[-margin:]])))
        above = prof > bg + 3.0
        if above.any():
            # biggest run
            runs = []
            i = 0
            while i < len(above):
                if above[i]:
                    j = i
                    while j < len(above) and above[j]:
                        j += 1
                    runs.append((i, j))
                    i = j
                else:
                    i += 1
            s_idx, e_idx = max(runs, key=lambda r: r[1] - r[0])
            width = float(e_idx - s_idx)
        else:
            width = 0.0
        s_arr.append(s / width_px)
        w_arr.append(width)
        s += step
    return np.array(s_arr), np.array(w_arr)


# --- top level --------------------------------------------------------------


def analyze(bgr: np.ndarray) -> AnalysisResult:
    """Run the v2 detector on a BGR frame. Returns the same shape as
    `droplet_cv.analyze` so existing tools can use it transparently.
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_s = cv2.GaussianBlur(gray, (3, 3), 0)

    res = AnalysisResult(width=w, height=h)

    axis, length = _find_axis(gray_s)
    if axis is None:
        res.tube.reason = "no tube-like line found"
        res.debug = _make_debug(bgr, res)
        return res
    p0, p1 = axis

    # Sample both ends of the axis at a few distances from each
    # endpoint to gauge "column brightness" and width. The end where
    # the column *terminates* (the bright cap runs out into the
    # background) is the tip. The other end is the body.
    ux, uy, L = _axis_unit(p0, p1)

    # Try both orientations and pick the one that gives the best
    # droplet (largest, best attached to a clear tip).
    candidates = []
    for orientation in ("p0_tip", "p1_tip"):
        cand = _try_orientation(
            gray, p0, p1, orientation,
            ux=ux, uy=uy, L=L,
        )
        candidates.append(cand)

    # Score each candidate. A good candidate has the droplet both
    # (a) detected and (b) physically attached to the tip (i.e.
    # adjacent and on the OUTSIDE side). We prefer:
    #   1. Candidate where the droplet is detected.
    #   2. If both detect one, prefer the one whose droplet is *closer*
    #      to the tip (a detached blob further away is suspicious).
    #   3. If neither detects one, take the one with the lower
    #      "axial extent beyond the tip" — i.e. the end where the
    #      column actually terminates, not the end where the tube
    #      just runs off the image.
    def score(c):
        d = c["droplet"]
        if not c["tube_ok"] or c["w_tube"] <= 0:
            return (3, 0.0, 0.0, 0.0, 0.0)
        if not d.ok:
            # No droplet: prefer the orientation where the column
            # terminates close to the assumed tip (i.e. the bright
            # cap ends, not just runs off the image).
            return (1, c["terminus_score"], 0.0, 0.0, 0.0)
        # Has droplet. Score on:
        #   - perpendicular offset of the centroid from the axis
        #     (should be small — a droplet is centred on the tip's
        #     extension, a random scatter blob is not)
        #   - then area (prefer larger)
        #   - then height above tip
        body_dx, body_dy = c["body_dir"]
        nx, ny = -body_dy, body_dx
        dxc = d.center[0] - c["tip"][0]
        dyc = d.center[1] - c["tip"][1]
        perp_off = abs(dxc * nx + dyc * ny) / max(1.0, c["w_tube"])
        return (0, perp_off, -d.area_px, -d.vertical_extent_in_tube_widths, 0.0)

    candidates.sort(key=score)
    chosen = candidates[0]

    # Copy into the result shape.
    res.tube = V2TubeMeasurement(
        ok=chosen["tube_ok"], reason=chosen["tube_reason"],
        p0=p0, p1=p1,
        tip=chosen["tip"], body_dir=chosen["body_dir"],
        width_px=chosen["w_tube"],
        tip_end=chosen["tip_end"],
        roi=chosen["roi"],
    )
    res.droplet = V2DropletMeasurement(
        ok=chosen["droplet"].ok,
        reason=chosen["droplet"].reason,
        center=chosen["droplet"].center,
        radius_px=chosen["droplet"].radius_px,
        diameter_in_tube_widths=chosen["droplet"].diameter_in_tube_widths,
        area_px=chosen["droplet"].area_px,
        vertical_extent_in_tube_widths=chosen["droplet"].vertical_extent_in_tube_widths,
        diameter_source=chosen["droplet"].diameter_source,
        column_end_x=chosen["column_end"][0],
        column_end_y=chosen["column_end"][1],
    )
    res.debug = _make_debug(bgr, res)
    return res


def _try_orientation(
    gray: np.ndarray,
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    orientation: str,
    ux: float,
    uy: float,
    L: float,
):
    """Run the analysis assuming `orientation`: 'p0_tip' or 'p1_tip'.
    Returns the candidate dict; success/failure is recorded inside.
    """
    h, w = gray.shape
    nx, ny = _perp_unit(ux, uy)
    if orientation == "p0_tip":
        tip = p0
        body_dx, body_dy = ux, uy
    else:
        tip = p1
        body_dx, body_dy = -ux, -uy
    out_dx, out_dy = -body_dx, -body_dy

    # W_tube: median width across rows in the body half (well away
    # from the tip).
    w_tube = _measure_w_tube(gray, p0, p1, body_dx, body_dy)
    if w_tube < 6:
        return _empty_candidate(orientation, "could not measure tube width", (0, 0), tip, body_dx, body_dy)

    # ----- droplet detection ---------------------------------------
    # Look for a connected component of bright pixels on the OUTSIDE
    # of the tip. The droplet is a refraction-bright crescent.
    cx_b, cy_b, radius_b, area_b = _find_bright_blob_outside_tip(
        gray, tip, out_dx, out_dy, nx, ny, w_tube,
    )
    drop = V2DropletMeasurement(ok=False, reason="no droplet")
    if cx_b is not None and radius_b > 0.5 * w_tube:
        dxc = cx_b - tip[0]
        dyc = cy_b - tip[1]
        proj_out = dxc * out_dx + dyc * out_dy
        drop = V2DropletMeasurement(
            ok=True,
            center=(cx_b, cy_b),
            radius_px=radius_b,
            diameter_in_tube_widths=(2.0 * radius_b) / w_tube,
            area_px=area_b,
            vertical_extent_in_tube_widths=proj_out / w_tube,
            diameter_source="bright-outline",
        )
    else:
        # Fall back: column-bulge axial profile.
        s_arr, w_arr = _column_axial_profile(
            gray, tip, body_dx, body_dy, nx, ny, w_tube,
            extent_units=4.0,
        )
        # Outside of tip means s < 0 (we'll re-normalise after).
        if s_arr.size:
            out_mask = (s_arr < 0) & (s_arr > -3.5)
            if out_mask.any():
                w_out = w_arr[out_mask]
                s_out = s_arr[out_mask]
                if w_out.size and w_out.max() > 1.4 * w_tube:
                    peak_idx = int(np.argmax(w_out))
                    peak_w = float(w_out[peak_idx])
                    peak_s = float(s_out[peak_idx])
                    # Convert peak_s (in old "w_seed" units, but here
                    # s_arr is already in units of w_tube because we
                    # passed extent_units=4.0 and the function uses
                    # w_tube as the unit).
                    drop = V2DropletMeasurement(
                        ok=True,
                        center=(tip[0] + peak_s * w_tube * body_dx,
                                tip[1] + peak_s * w_tube * body_dy),
                        radius_px=peak_w / 2.0,
                        diameter_in_tube_widths=peak_w / w_tube,
                        area_px=float(np.pi * (peak_w / 2.0) ** 2),
                        vertical_extent_in_tube_widths=-peak_s,
                        diameter_source="column-bulge",
                    )
                    if not drop_ok(drop, w_tube):
                        drop = V2DropletMeasurement(ok=False, reason="bulge too small")

    # column_end: where the bright column drops to roughly W_tube
    # (i.e. the actual terminus of the column on the body side). We
    # measure this by walking perpendicular profiles from the tip
    # into the body and finding the first "narrow" one that's
    # within [0.7, 1.3] * w_tube. This is also where the tip
    # physically is (the lumen); useful for the runtime overlay.
    column_end_x = tip[0] + 1.5 * w_tube * body_dx
    column_end_y = tip[1] + 1.5 * w_tube * body_dy
    for k_body in (0.20, 0.30, 0.40, 0.50, 0.60, 0.70):
        bx = tip[0] + k_body * L * body_dx
        by = tip[1] + k_body * L * body_dy
        if not (0 <= bx < w and 0 <= by < h):
            continue
        prof = _sample_perp_profile(gray, bx, by, body_dx, body_dy, int(0.5 * min(h, w)))
        prof = cv2.GaussianBlur(prof.reshape(1, -1), (1, 5), 0).flatten()
        margin = max(15, len(prof) // 5)
        bg = float(np.median(np.concatenate([prof[:margin], prof[-margin:]])))
        sigma = max(1.0, 1.4826 * float(np.median(
            np.abs(np.concatenate([prof[:margin], prof[-margin:]]) - bg)
        )) + 1e-3)
        bright = prof > bg + 2.0 * sigma
        if not bright.any():
            continue
        # count width of run containing axis-centre
        centre = len(prof) // 2
        if not bright[centre]:
            continue
        s = e = centre
        while s > 0 and bright[s - 1]:
            s -= 1
        while e < len(bright) - 1 and bright[e + 1]:
            e += 1
        run_w = e - s + 1
        if 0.7 * w_tube <= run_w <= 1.3 * w_tube:
            column_end_x = bx
            column_end_y = by
            break

    # terminus_score: how close to image centre the column extends
    # before the first narrow row. We want tip_end == the end where
    # the column actually terminates (not runs off the image). The
    # norm of (column_end - tip) divided by L is the fraction of the
    # body-axial extent that was "cleanly measurable" — closer to 0
    # means the column terminates right at the tip.
    terminus_score = math.hypot(
        column_end_x - tip[0], column_end_y - tip[1],
    ) / max(1.0, w_tube)
    return {
        "tube_ok": True, "tube_reason": "",
        "tip": tip, "body_dir": (body_dx, body_dy),
        "tip_end": -1 if orientation == "p0_tip" else 1,
        "w_tube": w_tube,
        "roi": (0, 0, w, h),
        "droplet": drop,
        "column_end": (column_end_x, column_end_y),
        "terminus_score": terminus_score,
    }


def _empty_candidate(orientation, reason, wh, tip, body_dx, body_dy):
    return {
        "tube_ok": False, "tube_reason": reason,
        "tip": tip, "body_dir": (body_dx, body_dy),
        "tip_end": -1 if orientation == "p0_tip" else 1,
        "w_tube": 0.0, "roi": (0, 0, 0, 0),
        "droplet": V2DropletMeasurement(ok=False, reason=reason),
        "column_end": (0.0, 0.0),
        "terminus_score": 9999.0,
    }


def drop_ok(d: dc.DropletMeasurement, w_tube: float) -> bool:
    return d.ok and 0.3 < d.diameter_in_tube_widths < 10.0


def _find_bright_blob_outside_tip(
    gray: np.ndarray,
    tip: Tuple[float, float],
    out_dx: float,
    out_dy: float,
    nx: float,
    ny: float,
    width_px: float,
):
    """Look for a *brighter-than-background* blob on the outside of
    the tube tip. Returns (cx, cy, radius, area) or (None,)*4.

    The droplet has a refraction-bright crescent on its outline and
    is roughly round. We look for connected components of pixels that
    are brighter than the local background by a margin and lie on
    the OUTSIDE of the tip (projection onto out direction > 0).
    """
    h, w = gray.shape
    look_ahead = 4.0 * width_px
    look_side = 3.0 * width_px
    half_w = int(look_side / 2)
    ahead = int(look_ahead)
    p_out_x = tip[0] + out_dx * ahead
    p_out_y = tip[1] + out_dy * ahead
    poly = np.array([
        [tip[0] + nx * half_w, tip[1] + ny * half_w],
        [tip[0] - nx * half_w, tip[1] - ny * half_w],
        [p_out_x - nx * half_w, p_out_y - ny * half_w],
        [p_out_x + nx * half_w, p_out_y + ny * half_w],
    ], dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None, None, None, None
    bx0, by0, bx1, by1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    roi_mask = mask[by0:by1 + 1, bx0:bx1 + 1]
    roi_gray = gray[by0:by1 + 1, bx0:bx1 + 1]

    # background: the median intensity of the OUTER rim of the ROI.
    margin = max(3, int(0.18 * width_px))
    rim = np.concatenate([
        roi_gray[:margin, :].flatten(),
        roi_gray[-margin:, :].flatten(),
        roi_gray[:, :margin].flatten(),
        roi_gray[:, -margin:].flatten(),
    ])
    rim = rim[rim > 0]
    bg = float(np.median(rim)) if rim.size else float(np.median(roi_gray))
    mad = float(np.median(np.abs(rim - bg))) if rim.size else 1.0
    sigma = max(1.0, 1.4826 * mad + 1e-3)
    delta = max(6.0, 2.0 * sigma)

    fg = ((roi_gray > bg + delta) & (roi_mask > 0)).astype(np.uint8) * 255
    if fg.sum() < max(40, width_px * width_px * 0.3):
        return None, None, None, None
    # Light morphology to break the tube's wall (long thin bright band)
    # away from the droplet (round-ish).
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n <= 1:
        return None, None, None, None

    # Select the largest blob that:
    #   (a) is at least 0.4 * W_tube in equivalent diameter, and
    #   (b) has its centroid within ~2 W_tube of the tip on the out side.
    best_i = -1
    best_area = 0.0
    for i in range(1, n):
        bb_x, bb_y, bb_w, bb_h, area = stats[i]
        if area < 0.05 * (width_px * width_px):
            continue
        bb_cx = bb_x + bb_w / 2.0
        bb_cy = bb_y + bb_h / 2.0
        cxp = bb_cx + bx0
        cyp = bb_cy + by0
        ax = cxp - tip[0]
        ay = cyp - tip[1]
        proj = ax * out_dx + ay * out_dy
        if proj < 0.2 * width_px:
            continue
        if proj > 4.0 * width_px:
            continue
        # Round-ish: prefer blobby aspect ratios (not the tube wall
        # which is very tall and thin).
        aspect = max(bb_w, bb_h) / max(1, min(bb_w, bb_h))
        if aspect > 4.0:
            continue
        if area > best_area:
            best_area = area
            best_i = i

    if best_i < 0:
        return None, None, None, None
    ys2, xs2 = np.where(labels == best_i)
    pts = np.column_stack((xs2, ys2)).astype(np.float32)
    (mx, my), radius = cv2.minEnclosingCircle(pts)
    cx = float(mx + bx0)
    cy = float(my + by0)
    return cx, cy, float(radius), float(best_area)


def _make_debug(bgr: np.ndarray, res: AnalysisResult) -> np.ndarray:
    out = bgr.copy()
    h, w = out.shape[:2]
    p0 = tuple(int(v) for v in res.tube.p0)
    p1 = tuple(int(v) for v in res.tube.p1)
    if p0 != (0, 0) or p1 != (0, 0):
        cv2.line(out, p0, p1, (0, 255, 255), 1, cv2.LINE_AA)
    if res.tube.ok:
        tip = tuple(int(v) for v in res.tube.tip)
        body = res.tube.body_dir
        cv2.circle(out, tip, 5, (0, 255, 0), 2, cv2.LINE_AA)
        # arrow pointing into the body
        ax2 = int(tip[0] + 60 * body[0])
        ay2 = int(tip[1] + 60 * body[1])
        cv2.arrowedLine(out, tip, (ax2, ay2), (0, 200, 0), 1, cv2.LINE_AA)
        cv2.putText(
            out,
            f"tube W = {res.tube.width_px:.1f} px  (unit = W_tube)",
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
        # axial position of column_end (where the column drops to W_tube)
        ex = int(res.droplet.column_end_x)
        ey = int(res.droplet.column_end_y)
        cv2.circle(out, (ex, ey), 3, (255, 200, 0), -1, cv2.LINE_AA)
        msg = (
            f"drop D = {res.droplet.diameter_in_tube_widths:.2f} W_tube"
            f"   h = {res.droplet.vertical_extent_in_tube_widths:+.2f} W_tube"
        )
        cv2.putText(
            out, msg,
            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 1, cv2.LINE_AA,
        )
        cv2.putText(
            out, f"diam source: {res.droplet.diameter_source or 'circle'}",
            (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA,
        )
    else:
        cv2.putText(
            out, f"drop: {res.droplet.reason}",
            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA,
        )
    cv2.putText(
        out, f"{w}x{h}  tip_end={res.tube.tip_end:+d}",
        (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA,
    )
    return out
