#!/usr/bin/env python3
"""Generate a synthetic back-lit tube + droplet image and run the CV
detector on it. This lets us validate the algorithm with known ground
truth without having to actually grow a droplet on the microscope.

The synthetic scene has:
  - Flat grey background
  - Tilted translucent tube (a bright column with darker walls)
  - Optional droplet above the tube tip (a darker round blob)

The tube's "perpendicular width" is the ground truth W_tube; the droplet
diameter in W_tube units is also known. We compare both against the
detector's output.
"""
from __future__ import annotations

import math
import sys
import json
import os
from dataclasses import dataclass, asdict

import cv2
import numpy as np

import droplet_cv as dc


@dataclass
class GroundTruth:
    width_px: float
    droplet_diameter_px: float
    droplet_diameter_W: float
    droplet_offset_W: float   # vertical offset from tube tip, in W_tube


def render(
    width: int = 640,
    height: int = 480,
    tube_width_px: float = 40.0,
    tube_tilt_deg: float = 6.0,
    tip_y: float = 220.0,
    tube_bottom_y: float = 460.0,
    centre_x: float = 300.0,
    droplet_diameter_px: float = 0.0,   # 0 = no droplet
    droplet_offset_y_px: float = 0.0,  # +ve = above tip
    droplet_offset_x_px: float = 0.0,
    background: int = 168,
    tube_centre_grey: int = 210,
    wall_grey: int = 150,
    noise_sigma: float = 1.5,
    rng_seed: int = 1,
) -> tuple[np.ndarray, GroundTruth]:
    rng = np.random.default_rng(rng_seed)
    img = np.full((height, width, 3), background, dtype=np.uint8)
    grey = np.full((height, width), background, dtype=np.float32)

    tilt = math.radians(tube_tilt_deg)
    # Unit vector along the tube axis (down).
    ax, ay = math.sin(tilt), math.cos(tilt)
    # Perpendicular (left-of-axis when going down).
    nx, ny = -ay, ax

    # Perpendicular cross-section: 5 points from -W to +W with the centre
    # at tube_centre_grey, edges at wall_grey, smoothly varying.
    half = tube_width_px / 2.0
    profile_xs = np.linspace(-half - 8, half + 8, 401)
    # Two-slope cross-section: brighter centre, darker walls, flat outside.
    def cross_section(s):
        a = abs(s)
        if a < half * 0.4:
            return tube_centre_grey
        if a < half:
            # ramp from centre to wall
            t = (a - half * 0.4) / (half * 0.6)
            return tube_centre_grey * (1 - t) + wall_grey * t
        if a < half + 4:
            # soft fall-off to background
            t = (a - half) / 4.0
            return wall_grey * (1 - t) + background * t
        return float(background)
    profile = np.array([cross_section(s) for s in profile_xs], dtype=np.float32)

    # Build the tube by walking along the axis and stamping the profile.
    # Sample at sub-pixel resolution and accumulate with bilinear weights.
    # The stamping window covers from tip_y to tube_bottom_y in y, and
    # the full width in the perpendicular direction.
    axis_x0 = centre_x
    axis_y0 = tip_y
    axis_x1 = axis_x0 + (tube_bottom_y - tip_y) * math.tan(tilt)
    axis_y1 = tube_bottom_y
    # For each output pixel, compute (along, perp) and sample profile.
    ys, xs = np.mgrid[0:height, 0:width]
    # Vector "from top of axis to this pixel"
    dx = xs - axis_x0
    dy = ys - axis_y0
    along = dx * ax + dy * ay
    perp = dx * nx + dy * ny
    # Mask: only pixels where along is in [0, axis_length] and perp is
    # in profile range
    axis_len = math.hypot(axis_x1 - axis_x0, axis_y1 - axis_y0)
    in_along = (along >= 0) & (along <= axis_len)
    in_perp = (perp >= profile_xs[0]) & (perp <= profile_xs[-1])
    in_mask = in_along & in_perp
    # Index into profile
    idx = ((perp - profile_xs[0]) / (profile_xs[1] - profile_xs[0])).astype(np.int32)
    idx = np.clip(idx, 0, len(profile) - 1)
    tube_grey = np.where(in_mask, profile[idx], grey)
    grey = tube_grey

    # Add some random noise everywhere.
    grey += rng.normal(0, noise_sigma, grey.shape).astype(np.float32)
    grey = np.clip(grey, 0, 255).astype(np.uint8)
    img = np.stack([grey, grey, grey], axis=2)

    # Render droplet as a darker round blob. Drawn directly on img.
    gt = GroundTruth(
        width_px=tube_width_px,
        droplet_diameter_px=0.0,
        droplet_diameter_W=0.0,
        droplet_offset_W=0.0,
    )
    if droplet_diameter_px > 0:
        # Droplet sits in the "above tip" direction = -a (opposite of axis-down).
        # Use the tip's actual screen position, not the centre_x, so the
        # droplet lines up with the top of the rendered tube.
        # The tube top is at along=0 in our stamping parameterisation, which
        # corresponds to (axis_x0, axis_y0).
        tip_x_screen = axis_x0
        tip_y_screen = axis_y0
        cx = int(round(tip_x_screen - droplet_offset_y_px * ax + droplet_offset_x_px * nx))
        cy = int(round(tip_y_screen - droplet_offset_y_px * ay + droplet_offset_x_px * ny))
        r = int(round(droplet_diameter_px / 2))
        cv2.circle(img, (cx, cy), r, (140, 140, 140), thickness=-1)
        # Soft edge: blur a bit.
        img = cv2.GaussianBlur(img, (3, 3), 0)
        gt = GroundTruth(
            width_px=tube_width_px,
            droplet_diameter_px=float(droplet_diameter_px),
            droplet_diameter_W=droplet_diameter_px / tube_width_px,
            droplet_offset_W=droplet_offset_y_px / tube_width_px,
        )
    return img, gt


def main():
    cases = [
        # (label, kwargs)
        ("empty", {}),
        ("small_drop", dict(droplet_diameter_px=20, droplet_offset_y_px=30)),
        ("medium_drop", dict(droplet_diameter_px=40, droplet_offset_y_px=60)),
        ("big_drop", dict(droplet_diameter_px=80, droplet_offset_y_px=120)),
        ("tilted_tube", dict(tube_tilt_deg=12.0)),
        ("offcentre", dict(centre_x=200.0)),
    ]
    print(f"{'case':18s} {'tube_W':>10s} {'tip_x':>7s} {'tip_y':>7s} {'drop_D_W':>10s} {'drop_h_W':>10s} {'ok':>4s}  notes")
    os.makedirs("snapshots/synth", exist_ok=True)
    for label, kw in cases:
        img, gt = render(**kw)
        res = dc.analyze(img)
        cv2.imwrite(f"snapshots/synth/{label}.png", img)
        if res.debug is not None:
            cv2.imwrite(f"snapshots/synth/{label}_dbg.png", res.debug)
        notes = []
        if res.tube.ok:
            tube_W = res.tube.width_px
            tip_x, tip_y = res.tube.tip
        else:
            tube_W = 0
            tip_x = tip_y = 0
            notes.append("tube-FAIL: " + res.tube.reason)
        if res.droplet.ok:
            drop_D = res.droplet.diameter_in_tube_widths
            drop_h = res.droplet.vertical_extent_in_tube_widths
        else:
            drop_D = drop_h = 0
            if gt.droplet_diameter_px > 0:
                notes.append("drop-FAIL: " + res.droplet.reason)
        print(f"{label:18s} {tube_W:10.2f} {tip_x:7.1f} {tip_y:7.1f} {drop_D:10.3f} {drop_h:10.3f} {int(res.tube.ok):>4d}  {' / '.join(notes)}")
        if gt.droplet_diameter_px > 0:
            err = abs(drop_D - gt.droplet_diameter_W)
            print(f"{'':18s}  expected D={gt.droplet_diameter_W:.3f} h={gt.droplet_offset_W:.3f} -> D_err={err:.3f}")


if __name__ == "__main__":
    main()
