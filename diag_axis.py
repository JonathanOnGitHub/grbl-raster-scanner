#!/usr/bin/env python3
"""Diagnostic: dump the per-orientation analysis details."""
import sys
import math
import cv2
import numpy as np

import droplet_cv as dc
import droplet_v2 as dv2


def diagnose(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        sys.exit(f"can't read {path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_s = cv2.GaussianBlur(gray, (3, 3), 0)
    axis = dc._find_tube_axis_bright_col(gray_s)
    print(f"== {path} ==")
    if axis is None:
        print("no axis!")
        return
    p0, p1, length = axis
    print(f"axis p0={p0} p1={p1} length={length:.1f}")
    ux, uy, L = dv2._axis_unit(p0, p1)
    print(f"axis_unit=({ux:+.3f}, {uy:+.3f}) L={L:.1f}")
    nx, ny = dv2._perp_unit(ux, uy)
    # Sample axial profile along the axis
    s_arr, w_arr = dv2._column_axial_profile(
        gray, p0, ux, uy, nx, ny, max(30, L * 0.05),
        extent_units=L / max(30, L * 0.05) / 2,
    )
    # Print sparsely
    step = max(1, len(s_arr) // 40)
    for i in range(0, len(s_arr), step):
        bar = "#" * max(0, int(w_arr[i] / 4))
        print(f"  s={s_arr[i]:+6.2f}W  w={w_arr[i]:5.1f}px  {bar}")


for p in sys.argv[1:]:
    diagnose(p)
