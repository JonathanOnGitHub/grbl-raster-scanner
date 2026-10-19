#!/usr/bin/env python3
"""Run droplet_cv.analyze on a single image (or every PNG in a directory)
and save the debug overlay next to it.

Usage:
    python droplet_debug.py path/to/image.png [more.png ...]
    python droplet_debug.py snapshots/
"""
import sys
import os
import json
import glob
import cv2
import numpy as np

import droplet_cv as dc


def analyse_one(path: str) -> None:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"{path}: could not read", file=sys.stderr)
        return
    res = dc.analyze(bgr)
    base, ext = os.path.splitext(path)
    out = base + "_dbg.png"
    if res.debug is not None:
        cv2.imwrite(out, res.debug)
    summary = {
        "tube_ok": res.tube.ok,
        "tube_width_px": round(res.tube.width_px, 2),
        "tube_reason": res.tube.reason,
        "tube_tip": [round(v, 1) for v in res.tube.tip],
        "droplet_ok": res.droplet.ok,
        "droplet_diameter_W": round(res.droplet.diameter_in_tube_widths, 3),
        "droplet_height_W": round(res.droplet.vertical_extent_in_tube_widths, 3),
        "droplet_area_px": round(res.droplet.area_px, 1),
        "droplet_reason": res.droplet.reason,
    }
    print(f"{path}: {json.dumps(summary)}")
    print(f"  -> wrote {out}")


def main(argv):
    if not argv:
        argv = ["snapshots/"]
    targets = []
    for a in argv:
        if os.path.isdir(a):
            targets.extend(sorted(glob.glob(os.path.join(a, "*.png"))))
        else:
            targets.append(a)
    for t in targets:
        analyse_one(t)


if __name__ == "__main__":
    main(sys.argv[1:])
