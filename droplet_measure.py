#!/usr/bin/env python3
"""Run the droplet measurement on /dev/video2 and print one JSON line
per second to stdout. Suitable for piping into the closed-loop
controller (droplet_loop.py).

Each line is:
  {"ts": <unix_ms>, "tube_ok": bool, "tube_W_px": float, "tip_x": float,
   "tip_y": float, "droplet_ok": bool, "droplet_D_W": float,
   "droplet_h_W": float, "droplet_area_px": float}

Usage:
  python droplet_measure.py            # 1 Hz, default device
  python droplet_measure.py --rate 5   # 5 Hz
  python droplet_measure.py --no-flip  # do not flip the image
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import cv2

import droplet_cv as dc


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dev", default="/dev/video2")
    p.add_argument("--rate", type=float, default=1.0, help="Hz")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fourcc", default="MJPG")
    p.add_argument("--no-flip", action="store_true",
                   help="do NOT vertically flip the frame")
    return p.parse_args()


def open_cap(args):
    cap = cv2.VideoCapture(args.dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        sys.stderr.write(f"could not open {args.dev}\n")
        sys.exit(2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    return cap


def emit(res: dc.AnalysisResult):
    out = {
        "ts": int(time.time() * 1000),
        "tube_ok": res.tube.ok,
        "tube_W_px": round(res.tube.width_px, 2),
        "tip_x": round(res.tube.tip[0], 1),
        "tip_y": round(res.tube.tip[1], 1),
        "droplet_ok": res.droplet.ok,
        "droplet_D_W": round(res.droplet.diameter_in_tube_widths, 4),
        "droplet_h_W": round(res.droplet.vertical_extent_in_tube_widths, 4),
        "droplet_area_px": round(res.droplet.area_px, 1),
    }
    print(json.dumps(out), flush=True)


def main():
    args = parse_args()
    cap = open_cap(args)
    period = 1.0 / max(args.rate, 0.1)
    next_t = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            if not args.no_flip:
                frame = cv2.flip(frame, 0)
            res = dc.analyze(frame)
            emit(res)
            next_t += period
            sleep = next_t - time.time()
            if sleep > 0:
                time.sleep(sleep)
            else:
                # Fell behind; reset
                next_t = time.time()
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()


if __name__ == "__main__":
    main()
