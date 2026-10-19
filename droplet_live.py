#!/usr/bin/env python3
"""Live tube + droplet viewer for the back-lit USB microscope.

Opens /dev/video2, runs droplet_cv.analyze on every frame, and shows the
result in an OpenCV window. Press:
  s   save a snapshot to snapshots/live_<timestamp>.png
  d   save a snapshot AND a debug overlay
  q   quit
  p   pause/resume
  r   run a one-shot droplet measurement and print to stdout (JSON)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import cv2

import droplet_cv as dc


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dev", default="/dev/video2", help="V4L2 device")
    p.add_argument("--width", type=int, default=1280, help="capture width")
    p.add_argument("--height", type=int, default=720, help="capture height")
    p.add_argument("--fourcc", default="MJPG", help="fourcc to request")
    p.add_argument("--snapshot-dir", default="snapshots", help="where to save")
    p.add_argument("--show-debug", action="store_true",
                   help="show the detector's annotated overlay instead of the raw frame")
    p.add_argument("--measure-only", action="store_true",
                   help="print one measurement as JSON and exit (no GUI)")
    p.add_argument("--no-flip", action="store_true",
                   help="do NOT vertically flip the frame; by default the live "
                        "tool flips the camera image vertically so that the "
                        "tube tip is at the top of the image (matching the "
                        "synth convention). Pass --no-flip if the camera is "
                        "already oriented tip-up in the image.")
    return p.parse_args()


def open_cap(args):
    cap = cv2.VideoCapture(args.dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        sys.stderr.write(f"could not open {args.dev}\n")
        sys.exit(2)
    fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    return cap


def measure_and_print(res: dc.AnalysisResult):
    out = {
        "tube_ok": res.tube.ok,
        "tube_width_px": round(res.tube.width_px, 2),
        "tube_tip": [round(v, 1) for v in res.tube.tip],
        "tube_reason": res.tube.reason,
        "droplet_ok": res.droplet.ok,
        "droplet_diameter_W": round(res.droplet.diameter_in_tube_widths, 4),
        "droplet_height_W": round(res.droplet.vertical_extent_in_tube_widths, 4),
        "droplet_area_px": round(res.droplet.area_px, 1),
        "droplet_reason": res.droplet.reason,
    }
    print(json.dumps(out))


def main():
    args = parse_args()
    os.makedirs(args.snapshot_dir, exist_ok=True)
    cap = open_cap(args)
    if args.measure_only:
        ok, frame = cap.read()
        if not ok:
            sys.stderr.write("read failed\n")
            sys.exit(3)
        if not args.no_flip:
            frame = cv2.flip(frame, 0)
        res = dc.analyze(frame)
        measure_and_print(res)
        cap.release()
        return
    paused = False
    print("Live tube/droplet viewer. Keys: q=quit, s=snapshot, d=snapshot+debug, p=pause, r=print measurement")
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("read failed; retrying", file=sys.stderr)
                time.sleep(0.1)
                continue
            if not args.no_flip:
                frame = cv2.flip(frame, 0)
            res = dc.analyze(frame)
            show = res.debug if (res.debug is not None and args.show_debug) else frame
            # If we have debug but the user wants raw, still overlay the text
            # (the live overlay always shows the current measurement).
            if not args.show_debug:
                show = frame.copy()
                # Render only the measurement text + axes overlay, not the
                # full debug rendering.
                if res.tube.p0 != (0.0, 0.0) or res.tube.p1 != (0.0, 0.0):
                    p0 = tuple(int(v) for v in res.tube.p0)
                    p1 = tuple(int(v) for v in res.tube.p1)
                    cv2.line(show, p0, p1, (0, 255, 255), 1, cv2.LINE_AA)
                if res.tube.ok:
                    tip = tuple(int(v) for v in res.tube.tip)
                    cv2.circle(show, tip, 4, (0, 255, 0), 2, cv2.LINE_AA)
                if res.droplet.ok:
                    c = tuple(int(v) for v in res.droplet.center)
                    r = int(round(res.droplet.radius_px))
                    cv2.circle(show, c, max(2, r), (255, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(
                show, "q:quit  s:snap  d:snap+dbg  p:pause  r:measure",
                (10, show.shape[0] - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (200, 200, 200), 1, cv2.LINE_AA,
            )
            cv2.putText(
                show, f"{res.tube.width_px:.1f}px W_tube  drop D={res.droplet.diameter_in_tube_widths:.2f}W  h={res.droplet.vertical_extent_in_tube_widths:.2f}W",
                (10, show.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0) if res.droplet.ok else (0, 0, 255), 1, cv2.LINE_AA,
            )
            cv2.imshow("droplet_cv", show)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord("p"):
            paused = not paused
            print(f"paused={paused}")
        if key == ord("s") and "frame" in locals():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(args.snapshot_dir, f"live_{ts}.png")
            cv2.imwrite(path, frame)
            print(f"saved {path}")
        if key == ord("d") and "frame" in locals():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_path = os.path.join(args.snapshot_dir, f"live_{ts}.png")
            dbg_path = os.path.join(args.snapshot_dir, f"live_{ts}_dbg.png")
            cv2.imwrite(raw_path, frame)
            if res.debug is not None:
                cv2.imwrite(dbg_path, res.debug)
            print(f"saved {raw_path} + {dbg_path}")
        if key == ord("r") and "res" in locals():
            measure_and_print(res)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
