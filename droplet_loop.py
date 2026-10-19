#!/usr/bin/env python3
"""Closed-loop controller: drive a GRBL stepper to create a droplet of
a target size on the back-lit USB microscope.

Reads measurements from `droplet_measure.py` (one JSON per line on
stdin) and issues X-axis moves via `grbl_x_move.py` to bring the
droplet to the target diameter D_target (in tube-width units, W_tube).

Control law
-----------
The plant (stepper -> plunger -> droplet) is a slow, monotonic, and
noisy system. A simple integrator with dead-band is plenty:

  err = D_target - D_measured
  if |err| < dead_band: do nothing
  else:                 move by K_p * err, clamped to max_step

The stepper's response per millimetre depends on the syringe/plunger
geometry, so K_p and max_step are CLI-tunable defaults. Tune by
observation:

  python droplet_loop.py \\
      --target-D 1.5  --dead-band 0.05  --gain 0.04  --max-step 0.2

Sign convention
--------------
A positive X move pushes water OUT (droplet grows); a negative X
move sucks water IN (droplet shrinks). Adjust --invert if your
mechanics are the other way around.

Run the measure script in another terminal:

  python droplet_measure.py --rate 5 | python droplet_loop.py \\
      --target-D 1.5 --port /dev/ttyACM0
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-D", type=float, required=True,
                   help="target droplet diameter in tube-widths (W_tube)")
    p.add_argument("--dead-band", type=float, default=0.05,
                   help="|D_target - D_measured| below this is a no-op "
                        "(W_tube units)")
    p.add_argument("--gain", type=float, default=0.04,
                   help="X-step per W_tube of error, in mm")
    p.add_argument("--max-step", type=float, default=0.2,
                   help="maximum |X-step| per iteration, in mm")
    p.add_argument("--port", default="/dev/ttyACM0",
                   help="serial port for grbl_x_move.py")
    p.add_argument("--smooth", type=int, default=1,
                   help="median-filter window over the last N measurements")
    p.add_argument("--min-step", type=float, default=0.01,
                   help="don't move for |step| smaller than this (mm). "
                        "Stops micro-jittering once we're inside dead-band.")
    p.add_argument("--invert", action="store_true",
                   help="invert the sign of every X move (if your mechanics "
                        "are reversed: positive X = suck in, negative = push out)")
    p.add_argument("--dry-run", action="store_true",
                   help="compute the move but do not actually send it to GRBL")
    p.add_argument("--settle", type=float, default=0.0,
                   help="seconds to wait after a move before reading the next "
                        "measurement (lets the stepper + fluid settle)")
    return p.parse_args()


def send_move(port: str, dx_mm: float, dry_run: bool, invert: bool) -> bool:
    """Send a G91 X-move via grbl_x_move.py. Returns True on ok."""
    if invert:
        dx_mm = -dx_mm
    if abs(dx_mm) < 1e-6:
        return True
    if dry_run:
        sys.stderr.write(f"[dry-run] would move X by {dx_mm:+.4f} mm\n")
        return True
    cmd = ["python", "grbl_x_move.py", port, f"{dx_mm:.4f}"]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            sys.stderr.write(
                f"grbl_x_move.py failed (rc={result.returncode}): "
                f"{result.stderr.strip()}\n",
            )
            return False
        sys.stderr.write(f"moved X {dx_mm:+.4f} mm: {result.stdout.strip()}\n")
        return True
    except subprocess.TimeoutExpired:
        sys.stderr.write("grbl_x_move.py timed out\n")
        return False
    except FileNotFoundError:
        sys.stderr.write("grbl_x_move.py not found in current directory\n")
        return False


def main():
    args = parse_args()
    history = []  # recent measurements
    last_move_t = 0.0
    sys.stderr.write(
        f"target D = {args.target_D} W_tube, dead-band = {args.dead_band}, "
        f"gain = {args.gain} mm/W_tube, max-step = {args.max_step} mm\n"
    )
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            m = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not m.get("tube_ok"):
            sys.stderr.write("tube not found; skipping\n")
            continue
        if not m.get("droplet_ok"):
            # No droplet detected: don't move. Print a heartbeat so the
            # user can see we're alive.
            sys.stderr.write(
                f"  no droplet (tube W={m.get('tube_W_px', '?')}px)\n"
            )
            continue
        d = m["droplet_D_W"]
        history.append(d)
        if len(history) > max(1, args.smooth):
            history = history[-args.smooth:]
        if args.smooth > 1:
            d_used = statistics.median(history)
        else:
            d_used = d
        err = args.target_D - d_used
        sys.stderr.write(
            f"  D = {d:.3f} (used {d_used:.3f}), err = {err:+.3f} W_tube\n"
        )
        if abs(err) < args.dead_band:
            sys.stderr.write("  inside dead-band, no move\n")
            continue
        step = args.gain * err
        # Clamp.
        if abs(step) > args.max_step:
            step = math.copysign(args.max_step, step)
        if abs(step) < args.min_step:
            sys.stderr.write(
                f"  step {step:+.4f} mm below min-step {args.min_step} mm, "
                "no move\n",
            )
            continue
        # Don't double-move if we just moved (settle).
        if time.time() - last_move_t < args.settle:
            sys.stderr.write("  settle timer not elapsed, skipping\n")
            continue
        ok = send_move(args.port, step, args.dry_run, args.invert)
        last_move_t = time.time()
        if not ok and not args.dry_run:
            sys.stderr.write("  move failed; bailing\n")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
