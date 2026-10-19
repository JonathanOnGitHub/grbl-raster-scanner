#!/usr/bin/env python3
"""Send a G91 (relative) Y move to a standalone GRBL controller on a serial port.

Usage:
    python3 grbl_y_move.py <port> <y> [--baud=115200] [--feed=500] [--no-wait]

Examples:
    python3 grbl_y_move.py /dev/ttyACM0 1
    python3 grbl_y_move.py /dev/ttyACM0 -2.5 --feed=300
    python3 grbl_y_move.py /dev/ttyACM0 0.1 --no-wait

Sends "G91 G21 G0 Y<y> F<feed>" and waits for GRBL's "ok" reply.
Defaults: baud=115200, feed=500 mm/min, waits for ok.
"""
import argparse
import sys
import time
import serial


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("port", help="Serial port device (e.g. /dev/ttyACM0)")
    p.add_argument("y", type=float, help="Y distance to move (mm). Negative for Y-. 0 is a no-op.")
    p.add_argument("--baud", type=int, default=115200, help="Baud rate (default 115200)")
    p.add_argument("--feed", type=float, default=500.0, help="Feed rate mm/min (default 500)")
    p.add_argument("--no-wait", action="store_true", help="Do not wait for GRBL 'ok' reply")
    p.add_argument("--timeout", type=float, default=10.0, help="Reply timeout seconds (default 10)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.y == 0:
        print("y=0: nothing to send")
        return 0

    sign = "" if args.y >= 0 else "-"
    cmd = f"G91 G21 G0 Y{sign}{abs(args.y)} F{args.feed}\n".encode("ascii")

    try:
        with serial.Serial(args.port, baudrate=args.baud, timeout=2.0) as ser:
            # GRBL/Arduino boards reset when the port opens — give them a moment.
            time.sleep(2.0)
            ser.reset_input_buffer()
            ser.write(cmd)
            ser.flush()

            if args.no_wait:
                print(f"Sent: {cmd.decode('ascii').strip()!r} (not waiting for ok)")
                return 0

            deadline = time.time() + args.timeout
            buf = b""
            while time.time() < deadline:
                line = ser.readline()
                if not line:
                    continue
                buf += line
                text = line.decode("ascii", errors="replace").strip()
                if text.startswith("ok"):
                    print(f"OK ({text})")
                    return 0
                if text.startswith("error"):
                    sys.stderr.write(f"GRBL error: {text}\n")
                    return 1
            sys.stderr.write(f"timed out after {args.timeout}s. Reply so far: {buf!r}\n")
            return 1

    except serial.SerialException as e:
        sys.stderr.write(f"serial error on {args.port}: {e}\n")
        return 1
    except KeyboardInterrupt:
        sys.stderr.write("\ninterrupted\n")
        return 130


if __name__ == "__main__":
    sys.exit(main())
