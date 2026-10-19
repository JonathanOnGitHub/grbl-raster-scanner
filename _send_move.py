#!/usr/bin/env python3
"""Send a stepper move command to the Leonardo stepper driver.

Protocol (115200 baud, newline terminated):
    X <steps> <speed>\\n
    -> responds with "OK" on accept, or "ERR ..." on bad input.

Usage:
    python3 _send_move.py /dev/ttyACM0 200 100
    python3 _send_move.py /dev/ttyACM0 -3200 800
"""
import sys
import time
import serial


def main() -> int:
    if len(sys.argv) != 4:
        sys.stderr.write("usage: _send_move.py <port> <steps> <speed>\n")
        return 2

    port, steps_str, speed_str = sys.argv[1], sys.argv[2], sys.argv[3]

    try:
        steps = int(steps_str)
    except ValueError:
        sys.stderr.write(f"error: steps must be an integer, got {steps_str!r}\n")
        return 2
    try:
        speed = float(speed_str)
    except ValueError:
        sys.stderr.write(f"error: speed must be a number, got {speed_str!r}\n")
        return 2

    if steps == 0:
        sys.stderr.write("error: steps must be non-zero\n")
        return 2
    if speed <= 0:
        sys.stderr.write("error: speed must be > 0\n")
        return 2

    cmd = f"X {steps} {speed}\n".encode("ascii")

    try:
        with serial.Serial(port, baudrate=115200, timeout=2.0) as ser:
            time.sleep(2.0)  # let Leonardo finish bootloader reset
            ser.reset_input_buffer()
            ser.write(cmd)
            ser.flush()
            line = ser.readline().decode("ascii", errors="replace").strip()
    except serial.SerialException as e:
        sys.stderr.write(f"error: serial: {e}\n")
        return 1

    if line == "OK":
        print("OK")
        return 0

    sys.stderr.write(f"error: device replied: {line!r}\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
