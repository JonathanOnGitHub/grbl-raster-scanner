#!/usr/bin/env python3
"""
GRBL Raster Scanner - Backend Server
Connects to GRBL via serial, serves a web UI, and executes raster scan jobs.
"""

import asyncio
import json
import os
import re
import serial
import serial.tools.list_ports
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, request, send_from_directory
from flask_sock import Sock

app = Flask(__name__, static_folder='static')
sock = Sock(app)

# ── Global state ──────────────────────────────────────────────────────────────
class State:
    serial_conn: Optional[serial.Serial] = None
    # Reentrant: invalidate_serial() takes this lock while the serial
    # helpers already hold it, so a plain Lock would deadlock.
    serial_lock = threading.RLock()
    ws_clients: list = []
    ws_lock = threading.Lock()
    job_running = False
    job_paused = False
    job_stop_flag = False
    current_job: dict = {}
    log_buffer: list = []          # ring buffer for new WS connections

state = State()

# ── Logging / broadcast ───────────────────────────────────────────────────────
def broadcast(msg: dict):
    txt = json.dumps(msg)
    state.log_buffer.append(txt)
    if len(state.log_buffer) > 500:
        state.log_buffer.pop(0)
    with state.ws_lock:
        dead = []
        for ws in state.ws_clients:
            try:
                ws.send(txt)
            except Exception:
                dead.append(ws)
        for ws in dead:
            state.ws_clients.remove(ws)

def log(level: str, text: str):
    print(f"[{level.upper()}] {text}")
    broadcast({"type": "log", "level": level, "text": text})

def status_update():
    broadcast({
        "type": "status",
        "job_running": state.job_running,
        "job_paused": state.job_paused,
        "job": state.current_job,
        "connected": state.serial_conn is not None and state.serial_conn.is_open,
    })

# ── Serial helpers ─────────────────────────────────────────────────────────────
def invalidate_serial(reason: str):
    """Drop the current serial handle after a fatal I/O failure.

    A USB device that has been unplugged and re-plugged leaves the old file
    descriptor pointing at a dead endpoint. pyserial's ``is_open`` still
    reports True for that handle, so every subsequent write fails with
    Errno 5 (Input/output error) until the handle is explicitly discarded.
    Clearing it here makes the UI report "Not connected" and prompts the
    user to press Connect again, instead of failing every request with 400.
    """
    with state.serial_lock:
        conn = state.serial_conn
        state.serial_conn = None
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
    log("error", f"Serial connection lost ({reason}); press Connect to reconnect")
    status_update()


def grbl_send(cmd: str, wait_ok=True, timeout=10.0) -> str:
    """Send a GRBL command and optionally wait for 'ok' or 'error'."""
    conn = state.serial_conn
    if conn is None or not conn.is_open:
        raise RuntimeError("Serial not connected")
    cmd = cmd.strip() + "\n"
    with state.serial_lock:
        try:
            conn.write(cmd.encode())
            conn.flush()
        except (OSError, serial.SerialException) as e:
            invalidate_serial(str(e))
            raise RuntimeError(f"Serial connection lost: {e}") from e
        if not wait_ok:
            return ""
        deadline = time.time() + timeout
        buf = ""
        while time.time() < deadline:
            try:
                line = conn.readline().decode(errors="replace").strip()
            except (OSError, serial.SerialException) as e:
                invalidate_serial(str(e))
                raise RuntimeError(f"Serial connection lost: {e}") from e
            if line:
                log("grbl", f"<< {line}")
            if line.startswith("ok") or line.startswith("error"):
                return line
            if line:
                buf += line + "\n"
        raise TimeoutError(f"GRBL did not respond to: {cmd.strip()}")

def grbl_wait_for_idle(timeout=120.0):
    """Poll GRBL status until state is Idle."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if state.job_stop_flag:
            raise InterruptedError("Job stopped by user")
        conn = state.serial_conn
        if conn is None or not conn.is_open:
            raise RuntimeError("Serial not connected")
        with state.serial_lock:
            try:
                conn.write(b"?")
                conn.flush()
                line = conn.readline().decode(errors="replace").strip()
            except (OSError, serial.SerialException) as e:
                invalidate_serial(str(e))
                raise RuntimeError(f"Serial connection lost: {e}") from e
        if "<Idle" in line:
            return
        time.sleep(0.2)
    raise TimeoutError("Machine did not reach Idle state")


def grbl_get_position() -> tuple:
    """Query GRBL for current position. Returns (x, y, z).

    Prefers WPos (work position, respects G92 offsets) so callers
    see coordinates consistent with what the user sees in the UI.
    Falls back to MPos if WPos is not present in the status report.

    Waits until GRBL reports `<Idle` with the *same* WPos on two
    consecutive polls before returning. This avoids a known GRBL quirk
    where the planner transitions to Idle just before the final position
    update is committed to the status report — sampling at that exact
    moment gives a half-step value. Also drains the input buffer before
    each poll so a long-lived session never returns a stale line.
    """
    deadline = time.time() + 15.0
    last_status = ""
    settled = None
    while time.time() < deadline:
        # Hold a local reference to the connection. invalidate_serial() sets
        # state.serial_conn to None from another thread (or from inside the
        # exception handler below), so re-reading the attribute later would
        # raise AttributeError on None instead of the real I/O error.
        conn = state.serial_conn
        if conn is None or not conn.is_open:
            raise RuntimeError("Serial not connected")
        with state.serial_lock:
            try:
                conn.reset_input_buffer()
            except Exception:
                pass
            try:
                conn.write(b"?")
                conn.flush()
            except (OSError, serial.SerialException) as e:
                invalidate_serial(str(e))
                raise RuntimeError(f"Serial connection lost: {e}") from e
            old_timeout = conn.timeout
            conn.timeout = 0.2
            try:
                status = ""
                t_end = time.time() + 2.0
                while time.time() < t_end and not status.startswith("<"):
                    line = conn.readline()
                    if not line:
                        continue
                    status = line.decode(errors="replace").strip()
                for _ in range(6):
                    extra = conn.readline()
                    if not extra:
                        break
            except (OSError, serial.SerialException) as e:
                invalidate_serial(str(e))
                raise RuntimeError(f"Serial connection lost: {e}") from e
            finally:
                conn.timeout = old_timeout
        if not status:
            continue
        last_status = status
        if "<Idle" not in status:
            log("grbl", f"<< {status} (still in motion, waiting)")
            settled = None
            time.sleep(0.1)
            continue
        m = re.search(r'WPos:([-\d.]+),([-\d.]+),([-\d.]+)', status)
        if not m:
            m = re.search(r'MPos:([-\d.]+),([-\d.]+),([-\d.]+)', status)
        if not m:
            time.sleep(0.05)
            continue
        x, y, z = float(m.group(1)), float(m.group(2)), float(m.group(3))
        if settled is not None and (settled[0] == x and settled[1] == y and settled[2] == z):
            log("grbl", f"<< {status} (settled)")
            return x, y, z
        settled = (x, y, z)
        log("grbl", f"<< {status} (idle, awaiting confirmation)")
        time.sleep(0.1)
    raise TimeoutError(f"Could not read settled position from GRBL (last: {last_status!r})")

def wait_subprocess(proc, timeout):
    """Wait for a subprocess to exit, but abort immediately on Stop or timeout.

    Returns the captured (stdout, stderr) tuple on normal exit.
    Raises InterruptedError if the job-stop flag is set (and kills the process).
    Raises TimeoutError if the timeout elapses (and kills the process).
    """
    deadline = time.time() + timeout
    while proc.poll() is None:
        if state.job_stop_flag:
            proc.kill()
            proc.wait()
            raise InterruptedError("Job stopped by user")
        if time.time() > deadline:
            proc.kill()
            proc.wait()
            raise TimeoutError(f"User program timed out after {timeout}s")
        time.sleep(0.05)
    return proc.communicate()

def grbl_wait_for_response(pattern: str, timeout=300.0):
    """Wait for a line matching a regex pattern on the serial port."""
    rx = re.compile(pattern)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if state.job_stop_flag:
            raise InterruptedError("Job stopped by user")
        conn = state.serial_conn
        if conn is None or not conn.is_open:
            raise RuntimeError("Serial not connected")
        with state.serial_lock:
            try:
                line = conn.readline().decode(errors="replace").strip()
            except (OSError, serial.SerialException) as e:
                invalidate_serial(str(e))
                raise RuntimeError(f"Serial connection lost: {e}") from e
        if line:
            log("serial", f"<< {line}")
            if rx.search(line):
                return line
        time.sleep(0.05)
    raise TimeoutError(f"Timed out waiting for pattern: {pattern}")

# ── REST API ───────────────────────────────────────────────────────────────────
@app.route("/api/ports")
def list_ports():
    ports = [{"device": p.device, "description": p.description}
             for p in serial.tools.list_ports.comports()]
    return jsonify(ports)

@app.route("/api/connect", methods=["POST"])
def connect():
    data = request.json or {}
    port = data.get("port")
    if not port:
        return jsonify({"ok": False, "error": "No port selected"}), 400
    try:
        baud = int(data.get("baud", 115200))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Invalid baud rate"}), 400
    try:
        # Drop any previous handle first — including a stale one left behind
        # by a device that was unplugged — so the new connection starts clean.
        if state.serial_conn is not None:
            try:
                state.serial_conn.close()
            except Exception:
                pass
            state.serial_conn = None
        conn = serial.Serial(port, baud, timeout=2)
        time.sleep(2)  # GRBL boot
        conn.flushInput()
        state.serial_conn = conn
        log("info", f"Connected to {port} @ {baud} baud")
        status_update()
        return jsonify({"ok": True})
    except Exception as e:
        state.serial_conn = None
        log("error", f"Connect to {port} failed: {e}")
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route("/api/disconnect", methods=["POST"])
def disconnect():
    # A running job thread may still be using the handle. Closing it here
    # while the job is mid-move leaves that thread writing to a None/closed
    # object, which kills the job with an unhandled exception. Stop the job
    # first so the thread unwinds cleanly before the handle goes away.
    if state.job_running:
        state.job_stop_flag = True
        log("warn", "Stop requested before disconnecting")
        deadline = time.time() + 5.0
        while state.job_running and time.time() < deadline:
            time.sleep(0.05)
    if state.serial_conn is not None:
        try:
            state.serial_conn.close()
        except Exception:
            pass
        state.serial_conn = None
    log("info", "Disconnected")
    status_update()
    return jsonify({"ok": True})

@app.route("/api/jog", methods=["POST"])
def jog():
    data = request.json or {}
    cmd = data.get("cmd", "")
    if not cmd:
        return jsonify({"ok": False, "error": "No command supplied"}), 400
    try:
        grbl_send(cmd, wait_ok=True)
        return jsonify({"ok": True})
    except Exception as e:
        # A dropped connection is a distinct, recoverable condition: the UI
        # shows "press Connect" rather than a generic bad-request message.
        if state.serial_conn is None:
            return jsonify({"ok": False, "error": str(e),
                            "reconnect_required": True}), 409
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route("/api/estop", methods=["POST"])
def estop():
    """Emergency stop: send a soft-reset (Ctrl-X 0x18) and request a feed hold
    first so the controller aborts the move in progress. Safe to call at any
    time, even during a job (though the job loop may also be cancelled)."""
    if not state.serial_conn or not state.serial_conn.is_open:
        return jsonify({"ok": False, "error": "Not connected"}), 400
    try:
        with state.serial_lock:
            # 1. Feed hold: stop motion immediately without resetting state.
            try:
                state.serial_conn.write(b"!")
                state.serial_conn.flush()
                time.sleep(0.05)
            except Exception:
                pass
            # 2. Drain the bootloader reset banner that GRBL emits on open.
            try:
                state.serial_conn.reset_input_buffer()
            except Exception:
                pass
            # 3. Soft reset: returns GRBL to a known Idle state.
            state.serial_conn.write(b"\x18")
            state.serial_conn.flush()
            time.sleep(0.2)
            try:
                state.serial_conn.reset_input_buffer()
            except Exception:
                pass
        # Also flag any in-flight job as stopped so the runner loop notices.
        if state.job_running and not state.job_stop_flag:
            state.job_stop_flag = True
            log("warn", "Emergency stop requested during a job")
        log("warn", "EMERGENCY STOP issued (soft reset)")
        return jsonify({"ok": True})
    except Exception as e:
        log("error", f"Emergency stop failed: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/position", methods=["GET"])
def position():
    """Return current GRBL work position as JSON. Used by the Jog UI
    to display live coordinates while the Jog tab is open."""
    if not state.serial_conn or not state.serial_conn.is_open:
        return jsonify({"ok": False, "error": "Not connected",
                        "reconnect_required": True}), 409
    try:
        x, y, z = grbl_get_position()
        return jsonify({"ok": True, "x": x, "y": y, "z": z})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e),
                        "reconnect_required": state.serial_conn is None}), 500

@app.route("/api/set_z_work", methods=["POST"])
def set_z_work():
    """Read the current GRBL Z (WPos) and return it to the caller. No motion,
    no G92 — the UI uses the returned value to populate the Grid tab's
    z_work input."""
    if not state.serial_conn or not state.serial_conn.is_open:
        return jsonify({"ok": False, "error": "Not connected",
                        "reconnect_required": True}), 409
    try:
        x, y, z = grbl_get_position()
        return jsonify({"ok": True, "x": x, "y": y, "z": z})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e),
                        "reconnect_required": state.serial_conn is None}), 500

@app.route("/api/home", methods=["POST"])
def home():
    try:
        grbl_send("$H", wait_ok=True, timeout=60)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route("/api/unlock", methods=["POST"])
def unlock():
    try:
        grbl_send("$X", wait_ok=True)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route("/api/start", methods=["POST"])
def start_job():
    if state.job_running:
        return jsonify({"ok": False, "error": "Job already running"}), 400
    config = request.json
    t = threading.Thread(target=run_job, args=(config,), daemon=True)
    t.start()
    return jsonify({"ok": True})

@app.route("/api/pause", methods=["POST"])
def pause_job():
    state.job_paused = not state.job_paused
    log("info", "Job paused" if state.job_paused else "Job resumed")
    status_update()
    return jsonify({"ok": True, "paused": state.job_paused})

@app.route("/api/stop", methods=["POST"])
def stop_job():
    state.job_stop_flag = True
    log("warn", "Stop requested")
    return jsonify({"ok": True})

@app.route("/api/finish_signal", methods=["POST"])
def manual_finish():
    """Allow the UI to send a manual 'finish' signal."""
    state._manual_finish = True
    log("info", "Manual finish signal received")
    return jsonify({"ok": True})

@app.route("/api/set_origin", methods=["POST"])
def set_origin():
    """Set the current head position as the work origin (G92 X0 Y0 Z0)."""
    if not state.serial_conn or not state.serial_conn.is_open:
        return jsonify({"ok": False, "error": "Not connected"}), 400
    if state.job_running:
        return jsonify({"ok": False, "error": "Cannot set origin while a job is running"}), 400
    try:
        grbl_send("G92 X0 Y0 Z0")
        grbl_wait_for_idle()
        log("info", "Work origin set at current position (A1)")
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# ── WebSocket ──────────────────────────────────────────────────────────────────
@sock.route("/ws")
def ws_handler(ws):
    with state.ws_lock:
        state.ws_clients.append(ws)
    # Replay recent log
    for msg in state.log_buffer[-50:]:
        try:
            ws.send(msg)
        except Exception:
            break
    status_update()
    try:
        while True:
            ws.receive(timeout=30)
    except Exception:
        pass
    with state.ws_lock:
        if ws in state.ws_clients:
            state.ws_clients.remove(ws)

# ── Job runner ─────────────────────────────────────────────────────────────────
def run_job(config: dict):
    state.job_running = True
    state.job_paused = False
    state.job_stop_flag = False
    state._manual_finish = False

    try:
        # Parse config
        x_start  = float(config["x_start"])
        x_end    = float(config["x_end"])
        x_step   = float(config["x_step"])
        y_start  = float(config["y_start"])
        y_end    = float(config["y_end"])
        y_step   = float(config["y_step"])
        z_travel = float(config["z_travel"])   # Z when moving
        z_work   = float(config["z_work"])     # Z when working
        feed_xy  = float(config.get("feed_xy", 1000))
        feed_z   = float(config.get("feed_z", 300))
        user_cmd = config.get("user_cmd", "").strip()
        finish_pattern = config.get("finish_pattern", "SCAN_DONE").strip()
        finish_mode = config.get("finish_mode", "serial")  # serial|manual|none
        cmd_timeout = float(config.get("cmd_timeout", 120))
        serpentine = config.get("serpentine", True)
        relative_origin = config.get("relative_origin", False)
        relative_move = config.get("relative_move", False)
        return_to_start = config.get("return_to_start", False)
        run_program = config.get("run_program", True)
        post_exit_delay = max(0.0, float(config.get("post_exit_delay", 1.0)))

        # Build point grid
        xs = []
        x = x_start
        while x <= x_end + 1e-9:
            xs.append(round(x, 4))
            x += x_step

        ys = []
        y = y_start
        while y <= y_end + 1e-9:
            ys.append(round(y, 4))
            y += y_step

        total = len(xs) * len(ys)
        done = 0
        state.current_job = {"total": total, "done": 0, "x": None, "y": None}
        status_update()

        log("info", f"Starting raster scan: {len(xs)}\u00d7{len(ys)} = {total} points" +
            (" (dry-run: program disabled)" if not run_program else ""))

        # If relative origin, offset all points by current work position.
        # Uses WPos so that a prior G92 (e.g. "Set origin at A1") is respected.
        if relative_origin:
            ox, oy, _ = grbl_get_position()
            ox, oy = round(ox, 4), round(oy, 4)
            xs = [round(x + ox, 4) for x in xs]
            ys = [round(y + oy, 4) for y in ys]
            log("info", f"Relative origin: offsetting by X{ox} Y{oy} (first point = current position, no XY move)")
            broadcast({"type": "origin", "ox": ox, "oy": oy})
        else:
            broadcast({"type": "origin", "ox": 0, "oy": 0})

        # Distance mode for all scan moves: G90 (absolute) or G91 (relative/incremental).
        # In G91 mode we issue deltas from the current position instead of target coordinates.
        dist_mode = "G91" if relative_move else "G90"

        # Track current position locally so G91 deltas are computed correctly
        # without having to poll GRBL after every move.
        cur_x, cur_y, cur_z = grbl_get_position()
        cur_x, cur_y, cur_z = round(cur_x, 4), round(cur_y, 4), round(cur_z, 4)

        # Remember the scan start position for the optional "return to start" move.
        start_x, start_y, start_z = cur_x, cur_y, cur_z
        job_completed_normally = False

        # Raise Z to travel height first, unless the user has explicitly anchored
        # the job at the current position — in that case we trust them to have
        # the head already at a safe height and skip the pre-scan Z move.
        if not relative_origin:
            if relative_move:
                dz = round(z_travel - cur_z, 4)
                grbl_send(f"G91 G0 Z{dz} F{feed_z}")
            else:
                grbl_send(f"G90 G0 Z{z_travel} F{feed_z}")
            grbl_wait_for_idle()
            cur_z = z_travel

        for row_i, y in enumerate(ys):
            col_iter = reversed(xs) if (serpentine and row_i % 2 == 1) else iter(xs)

            for x in col_iter:
                # Pause support
                while state.job_paused and not state.job_stop_flag:
                    time.sleep(0.2)

                if state.job_stop_flag:
                    raise InterruptedError("Job stopped")

                log("info", f"→ Moving to X{x} Y{y}")

                # Move XY at travel height
                if relative_move:
                    dx = round(x - cur_x, 4)
                    dy = round(y - cur_y, 4)
                    if dx != 0 or dy != 0:
                        grbl_send(f"G91 G0 X{dx} Y{dy} F{feed_xy}")
                else:
                    grbl_send(f"G90 G0 X{x} Y{y} F{feed_xy}")
                grbl_wait_for_idle()
                cur_x, cur_y = x, y

                # Lower Z to work height
                if relative_move:
                    dz = round(z_work - cur_z, 4)
                    if dz != 0:
                        grbl_send(f"G91 G0 Z{dz} F{feed_z}")
                else:
                    grbl_send(f"G90 G0 Z{z_work} F{feed_z}")
                grbl_wait_for_idle()
                cur_z = z_work

                broadcast({"type": "point", "x": x, "y": y, "done": done, "total": total})

                # Run user program (unless dry-run)
                if run_program and user_cmd:
                    env = {**os.environ, "SCAN_X": str(x), "SCAN_Y": str(y),
                           "SCAN_Z_WORK": str(z_work)}
                    log("info", f"  Running: {user_cmd}")
                    proc = subprocess.Popen(user_cmd, shell=True, env=env,
                                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    out, _ = wait_subprocess(proc, timeout=cmd_timeout)
                    if out:
                        log("cmd", out.decode(errors="replace").strip())

                # Wait for finish signal (skipped in dry-run)
                if run_program:
                    if finish_mode == "serial":
                        log("info", f"  Waiting for serial pattern: '{finish_pattern}'")
                        grbl_wait_for_response(finish_pattern, timeout=cmd_timeout)
                    elif finish_mode == "subprocess":
                        # Already waited in proc.communicate() above — nothing more to do.
                        pass
                    elif finish_mode == "manual":
                        log("info", "  Waiting for manual finish signal…")
                        deadline = time.time() + cmd_timeout
                        while not state._manual_finish and not state.job_stop_flag:
                            if time.time() > deadline:
                                raise TimeoutError("Manual finish timed out")
                            time.sleep(0.1)
                        state._manual_finish = False
                else:
                    log("info", "  (dry-run: skipping program + finish wait)")

                # Settle delay after the program/finish signal — gives downstream
                # hardware (camera, valves, etc.) a moment to stabilise before
                # the next XY move. Cancelled instantly if the user hits Stop.
                if post_exit_delay > 0 and run_program:
                    log("info", f"  Waiting {post_exit_delay:g}s after program exit…")
                    waited = 0.0
                    step = 0.05
                    while waited < post_exit_delay:
                        if state.job_stop_flag:
                            raise InterruptedError("Job stopped")
                        time.sleep(step)
                        waited += step

                # Raise Z
                if relative_move:
                    dz = round(z_travel - cur_z, 4)
                    if dz != 0:
                        grbl_send(f"G91 G0 Z{dz} F{feed_z}")
                else:
                    grbl_send(f"G90 G0 Z{z_travel} F{feed_z}")
                grbl_wait_for_idle()
                cur_z = z_travel

                done += 1
                state.current_job = {"total": total, "done": done, "x": x, "y": y}
                status_update()

        log("info", "✓ Scan complete!")
        job_completed_normally = True

        # Optional: return XY to the position the head was at when the job started.
        if return_to_start and (start_x, start_y) != (cur_x, cur_y):
            log("info", f"↩ Returning to start position X{start_x} Y{start_y}")
            if relative_move:
                dx = round(start_x - cur_x, 4)
                dy = round(start_y - cur_y, 4)
                if dx != 0 or dy != 0:
                    grbl_send(f"G91 G0 X{dx} Y{dy} F{feed_xy}")
            else:
                grbl_send(f"G90 G0 X{start_x} Y{start_y} F{feed_xy}")
            grbl_wait_for_idle()

    except InterruptedError:
        log("warn", "Job stopped by user")
    except RuntimeError as e:
        # Raised by grbl_send/grbl_get_position when the serial link dies.
        # Report it as a connection loss rather than an opaque job failure.
        log("error", f"Job aborted: {e}")
    except Exception as e:
        log("error", f"Job failed: {e}")
    finally:
        state.job_running = False
        state.job_paused = False
        state.job_stop_flag = False
        state.current_job = {}
        status_update()

# ── Static files ───────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    print("GRBL Raster Scanner running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
