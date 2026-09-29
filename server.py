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
    serial_lock = threading.Lock()
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
def grbl_send(cmd: str, wait_ok=True, timeout=10.0) -> str:
    """Send a GRBL command and optionally wait for 'ok' or 'error'."""
    if not state.serial_conn or not state.serial_conn.is_open:
        raise RuntimeError("Serial not connected")
    cmd = cmd.strip() + "\n"
    with state.serial_lock:
        state.serial_conn.write(cmd.encode())
        state.serial_conn.flush()
        if not wait_ok:
            return ""
        deadline = time.time() + timeout
        buf = ""
        while time.time() < deadline:
            line = state.serial_conn.readline().decode(errors="replace").strip()
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
        with state.serial_lock:
            state.serial_conn.write(b"?")
            state.serial_conn.flush()
            line = state.serial_conn.readline().decode(errors="replace").strip()
        if "<Idle" in line:
            return
        time.sleep(0.2)
    raise TimeoutError("Machine did not reach Idle state")


def grbl_get_position() -> tuple:
    """Query GRBL for current position. Returns (x, y, z).

    Prefers WPos (work position, respects G92 offsets) so callers
    see coordinates consistent with what the user sees in the UI.
    Falls back to MPos if WPos is not present in the status report.
    """
    deadline = time.time() + 5.0
    while time.time() < deadline:
        with state.serial_lock:
            state.serial_conn.write(b"?")
            state.serial_conn.flush()
            line = state.serial_conn.readline().decode(errors="replace").strip()
        # Status line looks like: <Idle|MPos:1,2,3|WPos:0,0,0|...>
        m = re.search(r'WPos:([-\d.]+),([-\d.]+),([-\d.]+)', line)
        if not m:
            m = re.search(r'MPos:([-\d.]+),([-\d.]+),([-\d.]+)', line)
        if m:
            return float(m.group(1)), float(m.group(2)), float(m.group(3))
        time.sleep(0.1)
    raise TimeoutError("Could not read position from GRBL")

def grbl_wait_for_response(pattern: str, timeout=300.0):
    """Wait for a line matching a regex pattern on the serial port."""
    rx = re.compile(pattern)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if state.job_stop_flag:
            raise InterruptedError("Job stopped by user")
        with state.serial_lock:
            line = state.serial_conn.readline().decode(errors="replace").strip()
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
    data = request.json
    port = data.get("port")
    baud = int(data.get("baud", 115200))
    try:
        if state.serial_conn and state.serial_conn.is_open:
            state.serial_conn.close()
        conn = serial.Serial(port, baud, timeout=2)
        time.sleep(2)  # GRBL boot
        conn.flushInput()
        state.serial_conn = conn
        log("info", f"Connected to {port} @ {baud} baud")
        status_update()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route("/api/disconnect", methods=["POST"])
def disconnect():
    if state.serial_conn:
        state.serial_conn.close()
        state.serial_conn = None
    log("info", "Disconnected")
    status_update()
    return jsonify({"ok": True})

@app.route("/api/jog", methods=["POST"])
def jog():
    data = request.json
    cmd = data.get("cmd", "")
    try:
        grbl_send(cmd, wait_ok=True)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

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
    if state.ser is None:
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
        finish_mode = config.get("finish_mode", "serial")  # serial|http|manual
        cmd_timeout = float(config.get("cmd_timeout", 120))
        serpentine = config.get("serpentine", True)
        relative_origin = config.get("relative_origin", False)
        run_program = config.get("run_program", True)

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

        # Raise Z to travel height first
        grbl_send(f"G90 G0 Z{z_travel} F{feed_z}")
        grbl_wait_for_idle()

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
                grbl_send(f"G90 G0 X{x} Y{y} F{feed_xy}")
                grbl_wait_for_idle()

                # Lower Z to work height
                grbl_send(f"G90 G0 Z{z_work} F{feed_z}")
                grbl_wait_for_idle()

                broadcast({"type": "point", "x": x, "y": y, "done": done, "total": total})

                # Run user program (unless dry-run)
                if run_program and user_cmd:
                    env = {**os.environ, "SCAN_X": str(x), "SCAN_Y": str(y),
                           "SCAN_Z_WORK": str(z_work)}
                    log("info", f"  Running: {user_cmd}")
                    proc = subprocess.Popen(user_cmd, shell=True, env=env,
                                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    out, _ = proc.communicate(timeout=cmd_timeout)
                    if out:
                        log("cmd", out.decode(errors="replace").strip())

                # Wait for finish signal (skipped in dry-run)
                if run_program:
                    if finish_mode == "serial":
                        log("info", f"  Waiting for serial pattern: '{finish_pattern}'")
                        grbl_wait_for_response(finish_pattern, timeout=cmd_timeout)
                    elif finish_mode == "manual":
                        log("info", "  Waiting for manual finish signal…")
                        deadline = time.time() + cmd_timeout
                        while not state._manual_finish and not state.job_stop_flag:
                            if time.time() > deadline:
                                raise TimeoutError("Manual finish timed out")
                            time.sleep(0.1)
                        state._manual_finish = False
                    # http mode: just trust the subprocess completed
                else:
                    log("info", "  (dry-run: skipping program + finish wait)")

                # Raise Z
                grbl_send(f"G90 G0 Z{z_travel} F{feed_z}")
                grbl_wait_for_idle()

                done += 1
                state.current_job = {"total": total, "done": done, "x": x, "y": y}
                status_update()

        log("info", "✓ Scan complete!")

    except InterruptedError:
        log("warn", "Job stopped by user")
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
