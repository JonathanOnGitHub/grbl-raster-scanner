# GRBL Raster Scanner

A web app to raster-scan an XY grid with a GRBL-controlled CNC robot. At each point it lowers Z, runs your custom program, waits for a finish signal, raises Z, and moves on.

## Setup

### 1. Serial port permissions (Debian/Ubuntu — do this once)

```bash
sudo usermod -aG dialout $USER
# log out and back in for this to take effect
```

### 2. Find your serial port

Plug in the Arduino, then:

```bash
ls /dev/ttyUSB* /dev/ttyACM*
# or check dmesg for the device name:
dmesg | tail -20
```

The Arduino will typically appear as `/dev/ttyACM0` or `/dev/ttyUSB0`.

### 3. Create and activate a virtual environment

```bash
# Create the venv (do this once)
python3 -m venv venv

# Activate it (do this every time you open a new terminal)
source venv/bin/activate

# Your prompt will change to show (venv) when active
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the server

```bash
python server.py
```

Then open **http://localhost:5000** in your browser.

### 6. Stopping and restarting

To stop the server: `Ctrl+C`

To start again later:
```bash
source venv/bin/activate
python server.py
```

To deactivate the venv when you're done: `deactivate`

---

## How it works

1. **Connect** — select your Arduino's serial port and click Connect
2. **Unlock** — click Unlock ($X) if GRBL starts in alarm state
3. **Configure Grid** — set X/Y start, end, step, Z travel height, and Z work height
4. **User Program** — a shell command run at each point. Env vars injected:
   - `SCAN_X` — current X coordinate
   - `SCAN_Y` — current Y coordinate
   - `SCAN_Z_WORK` — the configured work Z
5. **Finish Mode** — choose how to detect "done":
   - **Serial Pattern** — regex matched against incoming serial data
   - **Manual Button** — a "✓ DONE" button appears in the UI
   - **Trust Exit Code** — the shell command blocks until done (recommended)
6. **Start Scan** — watch the grid visualiser track progress

## Per-point sequence

```
Move XY (at Z_travel)
→ Lower to Z_work
→ Run user_cmd (with SCAN_X, SCAN_Y env vars)
→ Wait for finish signal
→ Raise to Z_travel
→ Move to next point
```

## Finish protocol

The recommended approach is **Trust Exit Code** — set Finish Mode to this and let your shell script simply exit when done. No serial port conflicts.

```bash
#!/usr/bin/env bash
echo "At X=${SCAN_X} Y=${SCAN_Y}"
sleep 2          # or run your measurement here
# script exits normally — scanner moves to next point
```

Alternatively, signal via HTTP:
```bash
curl -s -X POST http://localhost:5000/api/finish_signal
```

See **`example_measure.sh`** for a ready-to-use template.

## Relative origin

Tick **"Start from current pos"** in the Scan Grid section to use your current head position as the origin. Jog to the corner of your sample, tick the box, and the scan coordinates become offsets from there.

## Serpentine path

When enabled, alternating rows scan in reverse — minimising travel distance (boustrophedon pattern).

## GRBL notes

- Jog commands use `G91 G21 G0` (relative mode). GRBL is reset to absolute mode (`G90`) at the start of each scan.
- If GRBL reports an alarm on startup, click **Unlock ($X)** before jogging or scanning.
- The four-axis setup (paired/mirrored actuators) is handled entirely in GRBL firmware — no changes needed in this app.
