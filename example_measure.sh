#!/usr/bin/env bash
#
# example_measure.sh
#
# Called by the raster scanner at each XY point.
# Waits 2 seconds, then sends SCAN_DONE back over serial.
#
# Environment variables provided by the scanner:
#   SCAN_X        current X coordinate
#   SCAN_Y        current Y coordinate
#   SCAN_Z_WORK   work Z height
#
# Usage: set "User Program" in the web UI to:
#   bash /path/to/example_measure.sh
#
# Set "Finish Mode" to "Wait for Serial Pattern", pattern: SCAN_DONE

SERIAL_PORT="${GRBL_PORT:-/dev/ttyUSB0}"
BAUD="${GRBL_BAUD:-115200}"

echo "At X=${SCAN_X} Y=${SCAN_Y} — waiting 2 seconds..."

sleep 2

# Send completion signal back to the scanner
stty -F "$SERIAL_PORT" "$BAUD" raw
echo "SCAN_DONE" > "$SERIAL_PORT"

echo "Sent SCAN_DONE to $SERIAL_PORT"
