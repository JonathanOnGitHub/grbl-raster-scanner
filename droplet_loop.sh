#!/usr/bin/env bash
# Measure-and-print wrapper that combines droplet_measure.py and the
# loop controller into a single command. Used as:
#
#   ./droplet_loop.sh --target-D 1.5 --port /dev/ttyACM0 --rate 5
#
# All arguments after the script name are forwarded to droplet_loop.py
# (with the exception of --rate which is forwarded to
# droplet_measure.py).
set -euo pipefail
cd "$(dirname "$0")"

RATE=5
PORT=/dev/ttyACM0
DRY_RUN=0
TARGET=
ARGS=()
for a in "$@"; do
  case "$a" in
    --rate) shift; RATE="$1"; shift;;
    --port) shift; PORT="$1"; shift;;
    --dry-run) DRY_RUN=1; shift;;
    --target-D) shift; TARGET="$1"; shift;;
    *) ARGS+=("$a"); shift;;
  esac
done
if [ -z "$TARGET" ]; then
  echo "missing --target-D" >&2
  exit 2
fi
DRY_FLAG=""
if [ "$DRY_RUN" = "1" ]; then
  DRY_FLAG="--dry-run"
fi
exec python droplet_measure.py --rate "$RATE" --port "$PORT" \
  | python droplet_loop.py --port "$PORT" --rate-hint "$RATE" \
    --target-D "$TARGET" $DRY_FLAG "${ARGS[@]}"
