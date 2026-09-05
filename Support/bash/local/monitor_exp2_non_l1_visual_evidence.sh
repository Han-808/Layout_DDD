#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/Support/artifacts/outputs/exp2_non_l1_visual_evidence_gpt56}"
RUN_LOG="${RUN_LOG:-/private/tmp/layoutddd-exp2-non-l1.log}"
PID_FILE="${PID_FILE:-/private/tmp/layoutddd-exp2-non-l1.pid}"
INTERVAL="${INTERVAL:-300}"
EXPECTED_PER_REPEAT=480

if ! [[ "$INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
  echo "INTERVAL must be a positive integer number of seconds." >&2
  exit 2
fi

while true; do
  clear
  date '+%Y-%m-%d %H:%M:%S %Z'
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "launcher: running (PID $pid)"
  elif [[ -n "$pid" ]]; then
    echo "launcher: stopped (last PID $pid)"
  else
    echo "launcher: PID file not found"
  fi

  if lsof -nP -iTCP:4010 -sTCP:LISTEN >/dev/null 2>&1; then
    echo "GPT-5.6-Sol proxy: listening on 127.0.0.1:4010"
  else
    echo "GPT-5.6-Sol proxy: not listening"
  fi

  for repeat_index in 1 2; do
    repeat_root="$RUN_ROOT/repeat_${repeat_index}"
    if [[ -d "$repeat_root/events" ]]; then
      counts="$(
        "$PYTHON" -c '
import json
import pathlib
import sys

files = list((pathlib.Path(sys.argv[1]) / "events").glob("*/*.json"))
ready = 0
failed = 0
for path in files:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        failed += 1
        continue
    if value.get("error"):
        failed += 1
    else:
        ready += 1
print(f"{ready} {failed} {len(files)}")
' "$repeat_root"
      )"
      read -r ready failed materialized <<<"$counts"
      echo "repeat_${repeat_index}: ready=$ready failed=$failed materialized=$materialized/$EXPECTED_PER_REPEAT"
    else
      echo "repeat_${repeat_index}: not started (0/$EXPECTED_PER_REPEAT)"
    fi
  done

  if [[ -f "$RUN_ROOT/analysis/report.md" ]]; then
    echo "repeat analysis: ready"
  else
    echo "repeat analysis: pending"
  fi

  echo
  echo "Recent launcher output:"
  tail -n 24 "$RUN_LOG" 2>/dev/null || echo "(launcher log not created yet)"
  echo
  echo "Refreshing in ${INTERVAL}s. Ctrl-C stops only this monitor."
  sleep "$INTERVAL"
done
