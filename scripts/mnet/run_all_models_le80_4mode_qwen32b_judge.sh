#!/usr/bin/env bash
set -Eeuo pipefail

# Compatibility wrapper. The old filename used to mean "all generators with
# fixed Qwen32B judge"; the current job is Qwen-only generator/judge matrix.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_qwen_matrix_le80_4mode_flush.sh" "$@"
