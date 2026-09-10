#!/usr/bin/env bash
set -euo pipefail
set +x
TASK_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
# This is a local campaign recipe, not a self-contained public dataset launcher.
if [[ ! -f "$TASK_REPO_ROOT/Support/artifacts/releases/complicated_eval_combined142_v1/run_combined.py" ]]; then
  echo "Missing sealed combined142 release; see docs/evaluator_publication_20260910.md. No evaluation started." >&2
  exit 2
fi
export PYTHONDONTWRITEBYTECODE=1
exec "$TASK_REPO_ROOT/.venv/bin/python" -B \
  "$TASK_REPO_ROOT/scripts/run_complicated_combined142_hardened.py" "$@"
