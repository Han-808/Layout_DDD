from __future__ import annotations

import runpy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    runpy.run_path(str(PROJECT_ROOT / "scripts" / "run_hssd_hab_10_qwen_validity.py"), run_name="__main__")
