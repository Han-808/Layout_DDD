#!/usr/bin/env python3
"""Compatibility CLI for :mod:`benchmark.api.evaluation`."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.api.evaluation import main, run_evaluate


__all__ = ["main", "run_evaluate"]


if __name__ == "__main__":
    main()
