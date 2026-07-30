#!/usr/bin/env python3
"""Compatibility CLI for :mod:`benchmark.api.generation`."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.api.generation import main, run_generate, run_generate_from_natural_language


__all__ = ["main", "run_generate", "run_generate_from_natural_language"]


if __name__ == "__main__":
    main()
