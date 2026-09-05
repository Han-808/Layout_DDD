#!/usr/bin/env python3
"""Freeze a pinned public LayoutGPT training subset; no model calls."""
from benchmark.generation_comparison.layoutgpt_icl import main

if __name__ == "__main__":
    main()
