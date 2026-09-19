#!/usr/bin/env python3
"""The sole new-experiment Model Open-space/Multi-room evaluation entrypoint."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from benchmark.camera_cal_scene_level.uniform import main

if __name__ == '__main__':
    main()
