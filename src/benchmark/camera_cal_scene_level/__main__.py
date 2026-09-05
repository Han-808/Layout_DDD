"""Executable package entrypoint for the camera-cal scene evaluator."""

from benchmark.camera_cal_scene_level.composition import (
    orchestrator_dependencies,
)
from benchmark.camera_cal_scene_level.orchestrator import run_main


if __name__ == "__main__":
    run_main(deps=orchestrator_dependencies())
