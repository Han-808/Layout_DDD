from __future__ import annotations

import runpy
import sys
from pathlib import Path

LEGEND_INPUT_CHAIN = True
CURRENT_INPUT_CHAIN = "natural_language"
LEGEND_SCRIPT = Path(__file__).resolve().parent / "legend" / "legend_prepare_hssd_hab.py"


def main() -> None:
    print(
        "LEGEND compatibility entry point: HSSD input is no longer the current input path; "
        "use the canonical harness via scripts/run_scene_harness.py for current work. "
        f"Forwarding to {LEGEND_SCRIPT.relative_to(Path(__file__).resolve().parent.parent)}.",
        file=sys.stderr,
    )
    runpy.run_path(str(LEGEND_SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
