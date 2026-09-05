"""Open an existing benchmark blend in non-semantic Material Preview mode."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import bpy


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RENDERING_DIR = PROJECT_ROOT / "src/benchmark/rendering"
if str(RENDERING_DIR) not in sys.path:
    sys.path.insert(0, str(RENDERING_DIR))

from saved_blend_view import configure_textured_inspection_view  # noqa: E402


def main() -> None:
    result = configure_textured_inspection_view(bpy)
    print(
        "benchmark textured inspection view: "
        + json.dumps(result, sort_keys=True)
    )


main()
