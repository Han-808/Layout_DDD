"""Run the config-only frozen generator CLI.

See ``docs/generation_transport_compatibility.md``.  This module is deliberately
not registered as the public ``layout-ddd-generate`` entrypoint.
"""

from benchmark.scene_generation.frozen_two_stage.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
