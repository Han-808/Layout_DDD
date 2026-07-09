# Legend HSSD Input Chain

The current input path for this repo is natural language:

```text
natural language instruction -> scene_spec -> asset retrieval -> generation input -> scene evaluation
```

HSSD-HAB is retained only as a legend compatibility chain for old benchmark
cases, regression tests, and historical sweeps. It should not be treated as the
default product input.

Use these modules for HSSD work:

```python
from benchmark.legend.hssd.hssd_hab_converter import convert_hssd_hab
from benchmark.legend.hssd.hssd_small_selector import convert_selected_small_hssd_scene
from benchmark.legend.hssd.estimated_relations import build_estimated_spatial_cues
```

The old `benchmark.datasets.*` HSSD modules still exist as compatibility
wrappers, but they emit deprecation warnings when called.

Use these CLI entry points for HSSD work:

```bash
python scripts/legend/legend_prepare_hssd_hab.py
python scripts/legend/legend_convert_hssd_hab.py
python scripts/legend/legend_select_small_hssd_hab_scene.py
python scripts/legend/legend_run_hssd_hab_10_qwen_validity.py
```

The old top-level HSSD scripts remain as forwarding wrappers for existing
automation.
