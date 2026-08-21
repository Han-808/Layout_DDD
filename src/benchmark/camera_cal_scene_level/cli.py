"""Run the promptless L1/L3 camera-cal experiment at scene level.

The frozen camera-cal cases contain generation prompts, but this experiment
does not evaluate prompt fidelity and never supplies those prompts to the
Judge. L1 remains scene-level deterministic evidence plus conditional VLM
adjudication. L3 runs the existing metric-specific scope policy, judges every
eligible group required by that metric, and preserves the evaluator's existing
scene-level aggregation.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Sequence

from benchmark.rendering.blender import CYCLES_DEVICES, RENDER_ENGINES
from benchmark.scoring_profiles import DEFAULT_DEDUCTION_MULTIPLIER


# ``scripts/run_camera_cal_scene_level.py`` historically used the checkout
# root as PROJECT_ROOT.  Resolve the same root from this source package.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

ANNOTATED_L3_METRICS = (
    "scale_consistency",
    "object_pairing_consistency",
    "style_consistency",
    "functional_consistency",
    "semantic_placement_consistency",
)

DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT / "Support" / "datasets" / "camera_cal_scenesets"
)
DEFAULT_GROUPING_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "grouping"
    / "vlm_visual_evidence_scope_v2.yaml"
)
DEFAULT_BLENDER_BIN = Path(
    os.environ.get(
        "BLENDER_BIN",
        "/Applications/Blender.app/Contents/MacOS/Blender",
    )
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError(
            "value must be finite and greater than zero"
        )
    return parsed


def _parse_args_impl(
    argv: list[str] | None = None,
    *,
    description: str | None = None,
    default_dataset_root: Path = DEFAULT_DATASET_ROOT,
    default_grouping_config: Path = DEFAULT_GROUPING_CONFIG,
    default_blender_bin: Path = DEFAULT_BLENDER_BIN,
    annotated_l3_metrics: Sequence[str] = ANNOTATED_L3_METRICS,
    render_engines: Sequence[str] = RENDER_ENGINES,
    cycles_devices: Sequence[str] = CYCLES_DEVICES,
    default_deduction_multiplier: float = DEFAULT_DEDUCTION_MULTIPLIER,
) -> argparse.Namespace:
    """Build and parse the frozen camera-cal CLI.

    The keyword-only configuration is deliberately an implementation seam for
    the legacy script façade.  The public ``parse_args`` function below keeps
    the historical one-argument signature.
    """

    parser = argparse.ArgumentParser(
        description=__doc__ if description is None else description
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=default_dataset_root,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="New or resumable run-level output directory.",
    )
    parser.add_argument(
        "--grouping-config",
        type=Path,
        default=default_grouping_config,
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Repeat to select cases. Omit to run every ready case.",
    )
    parser.add_argument(
        "--metric",
        action="append",
        choices=annotated_l3_metrics,
        default=[],
        help="Repeat to select L3 metrics. Omit to run all five annotations.",
    )
    parser.add_argument(
        "--functional-group-local-granularity",
        choices=("per_check", "batched"),
        default="per_check",
        help=(
            "Functional group-local scheduling: per_check gives every typed "
            "check an independent Judge episode with the same base group "
            "evidence; batched judges all checks in one group call."
        ),
    )
    parser.add_argument(
        "--functional-group-local-evidence-policy",
        choices=("isolated_episode", "shared_group_bank"),
        default="shared_group_bank",
        help=(
            "Functional per-check evidence sharing: isolated_episode keeps "
            "camera follow-ups private to each check; shared_group_bank "
            "is the default and reuses relevant group evidence through a "
            "bounded six-image active window."
        ),
    )
    parser.add_argument(
        "--deduction-multiplier",
        type=positive_float,
        default=default_deduction_multiplier,
        help=(
            "Multiply final deductions for Collision, Support, OOB, Scale, "
            "Style, and Object Pairing (default: 2.0; use 1.0 for the "
            "unscaled projection)."
        ),
    )
    parser.add_argument(
        "--l3-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Recovery mode: disable L1 and execute only selected L3 metrics. "
            "The resulting benchmark score is L3-only and must be merged "
            "post-hoc with a separately retained L1 result when needed."
        ),
    )
    parser.add_argument("--max-cases", type=positive_int, default=None)
    parser.add_argument("--max-workers", type=positive_int, default=1)
    parser.add_argument(
        "--endpoint-preflight-attempts",
        type=positive_int,
        default=10,
        help=(
            "Required consecutive real-image endpoint calls before any case "
            "starts (default: 10). All attempts must succeed."
        ),
    )
    parser.add_argument(
        "--endpoint-preflight-timeout-seconds",
        type=positive_int,
        default=300,
        help="Per-call timeout for the pre-run endpoint stability gate.",
    )
    parser.add_argument(
        "--terminal-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Mirror progress.jsonl events to stdout. Persistent progress "
            "events are always written."
        ),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--export-audit-graphs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Post-hoc export RelationCandidateGraph and "
            "EvaluationQueryGraph artifacts. Disabled by default and never "
            "used by evaluation or scoring."
        ),
    )
    parser.add_argument(
        "--blender-bin",
        type=Path,
        default=default_blender_bin,
    )
    parser.add_argument("--render-width", type=positive_int, default=768)
    parser.add_argument("--render-height", type=positive_int, default=768)
    parser.add_argument(
        "--render-engine",
        choices=render_engines,
        default="BLENDER_EEVEE_NEXT",
    )
    parser.add_argument(
        "--cycles-device",
        choices=cycles_devices,
        default="CPU",
    )
    parser.add_argument("--cycles-samples", type=positive_int, default=16)
    parser.add_argument(
        "--cycles-denoising",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--preview-render-engine",
        choices=render_engines,
        default="BLENDER_EEVEE_NEXT",
    )
    parser.add_argument("--preview-width", type=positive_int, default=256)
    parser.add_argument("--preview-height", type=positive_int, default=256)
    parser.add_argument(
        "--preview-cycles-samples",
        type=positive_int,
        default=1,
    )
    parser.add_argument(
        "--blender-timeout-seconds",
        type=positive_int,
        default=900,
    )
    return parser.parse_args(argv)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments using the frozen defaults."""

    # Resolve these at call time so a package-level test or façade can
    # monkeypatch a default without this function retaining an import-time
    # snapshot.
    return _parse_args_impl(
        argv,
        default_dataset_root=DEFAULT_DATASET_ROOT,
        default_grouping_config=DEFAULT_GROUPING_CONFIG,
        default_blender_bin=DEFAULT_BLENDER_BIN,
        annotated_l3_metrics=ANNOTATED_L3_METRICS,
        render_engines=RENDER_ENGINES,
        cycles_devices=CYCLES_DEVICES,
        default_deduction_multiplier=DEFAULT_DEDUCTION_MULTIPLIER,
    )


__all__ = [
    "ANNOTATED_L3_METRICS",
    "CYCLES_DEVICES",
    "DEFAULT_BLENDER_BIN",
    "DEFAULT_DATASET_ROOT",
    "DEFAULT_DEDUCTION_MULTIPLIER",
    "DEFAULT_GROUPING_CONFIG",
    "PROJECT_ROOT",
    "RENDER_ENGINES",
    "parse_args",
    "positive_float",
    "positive_int",
]
