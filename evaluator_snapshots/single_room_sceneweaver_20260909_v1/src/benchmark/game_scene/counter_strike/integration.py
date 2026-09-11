"""One-capture integration of canonical Game metrics and Counter-Strike L4.

The browser capture must already exist.  The supplied renderer is required to
be a frozen-capture adapter rooted at that directory; this module never opens a
browser and never creates another observation.  A collision-capable adapter is
required because the generic ``FrozenBrowserCaptureRenderer`` exposes only
style-local evidence and is not, by itself, a valid callable P0b provider.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

from benchmark.api.submission import (
    SubmissionEvaluationError,
    evaluate_submission,
)
from benchmark.rendering.browser import FrozenBrowserCaptureRenderer
from benchmark.utils.io import read_json, write_json

from .collision_evidence import CounterStrikeFrozenCaptureRenderer
from .evaluator import (
    evaluate_counter_strike_l4,
    merge_counter_strike_evaluation,
)
from .evidence import load_counter_strike_frozen_evidence
from .loader import CounterStrikeBenchmarkConfig, CounterStrikeCaseContract


COUNTER_STRIKE_CAPTURE_INTEGRATION_VERSION = (
    "counter_strike_capture_integration_v1"
)


class CounterStrikeIntegrationError(RuntimeError):
    """Raised when the one-capture integration boundary is violated."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"Counter-Strike integration failed [{code}]: {message}")


def evaluate_counter_strike_frozen_capture(
    *,
    out_dir: str | Path,
    capture_dir: str | Path,
    canonical_case_bundle: Any,
    benchmark_config: CounterStrikeBenchmarkConfig,
    case_contract: CounterStrikeCaseContract,
    canonical_vlm_judge: Any,
    counter_strike_visual_judge: Any,
    renderer: Any,
    official_mode: bool = True,
    canonical_evaluator: Callable[..., dict[str, Any]] = evaluate_submission,
) -> dict[str, Any]:
    """Run canonical L1/L3 and CS L4 over one immutable browser capture.

    ``renderer`` is intentionally injected.  It must reuse ``capture_dir`` and
    must be callable as the collision evidence provider in addition to
    exposing ``provide_scene_quality_evidence``.  This lets the CS-specific
    collision adapter remain independent while preventing this integration
    layer from silently falling back to style views for Collision.
    """

    if not isinstance(benchmark_config, CounterStrikeBenchmarkConfig):
        raise CounterStrikeIntegrationError(
            "benchmark_config_unvalidated",
            "benchmark_config must be prevalidated",
        )
    if not isinstance(case_contract, CounterStrikeCaseContract):
        raise CounterStrikeIntegrationError(
            "case_contract_unvalidated",
            "case_contract must be prevalidated",
        )
    destination = Path(out_dir).expanduser().resolve()
    capture = Path(capture_dir).expanduser().resolve()
    expected_capture = destination / "renders"
    if capture != expected_capture:
        raise CounterStrikeIntegrationError(
            "capture_directory_mismatch",
            "the canonical submission runner can reuse a frozen capture only "
            "from <out_dir>/renders",
        )
    if not callable(getattr(renderer, "render_scene", None)):
        raise CounterStrikeIntegrationError(
            "frozen_renderer_invalid",
            "renderer must expose render_scene",
        )
    renderer_capture = getattr(renderer, "capture_dir", None)
    if renderer_capture is None or Path(renderer_capture).resolve() != capture:
        raise CounterStrikeIntegrationError(
            "frozen_renderer_invalid",
            "renderer.capture_dir must match the supplied frozen capture",
        )
    if not (
        isinstance(renderer, FrozenBrowserCaptureRenderer)
        or isinstance(renderer, CounterStrikeFrozenCaptureRenderer)
        or bool(getattr(renderer, "is_frozen_capture_renderer", False))
    ):
        raise CounterStrikeIntegrationError(
            "frozen_renderer_invalid",
            "renderer must be a FrozenBrowserCaptureRenderer or explicitly "
            "declare is_frozen_capture_renderer=True",
        )
    if not callable(renderer):
        raise CounterStrikeIntegrationError(
            "collision_evidence_provider_missing",
            "the CS renderer must be callable as a metric-local Collision "
            "evidence provider; the style-only frozen renderer is insufficient",
        )
    if not callable(getattr(renderer, "provide_scene_quality_evidence", None)):
        raise CounterStrikeIntegrationError(
            "style_evidence_provider_missing",
            "the CS renderer must expose frozen style evidence",
        )

    destination.mkdir(parents=True, exist_ok=True)
    frozen_evidence = load_counter_strike_frozen_evidence(
        capture,
        benchmark_config=benchmark_config,
    )
    manifest = read_json(frozen_evidence.manifest_path)
    exported_scene_path = Path(str(manifest["exported_scene"])).resolve()
    scene = read_json(exported_scene_path)

    canonical_report: dict[str, Any]
    canonical_incomplete_exception: str | None = None
    canonical_report_path = destination / "evaluation_report.json"
    prior_report_mtime_ns = (
        canonical_report_path.stat().st_mtime_ns
        if canonical_report_path.is_file()
        else None
    )
    try:
        canonical_result = canonical_evaluator(
            scene=exported_scene_path,
            case_bundle=canonical_case_bundle,
            out_dir=destination,
            renderer=renderer,
            vlm_judge=canonical_vlm_judge,
            official_mode=official_mode,
        )
        canonical_report = canonical_result["evaluation_report"]
    except SubmissionEvaluationError as exc:
        if not canonical_report_path.is_file():
            raise
        if (
            prior_report_mtime_ns is not None
            and canonical_report_path.stat().st_mtime_ns
            == prior_report_mtime_ns
        ):
            raise CounterStrikeIntegrationError(
                "canonical_report_stale",
                "canonical evaluator failed before refreshing its durable report",
            ) from exc
        canonical_report = read_json(canonical_report_path)
        canonical_incomplete_exception = type(exc).__name__

    l4_report = evaluate_counter_strike_l4(
        scene,
        case_contract=case_contract,
        benchmark_config=benchmark_config,
        visual_judge=counter_strike_visual_judge,
        frozen_evidence=frozen_evidence,
        out_dir=destination / "counter_strike_l4",
    )
    report = merge_counter_strike_evaluation(
        canonical_report,
        l4_report,
        benchmark_config=benchmark_config,
    )
    report["protocol_scope"] = (
        "official_counter_strike_submission"
        if official_mode
        else "trusted_counter_strike_diagnostic"
    )
    report["official_submission"] = bool(
        official_mode and report.get("benchmark_score_status") == "complete"
    )
    report["integration"] = {
        "version": COUNTER_STRIKE_CAPTURE_INTEGRATION_VERSION,
        "capture_performed": False,
        "frozen_capture_reused": True,
        "capture_directory": capture.as_posix(),
        "render_manifest": frozen_evidence.manifest_path.as_posix(),
        "render_manifest_sha256": frozen_evidence.manifest_sha256,
        "ordered_evidence": [
            {
                "id": item.id,
                "role": item.role,
                "sha256": item.sha256,
            }
            for item in frozen_evidence.ordered
        ],
        "canonical_evaluator_invocations": 1,
        "canonical_incomplete_exception": canonical_incomplete_exception,
        "canonical_report": (destination / "evaluation_report.json").as_posix(),
        "canonical_report_sha256": _sha256(
            canonical_report_path
        ),
        "game_profile_modified": False,
        "collision_evidence_provider": type(renderer).__name__,
    }
    report_path = write_json(
        destination / "counter_strike_evaluation_report.json",
        report,
    )
    return {
        "evaluation_report": report,
        "report_path": report_path.as_posix(),
        "canonical_report": canonical_report,
        "l4_report": l4_report,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
