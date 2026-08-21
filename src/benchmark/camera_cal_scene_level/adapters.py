"""External component wiring for the camera-cal scene evaluator.

The evaluator's case runtime owns policy, prompts, scoring, persistence, and
``run_evaluate``.  This module only constructs the external objects that are
passed into that runtime.  Every concrete class/builder and the two observed
wrapper factories are supplied by the caller so the legacy runner can build
this dependency object from its current module globals on every case.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class AdapterFactories:
    """Factories and wrappers used by :func:`build_adapters`.

    No defaults are provided intentionally: the compatibility façade must
    supply the current runner globals for every ``run_case`` call.  This keeps
    class/builder monkeypatches live instead of freezing import-time aliases.
    """

    model_config: Callable[..., dict[str, Any]]
    build_grouping_model: Callable[..., Any]
    build_openai_compatible_vlm_judge: Callable[..., Any]
    build_openai_compatible_camera_selector: Callable[..., Any]
    BlenderRenderer: Callable[..., Any]
    CameraEvidenceProvider: Callable[..., Any]
    DeterministicLocalCameraSelector: Callable[..., Any]
    CameraViewEvidenceRenderer: Callable[..., Any]
    CameraCandidatePreviewRenderer: Callable[..., Any]
    ObservedEvidenceProvider: Callable[..., Any]
    ObservedRenderer: Callable[..., Any]


@dataclass(frozen=True)
class AdapterBundle:
    """External objects constructed for one case."""

    grouping_model: Any
    raw_judge: Any
    vlm_selector: Any
    renderer: Any
    l1_provider: Any
    l3_provider: Any
    functional_probe_provider: Any
    deterministic_selector: Any
    evidence_renderer: Any
    preview_renderer: Any


def build_adapters(
    *,
    case_id: str,
    paths: Mapping[str, Path],
    case_out: Path,
    output_root: Path,
    route: dict[str, Any],
    renderer_config: dict[str, Any],
    api_tracker: Any,
    collision_geometry: dict[str, Any],
    progress: Any,
    factories: AdapterFactories,
) -> AdapterBundle:
    """Construct the external case components in the frozen runner order.

    The order and keyword arguments intentionally mirror the original
    ``run_case`` body:

    ``model configs -> grouping observe -> judge build/observe -> selector
    build/observe -> renderer -> L1 provider -> L3 provider -> functional
    probe -> deterministic selector -> final renderer -> preview renderer``.

    ``api_tracker`` is an already-created tracker.  Its ``observe_model``
    method is used exactly where the runner historically used it, including
    the conditional wrapping of judge/selector model objects.
    """

    judge_config = factories.model_config(route, role="judge")
    selector_config = factories.model_config(route, role="camera-selector")

    grouping_model = api_tracker.observe_model(
        factories.build_grouping_model(route),
        role="grouping",
    )

    raw_judge = factories.build_openai_compatible_vlm_judge(judge_config)
    if callable(
        getattr(getattr(raw_judge, "model", None), "chat_messages", None)
    ):
        raw_judge.model = api_tracker.observe_model(
            raw_judge.model,
            role="judge",
        )

    vlm_selector = factories.build_openai_compatible_camera_selector(
        selector_config
    )
    if callable(
        getattr(getattr(vlm_selector, "model", None), "chat_messages", None)
    ):
        vlm_selector.model = api_tracker.observe_model(
            vlm_selector.model,
            role="camera_selector",
        )

    renderer = factories.BlenderRenderer(**renderer_config)

    l1_provider = factories.ObservedEvidenceProvider(
        factories.CameraEvidenceProvider(
            renderer=renderer,
            blend_file=paths["blend"],
            out_dir=case_out / "l1_camera",
            mode="auto",
            selector=None,
            max_views=2,
            max_steps=1,
            candidate_count=6,
            collision_overlay=True,
            collision_contour=True,
            collision_geometry=collision_geometry,
        ),
        phase="l1_initial_evidence",
        case_id=case_id,
        progress=progress,
    )

    l3_provider = factories.ObservedEvidenceProvider(
        factories.CameraEvidenceProvider(
            renderer=renderer,
            blend_file=paths["blend"],
            out_dir=case_out / "l3_initial_camera",
            mode="visibility_ranked",
            selector=None,
            max_views=2,
            max_steps=1,
            candidate_count=6,
            collision_overlay=False,
            collision_contour=False,
            collision_geometry=collision_geometry,
            active_repair=False,
        ),
        phase="l3_initial_evidence",
        case_id=case_id,
        progress=progress,
    )

    functional_probe_provider = factories.ObservedEvidenceProvider(
        factories.CameraEvidenceProvider(
            renderer=renderer,
            blend_file=paths["blend"],
            out_dir=case_out / "l3_functional_probes",
            mode="query_cov",
            selector=vlm_selector,
            max_views=1,
            max_steps=0,
            candidate_count=6,
            collision_overlay=False,
            collision_contour=False,
            collision_geometry=collision_geometry,
            active_repair=False,
            usable_surface_cache_dir=(
                output_root / "_usable_surface_cache"
            ),
        ),
        phase="l3_functional_probe",
        case_id=case_id,
        progress=progress,
    )

    deterministic_selector = factories.DeterministicLocalCameraSelector(
        candidate_policy=l3_provider.candidate_policy
    )

    evidence_renderer = factories.ObservedRenderer(
        factories.CameraViewEvidenceRenderer(
            renderer=renderer,
            blend_file=paths["blend"],
            out_dir=case_out / "repair_camera",
        ),
        phase="final_evidence",
        case_id=case_id,
        progress=progress,
    )

    preview_renderer = factories.ObservedRenderer(
        factories.CameraCandidatePreviewRenderer(
            renderer=renderer,
            blend_file=paths["blend"],
            out_dir=case_out / "repair_camera",
        ),
        phase="candidate_preview",
        case_id=case_id,
        progress=progress,
    )

    return AdapterBundle(
        grouping_model=grouping_model,
        raw_judge=raw_judge,
        vlm_selector=vlm_selector,
        renderer=renderer,
        l1_provider=l1_provider,
        l3_provider=l3_provider,
        functional_probe_provider=functional_probe_provider,
        deterministic_selector=deterministic_selector,
        evidence_renderer=evidence_renderer,
        preview_renderer=preview_renderer,
    )


def build_runtime_adapters(
    *,
    case_id: str,
    paths: Mapping[str, Path],
    case_out: Path,
    output_root: Path,
    route: dict[str, Any],
    renderer_config: dict[str, Any],
    api_tracker: Any,
    collision_geometry: dict[str, Any],
    progress: Any,
    factories: AdapterFactories,
) -> AdapterBundle:
    """Compatibility-named entry point for case-runtime integration."""

    return build_adapters(
        case_id=case_id,
        paths=paths,
        case_out=case_out,
        output_root=output_root,
        route=route,
        renderer_config=renderer_config,
        api_tracker=api_tracker,
        collision_geometry=collision_geometry,
        progress=progress,
        factories=factories,
    )


__all__ = [
    "AdapterBundle",
    "AdapterFactories",
    "build_adapters",
    "build_runtime_adapters",
]
