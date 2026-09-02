"""Resilient model/scene/room coordinator for the additive nonrect route.

The coordinator owns only sanitized state. Generation artifacts remain
read-only, materialization and evaluation attempts are immutable, and final
scene aggregation is routed back through ``run_evaluate`` with the explicit
``non_rectangular_multi_room`` mode.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import threading
from typing import Any, Mapping, Protocol

from benchmark.api.evaluation import run_evaluate
from benchmark.materialization.catalog import FrozenCatalog, sha256_file, sha256_json
from benchmark.models.openai_compatible_model import (
    EndpointConfigurationError,
    EndpointConnectionError,
    EndpointHTTPError,
    EndpointMalformedResponseError,
    MissingAPIKeyError,
)
from benchmark.non_rectangular.blender_materialization import (
    BlenderNonRectangularRoomMaterializer,
)
from benchmark.non_rectangular.camera import NonRectangularCameraEvidenceExhausted
from benchmark.non_rectangular.contracts import (
    NON_RECTANGULAR_EVALUATION_MODE,
    NonRectangularContractError,
)
from benchmark.non_rectangular.evaluator import (
    NonRectangularRoomMetricIncomplete,
)
from benchmark.non_rectangular.geometry import POLYGON_ROOM_METADATA_KEY
from benchmark.non_rectangular.materialization import (
    NonRectangularMaterializationContractError,
    NonRectangularMaterializationInfrastructureError,
    RoomMaterializationBackend,
    RoomMaterializationResult,
    archive_incomplete_materialization_building,
    materialize_nonrect_room,
    verify_completed_nonrect_materialization,
)
from benchmark.non_rectangular.preflight import (
    NonRectangularEvaluationInput,
    NonRectangularPreflightError,
    NonRectangularPreflightResult,
    prepare_non_rectangular_evaluation,
)
from benchmark.non_rectangular.projection import project_room_unit_to_canonical_scene
from benchmark.non_rectangular.room_layout import RoomLayoutValidationError
from benchmark.non_rectangular.room_unit import (
    RoomEvaluationUnit,
    build_room_evaluation_units,
)
from benchmark.non_rectangular.workflow import (
    L1_METRICS,
    ROOM_METRIC_EXECUTION_ORDER,
    ROOM_REPORT_SCHEMA_VERSION,
    RoomEvaluator,
    RoomEvaluatorReportError,
    validate_complete_room_report,
)
from benchmark.rendering.blender import BlenderRenderError
from benchmark.visual_judge.interfaces.evidence import EvidenceRenderFailure


RUN_MANIFEST_VERSION = "non_rectangular_resilient_run_manifest_v1"
EVENT_VERSION = "non_rectangular_resilient_event_v1"
ROOM_SUMMARY_VERSION = "non_rectangular_resilient_room_summary_v1"
SCENE_SUMMARY_VERSION = "non_rectangular_resilient_scene_summary_v1"
MODEL_SUMMARY_VERSION = "non_rectangular_resilient_model_summary_v1"
TERMINAL_MANIFEST_VERSION = "non_rectangular_resilient_terminal_manifest_v1"
PROVIDER_TOTALS_VERSION = "non_rectangular_resilient_provider_totals_v1"
COORDINATOR_REVISION = "non_rectangular_resilient_coordinator_v2"
RETRYABLE_HTTP_STATUSES = frozenset({429, 500, 502, 503, 504})
_HTTP_STATUS_PATTERN = re.compile(r"\bHTTP\s+(\d{3})\b", re.IGNORECASE)
REQUIRED_SOURCE_ARTIFACTS = {
    "room_layout": Path("room_layout.json"),
    "room_program": Path("room_program.json"),
    "object_plan": Path("stage_a/object_plan.json"),
    "asset_selection": Path("retrieval/asset_selection.json"),
    "generated_scene": Path("generated_scene.json"),
}
OPTIONAL_SOURCE_ARTIFACTS = {
    "compiled_architecture": Path("compiled_architecture.json"),
    "generation_preflight": Path("evaluation_preflight.json"),
}


class ResilientCampaignError(RuntimeError):
    """Campaign identity, locking, or terminal state is invalid."""


@dataclass(frozen=True, slots=True)
class FailureClassification:
    stage: str
    category: str
    error_type: str
    retryable: bool
    metric_id: str | None = None
    source_status: str | None = None
    event_keys: tuple[str, ...] = ()
    fallback_rejection_reasons: tuple[str, ...] = ()

    def public_dict(self) -> dict[str, Any]:
        result = {
            "stage": self.stage,
            "category": self.category,
            "error_type": self.error_type,
            "retryable": self.retryable,
        }
        if self.metric_id is not None:
            result["metric_id"] = self.metric_id
        if self.source_status is not None:
            result["source_status"] = self.source_status
        if self.event_keys:
            result["event_keys"] = list(self.event_keys)
        if self.fallback_rejection_reasons:
            result["fallback_rejection_reasons"] = list(
                self.fallback_rejection_reasons
            )
        return result


@dataclass(frozen=True, slots=True)
class GenerationBundle:
    model: str
    scene_id: str
    root: Path
    artifacts: dict[str, Path]
    file_sha256: dict[str, str]
    values: dict[str, dict[str, Any]]
    evaluation_input: NonRectangularEvaluationInput
    preflight: NonRectangularPreflightResult
    units: tuple[RoomEvaluationUnit, ...]
    selected_asset_ids: tuple[str, ...]

    def source_identity(self, *, room_id: str) -> dict[str, Any]:
        return {
            "schema_version": "non_rectangular_generation_bundle_identity_v1",
            "model": self.model,
            "layout_id": self.preflight.layout_id,
            "scene_id": self.scene_id,
            "room_id": room_id,
            "generation_root_read_only": True,
            "artifacts": {
                name: {
                    "path": str(path),
                    "sha256": self.file_sha256[name],
                }
                for name, path in self.artifacts.items()
            },
        }


@dataclass(frozen=True, slots=True)
class RejectedGenerationScene:
    model: str
    scene_id: str
    root: Path
    status: str
    terminal_failure: bool
    failure: FailureClassification
    artifact_sha256: dict[str, str]
    missing_artifacts: tuple[str, ...]

    def public_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "scene_id": self.scene_id,
            "generation_root": str(self.root),
            "status": self.status,
            "terminal_failure": self.terminal_failure,
            "failure": self.failure.public_dict(),
            "artifact_sha256": dict(self.artifact_sha256),
            "missing_artifacts": list(self.missing_artifacts),
        }


@dataclass(frozen=True, slots=True)
class RoomRuntimeContext:
    model: str
    scene_id: str
    unit: RoomEvaluationUnit
    materialization: RoomMaterializationResult
    attempt_root: Path


class RoomEvaluatorFactory(Protocol):
    def build(self, context: RoomRuntimeContext) -> RoomEvaluator: ...

    def identity(self) -> Mapping[str, Any]: ...

    def usage_totals(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class ResilientCampaignConfig:
    model_roots: tuple[tuple[str, Path], ...]
    output_root: Path
    asset_csv: Path
    asset_root: Path
    catalog_snapshot_id: str
    blender_bin: Path
    max_workers: int = 3
    max_materialization_attempts: int = 3
    max_room_attempts: int = 3
    api_max_retries: int = 5
    materialization_timeout_seconds: int = 900
    resume: bool = False
    recover_interrupted: bool = False

    @classmethod
    def create(
        cls,
        *,
        model_roots: Mapping[str, str | Path],
        output_root: str | Path,
        asset_csv: str | Path,
        asset_root: str | Path,
        catalog_snapshot_id: str,
        blender_bin: str | Path,
        max_workers: int = 3,
        max_materialization_attempts: int = 3,
        max_room_attempts: int = 3,
        api_max_retries: int = 5,
        materialization_timeout_seconds: int = 900,
        resume: bool = False,
        recover_interrupted: bool = False,
    ) -> "ResilientCampaignConfig":
        if not model_roots:
            raise ValueError("at least one model root is required")
        models: list[tuple[str, Path]] = []
        for model, raw_root in model_roots.items():
            name = str(model).strip()
            root = Path(raw_root).expanduser().resolve()
            if not name or root.is_symlink() or not root.is_dir():
                raise ValueError("model roots must be named real directories")
            models.append((name, root))
        for name, value in (
            ("max_workers", max_workers),
            ("max_materialization_attempts", max_materialization_attempts),
            ("max_room_attempts", max_room_attempts),
            ("materialization_timeout_seconds", materialization_timeout_seconds),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if api_max_retries != 5:
            raise ValueError("non-rectangular API logical calls require exactly five retries")
        snapshot = str(catalog_snapshot_id).strip()
        if not snapshot:
            raise ValueError("catalog_snapshot_id must be non-empty")
        return cls(
            model_roots=tuple(models),
            output_root=Path(output_root).expanduser().resolve(),
            asset_csv=Path(asset_csv).expanduser().resolve(),
            asset_root=Path(asset_root).expanduser().resolve(),
            catalog_snapshot_id=snapshot,
            blender_bin=Path(blender_bin).expanduser().resolve(),
            max_workers=max_workers,
            max_materialization_attempts=max_materialization_attempts,
            max_room_attempts=max_room_attempts,
            api_max_retries=api_max_retries,
            materialization_timeout_seconds=materialization_timeout_seconds,
            resume=bool(resume),
            recover_interrupted=bool(recover_interrupted),
        )


@dataclass(frozen=True, slots=True)
class ResilientCampaignResult:
    output_root: Path
    status: str
    model_count: int
    scene_count: int
    room_count: int
    complete_room_count: int
    terminal_manifest_path: Path

    def public_dict(self) -> dict[str, Any]:
        return {
            "output_root": str(self.output_root),
            "status": self.status,
            "model_count": self.model_count,
            "scene_count": self.scene_count,
            "room_count": self.room_count,
            "complete_room_count": self.complete_room_count,
            "terminal_manifest_path": str(self.terminal_manifest_path),
        }


class NoAPIMockMaterializer:
    """Deterministic no-Blender backend used only by workflow tests/dry runs."""

    def materialize(
        self,
        *,
        plan_path: Path,
        blend_path: Path,
        inspection_path: Path,
        blender_bin: Path,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        del blender_bin, timeout_seconds
        plan = _read_json(plan_path)
        blend_path.write_bytes(
            b"NONRECT_MOCK_BLEND\n" + sha256_file(plan_path).encode("ascii") + b"\n"
        )
        report = {
            "backend": "non_rectangular_no_api_mock_materializer_v1",
            "status": "passed",
            "render_invocation_count": 0,
            "room_id": plan["request"]["room_id"],
            "global_coordinates_preserved": True,
            "polygon_floor": True,
            "ordered_wall_segments": True,
            "instance_count": len(plan["instances"]),
        }
        _write_json_atomic(inspection_path, report)
        return report


class NoAPIMockEvaluatorFactory:
    """Whole-workflow room evaluator with no model or camera service calls."""

    def build(self, context: RoomRuntimeContext) -> RoomEvaluator:
        return _NoAPIMockRoomEvaluator(context)

    def identity(self) -> Mapping[str, Any]:
        return {
            "provider": "mock",
            "model": "no-api",
            "config_sha256": hashlib.sha256(b"nonrect-no-api-v1").hexdigest(),
        }

    def usage_totals(self) -> Mapping[str, Any]:
        return {
            "logical_calls": 0,
            "failed_calls": 0,
            "usage_missing_calls": 0,
            "tokens": 0,
        }


class _NoAPIMockRoomEvaluator:
    def __init__(self, context: RoomRuntimeContext) -> None:
        self.context = context

    def evaluate(self, unit: RoomEvaluationUnit) -> Mapping[str, Any]:
        if unit.room_id != self.context.unit.room_id:
            raise ValueError("mock evaluator room identity mismatch")
        scene = _read_json(self.context.materialization.canonical_scene_path)
        metadata = scene.get("metadata") if isinstance(scene.get("metadata"), dict) else {}
        polygon = metadata.get(POLYGON_ROOM_METADATA_KEY)
        if not isinstance(polygon, dict):
            raise ValueError("mock camera received no polygon metadata")
        wall_ids = [
            str(item.get("wall_id") or "")
            for item in polygon.get("wall_segments") or []
        ]
        if wall_ids != [str(item["wall_id"]) for item in unit.wall_segments]:
            raise ValueError("mock camera wall order mismatch")
        mock_camera_path = self.context.attempt_root / "mock_camera/evidence_manifest.json"
        _write_json_atomic(
            mock_camera_path,
            {
                "schema_version": "non_rectangular_mock_camera_evidence_v1",
                "status": "complete",
                "room_id": unit.room_id,
                "source_blend_sha256": self.context.materialization.output_sha256[
                    "room_blend"
                ],
                "floor_polygon_xy": polygon.get("floor_polygon_xy"),
                "wall_ids": wall_ids,
                "global_coordinates_preserved": True,
                "adjacent_room_objects_included": False,
            },
        )
        metrics: dict[str, dict[str, Any]] = {}
        for metric in ROOM_METRIC_EXECUTION_ORDER:
            item: dict[str, Any] = {
                "metric": metric,
                "status": "complete",
                "score": 1.0,
                "evaluated_object_count": unit.generated_object_count,
                "raw_report": {
                    "mock": True,
                    "metric": metric,
                    "room_id": unit.room_id,
                    "materialized_blend_sha256": (
                        self.context.materialization.output_sha256["room_blend"]
                    ),
                    "mock_camera_evidence_sha256": sha256_file(mock_camera_path),
                },
            }
            if metric in L1_METRICS:
                item["invalid_count"] = 0
            metrics[metric] = item
        return {
            "schema_version": ROOM_REPORT_SCHEMA_VERSION,
            "room_id": unit.room_id,
            "status": "complete",
            "metrics": metrics,
        }


def discover_generation_bundles(
    model_roots: tuple[tuple[str, Path], ...],
) -> tuple[GenerationBundle, ...]:
    """Load only structurally complete scenes and run strict preflight."""

    bundles, _ = _discover_generation_inventory(model_roots)
    if not bundles:
        raise ResilientCampaignError("no structurally complete generation scenes found")
    return bundles


def _discover_generation_inventory(
    model_roots: tuple[tuple[str, Path], ...],
) -> tuple[tuple[GenerationBundle, ...], tuple[RejectedGenerationScene, ...]]:
    """Return ready scenes plus sanitized incomplete/semantic rejections."""

    bundles: list[GenerationBundle] = []
    rejected: list[RejectedGenerationScene] = []
    for model, root in model_roots:
        scene_dirs = _ordered_scene_dirs(root)
        if not scene_dirs:
            raise ResilientCampaignError(f"model root has no scene directories: {root}")
        for scene_root in scene_dirs:
            artifacts = {
                name: scene_root / relative
                for name, relative in REQUIRED_SOURCE_ARTIFACTS.items()
            }
            missing = [name for name, path in artifacts.items() if not path.is_file()]
            if missing:
                rejected.append(
                    RejectedGenerationScene(
                        model=model,
                        scene_id=scene_root.name,
                        root=scene_root.resolve(),
                        status="excluded_incomplete_generation",
                        terminal_failure=False,
                        failure=FailureClassification(
                            "generation_preflight",
                            "incomplete_generation_bundle",
                            "MissingGenerationArtifact",
                            False,
                        ),
                        artifact_sha256={
                            name: sha256_file(path)
                            for name, path in artifacts.items()
                            if path.is_file() and not path.is_symlink()
                        },
                        missing_artifacts=tuple(missing),
                    )
                )
                continue
            for name, relative in OPTIONAL_SOURCE_ARTIFACTS.items():
                path = scene_root / relative
                if path.is_file():
                    artifacts[name] = path
            for path in artifacts.values():
                if path.is_symlink() or not path.is_file():
                    raise ResilientCampaignError(
                        f"generation artifact must be a regular file: {path}"
                    )
            file_hashes = {name: sha256_file(path) for name, path in artifacts.items()}
            try:
                values = {name: _read_json(path) for name, path in artifacts.items()}
                evaluation_input = NonRectangularEvaluationInput.from_artifacts(
                    room_layout=values["room_layout"],
                    room_program=values["room_program"],
                    object_plan=values["object_plan"],
                    generated_scene=values["generated_scene"],
                )
                preflight = prepare_non_rectangular_evaluation(evaluation_input)
                if not preflight.should_run_room_evaluation:
                    raise NonRectangularPreflightError(
                        str(preflight.failure_reason or "preflight_failed")
                    )
                if "generation_preflight" in values:
                    _verify_generation_preflight(values["generation_preflight"], preflight)
                units = build_room_evaluation_units(preflight)
                selected_assets = _selected_asset_ids(values["asset_selection"])
            except Exception as exc:
                failure = classify_failure(exc, stage="generation_preflight")
                rejected.append(
                    RejectedGenerationScene(
                        model=model,
                        scene_id=scene_root.name,
                        root=scene_root.resolve(),
                        status="nonretryable_generation_preflight_failure",
                        terminal_failure=True,
                        failure=FailureClassification(
                            failure.stage,
                            failure.category,
                            failure.error_type,
                            False,
                        ),
                        artifact_sha256=file_hashes,
                        missing_artifacts=(),
                    )
                )
                continue
            bundles.append(
                GenerationBundle(
                    model=model,
                    scene_id=scene_root.name,
                    root=scene_root.resolve(),
                    artifacts={name: path.resolve() for name, path in artifacts.items()},
                    file_sha256=file_hashes,
                    values=values,
                    evaluation_input=evaluation_input,
                    preflight=preflight,
                    units=units,
                    selected_asset_ids=selected_assets,
                )
            )
    return tuple(bundles), tuple(rejected)


def run_resilient_nonrect_campaign(
    config: ResilientCampaignConfig,
    *,
    evaluator_factory: RoomEvaluatorFactory,
    materializer_backend: RoomMaterializationBackend | None = None,
) -> ResilientCampaignResult:
    """Run or resume the complete nonrect materialize/evaluate state machine."""

    if not isinstance(config, ResilientCampaignConfig):
        raise TypeError("config must be ResilientCampaignConfig")
    backend = materializer_backend or BlenderNonRectangularRoomMaterializer()
    runtime_identity = _runtime_identity(evaluator_factory)
    _validate_static_paths(config)
    bundles, rejected = _discover_generation_inventory(config.model_roots)
    if not bundles and not any(item.terminal_failure for item in rejected):
        raise ResilientCampaignError("no structurally complete generation scenes found")
    _reject_output_source_overlap(
        config.output_root,
        bundles,
        rejected=rejected,
    )
    identity_payload = _campaign_identity(
        config,
        bundles,
        runtime_identity,
        rejected=rejected,
    )
    identity_sha256 = sha256_json(identity_payload)
    output = config.output_root
    output.mkdir(parents=True, exist_ok=True)
    with _CampaignLock(output / ".runner.lock"):
        run_manifest_path = output / "run_manifest.json"
        if run_manifest_path.exists():
            if not config.resume:
                raise ResilientCampaignError(
                    "output root exists; pass resume only for this exact campaign"
                )
            existing = _read_json(run_manifest_path)
            if existing.get("identity_sha256") != identity_sha256:
                raise ResilientCampaignError(
                    "resume refused because input/config identity drifted"
                )
        else:
            extras = [item for item in output.iterdir() if item.name != ".runner.lock"]
            if extras:
                raise ResilientCampaignError("fresh output root is not empty")
            _write_json_atomic(
                run_manifest_path,
                {
                    "schema_version": RUN_MANIFEST_VERSION,
                    "coordinator_revision": COORDINATOR_REVISION,
                    "evaluation_mode": NON_RECTANGULAR_EVALUATION_MODE,
                    "identity_sha256": identity_sha256,
                    "identity": identity_payload,
                    "output_root": str(output),
                    "events_path": str(output / "events.jsonl"),
                    "current_state_path": str(output / "current_state.json"),
                    "provider_model_totals_path": str(
                        output / "provider_model_totals.json"
                    ),
                    "terminal_manifest_path": str(output / "terminal_manifest.json"),
                },
            )
        coordinator = _Coordinator(
            config=config,
            bundles=bundles,
            rejected=rejected,
            evaluator_factory=evaluator_factory,
            materializer_backend=backend,
            output_root=output,
        )
        return coordinator.run()


class _Coordinator:
    def __init__(
        self,
        *,
        config: ResilientCampaignConfig,
        bundles: tuple[GenerationBundle, ...],
        rejected: tuple[RejectedGenerationScene, ...],
        evaluator_factory: RoomEvaluatorFactory,
        materializer_backend: RoomMaterializationBackend,
        output_root: Path,
    ) -> None:
        self.config = config
        self.bundles = bundles
        self.rejected = rejected
        self.evaluator_factory = evaluator_factory
        self.materializer_backend = materializer_backend
        self.output_root = output_root
        self.events_path = output_root / "events.jsonl"
        self._event_lock = threading.Lock()

    def run(self) -> ResilientCampaignResult:
        self._state("generation_preflight", status="running")
        for bundle in self.bundles:
            self._write_scene_preflight(bundle)
        for item in self.rejected:
            self._write_rejected_scene(item)
        self._event("campaign_started", stage="generation_preflight")

        self._state("initial_pass", status="running")
        for bundle in self.bundles:
            pending = [
                unit
                for unit in bundle.units
                if self._room_selected_report(bundle, unit) is None
                and not self._has_terminal_attempt(bundle, unit)
            ]
            self._run_units(bundle, pending, retry_round=0)
            self._write_scene_summary(bundle)
            self._write_model_summaries()

        retry_round = 0
        while True:
            pending = self._retryable_units()
            if not pending:
                break
            retry_round += 1
            self._state("global_retry", status="running", retry_round=retry_round)
            any_run = False
            for bundle in self.bundles:
                units = [unit for item, unit in pending if item is bundle]
                if units:
                    any_run = True
                    self._run_units(bundle, units, retry_round=retry_round)
                    self._write_scene_summary(bundle)
            self._write_model_summaries()
            if not any_run:
                break

        self._state("scene_aggregation", status="running", retry_round=retry_round)
        for bundle in self.bundles:
            self._aggregate_scene(bundle)
            self._write_scene_summary(bundle)
        self._write_model_summaries()
        provider_totals = _usage_totals(self.evaluator_factory)
        generation_model_totals = {}
        for model, _ in self.config.model_roots:
            summary_path = self.output_root / "models" / model / "summary.json"
            summary = _read_json(summary_path)
            generation_model_totals[model] = {
                "scene_count": int(summary["scene_count"]),
                "room_count": int(summary["room_count"]),
                "complete_room_count": int(summary["complete_room_count"]),
                "failed_room_count": int(summary["failed_room_count"]),
                "nonretryable_scene_failure_count": int(
                    summary["nonretryable_scene_failure_count"]
                ),
            }
        _write_json_atomic(
            self.output_root / "provider_model_totals.json",
            {
                "schema_version": PROVIDER_TOTALS_VERSION,
                "runtime": _runtime_identity(self.evaluator_factory),
                "totals": provider_totals,
                "generation_models": generation_model_totals,
            },
        )
        room_count = sum(len(bundle.units) for bundle in self.bundles)
        complete = sum(
            self._room_selected_report(bundle, unit) is not None
            for bundle in self.bundles
            for unit in bundle.units
        )
        semantic_scene_failures = sum(item.terminal_failure for item in self.rejected)
        status = (
            "complete"
            if complete == room_count and semantic_scene_failures == 0
            else "failed"
        )
        terminal_path = self.output_root / "terminal_manifest.json"
        terminal = {
            "schema_version": TERMINAL_MANIFEST_VERSION,
            "status": status,
            "evaluation_mode": NON_RECTANGULAR_EVALUATION_MODE,
            "model_count": len({bundle.model for bundle in self.bundles}),
            "scene_count": len(self.bundles) + semantic_scene_failures,
            "excluded_incomplete_scene_count": sum(
                not item.terminal_failure for item in self.rejected
            ),
            "nonretryable_scene_failure_count": semantic_scene_failures,
            "room_count": room_count,
            "complete_room_count": complete,
            "failed_room_count": room_count - complete,
            "metric_execution_order": list(ROOM_METRIC_EXECUTION_ORDER),
            "output_paths": {
                "root": str(self.output_root),
                "run_manifest": str(self.output_root / "run_manifest.json"),
                "events": str(self.events_path),
                "current_state": str(self.output_root / "current_state.json"),
                "provider_model_totals": str(
                    self.output_root / "provider_model_totals.json"
                ),
                "terminal_manifest": str(terminal_path),
            },
        }
        _write_json_atomic(terminal_path, terminal)
        self._state("terminal", status=status, retry_round=retry_round)
        self._event("campaign_terminal", stage="terminal", status=status)
        return ResilientCampaignResult(
            output_root=self.output_root,
            status=status,
            model_count=terminal["model_count"],
            scene_count=terminal["scene_count"],
            room_count=room_count,
            complete_room_count=complete,
            terminal_manifest_path=terminal_path,
        )

    def _run_units(
        self,
        bundle: GenerationBundle,
        units: list[RoomEvaluationUnit],
        *,
        retry_round: int,
    ) -> None:
        if not units:
            return
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(
                    self._run_room_once,
                    bundle,
                    unit,
                    retry_round,
                ): unit
                for unit in units
            }
            for future in as_completed(futures):
                unit = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    # _run_room_once persists a classified terminal attempt.
                    # An unclassified coordinator exception is also fail-closed.
                    self._event(
                        "room_coordinator_failure",
                        stage="coordinator",
                        model=bundle.model,
                        scene_id=bundle.scene_id,
                        room_id=unit.room_id,
                        error_type=type(exc).__name__,
                    )
                self._write_room_summary(bundle, unit)

    def _run_room_once(
        self,
        bundle: GenerationBundle,
        unit: RoomEvaluationUnit,
        retry_round: int,
    ) -> None:
        del retry_round
        self._verify_bundle_sources(bundle)
        room_root = self._room_root(bundle, unit)
        room_root.mkdir(parents=True, exist_ok=True)
        materialization = self._selected_materialization(bundle, unit)
        if materialization is None:
            materialization = self._materialize_once(bundle, unit)
        if materialization is None:
            return
        if self._room_selected_report(bundle, unit) is not None:
            return
        self._evaluate_once(bundle, unit, materialization)

    def _materialize_once(
        self,
        bundle: GenerationBundle,
        unit: RoomEvaluationUnit,
    ) -> RoomMaterializationResult | None:
        room_root = self._room_root(bundle, unit)
        attempts_root = room_root / "materialization_attempts"
        attempts_root.mkdir(parents=True, exist_ok=True)
        attempt_number = _next_attempt_number(attempts_root)
        if attempt_number > self.config.max_materialization_attempts:
            return None
        attempt_root = attempts_root / f"attempt_{attempt_number:03d}"
        destination = attempt_root / "materialization"
        if self.config.recover_interrupted:
            archive_incomplete_materialization_building(
                destination,
                recovery_root=room_root / "recovered_partials",
            )
        attempt_root.mkdir(parents=True, exist_ok=False)
        self._event(
            "materialization_started",
            stage="room_materialization",
            model=bundle.model,
            scene_id=bundle.scene_id,
            room_id=unit.room_id,
            attempt=attempt_number,
        )
        try:
            catalog = FrozenCatalog(
                asset_csv=self.config.asset_csv,
                asset_root=self.config.asset_root,
                allowed_asset_ids=bundle.selected_asset_ids,
                snapshot_id=self.config.catalog_snapshot_id,
            )
            result = materialize_nonrect_room(
                unit,
                destination=destination,
                room_layout=bundle.values["room_layout"],
                asset_selection=bundle.values["asset_selection"],
                source_identity=bundle.source_identity(room_id=unit.room_id),
                catalog=catalog,
                blender_bin=self.config.blender_bin,
                backend=self.materializer_backend,
                timeout_seconds=self.config.materialization_timeout_seconds,
                compiled_architecture=bundle.values.get("compiled_architecture"),
            )
        except Exception as exc:
            failure = classify_failure(exc, stage="materialization")
            _write_json_atomic(
                attempt_root / "attempt_manifest.json",
                {
                    "schema_version": "non_rectangular_materialization_attempt_v1",
                    "status": "failed",
                    "attempt": attempt_number,
                    "failure": failure.public_dict(),
                    "output_root": str(attempt_root),
                },
            )
            self._event(
                "materialization_failed",
                stage="room_materialization",
                model=bundle.model,
                scene_id=bundle.scene_id,
                room_id=unit.room_id,
                attempt=attempt_number,
                category=failure.category,
                error_type=failure.error_type,
                retryable=failure.retryable,
            )
            return None
        _write_json_atomic(
            attempt_root / "attempt_manifest.json",
            {
                "schema_version": "non_rectangular_materialization_attempt_v1",
                "status": "complete",
                "attempt": attempt_number,
                "identity_sha256": result.identity_sha256,
                "output_root": str(attempt_root),
            },
        )
        _write_json_atomic(
            room_root / "materialization_selected.json",
            {
                "schema_version": "non_rectangular_materialization_selection_v1",
                "status": "complete",
                "attempt": attempt_number,
                "materialization_root": str(result.root),
                "identity_sha256": result.identity_sha256,
            },
        )
        self._event(
            "materialization_completed",
            stage="materialization_inspection",
            model=bundle.model,
            scene_id=bundle.scene_id,
            room_id=unit.room_id,
            attempt=attempt_number,
        )
        return result

    def _evaluate_once(
        self,
        bundle: GenerationBundle,
        unit: RoomEvaluationUnit,
        materialization: RoomMaterializationResult,
    ) -> None:
        room_root = self._room_root(bundle, unit)
        attempts_root = room_root / "evaluation_attempts"
        attempts_root.mkdir(parents=True, exist_ok=True)
        attempt_number = _next_attempt_number(attempts_root)
        if attempt_number > self.config.max_room_attempts:
            return
        attempt_root = attempts_root / f"attempt_{attempt_number:03d}"
        attempt_root.mkdir(parents=True, exist_ok=False)
        expected_scene_hash = sha256_json(project_room_unit_to_canonical_scene(unit))
        if sha256_json(_read_json(materialization.canonical_scene_path)) != expected_scene_hash:
            failure = FailureClassification(
                stage="evaluation",
                category="materialized_scene_identity_drift",
                error_type="NonRectangularMaterializationContractError",
                retryable=False,
            )
            _write_json_atomic(
                attempt_root / "attempt_manifest.json",
                {
                    "schema_version": "non_rectangular_evaluation_attempt_v1",
                    "status": "failed",
                    "attempt": attempt_number,
                    "failure": failure.public_dict(),
                },
            )
            return
        context = RoomRuntimeContext(
            model=bundle.model,
            scene_id=bundle.scene_id,
            unit=unit,
            materialization=materialization,
            attempt_root=attempt_root,
        )
        self._event(
            "evaluation_started",
            stage="metric_evaluation",
            model=bundle.model,
            scene_id=bundle.scene_id,
            room_id=unit.room_id,
            attempt=attempt_number,
        )
        try:
            evaluator = self.evaluator_factory.build(context)
            raw = evaluator.evaluate(unit)
            report = validate_complete_room_report(raw, unit=unit)
        except Exception as exc:
            failure = classify_failure(exc, stage="evaluation")
            _write_json_atomic(
                attempt_root / "attempt_manifest.json",
                {
                    "schema_version": "non_rectangular_evaluation_attempt_v1",
                    "status": "failed",
                    "attempt": attempt_number,
                    "failure": failure.public_dict(),
                    "materialization_identity_sha256": materialization.identity_sha256,
                    "output_root": str(attempt_root),
                },
            )
            self._event(
                "evaluation_failed",
                stage="metric_evaluation",
                model=bundle.model,
                scene_id=bundle.scene_id,
                room_id=unit.room_id,
                attempt=attempt_number,
                category=failure.category,
                error_type=failure.error_type,
                retryable=failure.retryable,
                metric_id=failure.metric_id,
                source_status=failure.source_status,
                event_keys=list(failure.event_keys),
                fallback_rejection_reasons=list(
                    failure.fallback_rejection_reasons
                ),
            )
            return
        report_path = attempt_root / "room_evaluation_report.json"
        _write_json_atomic(report_path, report)
        report_sha256 = sha256_file(report_path)
        _write_json_atomic(
            attempt_root / "attempt_manifest.json",
            {
                "schema_version": "non_rectangular_evaluation_attempt_v1",
                "status": "complete",
                "attempt": attempt_number,
                "room_id": unit.room_id,
                "metric_execution_order": list(ROOM_METRIC_EXECUTION_ORDER),
                "room_report_path": str(report_path),
                "room_report_sha256": report_sha256,
                "materialization_identity_sha256": materialization.identity_sha256,
                "output_root": str(attempt_root),
            },
        )
        _write_json_atomic(
            room_root / "room_report_selected.json",
            {
                "schema_version": "non_rectangular_room_report_selection_v1",
                "status": "complete",
                "attempt": attempt_number,
                "room_report_path": str(report_path),
                "room_report_sha256": report_sha256,
                "materialization_identity_sha256": materialization.identity_sha256,
            },
        )
        self._event(
            "evaluation_completed",
            stage="room_report",
            model=bundle.model,
            scene_id=bundle.scene_id,
            room_id=unit.room_id,
            attempt=attempt_number,
        )

    def _selected_materialization(
        self,
        bundle: GenerationBundle,
        unit: RoomEvaluationUnit,
    ) -> RoomMaterializationResult | None:
        room_root = self._room_root(bundle, unit)
        pointer = room_root / "materialization_selected.json"
        if not pointer.is_file():
            for attempt_root in reversed(
                _attempt_dirs(room_root / "materialization_attempts")
            ):
                manifest_path = attempt_root / "attempt_manifest.json"
                if not manifest_path.is_file():
                    continue
                manifest = _read_json(manifest_path)
                if manifest.get("status") != "complete":
                    continue
                root = attempt_root / "materialization"
                result = verify_completed_nonrect_materialization(
                    root,
                    expected_identity_sha256=str(
                        manifest.get("identity_sha256") or ""
                    ),
                )
                _write_json_atomic(
                    pointer,
                    {
                        "schema_version": (
                            "non_rectangular_materialization_selection_v1"
                        ),
                        "status": "complete",
                        "attempt": int(manifest.get("attempt") or 0),
                        "materialization_root": str(result.root),
                        "identity_sha256": result.identity_sha256,
                        "selection_recovered_from_verified_attempt": True,
                    },
                )
                return result
            return None
        value = _read_json(pointer)
        root = Path(str(value.get("materialization_root") or "")).resolve()
        try:
            return verify_completed_nonrect_materialization(
                root,
                expected_identity_sha256=str(value.get("identity_sha256") or ""),
            )
        except Exception as exc:
            raise ResilientCampaignError(
                f"selected materialization failed verification for {unit.room_id}"
            ) from exc

    def _room_selected_report(
        self,
        bundle: GenerationBundle,
        unit: RoomEvaluationUnit,
    ) -> dict[str, Any] | None:
        room_root = self._room_root(bundle, unit)
        pointer = room_root / "room_report_selected.json"
        if not pointer.is_file():
            for attempt_root in reversed(
                _attempt_dirs(room_root / "evaluation_attempts")
            ):
                manifest_path = attempt_root / "attempt_manifest.json"
                if not manifest_path.is_file():
                    continue
                manifest = _read_json(manifest_path)
                if manifest.get("status") != "complete":
                    continue
                report_path = attempt_root / "room_evaluation_report.json"
                if (
                    not report_path.is_file()
                    or sha256_file(report_path)
                    != manifest.get("room_report_sha256")
                ):
                    raise ResilientCampaignError(
                        f"completed room attempt hash drift for {unit.room_id}"
                    )
                report = validate_complete_room_report(
                    _read_json(report_path),
                    unit=unit,
                )
                _write_json_atomic(
                    pointer,
                    {
                        "schema_version": (
                            "non_rectangular_room_report_selection_v1"
                        ),
                        "status": "complete",
                        "attempt": int(manifest.get("attempt") or 0),
                        "room_report_path": str(report_path),
                        "room_report_sha256": sha256_file(report_path),
                        "materialization_identity_sha256": manifest.get(
                            "materialization_identity_sha256"
                        ),
                        "selection_recovered_from_verified_attempt": True,
                    },
                )
                return report
            return None
        selected = _read_json(pointer)
        report_path = Path(str(selected.get("room_report_path") or "")).resolve()
        if not report_path.is_file() or sha256_file(report_path) != selected.get(
            "room_report_sha256"
        ):
            raise ResilientCampaignError(
                f"selected room report hash drift for {unit.room_id}"
            )
        report = _read_json(report_path)
        return validate_complete_room_report(report, unit=unit)

    def _has_terminal_attempt(
        self,
        bundle: GenerationBundle,
        unit: RoomEvaluationUnit,
    ) -> bool:
        failure = self._latest_failure(bundle, unit)
        return failure is not None and not failure["retryable"]

    def _retryable_units(
        self,
    ) -> list[tuple[GenerationBundle, RoomEvaluationUnit]]:
        result: list[tuple[GenerationBundle, RoomEvaluationUnit]] = []
        for bundle in self.bundles:
            for unit in bundle.units:
                if self._room_selected_report(bundle, unit) is not None:
                    continue
                failure = self._latest_failure(bundle, unit)
                if failure is None or not failure["retryable"]:
                    continue
                stage = failure["stage"]
                attempts_root = self._room_root(bundle, unit) / (
                    "materialization_attempts"
                    if stage == "materialization"
                    else "evaluation_attempts"
                )
                maximum = (
                    self.config.max_materialization_attempts
                    if stage == "materialization"
                    else self.config.max_room_attempts
                )
                if len(_attempt_dirs(attempts_root)) < maximum:
                    result.append((bundle, unit))
        return result

    def _latest_failure(
        self,
        bundle: GenerationBundle,
        unit: RoomEvaluationUnit,
    ) -> dict[str, Any] | None:
        room_root = self._room_root(bundle, unit)
        candidates: list[tuple[int, int, dict[str, Any]]] = []
        for stage_rank, (stage, directory) in enumerate((
            ("materialization", room_root / "materialization_attempts"),
            ("evaluation", room_root / "evaluation_attempts"),
        )):
            for path in _attempt_dirs(directory):
                manifest_path = path / "attempt_manifest.json"
                if not manifest_path.is_file():
                    if not self.config.recover_interrupted:
                        return {
                            "stage": stage,
                            "category": "interrupted_partial_write",
                            "error_type": "InterruptedAttempt",
                            "retryable": False,
                        }
                    continue
                manifest = _read_json(manifest_path)
                if manifest.get("status") != "failed":
                    continue
                failure = manifest.get("failure")
                if isinstance(failure, dict):
                    candidates.append(
                        (int(manifest.get("attempt") or 0), stage_rank, failure)
                    )
        return max(candidates, key=lambda item: (item[0], item[1]))[2] if candidates else None

    def _aggregate_scene(self, bundle: GenerationBundle) -> None:
        reports: dict[str, dict[str, Any]] = {}
        for unit in bundle.units:
            report = self._room_selected_report(bundle, unit)
            if report is None:
                return
            reports[unit.room_id] = report
        scene_root = self._scene_root(bundle)
        final_path = scene_root / "evaluation_report.json"
        temporary = scene_root / ".evaluation_report.tmp.json"
        cached = _CachedRoomEvaluator(reports)
        report = run_evaluate(
            evaluation_mode=NON_RECTANGULAR_EVALUATION_MODE,
            evaluation_input=bundle.evaluation_input,
            out=temporary,
            room_evaluator=cached,
        )
        if cached.call_order != list(bundle.preflight.room_order):
            raise ResilientCampaignError("public nonrect aggregation room order drift")
        if report.get("terminal_status") != "complete":
            raise ResilientCampaignError("public nonrect aggregation did not complete")
        temporary.replace(final_path)

    def _write_scene_preflight(self, bundle: GenerationBundle) -> None:
        scene_root = self._scene_root(bundle)
        scene_root.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(scene_root / "preflight.json", bundle.preflight.public_dict())

    def _write_rejected_scene(self, item: RejectedGenerationScene) -> None:
        scene_root = (
            self.output_root / "models" / item.model / "scenes" / item.scene_id
        )
        _write_json_atomic(
            scene_root / "summary.json",
            {
                "schema_version": SCENE_SUMMARY_VERSION,
                "model": item.model,
                "scene_id": item.scene_id,
                "status": item.status,
                "current_stage": "generation_preflight",
                "terminal_failure": item.terminal_failure,
                "failure": item.failure.public_dict(),
                "missing_artifacts": list(item.missing_artifacts),
                "room_order": [],
                "room_count": 0,
                "complete_room_count": 0,
                "failed_room_count": 0,
                "rooms": [],
                "output_paths": {
                    "scene_root": str(scene_root),
                    "summary": str(scene_root / "summary.json"),
                },
            },
        )

    def _write_room_summary(
        self,
        bundle: GenerationBundle,
        unit: RoomEvaluationUnit,
    ) -> None:
        room_root = self._room_root(bundle, unit)
        selected = self._room_selected_report(bundle, unit)
        materialization_attempts = _attempt_dirs(
            room_root / "materialization_attempts"
        )
        evaluation_attempts = _attempt_dirs(room_root / "evaluation_attempts")
        latest_failure = self._latest_failure(bundle, unit)
        retry_exhausted = False
        if selected is None and latest_failure is not None and latest_failure.get(
            "retryable"
        ):
            if latest_failure.get("stage") == "materialization":
                retry_exhausted = (
                    len(materialization_attempts)
                    >= self.config.max_materialization_attempts
                )
            else:
                retry_exhausted = (
                    len(evaluation_attempts) >= self.config.max_room_attempts
                )
        if selected is not None:
            terminal_status = "complete"
        elif retry_exhausted:
            terminal_status = "failed_retry_exhausted"
        elif latest_failure is not None and not latest_failure.get("retryable"):
            terminal_status = "failed_nonretryable"
        elif latest_failure is not None:
            terminal_status = "retryable_failure_pending"
        else:
            terminal_status = "pending"
        summary = {
            "schema_version": ROOM_SUMMARY_VERSION,
            "model": bundle.model,
            "scene_id": bundle.scene_id,
            "room_id": unit.room_id,
            "room_index": unit.room_index,
            "status": terminal_status,
            "current_stage": "complete" if selected is not None else (
                latest_failure["stage"] if latest_failure else "pending"
            ),
            "materialization_attempt_count": len(materialization_attempts),
            "materialization_retry_count": max(0, len(materialization_attempts) - 1),
            "evaluation_attempt_count": len(evaluation_attempts),
            "evaluation_retry_count": max(0, len(evaluation_attempts) - 1),
            "latest_failure": latest_failure,
            "retry_exhausted": retry_exhausted,
            "metric_execution_order": list(ROOM_METRIC_EXECUTION_ORDER),
            "output_paths": {
                "room_root": str(room_root),
                "summary": str(room_root / "summary.json"),
                "materialization_selection": str(
                    room_root / "materialization_selected.json"
                ),
                "room_report_selection": str(room_root / "room_report_selected.json"),
            },
        }
        _write_json_atomic(room_root / "summary.json", summary)

    def _write_scene_summary(self, bundle: GenerationBundle) -> None:
        scene_root = self._scene_root(bundle)
        rooms = []
        complete = 0
        for unit in bundle.units:
            self._write_room_summary(bundle, unit)
            summary = _read_json(self._room_root(bundle, unit) / "summary.json")
            rooms.append(summary)
            complete += summary["status"] == "complete"
        report_path = scene_root / "evaluation_report.json"
        status = "complete" if complete == len(bundle.units) and report_path.is_file() else (
            "rooms_complete_pending_aggregation"
            if complete == len(bundle.units)
            else "incomplete"
        )
        _write_json_atomic(
            scene_root / "summary.json",
            {
                "schema_version": SCENE_SUMMARY_VERSION,
                "model": bundle.model,
                "scene_id": bundle.scene_id,
                "layout_id": bundle.preflight.layout_id,
                "status": status,
                "current_stage": (
                    "complete" if status == "complete" else "room_execution"
                ),
                "room_order": list(bundle.preflight.room_order),
                "room_count": len(bundle.units),
                "complete_room_count": complete,
                "failed_room_count": len(bundle.units) - complete,
                "rooms": rooms,
                "output_paths": {
                    "scene_root": str(scene_root),
                    "summary": str(scene_root / "summary.json"),
                    "evaluation_report": str(report_path),
                },
            },
        )

    def _write_model_summaries(self) -> None:
        for model, _ in self.config.model_roots:
            bundles = [item for item in self.bundles if item.model == model]
            rejected = [item for item in self.rejected if item.model == model]
            scenes = []
            room_count = 0
            complete = 0
            for bundle in bundles:
                self._write_scene_summary(bundle)
                summary = _read_json(self._scene_root(bundle) / "summary.json")
                scenes.append(summary)
                room_count += summary["room_count"]
                complete += summary["complete_room_count"]
            for item in rejected:
                summary_path = (
                    self.output_root
                    / "models"
                    / model
                    / "scenes"
                    / item.scene_id
                    / "summary.json"
                )
                if not summary_path.is_file():
                    self._write_rejected_scene(item)
                scenes.append(_read_json(summary_path))
            root = self.output_root / "models" / model
            _write_json_atomic(
                root / "summary.json",
                {
                    "schema_version": MODEL_SUMMARY_VERSION,
                    "model": model,
                    "status": (
                        "complete"
                        if room_count == complete
                        and not any(item.terminal_failure for item in rejected)
                        else "incomplete"
                    ),
                    "scene_count": len(scenes),
                    "excluded_incomplete_scene_count": sum(
                        not item.terminal_failure for item in rejected
                    ),
                    "nonretryable_scene_failure_count": sum(
                        item.terminal_failure for item in rejected
                    ),
                    "room_count": room_count,
                    "complete_room_count": complete,
                    "failed_room_count": room_count - complete,
                    "scenes": scenes,
                    "output_root": str(root),
                },
            )

    def _verify_bundle_sources(self, bundle: GenerationBundle) -> None:
        for name, path in bundle.artifacts.items():
            if path.is_symlink() or not path.is_file():
                raise ResilientCampaignError(
                    f"generation input disappeared before room execution: {name}"
                )
            if sha256_file(path) != bundle.file_sha256[name]:
                raise ResilientCampaignError(
                    f"generation input hash drift before room execution: {name}"
                )

    def _scene_root(self, bundle: GenerationBundle) -> Path:
        return self.output_root / "models" / bundle.model / "scenes" / bundle.scene_id

    def _room_root(self, bundle: GenerationBundle, unit: RoomEvaluationUnit) -> Path:
        return self._scene_root(bundle) / "rooms" / unit.room_id

    def _state(self, stage: str, *, status: str, retry_round: int = 0) -> None:
        _write_json_atomic(
            self.output_root / "current_state.json",
            {
                "schema_version": "non_rectangular_resilient_current_state_v1",
                "status": status,
                "current_stage": stage,
                "global_retry_round": retry_round,
            },
        )

    def _event(self, event: str, *, stage: str, **fields: Any) -> None:
        payload = {
            "schema_version": EVENT_VERSION,
            "sequence": 0,
            "event": event,
            "stage": stage,
            **fields,
        }
        with self._event_lock:
            sequence = 1
            if self.events_path.is_file():
                with self.events_path.open("rb") as handle:
                    sequence += sum(1 for _ in handle)
            payload["sequence"] = sequence
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())


class _CachedRoomEvaluator:
    def __init__(self, reports: Mapping[str, Mapping[str, Any]]) -> None:
        self.reports = {name: deepcopy(dict(value)) for name, value in reports.items()}
        self.call_order: list[str] = []

    def evaluate(self, unit: RoomEvaluationUnit) -> Mapping[str, Any]:
        self.call_order.append(unit.room_id)
        return deepcopy(self.reports[unit.room_id])


class _CampaignLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = None

    def __enter__(self) -> "_CampaignLock":
        self.handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.close()
            raise ResilientCampaignError(
                "another writer already holds the campaign output lock"
            ) from exc
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(json.dumps({"pid": os.getpid()}) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        assert self.handle is not None
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()


def classify_failure(exc: BaseException, *, stage: str) -> FailureClassification:
    """Return a fail-closed taxonomy without persisting exception text.

    Room retries are reserved for explicit infrastructure failures.  A
    missing metric, bounded camera exhaustion, malformed API payload, or an
    unknown exception is terminal for that room so one bad room cannot cause
    repeated whole-room evaluation sweeps.
    """

    error_type = type(exc).__name__
    if isinstance(exc, NonRectangularMaterializationInfrastructureError):
        return FailureClassification(stage, exc.category, error_type, True)
    if isinstance(
        exc,
        (
            NonRectangularMaterializationContractError,
            NonRectangularPreflightError,
            NonRectangularContractError,
            RoomLayoutValidationError,
            RoomEvaluatorReportError,
        ),
    ):
        return FailureClassification(stage, "semantic_or_contract", error_type, False)
    if isinstance(exc, NonRectangularRoomMetricIncomplete):
        return FailureClassification(
            stage,
            exc.failure_category,
            error_type,
            False,
            metric_id=exc.metric_id,
            source_status=exc.source_status,
            event_keys=exc.event_keys,
            fallback_rejection_reasons=exc.fallback_rejection_reasons,
        )
    if isinstance(exc, NonRectangularCameraEvidenceExhausted):
        return FailureClassification(stage, "evidence_exhaustion_unclosed", error_type, False)
    if isinstance(exc, ResilientCampaignError) and stage == "generation_preflight":
        return FailureClassification(stage, "semantic_or_contract", error_type, False)
    if isinstance(exc, (EndpointConfigurationError, MissingAPIKeyError)):
        return FailureClassification(stage, "api_configuration", error_type, False)
    if isinstance(exc, EndpointHTTPError):
        status = _endpoint_http_status(exc)
        return FailureClassification(
            stage,
            "retryable_http" if status in RETRYABLE_HTTP_STATUSES else "nonretryable_http",
            error_type,
            status in RETRYABLE_HTTP_STATUSES,
        )
    if isinstance(exc, EndpointMalformedResponseError):
        return FailureClassification(stage, "api_response_contract", error_type, False)
    if isinstance(exc, EndpointConnectionError):
        return FailureClassification(stage, "transport_or_timeout", error_type, True)
    if isinstance(exc, (TimeoutError, ConnectionError, BrokenPipeError)):
        return FailureClassification(stage, "transport_or_timeout", error_type, True)
    if isinstance(exc, (BlenderRenderError, EvidenceRenderFailure)):
        return FailureClassification(stage, "service_or_renderer", error_type, True)
    lowered = error_type.lower()
    if any(value in lowered for value in ("configuration", "apikey", "schema", "geometry")):
        return FailureClassification(stage, "semantic_or_configuration", error_type, False)
    if isinstance(exc, ValueError):
        return FailureClassification(stage, "semantic_or_contract", error_type, False)
    return FailureClassification(stage, "unclassified_failure", error_type, False)


def _endpoint_http_status(exc: EndpointHTTPError) -> int | None:
    match = _HTTP_STATUS_PATTERN.search(str(exc))
    return int(match.group(1)) if match is not None else None


def _ordered_scene_dirs(root: Path) -> list[Path]:
    candidates = {
        path.name: path
        for path in root.iterdir()
        if path.is_dir() and not path.is_symlink() and not path.name.startswith("_")
    }
    order: list[str] = []
    manifests = [
        root / "_runner_state/run_manifest.json",
        root.parent / "_runner_state/run_manifest.json",
    ]
    for manifest_path in manifests:
        if not manifest_path.is_file():
            continue
        manifest = _read_json(manifest_path)
        values = manifest.get("scene_order")
        if isinstance(values, list):
            order = [str(item) for item in values if str(item) in candidates]
            break
    order.extend(sorted(set(candidates) - set(order)))
    return [candidates[name] for name in order]


def _selected_asset_ids(asset_selection: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    rooms = asset_selection.get("rooms")
    if not isinstance(rooms, list):
        raise ResilientCampaignError("asset_selection.rooms must be an array")
    for room in rooms:
        for item in room.get("objects") or []:
            selected = item.get("selected_asset")
            ref = selected.get("asset_ref") if isinstance(selected, Mapping) else None
            asset_id = str(
                (ref.get("asset_key") if isinstance(ref, Mapping) else None)
                or (selected.get("jid") if isinstance(selected, Mapping) else None)
                or ""
            )
            if not asset_id:
                raise ResilientCampaignError("asset selection contains no asset identity")
            values.append(asset_id)
    return tuple(dict.fromkeys(values))


def _verify_generation_preflight(
    value: Mapping[str, Any],
    preflight: NonRectangularPreflightResult,
) -> None:
    if value.get("terminal_status") != "ready":
        raise ResilientCampaignError("generation evaluation_preflight is not ready")
    if value.get("layout_id") != preflight.layout_id:
        raise ResilientCampaignError("generation evaluation_preflight layout drift")
    if value.get("room_order") != list(preflight.room_order):
        raise ResilientCampaignError("generation evaluation_preflight room order drift")
    hashes = value.get("artifact_sha256")
    if not isinstance(hashes, Mapping) or dict(hashes) != preflight.artifact_sha256:
        raise ResilientCampaignError("generation evaluation_preflight hash drift")


def _campaign_identity(
    config: ResilientCampaignConfig,
    bundles: tuple[GenerationBundle, ...],
    runtime_identity: dict[str, Any],
    *,
    rejected: tuple[RejectedGenerationScene, ...],
) -> dict[str, Any]:
    asset_ids = tuple(
        dict.fromkeys(
            asset_id
            for bundle in bundles
            for asset_id in bundle.selected_asset_ids
        )
    )
    asset_inventory: dict[str, dict[str, str]] = {}
    if asset_ids:
        catalog = FrozenCatalog(
            asset_csv=config.asset_csv,
            asset_root=config.asset_root,
            allowed_asset_ids=asset_ids,
            snapshot_id=config.catalog_snapshot_id,
        )
        asset_inventory = {
            asset_id: catalog.resolve(asset_id).hashes for asset_id in asset_ids
        }
    return {
        "coordinator_revision": COORDINATOR_REVISION,
        "evaluation_mode": NON_RECTANGULAR_EVALUATION_MODE,
        "model_order": [model for model, _ in config.model_roots],
        "models": {
            model: {
                "generation_root": str(root),
                "scene_order": [
                    bundle.scene_id for bundle in bundles if bundle.model == model
                ],
            }
            for model, root in config.model_roots
        },
        "scenes": [
            {
                "model": bundle.model,
                "scene_id": bundle.scene_id,
                "room_order": list(bundle.preflight.room_order),
                "artifacts": {
                    name: {"path": str(path), "sha256": bundle.file_sha256[name]}
                    for name, path in bundle.artifacts.items()
                },
            }
            for bundle in bundles
        ],
        "rejected_or_incomplete_scenes": [item.public_dict() for item in rejected],
        "materialization": {
            "asset_csv": str(config.asset_csv),
            "asset_csv_sha256": sha256_file(config.asset_csv),
            "asset_root": str(config.asset_root),
            "catalog_snapshot_id": config.catalog_snapshot_id,
            "selected_asset_count": len(asset_inventory),
            "selected_asset_inventory_sha256": sha256_json(asset_inventory),
            "selected_asset_hashes": asset_inventory,
            "blender_bin": str(config.blender_bin),
            "blender_bin_sha256": sha256_file(config.blender_bin),
            "timeout_seconds": config.materialization_timeout_seconds,
        },
        "execution": {
            "max_workers": config.max_workers,
            "max_materialization_attempts": config.max_materialization_attempts,
            "max_room_attempts": config.max_room_attempts,
            "api_initial_attempts": 1,
            "api_max_retries": config.api_max_retries,
            "room_retry_scheduling": "all_initial_rooms_then_global_retry_sweeps",
            "continue_after_room_failure": True,
        },
        "runtime": runtime_identity,
    }


def _runtime_identity(factory: RoomEvaluatorFactory) -> dict[str, Any]:
    value = factory.identity()
    if not isinstance(value, Mapping):
        raise ResilientCampaignError("evaluator factory identity must be an object")
    allowed = {"provider", "model", "config_sha256"}
    if set(value) != allowed:
        raise ResilientCampaignError(
            "runtime identity must contain only provider, model, and config_sha256"
        )
    result = {name: str(value[name]) for name in sorted(allowed)}
    if not result["provider"] or not result["model"] or len(result["config_sha256"]) != 64:
        raise ResilientCampaignError("runtime identity fields are invalid")
    return result


def _usage_totals(factory: RoomEvaluatorFactory) -> dict[str, int]:
    value = factory.usage_totals()
    if not isinstance(value, Mapping):
        raise ResilientCampaignError("usage totals must be an object")
    result = {}
    for name in ("logical_calls", "failed_calls", "usage_missing_calls", "tokens"):
        raw = value.get(name, 0)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise ResilientCampaignError(f"usage total {name} is invalid")
        result[name] = raw
    return result


def _validate_static_paths(config: ResilientCampaignConfig) -> None:
    for path, label in (
        (config.asset_csv, "asset_csv"),
        (config.blender_bin, "blender_bin"),
    ):
        if path.is_symlink() or not path.is_file():
            raise ResilientCampaignError(f"{label} must be a regular file")
    if config.asset_root.is_symlink() or not config.asset_root.is_dir():
        raise ResilientCampaignError("asset_root must be a real directory")
    try:
        config.output_root.relative_to(config.asset_root)
    except ValueError:
        pass
    else:
        raise ResilientCampaignError("output root cannot be inside asset_root")
    try:
        config.asset_root.relative_to(config.output_root)
    except ValueError:
        pass
    else:
        raise ResilientCampaignError("asset_root cannot be inside output root")
    try:
        config.asset_csv.relative_to(config.output_root)
    except ValueError:
        pass
    else:
        raise ResilientCampaignError("asset_csv cannot be inside output root")


def _reject_output_source_overlap(
    output: Path,
    bundles: tuple[GenerationBundle, ...],
    *,
    rejected: tuple[RejectedGenerationScene, ...] = (),
) -> None:
    for source_root in [
        *(bundle.root for bundle in bundles),
        *(item.root for item in rejected),
    ]:
        try:
            output.relative_to(source_root)
        except ValueError:
            pass
        else:
            raise ResilientCampaignError("output root cannot be inside generation root")
        try:
            source_root.relative_to(output)
        except ValueError:
            pass
        else:
            raise ResilientCampaignError("generation root cannot be inside output root")


def _attempt_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        (path for path in root.glob("attempt_*") if path.is_dir()),
        key=lambda path: path.name,
    )


def _next_attempt_number(root: Path) -> int:
    values = _attempt_dirs(root)
    if not values:
        return 1
    try:
        return max(int(path.name.split("_")[-1]) for path in values) + 1
    except ValueError as exc:
        raise ResilientCampaignError("attempt directory naming drift") from exc


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ResilientCampaignError(f"cannot read JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ResilientCampaignError(f"JSON artifact must be an object: {path}")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    temporary.write_text(
        json.dumps(
            dict(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


__all__ = [
    "FailureClassification",
    "GenerationBundle",
    "NoAPIMockEvaluatorFactory",
    "NoAPIMockMaterializer",
    "ResilientCampaignConfig",
    "ResilientCampaignError",
    "ResilientCampaignResult",
    "RoomEvaluatorFactory",
    "RoomRuntimeContext",
    "classify_failure",
    "discover_generation_bundles",
    "run_resilient_nonrect_campaign",
]
