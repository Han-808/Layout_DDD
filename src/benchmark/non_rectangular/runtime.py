"""Production runtime factory for room-scoped nonrect evaluation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import threading
import time
from typing import Any, Mapping

from benchmark.models.openai_compatible_model import (
    EndpointConfigurationError,
    EndpointConnectionError,
    EndpointHTTPError,
    EndpointMalformedResponseError,
    OpenAICompatibleModel,
)
from benchmark.non_rectangular.evaluator import (
    CanonicalNonRectangularRoomEvaluator,
)
from benchmark.non_rectangular.camera import (
    NonRectangularCameraEvidenceExhausted,
)
from benchmark.non_rectangular.projection import (
    project_room_unit_to_canonical_scene,
)
from benchmark.non_rectangular.resilient import RoomRuntimeContext
from benchmark.rendering.blender import BlenderRenderer
from benchmark.rendering.camera_pose import generate_global_context_poses
from benchmark.visual_judge.openai_camera_selector import (
    OpenAICompatibleCameraSelector,
)
from benchmark.visual_judge.openai_compatible import OpenAICompatibleVLMJudge
from benchmark.visual_judge.render_views import CameraEvidenceProvider
from benchmark.materialization.catalog import sha256_json
from benchmark.utils.io import write_json


class APIUsageRecorder:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._totals = {
            "logical_calls": 0,
            "failed_calls": 0,
            "usage_missing_calls": 0,
            "tokens": 0,
        }

    def record(self, *, failed: bool, usage: Mapping[str, Any] | None) -> None:
        with self._lock:
            self._totals["logical_calls"] += 1
            if failed:
                self._totals["failed_calls"] += 1
                return
            if not isinstance(usage, Mapping):
                self._totals["usage_missing_calls"] += 1
                return
            total = usage.get("total_tokens")
            if isinstance(total, bool) or not isinstance(total, int) or total < 0:
                prompt = usage.get("prompt_tokens", 0)
                completion = usage.get("completion_tokens", 0)
                if all(
                    isinstance(value, int) and not isinstance(value, bool) and value >= 0
                    for value in (prompt, completion)
                ):
                    total = int(prompt) + int(completion)
                else:
                    self._totals["usage_missing_calls"] += 1
                    return
            self._totals["tokens"] += int(total)

    def totals(self) -> dict[str, int]:
        with self._lock:
            return dict(self._totals)


class RecordingOpenAICompatibleModel(OpenAICompatibleModel):
    def __init__(
        self,
        *,
        recorder: APIUsageRecorder,
        exact_max_retries: int,
        exact_retry_delay_seconds: float,
        **kwargs: Any,
    ) -> None:
        self._usage_recorder = recorder
        self._exact_max_retries = int(exact_max_retries)
        self._exact_retry_delay_seconds = float(exact_retry_delay_seconds)
        super().__init__(**kwargs)

    def chat_messages(self, *args: Any, **kwargs: Any) -> str:
        retries = 0
        while True:
            try:
                result = super().chat_messages(*args, **kwargs)
            except Exception as exc:
                if (
                    _retryable_exact_request_failure(exc)
                    and retries < self._exact_max_retries
                ):
                    retries += 1
                    if self._exact_retry_delay_seconds > 0.0:
                        time.sleep(self._exact_retry_delay_seconds)
                    continue
                self._usage_recorder.record(failed=True, usage=None)
                raise
            break
        self._usage_recorder.record(
            failed=False,
            usage=(
                self.last_request_metadata.get("usage")
                if isinstance(self.last_request_metadata, dict)
                else None
            ),
        )
        return result


class DefaultNonRectangularRuntimeFactory:
    """Build isolated Judge/camera providers for each room attempt.

    The runtime configuration is never copied into coordinator summaries;
    those receive only provider/model labels and a SHA-256 identity.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        if not isinstance(config, Mapping):
            raise TypeError("runtime config must be a JSON object")
        self.config = deepcopy(dict(config))
        judge = self.config.get("judge")
        if not isinstance(judge, Mapping):
            raise ValueError("runtime config requires judge")
        if "api_key" in judge:
            raise ValueError("runtime config must use api_key_env, never api_key")
        self.provider = str(self.config.get("provider") or "openai_compatible")
        self.model = str(judge.get("model") or judge.get("model_id") or "")
        if not self.model:
            raise ValueError("runtime judge model is required")
        self.recorder = APIUsageRecorder()

    def identity(self) -> Mapping[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "config_sha256": sha256_json(self.config),
        }

    def usage_totals(self) -> Mapping[str, Any]:
        return self.recorder.totals()

    def build(self, context: RoomRuntimeContext) -> CanonicalNonRectangularRoomEvaluator:
        judge_config = deepcopy(dict(self.config["judge"]))
        judge_model = _model_from_config(
            judge_config,
            recorder=self.recorder,
            default_name="nonrect-vlm-judge",
        )
        judge = OpenAICompatibleVLMJudge(
            judge_model,
            max_images=int(judge_config.get("max_images", 8)),
            max_context_chars=int(judge_config.get("max_context_chars", 30000)),
            response_format_json=bool(judge_config.get("response_format_json", True)),
        )
        # Grouping uses the same transport contract but a separate client so
        # concurrent telemetry and retry state cannot race with Judge calls.
        grouping_model = _model_from_config(
            deepcopy(dict(self.config.get("grouping_model") or judge_config)),
            recorder=self.recorder,
            default_name="nonrect-vlm-grouping",
        )

        render_config = deepcopy(dict(self.config.get("renderer") or {}))
        renderer = BlenderRenderer(
            blender_bin=str(
                render_config.pop(
                    "blender_bin",
                    self.config.get("blender_bin")
                    or "/Applications/Blender.app/Contents/MacOS/Blender",
                )
            ),
            timeout_seconds=int(render_config.pop("timeout_seconds", 900)),
            width=int(render_config.pop("width", 768)),
            height=int(render_config.pop("height", 768)),
            render_engine=str(render_config.pop("render_engine", "BLENDER_EEVEE_NEXT")),
            cycles_device=str(render_config.pop("cycles_device", "CPU")),
            cycles_samples=int(render_config.pop("cycles_samples", 16)),
            cycles_denoising=bool(render_config.pop("cycles_denoising", False)),
            preview_render_engine=render_config.pop("preview_render_engine", None),
            preview_width=int(render_config.pop("preview_width", 256)),
            preview_height=int(render_config.pop("preview_height", 256)),
            preview_cycles_samples=int(render_config.pop("preview_cycles_samples", 1)),
            require_asset_mesh=bool(render_config.pop("require_asset_mesh", True)),
        )
        if render_config:
            raise ValueError(
                f"unsupported nonrect renderer options: {sorted(render_config)}"
            )
        camera_config = deepcopy(dict(self.config.get("camera") or {}))
        selector = self._camera_selector()
        mode = str(camera_config.pop("mode", "visibility_ranked"))
        max_views = int(camera_config.pop("max_views", 4))
        max_steps = int(camera_config.pop("max_steps", 3))
        candidate_count = int(camera_config.pop("candidate_count", 6))
        metric_modes = camera_config.pop("metric_modes", {})
        active_repair = bool(camera_config.pop("active_repair", selector is not None))
        if camera_config:
            raise ValueError(
                f"unsupported nonrect camera options: {sorted(camera_config)}"
            )
        evidence_root = context.attempt_root / "camera_evidence"
        local_provider = _NonRectangularContinuityCameraProvider(
            CameraEvidenceProvider(
                renderer=renderer,
                blend_file=context.materialization.blend_path,
                out_dir=evidence_root / "l1",
                mode=mode,
                selector=selector,
                max_views=max_views,
                max_steps=max_steps,
                candidate_count=candidate_count,
                metric_modes=metric_modes,
                collision_overlay=True,
                collision_contour=True,
                active_repair=active_repair,
            )
        )
        l3_provider = _NonRectangularContinuityCameraProvider(
            CameraEvidenceProvider(
                renderer=renderer,
                blend_file=context.materialization.blend_path,
                out_dir=evidence_root / "l3",
                mode=mode,
                selector=selector,
                max_views=max_views,
                max_steps=max_steps,
                candidate_count=candidate_count,
                metric_modes=metric_modes,
                collision_overlay=False,
                collision_contour=False,
                active_repair=active_repair,
            )
        )
        room_scene = project_room_unit_to_canonical_scene(context.unit)
        global_evidence, global_evidence_audit = (
            _render_nonrect_global_evidence(
                renderer=renderer,
                blend_file=context.materialization.blend_path,
                scene=room_scene,
                out_dir=evidence_root / "global",
            )
        )
        return CanonicalNonRectangularRoomEvaluator(
            output_root=context.attempt_root / "evaluator_artifacts",
            vlm_judge=judge,
            grouping_model=grouping_model,
            local_view_provider=local_provider,
            l3_camera_evidence_provider=l3_provider,
            render_evidence={"global": global_evidence},
            evidence_continuity_context={
                "global_evidence": global_evidence_audit,
            },
            l1_config=deepcopy(dict(self.config.get("l1_config") or {})),
            scene_quality_config=deepcopy(
                dict(self.config.get("scene_quality_config") or {})
            ),
            evaluation_profile=deepcopy(
                dict(self.config.get("evaluation_profile") or {})
            ),
        )

    def _camera_selector(self) -> OpenAICompatibleCameraSelector | None:
        selector_config = self.config.get("camera_selector")
        if selector_config is None:
            return None
        if not isinstance(selector_config, Mapping):
            raise ValueError("camera_selector config must be an object")
        values = deepcopy(dict(selector_config))
        if "api_key" in values:
            raise ValueError("camera_selector must use api_key_env, never api_key")
        model = _model_from_config(
            values,
            recorder=self.recorder,
            default_name="nonrect-camera-selector",
        )
        return OpenAICompatibleCameraSelector(
            model,
            max_preview_images=int(values.get("max_preview_images", values.get("max_images", 8))),
            max_context_chars=int(values.get("max_context_chars", 30000)),
            response_format_json=bool(values.get("response_format_json", True)),
        )


class _NonRectangularContinuityCameraProvider:
    """Convert only proven bounded nonrect exhaustion into a typed signal."""

    def __init__(self, provider: CameraEvidenceProvider) -> None:
        self._provider = provider
        self._last_call_usage: dict[str, Any] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._provider, name)

    @property
    def last_call_usage(self) -> dict[str, Any] | None:
        return self._last_call_usage or getattr(
            self._provider,
            "last_call_usage",
            None,
        )

    def __call__(self, request: dict[str, Any]) -> list[Any]:
        self._last_call_usage = None
        try:
            return list(self._provider(request))
        except Exception as exc:
            usage = getattr(self._provider, "last_call_usage", None)
            usage = deepcopy(usage) if isinstance(usage, dict) else {}
            candidate_audit = usage.get("nonrect_candidate_audit")
            bounded_audit_exhausted = bool(
                isinstance(candidate_audit, dict)
                and candidate_audit.get("terminal_outcome")
                == "bounded_local_bank_exhausted"
            )
            generated = usage.get("candidate_count_generated")
            exact_no_feasible = str(exc).startswith(
                "no_feasible_candidate:"
            )
            if not (
                bounded_audit_exhausted
                or generated == 0
                or exact_no_feasible
            ):
                raise
            usage.update(
                selection_outcome="bounded_nonrect_camera_exhausted",
                recovered_as_typed_exhaustion=True,
                underlying_error_type=type(exc).__name__,
            )
            self._last_call_usage = usage
            raise NonRectangularCameraEvidenceExhausted(
                "bounded nonrect camera search produced no usable local view"
            ) from exc


def _render_nonrect_global_evidence(
    *,
    renderer: BlenderRenderer,
    blend_file: Path,
    scene: dict[str, Any],
    out_dir: Path,
) -> tuple[list[str], dict[str, Any]]:
    """Render a reusable room-global packet, salvaging poses independently."""

    poses = generate_global_context_poses(scene)
    paths: list[str] = []
    pose_records: list[dict[str, Any]] = []
    for pose in poses:
        pose_id = str(pose.get("id") or "global")
        try:
            manifest = renderer.render_camera_views(
                blend_file=blend_file,
                out_dir=out_dir / pose_id,
                camera_views=[pose],
                preview=False,
            )
            rendered = [
                str(item["path"])
                for item in manifest.get("views", [])
                if isinstance(item, dict) and item.get("path")
            ]
            if not rendered:
                raise NonRectangularCameraEvidenceExhausted(
                    "global pose renderer returned no view"
                )
        except Exception as exc:
            pose_records.append(
                {
                    "pose_id": pose_id,
                    "status": "failed_recovered",
                    "error_type": type(exc).__name__,
                }
            )
            continue
        paths.extend(rendered)
        pose_records.append(
            {
                "pose_id": pose_id,
                "status": "complete",
                "rendered_view_count": len(rendered),
            }
        )
    paths = list(dict.fromkeys(paths))
    audit = {
        "schema_version": "nonrect_global_evidence_packet_v1",
        "status": "complete" if paths else "unavailable_recovered",
        "policy": "global_top_then_polygon_perspective_independent_salvage",
        "visual_evidence_count": len(paths),
        "poses": pose_records,
        "geometry_only_fallback_allowed": not paths,
        "degraded": not paths or any(
            item["status"] != "complete" for item in pose_records
        ),
    }
    write_json(out_dir / "global_evidence_manifest.json", audit)
    return paths, audit


def _model_from_config(
    config: Mapping[str, Any],
    *,
    recorder: APIUsageRecorder,
    default_name: str,
) -> RecordingOpenAICompatibleModel:
    if "api_key" in config:
        raise ValueError("model config must use api_key_env, never api_key")
    endpoint = config.get("endpoint") or config.get("base_url")
    model = config.get("model") or config.get("model_id")
    if not endpoint or not model:
        raise ValueError("model config requires endpoint and model")
    return RecordingOpenAICompatibleModel(
        recorder=recorder,
        exact_max_retries=5,
        exact_retry_delay_seconds=float(
            config.get("retry_backoff_seconds", 30.0)
        ),
        name=str(config.get("name") or default_name),
        endpoint=str(endpoint),
        model_id=str(model),
        api_key_env=(str(config["api_key_env"]) if config.get("api_key_env") else None),
        temperature=float(config.get("temperature", 0.0)),
        max_tokens=int(config.get("max_tokens", 8192)),
        context_length=(
            int(config["context_length"])
            if config.get("context_length") is not None
            else None
        ),
        timeout_seconds=int(config.get("timeout_seconds", 300)),
        response_format_json=bool(config.get("response_format_json", True)),
        # The wrapper owns the single combined exact-request retry budget so
        # transport and malformed-response failures cannot multiply retries.
        max_retries=0,
        retry_backoff_seconds=0.0,
        min_request_interval_seconds=float(
            config.get("min_request_interval_seconds", 0.0)
        ),
        retry_on_status=list(config.get("retry_on_status", [429, 500, 502, 503, 504])),
        max_tokens_field=str(config.get("max_tokens_field", "max_tokens")),
        send_temperature=bool(config.get("send_temperature", True)),
        require_api_key=(
            bool(config["require_api_key"])
            if config.get("require_api_key") is not None
            else None
        ),
    )


def _retryable_exact_request_failure(exc: BaseException) -> bool:
    if isinstance(exc, EndpointConfigurationError):
        return False
    return isinstance(
        exc,
        (
            EndpointConnectionError,
            EndpointHTTPError,
            EndpointMalformedResponseError,
            TimeoutError,
            ConnectionError,
        ),
    )


__all__ = [
    "APIUsageRecorder",
    "DefaultNonRectangularRuntimeFactory",
    "RecordingOpenAICompatibleModel",
]
