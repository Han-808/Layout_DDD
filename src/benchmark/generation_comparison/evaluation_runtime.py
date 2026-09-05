"""Trusted post-generation wiring of the existing canonical evaluator runtime.

This module does not score, convert, retrieve, or modify scene state. It builds
the same renderer, grouping model, Judge and camera providers already used by
the canonical evaluation entrypoints. No runtime object/config enters adapters.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from benchmark.generation_comparison.identity import canonical_json_sha256
from benchmark.grouping import grouping_evidence_from_render_manifest
from benchmark.rendering import BlenderRenderer
from benchmark.scene_io.validate import ArtifactValidationError
from benchmark.utils.io import write_json
from benchmark.visual_judge import (
    CameraCandidatePreviewRenderer, CameraEvidenceProvider,
    CameraViewEvidenceRenderer, DeterministicLocalCameraSelector,
    build_openai_compatible_camera_selector, build_openai_compatible_vlm_judge,
)


RUNTIME_SCHEMA = "canonical_blender_evaluation_runtime_v1"
RUNTIME_KEYS = frozenset({
    "vlm_judge", "grouping_model", "camera_selector", "vlm_camera_selector",
    "p0b_local_view_provider", "l3_initial_evidence_provider",
    "functional_probe_evidence_provider", "deterministic_camera_selector",
    "evidence_renderer", "candidate_preview_renderer", "render_evidence",
    "grouping_visual_evidence", "collision_geometry",
    "functional_evidence_planner",
})


def evaluator_policy_readiness(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Fail before spending generation calls on a known incomplete-score policy.

    This is a diagnostic of the unchanged evaluator, not a scoring override.
    Currently a not-relevant L3 metric remains a defaulted, ungrounded term in
    its denominator. Honest frozen ownership therefore cannot yield complete
    coverage under the default profile. An operator decision is needed; never
    relabel ownership or silently remove metrics to turn this gate green.
    """
    from benchmark.evaluator.asset_policy import resolve_asset_policy, scene_quality_applicability
    from benchmark.evaluator.profile import L3, resolve_evaluation_profile

    static = dict(policy.get("static_kwargs") or {})
    profile = resolve_evaluation_profile(static.get("evaluation_profile"))
    applicability = scene_quality_applicability(resolve_asset_policy(static.get("asset_policy")))
    unresolved = [
        name for name, metric in profile[L3]["metrics"].items()
        if profile[L3]["enabled"] and profile["layer_weights"][L3] > 0
        and metric["enabled"] and metric["weight"] > 0
        and applicability[name]["applicability"] != "relevant"
    ]
    return {
        "ready": not unresolved,
        "required_score_status": "complete",
        "reasons": ["canonical_applicability_prevents_complete_coverage"] if unresolved else [],
        "unresolved_applicability_metrics": unresolved,
        "applicability": applicability,
        "scoring_modified": False,
    }


class CanonicalEvaluationRuntime:
    """Preflight constructs clients without calling them; rendering is post-hoc."""

    def __init__(self, config: Mapping[str, Any], *, require_credentials: bool = True):
        self.config = deepcopy(dict(config))
        if self.config.get("schema_version") != RUNTIME_SCHEMA:
            raise ArtifactValidationError("unsupported evaluator runtime schema")
        unknown = set(self.config) - {"schema_version", "judge", "camera_selector", "renderer", "asset_root"}
        if unknown:
            raise ArtifactValidationError(f"unsupported evaluator runtime fields: {sorted(unknown)}")
        self.config_sha256 = canonical_json_sha256(self.config)
        self._require_credentials = require_credentials
        self._preflight()

    def _preflight(self) -> None:
        renderer_config = deepcopy(self.config.get("renderer"))
        if not isinstance(renderer_config, dict):
            raise ArtifactValidationError("evaluator runtime requires renderer config")
        blender = Path(str(renderer_config.get("blender_bin") or "")).expanduser()
        if not blender.is_file() or not os.access(blender, os.X_OK):
            raise ArtifactValidationError("evaluator Blender executable missing/not executable")
        if renderer_config.get("require_asset_mesh", True) is not True:
            raise ArtifactValidationError("fixed-assets evaluator cannot disable asset mesh rendering")
        renderer_config["require_asset_mesh"] = True
        self.renderer = BlenderRenderer(**renderer_config)
        self.asset_root = self.config.get("asset_root")
        if self.asset_root is not None and not Path(self.asset_root).expanduser().is_dir():
            raise ArtifactValidationError("evaluator asset_root is missing")
        for role in ("judge", "camera_selector"):
            model = self.config.get(role)
            if not isinstance(model, dict):
                raise ArtifactValidationError(f"evaluator {role} config missing")
            if any(key in model for key in ("api_key", "headers", "authorization")):
                raise ArtifactValidationError(f"evaluator {role} must use environment credentials, not literals")
            endpoint = model.get("endpoint") or model.get("base_url")
            url = urlparse(str(endpoint or ""))
            if url.scheme not in {"http", "https"} or not url.netloc or url.username or url.password or url.query:
                raise ArtifactValidationError(f"evaluator {role} requires an uncredentialed HTTP endpoint")
            if not (model.get("model") or model.get("model_id")):
                raise ArtifactValidationError(f"evaluator {role} model missing")
            key_env = model.get("api_key_env")
            if self._require_credentials and model.get("require_api_key", True):
                if not key_env or not os.environ.get(str(key_env)):
                    raise ArtifactValidationError(f"evaluator {role} API credential environment unset")
        self.judge = build_openai_compatible_vlm_judge(self.config["judge"])
        self.selector = build_openai_compatible_camera_selector(self.config["camera_selector"])
        # Canonical grouping consumes the same chat-model interface as the
        # evaluator CLI's _grouping_chat_model(judge), with a separate prompt.
        if not callable(getattr(self.judge.model, "chat_messages", None)):
            raise ArtifactValidationError("evaluator grouping model interface unavailable")

    @property
    def readiness(self) -> dict[str, Any]:
        return {
            "schema_version": RUNTIME_SCHEMA, "ready": True,
            "config_sha256": self.config_sha256,
            "credentials_checked": self._require_credentials,
            "service_contacted": False, "real_service_verified": False,
            "entrypoint": "benchmark.api.evaluation.run_evaluate",
        }

    def for_scene(self, scene: Mapping[str, Any], out_dir: Path) -> dict[str, Any]:
        """Build fresh scene-bound evidence, also for every native iteration."""
        destination = Path(out_dir).resolve()
        if destination.exists():
            raise FileExistsError(f"evaluator runtime output already exists: {destination}")
        destination.mkdir(parents=True)
        before = canonical_json_sha256(scene)
        scene_path = write_json(destination / "render_input_scene.json", scene)
        input_bytes_hash = hashlib.sha256(scene_path.read_bytes()).hexdigest()
        manifest = self.renderer.render_scene(
            scene_path=scene_path, out_dir=destination / "renders", asset_root=self.asset_root,
        )
        if canonical_json_sha256(scene) != before or hashlib.sha256(scene_path.read_bytes()).hexdigest() != input_bytes_hash:
            raise ArtifactValidationError("evaluator runtime mutated the canonical scene")
        if not isinstance(manifest.get("collision_geometry"), dict):
            raise ArtifactValidationError("evaluator renderer omitted collision geometry")
        objects = manifest.get("objects") or []
        expected_ids = {str(item["id"]) for item in scene["objects"]}
        observed_ids = [str(item.get("id")) for item in objects]
        if set(observed_ids) != expected_ids or len(observed_ids) != len(expected_ids):
            raise ArtifactValidationError("evaluator rendered object inventory mismatch")
        if any(item.get("representation") != "asset_mesh" for item in objects):
            raise ArtifactValidationError("evaluator mesh import fell back to proxy geometry")

        def trusted_file(raw: Any) -> Path:
            path = Path(str(raw or "")).resolve()
            if not path.is_relative_to(destination) or not path.is_file():
                raise ArtifactValidationError("evaluator evidence missing/outside trusted output")
            return path

        blend = trusted_file(manifest.get("blend_file"))
        views = [trusted_file(view.get("path")).as_posix() for view in manifest.get("views", [])
                 if view.get("name") != "identity_map"]
        if not views:
            raise ArtifactValidationError("evaluator renderer omitted RGB evidence")
        grouping_evidence = grouping_evidence_from_render_manifest(manifest)
        for evidence in grouping_evidence:
            trusted_file(evidence["path"])
        geometry = manifest["collision_geometry"]

        def provider(name: str, **kwargs: Any) -> CameraEvidenceProvider:
            return CameraEvidenceProvider(
                renderer=self.renderer, blend_file=blend, out_dir=destination / name,
                max_views=2, max_steps=1, candidate_count=6,
                collision_geometry=geometry, **kwargs,
            )

        l1 = provider("l1_camera", mode="auto", selector=None,
                      collision_overlay=True, collision_contour=True)
        l3 = provider("l3_camera", mode="visibility_ranked", selector=None,
                      collision_overlay=False, collision_contour=False, active_repair=False)
        probe = CameraEvidenceProvider(
            renderer=self.renderer, blend_file=blend, out_dir=destination / "functional_probes",
            mode="query_cov", selector=self.selector, max_views=1, max_steps=0,
            candidate_count=6, collision_overlay=False, collision_contour=False,
            collision_geometry=geometry, active_repair=False,
            usable_surface_cache_dir=destination / "usable_surface_cache",
        )
        write_json(destination / "runtime_manifest.json", {
            **self.readiness, "phase": "post_generation_evaluation",
            "canonical_scene_sha256": before,
            "render_input_sha256": hashlib.sha256(scene_path.read_bytes()).hexdigest(),
            "render_manifest_sha256": canonical_json_sha256(manifest),
            "blend_file": blend.as_posix(), "benchmark_feedback_to_generator": False,
        })
        return {
            "vlm_judge": self.judge, "grouping_model": self.judge.model,
            "camera_selector": self.selector, "vlm_camera_selector": self.selector,
            "functional_evidence_planner": self.selector,
            "p0b_local_view_provider": l1, "l3_initial_evidence_provider": l3,
            "functional_probe_evidence_provider": probe,
            "deterministic_camera_selector": DeterministicLocalCameraSelector(candidate_policy=l3.candidate_policy),
            "evidence_renderer": CameraViewEvidenceRenderer(renderer=self.renderer, blend_file=blend,
                                                           out_dir=destination / "controller"),
            "candidate_preview_renderer": CameraCandidatePreviewRenderer(renderer=self.renderer, blend_file=blend,
                                                                         out_dir=destination / "controller"),
            "render_evidence": views, "grouping_visual_evidence": grouping_evidence,
            "collision_geometry": geometry,
        }


def runtime_evaluation_options(
    options: Mapping[str, Any], runtime: CanonicalEvaluationRuntime | None,
    *, scene: Mapping[str, Any], out_dir: Path,
) -> dict[str, Any]:
    """Keep evaluator-owned scene-bound runtime out of generic static kwargs."""
    result = dict(options)
    if runtime is not None:
        overlap = set(result) & RUNTIME_KEYS
        if overlap:
            raise ArtifactValidationError(f"static evaluation options override runtime: {sorted(overlap)}")
        result.update(runtime.for_scene(scene, out_dir))
    return result
