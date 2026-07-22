from __future__ import annotations

import base64
import hashlib
from io import BytesIO
import json
import math
import mimetypes
from pathlib import Path
from typing import Any

from benchmark.models import OpenAICompatibleModel, parse_json_object


SYSTEM_PROMPT = """You are the visual judge for a 3D scene-generation benchmark.
Use the rendered views as primary visual evidence. The canonical scene summary and deterministic
findings are navigation aids, not a replacement for inspecting the images. Do not invent hidden
objects or relations. Return exactly one JSON object with:
{"applicable":true,"score":0.0,"confidence":0.0,"summary":"...","issues":["..."],"evidence":["..."]}.
score and confidence must be between 0 and 1. If the supplied views cannot support this category,
return applicable=false, score=null, and explain why in summary."""

P0B_SYSTEM_PROMPT = """You adjudicate one ambiguous geometry event in a 3D scene benchmark.
Use the natural-language prompt and extracted relationships only to understand intended semantics.
Use detector evidence for measured geometry and inspect supplied images when present. Generator
relationships are claims, not automatic exemptions. Highlighted diagnostic views use the supplied
color legend; gray geometry is non-target context, not missing scene content. Decide whether this event constitutes a
structural error. Return exactly one JSON object:
{"verdict":"valid","confidence":0.0,"reason":"..."}.
verdict must be exactly valid or invalid. No abstention, not-applicable, insufficient-evidence,
continuous score, or third verdict is allowed. confidence must be between 0 and 1."""

RELATION_SYSTEM_PROMPT = """You adjudicate one explicit spatial relationship in a 3D scene benchmark.
The relation was routed because it is outside the frozen deterministic registry or because deterministic
proxy evidence could not safely resolve its semantic realization. Routing carries no valid or invalid
prior. Judge only whether that exact relationship is satisfied by the generated scene. Use the original
prompt, structured relation, detector evidence, and supplied rendered views. Canonical object geometry
is supporting context, not permission to invent hidden evidence. Return exactly one JSON object:
{"verdict":"valid","confidence":0.0,"reason":"..."}.
verdict must be exactly valid or invalid. No abstention, not-applicable, insufficient-evidence,
continuous score, or third verdict is allowed. confidence must be between 0 and 1."""

SPATIAL_FIDELITY_SYSTEM_PROMPT = """You adjudicate one candidate Spatial Fidelity issue in a
coarse-grained 3D scene benchmark. A statistical detector routed this event; routing is evidence to
inspect, not a valid/invalid prior. For scale, decide whether the object's visible real-world size is
semantically implausible, allowing legitimate variants and unusual but coherent designs. For
co-occurrence, decide whether the object-category combination is semantically incoherent in this
particular requested scene; dataset rarity or absence alone is never an error. Use the canonical
measurements and statistical packet together with the rendered views. Return exactly one JSON object:
{"verdict":"valid","confidence":0.0,"reason":"..."}.
verdict must be exactly valid or invalid. No abstention, not-applicable, insufficient-evidence,
continuous score, or third verdict is allowed. confidence must be between 0 and 1."""

CAMERA_SELECTION_SYSTEM_PROMPT = """You select camera evidence for a 3D scene benchmark.
Do not judge whether the metric event is valid or invalid. Select the candidate views that make the
specified event easiest to inspect. Prefer views where all target objects and the relevant contact,
gap, overlap, or room plane are visible and well framed. In highlighted previews, use the supplied
color legend and treat gray geometry as non-target context. A preview_warning_class means only
that deterministic highlight-pixel coverage was incomplete; do not infer target absence from it.
Candidates marked render_status=blank are unusable camera evidence. Do not select them when any
render_status=ok candidate exists.
You may request at most one listed discrete
camera action when adjustment is allowed. Return exactly one JSON object:
{"selected_view_ids":["candidate_id"],"action":null,"reason":"..."}.
selected_view_ids must contain between one and max_views candidate IDs. action must be null when
allow_adjustment is false; otherwise it may be null or
{"view_id":"candidate_id","type":"one allowed action"}. Do not return a metric verdict or score."""

CATEGORY_RUBRICS = {
    "visual_quality": (
        "Judge holistic visual coherence, commonsense plausibility, style/appearance consistency, "
        "proportions, and whether the scene looks intentionally composed. Do not deduct twice for "
        "a deterministic geometry issue unless its visual consequence is visible."
    ),
    "prompt_fidelity": (
        "Judge only whether the visible scene follows the supplied natural-language request. "
        "Check requested objects, attributes, rough counts, and explicitly requested spatial relations. "
        "Treat ordinary synonyms and subtypes as the same broad object identity. Internal object-mapping "
        "confidence, unmatched candidates, resolver requests, and retrieval metadata are non-scoring alignment "
        "diagnostics: do not reward or penalize the generator for them. Judge visible prompt adherence directly, "
        "including color, material, style, count, placement, and relations."
    ),
    "structural_validity": (
        "Use the images to resolve only ambiguous structural findings called out by deterministic evidence. "
        "Do not override exact schema, boundary, or collision calculations without visible contradictory evidence."
    ),
}


class OpenAICompatibleVLMJudge:
    """Multimodal judge shared by MNET localhost and remote OpenAI-style APIs."""

    def __init__(
        self,
        model: OpenAICompatibleModel,
        *,
        max_images: int = 6,
        max_context_chars: int = 30000,
        response_format_json: bool | None = None,
    ) -> None:
        self.model = model
        self.max_images = max(1, int(max_images))
        self.max_context_chars = max(1000, int(max_context_chars))
        self.response_format_json = (
            bool(getattr(model, "response_format_json", True))
            if response_format_json is None
            else bool(response_format_json)
        )

    def evaluate(self, request: dict) -> dict:
        if not isinstance(request, dict):
            raise TypeError("VLM judge request must be a JSON object")
        category = str(request.get("category") or "visual_quality")
        paths = [Path(str(value)).expanduser() for value in request.get("render_evidence", [])]
        selected = paths[: self.max_images]
        missing = [str(path) for path in selected if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"VLM render evidence does not exist: {missing}")

        context = {
            "category": category,
            "rubric": CATEGORY_RUBRICS.get(category, "Judge this category from the supplied evidence."),
            "natural_language_request": request.get("prompt"),
            "canonical_scene": request.get("scene_summary"),
            "deterministic_evidence": request.get("deterministic_evidence"),
            "view_names": [path.name for path in selected],
        }
        context_text = _budgeted_context_json(
            context,
            self.max_context_chars,
            priority_keys=(
                "category",
                "deterministic_evidence",
                "rubric",
                "natural_language_request",
                "view_names",
            ),
        )
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": "Evaluate this scene under the specified category.\n" + context_text,
            }
        ]
        content.extend(
            {"type": "image_url", "image_url": {"url": _image_data_url(path)}}
            for path in selected
        )
        raw = self.model.chat_messages(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            response_format_json=self.response_format_json,
            call_type=f"vlm_judge.{category}",
        )
        result = parse_json_object(raw)
        applicable = result.get("applicable", True)
        if not isinstance(applicable, bool):
            raise ValueError("VLM judge applicable must be boolean")
        if applicable:
            _score(result.get("score"), "score")
        elif result.get("score") is not None:
            raise ValueError("VLM judge score must be null when applicable is false")
        if result.get("confidence") is not None:
            _score(result.get("confidence"), "confidence")
        result["applicable"] = applicable
        result["model"] = self.model.model_id
        result["endpoint"] = self.model.endpoint
        result["images_used"] = [str(path.resolve()) for path in selected]
        result["request_metadata"] = dict(self.model.last_request_metadata)
        return result

    def adjudicate_p0b(self, request: dict) -> dict:
        if not isinstance(request, dict):
            raise TypeError("P0b judge request must be a JSON object")
        paths = [Path(str(value)).expanduser() for value in request.get("render_evidence", [])]
        selected = paths[: self.max_images]
        missing = [str(path) for path in selected if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"P0b render evidence does not exist: {missing}")
        context = {
            "metric": request.get("metric"),
            "metric_rubric": request.get("metric_rubric"),
            "candidate_selection_policy": request.get("candidate_selection_policy"),
            "collision_evidence_style_guide": request.get("collision_evidence_style_guide"),
            "visual_evidence_policy": request.get("visual_evidence_policy"),
            "event": request.get("event"),
            "detector_evidence": request.get("detector_evidence"),
            "objects": request.get("objects"),
            "architecture": request.get("architecture"),
            "natural_language_prompt": request.get("natural_language_prompt"),
            "extracted_relationships": request.get("extracted_relationships"),
            "view_names": [path.name for path in selected],
            "view_evidence": request.get("local_render_evidence_metadata"),
        }
        context_text = _budgeted_context_json(
            context,
            self.max_context_chars,
            priority_keys=(
                "metric",
                "detector_evidence",
                "event",
                "natural_language_prompt",
                "metric_rubric",
                "candidate_selection_policy",
                "collision_evidence_style_guide",
                "visual_evidence_policy",
                "view_names",
                "view_evidence",
            ),
        )
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": "Adjudicate this P0b event.\n" + context_text,
            }
        ]
        content.extend(
            {"type": "image_url", "image_url": {"url": _image_data_url(path)}}
            for path in selected
        )
        raw = self.model.chat_messages(
            [
                {"role": "system", "content": P0B_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            response_format_json=self.response_format_json,
            call_type=f"vlm_judge.p0b.{request.get('metric') or 'event'}",
        )
        result = parse_json_object(raw)
        if result.get("verdict") not in {"valid", "invalid"}:
            raise ValueError("P0b judge verdict must be exactly 'valid' or 'invalid'")
        _score(result.get("confidence"), "confidence")
        result["model"] = self.model.model_id
        result["endpoint"] = self.model.endpoint
        result["images_used"] = [str(path.resolve()) for path in selected]
        result["request_metadata"] = dict(self.model.last_request_metadata)
        return result

    def adjudicate_relation(self, request: dict) -> dict:
        if not isinstance(request, dict):
            raise TypeError("relationship judge request must be a JSON object")
        paths = [Path(str(value)).expanduser() for value in request.get("render_evidence", [])]
        selected = paths[: self.max_images]
        missing = [str(path) for path in selected if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"relationship render evidence does not exist: {missing}")
        if not selected:
            raise ValueError("relationship adjudication requires at least one rendered view")
        context = {
            "family": request.get("family"),
            "explicit_relation_claim": request.get("relation"),
            "natural_language_prompt": request.get("natural_language_prompt"),
            "detector_evidence": request.get("detector_evidence"),
            "involved_objects": request.get("involved_objects"),
            "canonical_scene": request.get("scene_summary"),
            "view_names": [path.name for path in selected],
        }
        context_text = _budgeted_context_json(
            context,
            self.max_context_chars,
            priority_keys=(
                "family",
                "detector_evidence",
                "explicit_relation_claim",
                "natural_language_prompt",
                "involved_objects",
                "view_names",
            ),
        )
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": "Adjudicate this explicit spatial relationship.\n" + context_text,
            }
        ]
        content.extend(
            {"type": "image_url", "image_url": {"url": _image_data_url(path)}}
            for path in selected
        )
        raw = self.model.chat_messages(
            [
                {"role": "system", "content": RELATION_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            response_format_json=self.response_format_json,
            call_type=f"vlm_judge.relationship.{request.get('family') or 'unknown'}",
        )
        result = parse_json_object(raw)
        if result.get("verdict") not in {"valid", "invalid"}:
            raise ValueError("relationship judge verdict must be exactly 'valid' or 'invalid'")
        _score(result.get("confidence"), "confidence")
        result["model"] = self.model.model_id
        result["endpoint"] = self.model.endpoint
        result["images_used"] = [str(path.resolve()) for path in selected]
        result["request_metadata"] = dict(self.model.last_request_metadata)
        return result

    def adjudicate_spatial_fidelity(self, request: dict) -> dict:
        if not isinstance(request, dict):
            raise TypeError("Spatial Fidelity judge request must be a JSON object")
        paths = [Path(str(value)).expanduser() for value in request.get("render_evidence", [])]
        selected = paths[: self.max_images]
        missing = [str(path) for path in selected if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Spatial Fidelity render evidence does not exist: {missing}")
        if not selected:
            raise ValueError("Spatial Fidelity adjudication requires at least one rendered view")
        context = {
            "metric": request.get("metric"),
            "event": request.get("event"),
            "detector_evidence": request.get("detector_evidence"),
            "involved_objects": request.get("involved_objects"),
            "canonical_scene": request.get("scene_summary"),
            "natural_language_prompt": request.get("natural_language_prompt"),
            "view_names": [path.name for path in selected],
        }
        context_text = _budgeted_context_json(
            context,
            self.max_context_chars,
            priority_keys=(
                "metric",
                "event",
                "detector_evidence",
                "involved_objects",
                "natural_language_prompt",
                "view_names",
            ),
        )
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": "Adjudicate this routed Spatial Fidelity event.\n" + context_text,
            }
        ]
        content.extend(
            {"type": "image_url", "image_url": {"url": _image_data_url(path)}}
            for path in selected
        )
        raw = self.model.chat_messages(
            [
                {"role": "system", "content": SPATIAL_FIDELITY_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            response_format_json=self.response_format_json,
            call_type=f"vlm_judge.spatial_fidelity.{request.get('metric') or 'event'}",
        )
        result = parse_json_object(raw)
        if result.get("verdict") not in {"valid", "invalid"}:
            raise ValueError("Spatial Fidelity judge verdict must be exactly 'valid' or 'invalid'")
        _score(result.get("confidence"), "confidence")
        result["model"] = self.model.model_id
        result["endpoint"] = self.model.endpoint
        result["images_used"] = [str(path.resolve()) for path in selected]
        result["request_metadata"] = dict(self.model.last_request_metadata)
        return result

    def select_camera_views(self, request: dict) -> dict:
        if not isinstance(request, dict):
            raise TypeError("camera selector request must be a JSON object")
        candidates = [item for item in request.get("candidates", []) if isinstance(item, dict)]
        if not candidates:
            raise ValueError("camera selector requires at least one candidate")
        usable_candidates = [
            item
            for item in candidates
            if str(item.get("render_status") or "ok") != "blank"
        ]
        if not usable_candidates:
            raise ValueError("camera selector received no non-blank candidate previews")
        internal_ids = [str(item.get("id") or "") for item in usable_candidates]
        if any(not value for value in internal_ids) or len(set(internal_ids)) != len(internal_ids):
            raise ValueError("camera selector candidates require unique non-empty internal IDs")
        candidate_paths = [Path(str(item.get("image_path"))).expanduser() for item in usable_candidates]
        missing = [str(path) for path in candidate_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"camera selector preview evidence does not exist: {missing}")
        selected_candidates = sorted(
            usable_candidates,
            key=_selector_candidate_order_key,
        )[: self.max_images]
        paths = [Path(str(item.get("image_path"))).expanduser() for item in selected_candidates]
        max_views = max(1, min(int(request.get("max_views") or 1), len(selected_candidates)))
        alias_to_internal = {
            f"candidate_{index:02d}": str(item.get("id"))
            for index, item in enumerate(selected_candidates)
        }
        aliases = list(alias_to_internal)
        allowed_actions = _selector_allowed_actions(request.get("allowed_actions"))
        allow_adjustment = bool(request.get("allow_adjustment")) and bool(allowed_actions)
        context = {
            "candidates": [
                {
                    "id": alias,
                    "pose": _minimal_selector_pose(item.get("pose")),
                }
                for alias, item in zip(aliases, selected_candidates)
            ],
            "max_views": max_views,
            "allow_adjustment": allow_adjustment,
            "allowed_actions": allowed_actions if allow_adjustment else [],
            "metric_family": _selector_metric_family(request.get("metric")),
            "preview_role": _selector_preview_role(request.get("preview_role")),
            "preview_warning_class": _selector_preview_warning_class(
                request.get("preview_visibility_warning")
                if request.get("preview_visibility_warning") is not None
                else request.get("preview_degradation")
            ),
            "color_legend": _sanitize_selector_legend(request.get("color_legend")),
        }
        context_text = _budgeted_context_json(
            context,
            self.max_context_chars,
            priority_keys=(
                "metric_family",
                "preview_role",
                "preview_warning_class",
                "color_legend",
                "candidates",
                "max_views",
                "allow_adjustment",
                "allowed_actions",
            ),
        )
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": "Select camera evidence only; do not adjudicate the event.\n" + context_text,
            }
        ]
        content.extend(
            {
                "type": "image_url",
                "image_url": {"url": _selector_image_data_url(path, alias=alias)},
            }
            for alias, path in zip(aliases, paths)
        )
        raw = self.model.chat_messages(
            [
                {"role": "system", "content": CAMERA_SELECTION_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            response_format_json=self.response_format_json,
            call_type="vlm_camera_pose.query_cov",
        )
        parsed = parse_json_object(raw)
        ids = parsed.get("selected_view_ids")
        available = set(alias_to_internal)
        if not isinstance(ids, list):
            raise ValueError("camera selector selected_view_ids must be a list")
        resolved_aliases = list(dict.fromkeys(str(value) for value in ids if str(value)))
        if (
            not resolved_aliases
            or len(resolved_aliases) > max_views
            or any(value not in available for value in resolved_aliases)
        ):
            raise ValueError("camera selector returned invalid selected_view_ids")
        action = parsed.get("action")
        resolved_action = None
        if action is not None:
            if not allow_adjustment or not isinstance(action, dict):
                raise ValueError("camera selector returned an action outside the adjustment contract")
            if str(action.get("view_id") or "") not in available:
                raise ValueError("camera selector action references an unknown candidate")
            if str(action.get("type") or "") not in set(allowed_actions):
                raise ValueError("camera selector requested an unsupported action")
            resolved_action = {
                "view_id": alias_to_internal[str(action["view_id"])],
                "type": str(action["type"]),
            }
        reason = parsed.get("reason")
        request_metadata = dict(self.model.last_request_metadata)
        request_metadata.update(
            {
                "selector_candidate_order_policy": "stable_pose_image_digest_v1",
                "selector_candidate_alias_policy": "per_request_sequential_alias_v1",
            }
        )
        return {
            "selected_view_ids": [alias_to_internal[value] for value in resolved_aliases],
            "action": resolved_action,
            "reason": str(reason)[:1000] if reason is not None else "",
            "model": self.model.model_id,
            "endpoint": self.model.endpoint,
            # Preserve an auditable image count without exposing local names or paths.
            "images_used": aliases,
            "request_metadata": request_metadata,
        }


def build_openai_compatible_vlm_judge(config: dict[str, Any]) -> OpenAICompatibleVLMJudge:
    if not isinstance(config, dict):
        raise TypeError("VLM judge config must be a JSON object")
    if "api_key" in config:
        raise ValueError(
            "VLM judge config must not contain literal api_key; use api_key_env instead"
        )
    endpoint = config.get("endpoint") or config.get("base_url")
    model_id = config.get("model") or config.get("model_id")
    if not endpoint or not model_id:
        raise ValueError("VLM judge config requires endpoint and model")
    model = OpenAICompatibleModel(
        name=str(config.get("name") or "openai-compatible-vlm-judge"),
        endpoint=str(endpoint),
        model_id=str(model_id),
        api_key_env=str(config["api_key_env"]) if config.get("api_key_env") else None,
        temperature=float(config.get("temperature", 0.0)),
        max_tokens=int(config.get("max_tokens", 2048)),
        context_length=int(config["context_length"]) if config.get("context_length") is not None else None,
        timeout_seconds=int(config.get("timeout_seconds", 300)),
        response_format_json=bool(config.get("response_format_json", True)),
        max_retries=int(config.get("max_retries", 1)),
        retry_backoff_seconds=float(config.get("retry_backoff_seconds", 1.0)),
        max_tokens_field=str(config.get("max_tokens_field", "max_tokens")),
        send_temperature=bool(config.get("send_temperature", True)),
        require_api_key=(
            bool(config["require_api_key"])
            if config.get("require_api_key") is not None
            else None
        ),
    )
    return OpenAICompatibleVLMJudge(
        model,
        max_images=int(config.get("max_images", 6)),
        max_context_chars=int(config.get("max_context_chars", 30000)),
        response_format_json=bool(config.get("response_format_json", True)),
    )


def _image_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _selector_image_data_url(path: Path, *, alias: str) -> str:
    """Return a metadata-free RGB PNG for the external camera selector."""

    try:
        from PIL import Image, ImageOps, UnidentifiedImageError
    except ImportError:  # pragma: no cover - exercised only in minimal installs
        return _image_data_url(path)

    try:
        with Image.open(path) as source:
            source.load()
            normalized = ImageOps.exif_transpose(source).convert("RGBA")
            flattened = Image.new("RGB", normalized.size, (255, 255, 255))
            flattened.paste(normalized, mask=normalized.getchannel("A"))
            output = BytesIO()
            flattened.save(output, format="PNG")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(
            f"camera selector preview {alias} is not a valid decodable image"
        ) from exc
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _selector_candidate_order_key(candidate: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            _minimal_selector_pose(candidate.get("pose")),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    path = Path(str(candidate.get("image_path"))).expanduser()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selector_metric_family(value: Any) -> str:
    metric = str(value or "").strip().lower()
    allowed = {
        "collision",
        "oob",
        "support",
        "object_architecture_penetration",
    }
    if metric not in allowed:
        raise ValueError(f"camera selector does not support metric family {metric!r}")
    return metric


def _selector_preview_role(value: Any) -> str:
    role = str(value or "").strip().lower()
    return role if role in {"highlighted_focus", "rgb_fallback"} else "unspecified"


def _selector_preview_warning_class(value: Any) -> str | None:
    if value is None or not str(value).strip():
        return None
    warning = str(value).lower()
    if "blank" in warning:
        return "blank_preview"
    if "visib" in warning or "highlight" in warning or "coverage" in warning:
        return "incomplete_target_visibility"
    return "preview_warning"


def _selector_allowed_actions(value: Any) -> list[str]:
    allowed = {
        "orbit_left",
        "orbit_right",
        "elevate",
        "lower",
        "dolly_in",
        "dolly_out",
    }
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(str(item) for item in value if str(item) in allowed))


def _minimal_selector_pose(value: Any) -> dict[str, Any]:
    pose = value if isinstance(value, dict) else {}
    result: dict[str, Any] = {}
    projection = str(pose.get("camera_type") or "PERSP").strip().upper()
    result["projection"] = "orthographic" if projection == "ORTHO" else "perspective"
    for source, target in (
        ("azimuth_degrees", "azimuth_degrees"),
        ("elevation_degrees", "elevation_degrees"),
        ("lens_mm", "lens_mm"),
        ("ortho_scale", "ortho_scale"),
    ):
        raw = pose.get(source)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool) and math.isfinite(float(raw)):
            result[target] = round(float(raw), 3)
    return result


def _sanitize_selector_legend(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    for item in value[:12]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        if role.startswith("related_target_"):
            role = "related_target"
        if role not in {
            "object_a",
            "object_b",
            "primary_target",
            "related_target",
            "architecture_plane",
            "measured_support_gap",
        }:
            role = "annotation"
        entry: dict[str, Any] = {"role": role}
        color = item.get("color")
        if isinstance(color, (list, tuple)) and len(color) >= 3:
            channels: list[float] = []
            for channel in color[:3]:
                if not isinstance(channel, (int, float)) or isinstance(channel, bool):
                    channels = []
                    break
                numeric = float(channel)
                if not math.isfinite(numeric):
                    channels = []
                    break
                channels.append(round(min(1.0, max(0.0, numeric)), 4))
            if channels:
                entry["rgb"] = channels
        representation = str(item.get("representation") or "").lower()
        if "mesh" in representation:
            entry["representation"] = "mesh"
        elif any(token in representation for token in ("obb", "bbox", "proxy")):
            entry["representation"] = "proxy"
        elif any(token in representation for token in ("plane", "boundary")):
            entry["representation"] = "architecture"
        elif representation:
            entry["representation"] = "annotation"
        result.append(entry)
    return result


def _budgeted_context_json(
    context: dict[str, Any],
    max_chars: int,
    *,
    priority_keys: tuple[str, ...] = (),
) -> str:
    """Serialize a valid JSON context while preserving high-value evidence.

    Raw prefix slicing can both corrupt JSON and drop a detector packet merely
    because a large scene summary appeared first.  Required/priority fields are
    therefore considered first, optional fields are added only while they fit,
    and an oversized priority value is represented by an explicit JSON-prefix
    summary instead of truncating the entire document.
    """

    limit = max(1, int(max_chars))
    full = _compact_json(context)
    if len(full) <= limit:
        return full

    marker = {"truncated": True}
    marker_key = "_benchmark_context_budget"
    priority = [key for key in priority_keys if key in context]
    priority_set = set(priority)
    # Seed every priority key before allocating the remaining budget. This
    # prevents one oversized earlier field (for example a prompt or raw
    # relation) from consuming the budget and deleting detector evidence.
    skeleton = {**{key: None for key in priority}, marker_key: marker}
    skeleton_size = len(_compact_json(skeleton))
    per_key_extra = max(0, limit - skeleton_size) // max(len(priority), 1)
    seed_limit = len("null") + per_key_extra
    result: dict[str, Any] = {
        key: _priority_seed_value(context[key], seed_limit)
        for key in priority
    }
    if len(_compact_json({**result, marker_key: marker})) > limit:
        minimal = _compact_json({marker_key: marker})
        return minimal if len(minimal) <= limit else "{}"

    # Enrich priority fields in declared order while all other priority keys
    # remain reserved in the document.
    for key in priority:
        value = context[key]
        trial = {**result, key: value, marker_key: marker}
        if len(_compact_json(trial)) <= limit:
            result[key] = value
            continue
        empty_trial = {**result, key: None, marker_key: marker}
        allowance = limit - len(_compact_json(empty_trial)) + len("null")
        compacted = _truncated_json_value(value, allowance)
        if compacted is not None:
            compacted_trial = {**result, key: compacted, marker_key: marker}
            if len(_compact_json(compacted_trial)) <= limit:
                result[key] = compacted

    # Optional fields never displace reserved evidence.
    for key, value in context.items():
        if key in priority_set:
            continue
        trial = {**result, key: value, marker_key: marker}
        if len(_compact_json(trial)) <= limit:
            result[key] = value

    rendered = _compact_json({**result, marker_key: marker})
    # ``max_context_chars`` is clamped to at least 1000 by the judge, but keep
    # this helper total for direct unit use with tiny budgets.
    if len(rendered) <= limit:
        return rendered
    minimal = _compact_json({marker_key: marker})
    return minimal if len(minimal) <= limit else "{}"


def _truncated_json_value(value: Any, max_chars: int) -> dict[str, Any] | None:
    if max_chars <= 0:
        return None
    raw = _compact_json(value)
    base = {
        "_truncated": True,
        "original_chars": len(raw),
        "json_prefix": "",
    }
    if len(_compact_json(base)) > max_chars:
        return None
    low, high = 0, len(raw)
    while low < high:
        middle = (low + high + 1) // 2
        candidate = {**base, "json_prefix": raw[:middle]}
        if len(_compact_json(candidate)) <= max_chars:
            low = middle
        else:
            high = middle - 1
    return {**base, "json_prefix": raw[:low]}


def _priority_seed_value(value: Any, max_chars: int) -> Any:
    raw = _compact_json(value)
    if len(raw) <= max_chars:
        return value
    compacted = _truncated_json_value(value, max_chars)
    if compacted is not None:
        return compacted
    marker = {"_truncated": True}
    return marker if len(_compact_json(marker)) <= max_chars else None


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _score(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"VLM judge {name} must be numeric")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"VLM judge {name} must be between 0 and 1")
    return result
