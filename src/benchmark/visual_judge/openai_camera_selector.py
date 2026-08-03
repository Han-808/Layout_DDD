from __future__ import annotations

import base64
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Mapping

from benchmark.models import OpenAICompatibleModel, parse_json_object


CAMERA_SELECTOR_SYSTEM_PROMPT = """You select visual evidence, not a benchmark
verdict. Follow selection_mode exactly. Never judge validity or quality, never
return a metric score, never change the scene or group membership, and never
invent a candidate, plan, action, or pose.

candidate_only response:
{"selected_view_ids":["trusted_candidate_id"],"reason":"...",
"provenance":{}}

repair_plan response:
{"selected_plan_id":"trusted_plan_id","reason":"...","provenance":{}}

freeform_pose response:
{"camera_proposal":{"location":[0,0,0],"target":[0,0,0],
"lens_mm":50,"camera_type":"PERSP"},"reason":"...","provenance":{}}

Return exactly one JSON object and no other fields."""

_MODE_FIELDS = {
    "candidate_only": frozenset(
        {"selected_view_ids", "reason", "provenance"}
    ),
    "repair_plan": frozenset(
        {"selected_plan_id", "reason", "provenance"}
    ),
    "freeform_pose": frozenset(
        {"camera_proposal", "reason", "provenance"}
    ),
}
_FORBIDDEN_DECISION_OR_MUTATION_KEYS = frozenset(
    {
        "verdict",
        "score",
        "metric_verdict",
        "metric_score",
        "scene_mutation",
        "mutated_scene",
        "scene_patch",
    }
)
_FORBIDDEN_RESPONSE_KEYS = (
    _FORBIDDEN_DECISION_OR_MUTATION_KEYS
    | frozenset(
        {
            "status",
            "group_members",
            "member_ids",
        }
    )
)


class OpenAICompatibleCameraSelector:
    """Production transport for one ActiveVLMCameraSelector policy call."""

    backend = "openai_compatible_camera_selector"
    production_camera_selector_transport = True
    requires_candidate_previews = True

    def __init__(
        self,
        model: OpenAICompatibleModel,
        *,
        max_preview_images: int = 8,
        max_context_chars: int = 30000,
        response_format_json: bool | None = None,
    ) -> None:
        self.model = model
        self.max_preview_images = _positive_int(
            max_preview_images,
            "max_preview_images",
        )
        # Legacy P0b providers inspect ``max_images`` and call
        # ``select_camera_views``.  Keep that public compatibility surface on
        # the dedicated selector transport so the CLI does not have to build a
        # metric Judge merely to select camera evidence.
        self.max_images = self.max_preview_images
        self.max_context_chars = _positive_int(
            max_context_chars,
            "max_context_chars",
        )
        self.response_format_json = (
            bool(getattr(model, "response_format_json", True))
            if response_format_json is None
            else bool(response_format_json)
        )
        self.last_request_metadata: dict[str, Any] = {}

    def select(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise TypeError(
                "OpenAICompatibleCameraSelector requires a payload mapping"
            )
        mode = str(payload.get("selection_mode") or "").strip()
        if mode not in _MODE_FIELDS:
            raise ValueError(
                "camera selector selection_mode must be candidate_only, "
                "repair_plan, or freeform_pose"
            )
        structured = _validated_payload(payload, mode=mode)
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Select camera evidence from this trusted request.\n"
                    + _bounded_json(
                        _payload_without_image_paths(structured),
                        limit=self.max_context_chars,
                    )
                ),
            }
        ]
        preview_aliases: list[str] = []
        if mode == "candidate_only":
            candidates = structured["candidate_views"]
            if len(candidates) > self.max_preview_images:
                raise ValueError(
                    "trusted candidate bank exceeds configured preview-image "
                    "capacity; refusing to silently truncate it"
                )
            for candidate in candidates:
                alias = str(candidate["id"])
                preview_aliases.append(alias)
                content.extend(
                    [
                        {
                            "type": "text",
                            "text": f"Trusted candidate preview: {alias}",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": _normalized_image_data_url(
                                    Path(candidate["image_path"]),
                                    alias=alias,
                                )
                            },
                        },
                    ]
                )
        raw_text = self.model.chat_messages(
            [
                {
                    "role": "system",
                    "content": CAMERA_SELECTOR_SYSTEM_PROMPT,
                },
                {"role": "user", "content": content},
            ],
            response_format_json=self.response_format_json,
            call_type=f"camera_selector_{mode}",
            case={
                "case_id": str(
                    structured.get("group_scope", {}).get("group_id")
                    if isinstance(
                        structured.get("group_scope"), dict
                    )
                    else structured.get("metric") or "camera_selection"
                ),
                "scene_id": str(
                    structured.get("scene", {}).get("scene_id")
                    if isinstance(structured.get("scene"), dict)
                    else ""
                ),
                "objects": (
                    structured.get("scene", {}).get("objects", [])
                    if isinstance(structured.get("scene"), dict)
                    else []
                ),
            },
        )
        response = _validated_response(
            parse_json_object(raw_text),
            mode=mode,
            payload=structured,
        )
        self.last_request_metadata = {
            "selection_mode": mode,
            "candidate_ids": [
                str(candidate["id"])
                for candidate in structured.get("candidate_views", [])
            ],
            "preview_aliases": preview_aliases,
            "trusted_plan_ids": [
                str(plan["plan_id"])
                for plan in structured.get(
                    "trusted_repair_plans", []
                )
            ],
            "model": getattr(self.model, "model_id", None),
            "endpoint": getattr(self.model, "endpoint", None),
            "transport": (
                f"{type(self).__module__}.{type(self).__qualname__}"
            ),
        }
        return response

    def select_camera_views(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        """Serve the frozen P0b selector contract through this transport.

        The compatibility implementation remains the established,
        camera-only prompt and validator; only its construction boundary moves
        from the metric-Judge builder to the dedicated camera-selector
        builder.
        """

        from benchmark.visual_judge.openai_compatible import (
            select_openai_compatible_camera_views,
        )

        result = select_openai_compatible_camera_views(
            model=self.model,
            request=request,
            max_images=self.max_preview_images,
            max_context_chars=self.max_context_chars,
            response_format_json=self.response_format_json,
        )
        self.last_request_metadata = {
            **dict(result.get("request_metadata") or {}),
            "selection_mode": "legacy_query_cov",
            "transport": (
                f"{type(self).__module__}.{type(self).__qualname__}"
            ),
        }
        return result


def build_openai_compatible_camera_selector(
    config: dict[str, Any],
) -> OpenAICompatibleCameraSelector:
    if not isinstance(config, dict):
        raise TypeError("VLM camera selector config must be a JSON object")
    if "api_key" in config:
        raise ValueError(
            "VLM camera selector config must not contain literal api_key; "
            "use api_key_env instead"
        )
    endpoint = config.get("endpoint") or config.get("base_url")
    model_id = config.get("model") or config.get("model_id")
    if not endpoint or not model_id:
        raise ValueError(
            "VLM camera selector config requires endpoint and model"
        )
    model = OpenAICompatibleModel(
        name=str(
            config.get("name")
            or "openai-compatible-camera-selector"
        ),
        endpoint=str(endpoint),
        model_id=str(model_id),
        api_key_env=(
            str(config["api_key_env"])
            if config.get("api_key_env")
            else None
        ),
        temperature=float(config.get("temperature", 0.0)),
        max_tokens=int(config.get("max_tokens", 1024)),
        context_length=(
            int(config["context_length"])
            if config.get("context_length") is not None
            else None
        ),
        timeout_seconds=int(config.get("timeout_seconds", 300)),
        response_format_json=bool(
            config.get("response_format_json", True)
        ),
        max_retries=int(config.get("max_retries", 1)),
        retry_backoff_seconds=float(
            config.get("retry_backoff_seconds", 1.0)
        ),
        max_tokens_field=str(
            config.get("max_tokens_field", "max_tokens")
        ),
        send_temperature=bool(
            config.get("send_temperature", True)
        ),
        require_api_key=(
            bool(config["require_api_key"])
            if config.get("require_api_key") is not None
            else None
        ),
    )
    return OpenAICompatibleCameraSelector(
        model,
        max_preview_images=int(
            config.get(
                "max_preview_images",
                config.get("max_images", 8),
            )
        ),
        max_context_chars=int(
            config.get("max_context_chars", 30000)
        ),
        response_format_json=bool(
            config.get("response_format_json", True)
        ),
    )


def _validated_payload(
    payload: dict[str, Any],
    *,
    mode: str,
) -> dict[str, Any]:
    result = deepcopy(payload)
    if _contains_forbidden_key(
        result,
        forbidden=_FORBIDDEN_DECISION_OR_MUTATION_KEYS,
    ):
        raise ValueError(
            "camera selector request contains metric-decision, mutation, "
            "or group-membership authority"
        )
    constraints = result.get("camera_constraints")
    if not isinstance(constraints, dict):
        raise ValueError(
            "camera selector requires structured camera_constraints"
        )
    targets = result.get("target_ids")
    if (
        not isinstance(targets, list)
        or not targets
        or any(not str(value).strip() for value in targets)
    ):
        raise ValueError(
            "camera selector requires non-empty target_ids"
        )
    if mode == "candidate_only":
        candidates = result.get("candidate_views")
        if (
            not isinstance(candidates, list)
            or not candidates
            or any(not isinstance(item, dict) for item in candidates)
        ):
            raise ValueError(
                "candidate_only requires trusted candidate_views"
            )
        seen: set[str] = set()
        for candidate in candidates:
            candidate_id = str(candidate.get("id") or "").strip()
            if not candidate_id or candidate_id in seen:
                raise ValueError(
                    "candidate_only requires unique candidate IDs"
                )
            seen.add(candidate_id)
            if candidate.get("technical_feasibility") is not True:
                raise ValueError(
                    "candidate_only accepts only technically feasible views"
                )
            if not isinstance(candidate.get("pose"), dict):
                raise ValueError(
                    "candidate_only candidate requires a trusted pose"
                )
            if candidate.get("render_status") != "ok":
                raise ValueError(
                    "candidate_only requires successful candidate previews"
                )
            path = str(candidate.get("image_path") or "").strip()
            if not path:
                raise ValueError(
                    "candidate_only candidate requires image_path"
                )
    elif mode == "repair_plan":
        plans = result.get("trusted_repair_plans")
        if (
            not isinstance(plans, list)
            or not plans
            or any(not isinstance(item, dict) for item in plans)
        ):
            raise ValueError(
                "repair_plan requires trusted_repair_plans"
            )
        plan_ids = [
            str(item.get("plan_id") or "").strip() for item in plans
        ]
        if any(not value for value in plan_ids) or len(plan_ids) != len(
            set(plan_ids)
        ):
            raise ValueError(
                "repair_plan requires unique trusted plan IDs"
            )
    return result


def _validated_response(
    value: Any,
    *,
    mode: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("camera selector response must be a JSON object")
    result = deepcopy(dict(value))
    unknown = set(result) - _MODE_FIELDS[mode]
    if unknown or _contains_forbidden_key(
        result,
        forbidden=_FORBIDDEN_RESPONSE_KEYS,
    ):
        raise ValueError(
            "camera selector response contains forbidden or unknown fields: "
            f"{sorted(unknown)}"
        )
    reason = result.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("camera selector response requires reason")
    provenance = result.get("provenance")
    if provenance is not None and not isinstance(provenance, dict):
        raise ValueError(
            "camera selector response provenance must be an object"
        )
    if mode == "candidate_only":
        selected = result.get("selected_view_ids")
        if (
            not isinstance(selected, list)
            or not selected
            or any(not isinstance(item, str) or not item for item in selected)
            or len(selected) != len(set(selected))
        ):
            raise ValueError(
                "candidate_only response requires unique selected_view_ids"
            )
        trusted = {
            str(item["id"]) for item in payload["candidate_views"]
        }
        unknown_ids = set(selected) - trusted
        if unknown_ids:
            raise ValueError(
                "camera selector selected an untrusted candidate ID"
            )
    elif mode == "repair_plan":
        selected = result.get("selected_plan_id")
        trusted = {
            str(item["plan_id"])
            for item in payload["trusted_repair_plans"]
        }
        if not isinstance(selected, str) or selected not in trusted:
            raise ValueError(
                "camera selector selected an untrusted repair-plan ID"
            )
    elif not isinstance(result.get("camera_proposal"), dict):
        raise ValueError(
            "freeform_pose response requires camera_proposal"
        )
    return result


def _payload_without_image_paths(
    payload: dict[str, Any],
) -> dict[str, Any]:
    result = deepcopy(payload)
    for candidate in result.get("candidate_views", []):
        if isinstance(candidate, dict):
            candidate.pop("image_path", None)
    return result


def _bounded_json(value: Any, *, limit: int) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    if len(encoded) > limit:
        raise ValueError(
            "camera selector structured context exceeds max_context_chars"
        )
    return encoded


def _normalized_image_data_url(path: Path, *, alias: str) -> str:
    try:
        from PIL import Image, ImageOps, UnidentifiedImageError
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "Pillow is required for camera selector previews"
        ) from exc
    try:
        with Image.open(path.expanduser()) as source:
            source.load()
            normalized = ImageOps.exif_transpose(source).convert("RGBA")
            flattened = Image.new(
                "RGB", normalized.size, (255, 255, 255)
            )
            flattened.paste(
                normalized, mask=normalized.getchannel("A")
            )
            output = BytesIO()
            flattened.save(output, format="PNG")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(
            f"candidate preview {alias!r} is not a decodable image"
        ) from exc
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _contains_forbidden_key(
    value: Any,
    *,
    forbidden: frozenset[str],
) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).strip().lower() in forbidden:
                return True
            if _contains_forbidden_key(
                nested,
                forbidden=forbidden,
            ):
                return True
    elif isinstance(value, (list, tuple)):
        return any(
            _contains_forbidden_key(item, forbidden=forbidden)
            for item in value
        )
    return False


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value
