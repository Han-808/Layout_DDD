"""VLM decoding of directed usable surfaces from trusted side previews."""

from __future__ import annotations

import base64
import hashlib
import json
import math
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from benchmark.models import OpenAICompatibleModel, parse_json_object
from benchmark.visual_judge.functional_discovery import (
    FUNCTIONAL_SURFACE_ROLES,
)
from benchmark.visual_judge.roles import (
    DecisionContract,
    VLMRole,
    vlm_audit_metadata,
)


USABLE_SURFACE_SCHEMA_VERSION = "usable_surface_decode_v2"
USABLE_SURFACE_PROMPT_VERSION = "usable_surface_decoder_v2"
USABLE_SURFACE_MAX_TOKENS = 1200
USABLE_SURFACE_DETECTOR_INTERFACE_VERSION = (
    "usable_surface_detector_v1"
)
USABLE_SURFACE_EVIDENCE_LOOP_VERSION = (
    "usable_surface_evidence_loop_v1"
)
USABLE_SURFACE_MAX_EVIDENCE_ROUNDS = 2
DEFAULT_USABLE_SURFACE_DETECTOR_BACKEND = "vlm_trusted_side_ids"
USABLE_SURFACE_SIDE_IDS = (
    "local_pos_x",
    "local_neg_x",
    "local_pos_y",
    "local_neg_y",
)
USABLE_SURFACE_STATUSES = frozenset(
    {
        "identified",
        "ambiguous",
        "no_directed_surface",
        "insufficient_comparison",
        "surface_unavailable",
    }
)
USABLE_SURFACE_PENDING_STATUSES = frozenset(
    {
        "ambiguous",
        "insufficient_comparison",
        "surface_unavailable",
    }
)
USABLE_SURFACE_TERMINAL_STATUSES = (
    USABLE_SURFACE_STATUSES - USABLE_SURFACE_PENDING_STATUSES
)

_TOP_LEVEL_FIELDS = frozenset({"status", "surfaces", "reason"})
_SURFACE_FIELDS = frozenset(
    {"surface_role", "side_id", "visual_cues", "confidence"}
)
_FORBIDDEN_FIELDS = frozenset(
    {
        "verdict",
        "validity",
        "score",
        "defect",
        "defects",
        "metric_verdict",
        "metric_score",
        "camera",
        "camera_pose",
        "pose",
        "location",
        "target",
        "lens",
        "lens_mm",
        "rotation",
        "direction",
        "normal",
        "azimuth",
        "elevation",
        "scene_patch",
        "scene_mutation",
        "mutation",
    }
)


USABLE_SURFACE_SYSTEM_PROMPT = """Decode an object's directed usable surface
from the available trusted local-side previews. Do not judge any benchmark
metric and do not select or author a camera pose.

The preview IDs are fixed object-local sides:
local_pos_x, local_neg_x, local_pos_y, local_neg_y.
Select only these trusted IDs.

Use status:
- identified: one side is visually supported as the requested usable surface;
- ambiguous: one or two available sides remain plausible and complementary
  evidence is needed;
- no_directed_surface: the object has no visually supported directed surface
  of the requested role; this status is allowed only when bank_complete=true;
- insufficient_comparison: the available subset cannot support a trustworthy
  comparison.

Use only requested_surface_roles. Surface roles describe access, opening,
control, display, seating, interaction, reflective, or service sides. A
surface result is an observation hypothesis, never evidence that the authored
placement is valid or invalid.

For identified return exactly one available side. For ambiguous return one or
two different available sides. For no_directed_surface or
insufficient_comparison return no surfaces. Never return surface_unavailable;
the deterministic adapter owns the zero-preview case.
Each visual_cues list must describe visible geometry or appearance cues only.

Return exactly:
{"status":"identified",
"surfaces":[{"surface_role":"access_side","side_id":"local_pos_x",
"visual_cues":["visible cue"],"confidence":0.0}],
"reason":"brief surface-identification explanation"}

Return no other fields. Never return a verdict, defect, score, pose, vector,
camera action, scene edit, or claim that evidence is sufficient."""


@runtime_checkable
class UsableSurfaceDetector(Protocol):
    """Decision-free detector over a trusted object-local side bank."""

    implementation_id: str
    version: str

    def detect(self, request: dict[str, Any]) -> dict[str, Any]:
        ...

    def manifest(self) -> dict[str, Any]:
        ...


class VLMTrustedSideUsableSurfaceDetector:
    """Default adapter around the existing trusted-side VLM decoder."""

    implementation_id = DEFAULT_USABLE_SURFACE_DETECTOR_BACKEND
    version = USABLE_SURFACE_PROMPT_VERSION

    def __init__(
        self,
        decoder: Any,
        *,
        configuration: dict[str, Any] | None = None,
    ) -> None:
        call = getattr(decoder, "decode_usable_surface", None)
        if not callable(call):
            raise TypeError(
                "vlm_trusted_side_ids requires "
                "decode_usable_surface(request)"
            )
        self.decoder = decoder
        self.configuration = deepcopy(configuration or {})

    def detect(self, request: dict[str, Any]) -> dict[str, Any]:
        normalized = validate_usable_surface_request(request)
        raw = self.decoder.decode_usable_surface(
            {
                "scene_id": normalized.get("scene_id"),
                "target_id": normalized["target_id"],
                "target_category": normalized["target_category"],
                "surface_roles": list(normalized["surface_roles"]),
                "previews": list(deepcopy(normalized["previews"])),
            }
        )
        if not isinstance(raw, dict):
            raise ValueError(
                "usable-surface detector result must be a JSON object"
            )
        validated = validate_usable_surface_response(
            {
                key: deepcopy(raw.get(key))
                for key in ("status", "surfaces", "reason")
            },
            allowed_surface_roles=set(normalized["surface_roles"]),
            available_side_ids=set(normalized["available_side_ids"]),
            bank_complete=normalized["bank_complete"],
        )
        return {
            **validated,
            "schema_version": raw.get(
                "schema_version", USABLE_SURFACE_SCHEMA_VERSION
            ),
            "target_id": normalized["target_id"],
            "detector_implementation_id": self.implementation_id,
            "detector_version": self.version,
            "decision_authority": "none",
            "provenance": {
                **deepcopy(raw.get("provenance") or {}),
                "detector_interface_version": (
                    USABLE_SURFACE_DETECTOR_INTERFACE_VERSION
                ),
                "detector_implementation_id": self.implementation_id,
                "detector_version": self.version,
            },
        }

    def manifest(self) -> dict[str, Any]:
        model = getattr(self.decoder, "model", None)
        return {
            **vlm_audit_metadata(
                VLMRole.USABLE_SURFACE_DECODER,
                decision_contract=DecisionContract.USABLE_SURFACE_DECODE,
                judge_method="decode_usable_surface",
            ),
            "interface_version": (
                USABLE_SURFACE_DETECTOR_INTERFACE_VERSION
            ),
            "implementation_id": self.implementation_id,
            "version": self.version,
            "configuration": deepcopy(self.configuration),
            "model": (
                str(getattr(model, "model_id", ""))
                if model is not None
                else None
            ),
            "endpoint": (
                str(getattr(model, "endpoint", ""))
                if model is not None
                else None
            ),
            "decision_authority": "none",
        }


def build_usable_surface_detector(
    *,
    backend: str = DEFAULT_USABLE_SURFACE_DETECTOR_BACKEND,
    decoder: Any = None,
    configuration: dict[str, Any] | None = None,
) -> UsableSurfaceDetector:
    """Resolve one detector backend without an implicit fallback."""

    resolved = str(backend or "").strip()
    if resolved != DEFAULT_USABLE_SURFACE_DETECTOR_BACKEND:
        raise ValueError(
            "unsupported usable-surface detector backend: "
            f"{resolved!r}"
        )
    return VLMTrustedSideUsableSurfaceDetector(
        decoder,
        configuration=configuration,
    )


def decode_openai_compatible_usable_surface(
    *,
    model: OpenAICompatibleModel,
    request: dict[str, Any],
    max_context_chars: int = 12000,
    response_format_json: bool | None = None,
) -> dict[str, Any]:
    """Decode one target's local usable side from a trusted preview subset."""

    normalized = validate_usable_surface_request(request)
    audit = vlm_audit_metadata(
        VLMRole.USABLE_SURFACE_DECODER,
        decision_contract=DecisionContract.USABLE_SURFACE_DECODE,
        judge_method="decode_usable_surface",
    )
    context = {
        **audit,
        "role": "usable_surface_decoder",
        "decision_authority": "none",
        "scene_access": "read_only",
        "metric": "functional_consistency",
        "prompt_version": USABLE_SURFACE_PROMPT_VERSION,
        "scene_id": normalized.get("scene_id"),
        "target_id": normalized["target_id"],
        "target_category": normalized["target_category"],
        "requested_surface_roles": normalized["surface_roles"],
        "trusted_side_ids": list(normalized["available_side_ids"]),
        "bank_complete": normalized["bank_complete"],
    }
    context_text = json.dumps(
        context,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    if len(context_text) > max(1000, int(max_context_chars)):
        raise ValueError(
            "usable-surface context exceeds max_context_chars"
        )
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Decode the directed usable surface from this trusted "
                "request.\n" + context_text
            ),
        }
    ]
    for preview in normalized["previews"]:
        alias = preview["side_id"]
        content.extend(
            [
                {
                    "type": "text",
                    "text": f"Trusted local-side preview: {alias}",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _image_data_url(
                            Path(preview["image_path"]),
                            alias=alias,
                        )
                    },
                },
            ]
        )
    use_json_response = (
        bool(getattr(model, "response_format_json", True))
        if response_format_json is None
        else bool(response_format_json)
    )
    raw = model.chat_messages(
        [
            {
                "role": "system",
                "content": USABLE_SURFACE_SYSTEM_PROMPT,
            },
            {"role": "user", "content": content},
        ],
        response_format_json=use_json_response,
        call_type="vlm_camera_pose.usable_surface_decode",
        max_tokens=max(
            USABLE_SURFACE_MAX_TOKENS,
            int(getattr(model, "max_tokens", None) or 0),
        ),
        max_tokens_source="usable_surface_decoder_minimum",
        case={
            "case_id": str(
                normalized.get("scene_id") or normalized["target_id"]
            ),
            "scene_id": str(normalized.get("scene_id") or ""),
            "objects": [
                {
                    "id": normalized["target_id"],
                    "category": normalized["target_category"],
                }
            ],
        },
    )
    result = validate_usable_surface_response(
        parse_json_object(raw),
        allowed_surface_roles=set(normalized["surface_roles"]),
        available_side_ids=set(normalized["available_side_ids"]),
        bank_complete=normalized["bank_complete"],
    )
    return {
        **result,
        "schema_version": USABLE_SURFACE_SCHEMA_VERSION,
        "target_id": normalized["target_id"],
        "decision_authority": "none",
        "provenance": {
            **audit,
            "prompt_version": USABLE_SURFACE_PROMPT_VERSION,
            "backend": "openai_compatible",
            "trusted_side_ids": list(normalized["available_side_ids"]),
            "bank_complete": normalized["bank_complete"],
            "request_metadata": {
                **dict(getattr(model, "last_request_metadata", {})),
                **audit,
                "usable_surface_prompt_version": (
                    USABLE_SURFACE_PROMPT_VERSION
                ),
            },
        },
    }


def validate_usable_surface_request(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("usable-surface request must be a JSON object")
    target_id = str(value.get("target_id") or "").strip()
    category = str(value.get("target_category") or "").strip()
    if not target_id or not category:
        raise ValueError(
            "usable-surface request requires target_id and target_category"
        )
    roles = _validated_text_list(
        value.get("surface_roles"),
        allowed=FUNCTIONAL_SURFACE_ROLES,
        label="surface_roles",
    )
    raw_previews = value.get("previews")
    if (
        not isinstance(raw_previews, list)
        or not 1 <= len(raw_previews) <= len(USABLE_SURFACE_SIDE_IDS)
    ):
        raise ValueError(
            "usable-surface request requires one to four trusted previews"
        )
    by_id: dict[str, dict[str, Any]] = {}
    for item in raw_previews:
        if not isinstance(item, dict):
            raise ValueError(
                "usable-surface previews must be JSON objects"
            )
        unknown = set(item) - {"side_id", "image_path", "pose"}
        if unknown:
            raise ValueError(
                "usable-surface preview contains unsupported fields: "
                f"{sorted(unknown)}"
            )
        side_id = str(item.get("side_id") or "").strip()
        if side_id not in USABLE_SURFACE_SIDE_IDS or side_id in by_id:
            raise ValueError(
                "usable-surface preview IDs must be unique trusted local "
                "side IDs"
            )
        path = Path(str(item.get("image_path") or "")).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"usable-surface preview does not exist: {path}"
            )
        by_id[side_id] = {
            "side_id": side_id,
            "image_path": str(path),
            "pose": deepcopy(item.get("pose") or {}),
        }
    available_side_ids = [
        side_id for side_id in USABLE_SURFACE_SIDE_IDS if side_id in by_id
    ]
    declared_available = value.get("available_side_ids")
    if declared_available is not None and list(declared_available) != (
        available_side_ids
    ):
        raise ValueError(
            "usable-surface available_side_ids must match trusted previews"
        )
    bank_complete = len(available_side_ids) == len(
        USABLE_SURFACE_SIDE_IDS
    )
    if (
        value.get("bank_complete") is not None
        and bool(value.get("bank_complete")) != bank_complete
    ):
        raise ValueError(
            "usable-surface bank_complete must match trusted previews"
        )
    read_only_provenance = value.get("read_only_provenance")
    if (
        read_only_provenance is not None
        and (
            not isinstance(read_only_provenance, dict)
            or read_only_provenance.get("scene_access")
            != "read_only"
        )
    ):
        raise ValueError(
            "usable-surface read_only_provenance must declare "
            "scene_access=read_only"
        )
    return {
        "scene_id": (
            str(value["scene_id"])
            if value.get("scene_id") is not None
            else None
        ),
        "target_id": target_id,
        "target_category": category,
        "surface_roles": roles,
        "available_side_ids": available_side_ids,
        "bank_complete": bank_complete,
        "previews": [by_id[side_id] for side_id in available_side_ids],
        "read_only_provenance": deepcopy(
            read_only_provenance or {"scene_access": "read_only"}
        ),
    }


def validate_usable_surface_response(
    value: Any,
    *,
    allowed_surface_roles: set[str],
    available_side_ids: set[str] | None = None,
    bank_complete: bool = True,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(
            "usable-surface response must be a JSON object"
        )
    _reject_forbidden_fields(value)
    unknown = set(value) - _TOP_LEVEL_FIELDS
    if unknown:
        raise ValueError(
            "usable-surface response contains unsupported fields: "
            f"{sorted(unknown)}"
        )
    status = str(value.get("status") or "").strip()
    if status not in USABLE_SURFACE_STATUSES:
        raise ValueError(
            "usable-surface status is unsupported"
        )
    if status == "surface_unavailable" and available_side_ids:
        raise ValueError(
            "surface_unavailable requires an empty trusted preview bank"
        )
    if status == "no_directed_surface" and not bank_complete:
        raise ValueError(
            "no_directed_surface requires a complete trusted side bank"
        )
    surfaces = value.get("surfaces")
    if not isinstance(surfaces, list) or not all(
        isinstance(item, dict) for item in surfaces
    ):
        raise ValueError(
            "usable-surface surfaces must be a JSON list of objects"
        )
    valid_count = (
        len(surfaces) == 1
        if status == "identified"
        else 1 <= len(surfaces) <= 2
        if status == "ambiguous"
        else len(surfaces) == 0
    )
    if not valid_count:
        raise ValueError(
            f"usable-surface {status} returned an invalid surface count"
        )
    normalized: list[dict[str, Any]] = []
    seen_sides: set[str] = set()
    for item in surfaces:
        unknown_surface = set(item) - _SURFACE_FIELDS
        if unknown_surface:
            raise ValueError(
                "usable-surface hypothesis contains unsupported fields: "
                f"{sorted(unknown_surface)}"
            )
        role = str(item.get("surface_role") or "").strip()
        if role not in allowed_surface_roles:
            raise ValueError(
                "usable-surface response returned an unrequested surface role"
            )
        side_id = str(item.get("side_id") or "").strip()
        if (
            side_id not in USABLE_SURFACE_SIDE_IDS
            or side_id in seen_sides
        ):
            raise ValueError(
                "usable-surface response must use unique trusted side IDs"
            )
        if (
            available_side_ids is not None
            and side_id not in available_side_ids
        ):
            raise ValueError(
                "usable-surface response selected a side without an "
                "available trusted preview"
            )
        cues = item.get("visual_cues")
        if not isinstance(cues, list) or not cues or any(
            not isinstance(cue, str) or not cue.strip() for cue in cues
        ):
            raise ValueError(
                "usable-surface visual_cues must be non-empty strings"
            )
        confidence = item.get("confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise ValueError(
                "usable-surface confidence must be between 0 and 1"
            )
        seen_sides.add(side_id)
        normalized.append(
            {
                "surface_role": role,
                "side_id": side_id,
                "visual_cues": [str(cue).strip()[:300] for cue in cues],
                "confidence": float(confidence),
            }
        )
    reason = str(value.get("reason") or "").strip()
    if not reason:
        raise ValueError("usable-surface response requires reason")
    return {
        "status": status,
        "surfaces": normalized,
        "reason": reason[:1000],
        "bank_complete": bool(bank_complete),
        "available_side_ids": sorted(
            available_side_ids
            if available_side_ids is not None
            else set(USABLE_SURFACE_SIDE_IDS)
        ),
    }


def usable_surface_cache_identity(
    object_record: dict[str, Any],
    *,
    requested_surface_roles: list[str] | tuple[str, ...] | None = None,
    trusted_preview_identity: dict[str, Any] | None = None,
    detector_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a rotation-independent content identity for one asset surface."""

    asset_ref = (
        object_record.get("asset_ref")
        if isinstance(object_record.get("asset_ref"), dict)
        else {}
    )
    metadata = (
        object_record.get("metadata")
        if isinstance(object_record.get("metadata"), dict)
        else {}
    )
    asset_metadata = (
        metadata.get("asset_metadata")
        if isinstance(metadata.get("asset_metadata"), dict)
        else {}
    )
    catalog_hashes = (
        asset_metadata.get("catalog_hashes")
        if isinstance(asset_metadata.get("catalog_hashes"), dict)
        else {}
    )
    catalog_placement = (
        metadata.get("catalog_placement")
        if isinstance(metadata.get("catalog_placement"), dict)
        else {}
    )
    scene_materialization = (
        metadata.get("scene_materialization")
        if isinstance(metadata.get("scene_materialization"), dict)
        else {}
    )
    materialization = (
        metadata.get("materialization")
        if isinstance(metadata.get("materialization"), dict)
        else {}
    )
    asset_key = str(
        asset_ref.get("asset_key")
        or object_record.get("jid")
        or ""
    ).strip()
    geometry_hash = str(
        object_record.get("geometry_sha256")
        or metadata.get("geometry_sha256")
        or scene_materialization.get("geometry_sha256")
        or materialization.get("geometry_sha256")
        or catalog_placement.get("geometry_sha256")
        or ""
    ).strip()
    if not geometry_hash:
        # A source mesh digest alone is not a materialized-geometry identity:
        # the same catalog asset may be instantiated at a different scale.
        # This deterministic proxy includes only local geometry/materialization
        # facts and deliberately excludes world translation and rotation.
        geometry_payload = {
            "asset_key": asset_key,
            "source_mesh_sha256": catalog_hashes.get("mesh_sha256"),
            "size": object_record.get("size"),
            "asset_proxy": object_record.get("asset_proxy"),
            "geometry_provenance": object_record.get(
                "geometry_provenance"
            ),
            "actual_local_bbox_size_m": catalog_placement.get(
                "actual_local_bbox_size_m"
            ),
            "effective_uniform_scale": catalog_placement.get(
                "effective_uniform_scale"
            ),
        }
        geometry_hash = _canonical_json_sha256(geometry_payload)
    appearance_hash = str(
        object_record.get("appearance_sha256")
        or object_record.get("material_sha256")
        or metadata.get("appearance_sha256")
        or metadata.get("material_sha256")
        or scene_materialization.get("appearance_sha256")
        or scene_materialization.get("material_sha256")
        or materialization.get("appearance_sha256")
        or materialization.get("material_sha256")
        or catalog_hashes.get("material_sha256")
        or ""
    ).strip()
    if not appearance_hash:
        appearance_hash = _canonical_json_sha256(
            {
                "materials": object_record.get("materials"),
                "material_slots": object_record.get("material_slots"),
                "appearance": metadata.get("appearance"),
                "catalog_materials": asset_metadata.get("materials"),
                "texture_sha256": catalog_hashes.get("texture_sha256"),
            }
        )
    payload = {
        "asset_key": asset_key or "instance_geometry",
        "geometry_hash": geometry_hash,
        "appearance_hash": appearance_hash,
        "prompt_version": USABLE_SURFACE_PROMPT_VERSION,
        "requested_surface_roles": (
            sorted(
                {
                    str(item)
                    for item in requested_surface_roles or []
                    if str(item).strip()
                }
            )
            if requested_surface_roles is not None
            else None
        ),
        "trusted_preview_identity": deepcopy(
            trusted_preview_identity
        ),
        "detector": (
            _detector_cache_manifest(detector_manifest)
            if detector_manifest is not None
            else None
        ),
    }
    return {
        **payload,
        "cache_key": _canonical_json_sha256(payload),
    }


def trusted_side_preview_cache_identity(
    *,
    preview_render_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Describe the deterministic side bank without rendering it.

    The identity is deliberately asset-local: it excludes world translation
    and rotation while covering the trusted side IDs, side-bank policy, and
    preview-render configuration.
    """

    return {
        "policy": "usable_surface_bounded_comparison_loop_v1",
        "primary_bank": "usable_surface_local_side_bank_v1",
        "deterministic_fallback_bank": (
            "usable_surface_elevated_detail_repair_v1"
        ),
        "max_evidence_rounds": USABLE_SURFACE_MAX_EVIDENCE_ROUNDS,
        "trusted_side_ids": list(USABLE_SURFACE_SIDE_IDS),
        "preview_render_config": deepcopy(
            preview_render_config or {}
        ),
    }


def _detector_cache_manifest(
    value: dict[str, Any] | None,
) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    return {
        "implementation_id": str(
            source.get("implementation_id") or ""
        ),
        "version": str(source.get("version") or ""),
        "configuration": deepcopy(source.get("configuration") or {}),
        "model": source.get("model"),
        "endpoint": source.get("endpoint"),
    }


def _validated_text_list(
    value: Any,
    *,
    allowed: frozenset[str],
    label: str,
) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty JSON list")
    result = [str(item).strip() for item in value]
    if any(not item for item in result) or len(result) != len(set(result)):
        raise ValueError(f"{label} must contain unique non-empty strings")
    unknown = sorted(set(result) - allowed)
    if unknown:
        raise ValueError(f"{label} contains unsupported values: {unknown}")
    return result


def _reject_forbidden_fields(value: Any, *, path: str = "response") -> None:
    if isinstance(value, dict):
        for raw_key, item in value.items():
            key = str(raw_key).strip().lower()
            if key in _FORBIDDEN_FIELDS:
                raise ValueError(
                    f"usable-surface decoder may not return {path}.{raw_key}"
                )
            _reject_forbidden_fields(item, path=f"{path}.{raw_key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_forbidden_fields(item, path=f"{path}[{index}]")


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _image_data_url(path: Path, *, alias: str) -> str:
    try:
        from PIL import Image, ImageOps, UnidentifiedImageError
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Pillow is required to sanitize usable-surface previews"
        ) from exc
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
            f"usable-surface preview {alias!r} is not decodable"
        ) from exc
    return (
        "data:image/png;base64,"
        + base64.b64encode(output.getvalue()).decode("ascii")
    )
