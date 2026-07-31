"""Canonical visual-input protocol for VLM evidence-scope grouping."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


GROUPING_EVIDENCE_PROTOCOL_VERSION = "grouping_evidence_protocol_v1"


@dataclass(frozen=True)
class GroupingEvidencePacket:
    visual_evidence: tuple[dict[str, Any], ...]
    identity_legend: dict[str, str]
    input_mode: str
    available_roles: tuple[str, ...]
    degraded_reasons: tuple[str, ...]

    def provenance(self) -> dict[str, Any]:
        return {
            "protocol_version": GROUPING_EVIDENCE_PROTOCOL_VERSION,
            "input_mode": self.input_mode,
            "available_roles": list(self.available_roles),
            "degraded_reasons": list(self.degraded_reasons),
            "identity_legend_available": bool(self.identity_legend),
            "image_count": len(self.visual_evidence),
        }


def prepare_grouping_evidence(
    value: Any,
    *,
    identity_legend: dict[str, Any] | None = None,
) -> GroupingEvidencePacket:
    """Normalize role-aware grouping evidence without inventing identity data."""

    explicit_legend = _identity_legend(identity_legend)
    items = _evidence_items(value)
    normalized: list[dict[str, Any]] = []
    discovered_legend = dict(explicit_legend)
    for index, raw in enumerate(items):
        item = _normalize_item(raw, index=index)
        if item is None:
            continue
        item_legend = _identity_legend(item.get("identity_legend"))
        if item_legend:
            if discovered_legend and discovered_legend != item_legend:
                raise ValueError(
                    "grouping evidence contains conflicting identity legends"
                )
            discovered_legend = item_legend
        if (
            item.get("representation") == "identity_map"
            and discovered_legend
        ):
            item["identity_legend"] = deepcopy(discovered_legend)
        normalized.append(item)

    normalized = _deduplicate_by_path(normalized)
    roles = tuple(
        dict.fromkeys(str(item["role"]) for item in normalized)
    )
    role_set = set(roles)
    has_perspective = "global_perspective_rgb" in role_set
    has_top = "global_top_rgb" in role_set
    has_identity = "global_identity_overlay" in role_set
    degraded: list[str] = []
    if not has_perspective:
        degraded.append("global_perspective_missing")
    if not has_top:
        degraded.append("global_top_missing")
    if not has_identity:
        degraded.append("identity_map_missing")
    if has_identity and not discovered_legend:
        degraded.append("identity_legend_missing")

    if (
        has_perspective
        and has_top
        and has_identity
        and discovered_legend
    ):
        mode = "identity_aware_perspective_top"
    elif has_perspective and has_top:
        mode = "perspective_top_degraded_no_identity"
    elif normalized:
        mode = "generic_overview_degraded"
    else:
        mode = "no_grouping_visual_evidence"
    return GroupingEvidencePacket(
        visual_evidence=tuple(deepcopy(normalized)),
        identity_legend=discovered_legend,
        input_mode=mode,
        available_roles=roles,
        degraded_reasons=tuple(degraded),
    )


def grouping_evidence_from_render_manifest(
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract explicit perspective/top/identity roles from a render manifest."""

    if not isinstance(manifest, dict):
        raise TypeError("grouping render manifest must be a JSON object")
    result: list[dict[str, Any]] = []
    legend = _identity_legend(
        manifest.get("identity_legend")
        or manifest.get("object_identity_legend")
    )
    for item in manifest.get("views") or []:
        if not isinstance(item, dict):
            continue
        path = item.get("path") or item.get("image_path")
        if path is None or not str(path).strip():
            continue
        name = str(
            item.get("name")
            or item.get("role")
            or item.get("id")
            or ""
        ).strip()
        role, representation, view_id = _role_from_name(name)
        if role is None:
            continue
        record = {
            "path": str(path),
            "role": role,
            "representation": representation,
            "view_id": view_id,
            "camera_scope": "global",
        }
        if representation == "identity_map":
            record["identity_overlay"] = True
            if legend:
                record["identity_legend"] = deepcopy(legend)
        result.append(record)
    return result


def _evidence_items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(deepcopy(value))
    if not isinstance(value, dict):
        raise TypeError(
            "grouping visual evidence must be a list or role-keyed mapping"
        )
    for key in ("grouping_visual_evidence", "grouping"):
        selected = value.get(key)
        if isinstance(selected, (list, tuple)):
            return list(deepcopy(selected))
    result: list[Any] = []
    role_keys = (
        "global_perspective",
        "perspective",
        "global_top",
        "top",
        "identity_map",
        "identity_overlay",
    )
    for key in role_keys:
        selected = value.get(key)
        if isinstance(selected, (list, tuple)):
            result.extend(
                {"path": item, "role_hint": key}
                if isinstance(item, (str, Path))
                else deepcopy(item)
                for item in selected
            )
        elif isinstance(selected, (str, Path)):
            result.append({"path": selected, "role_hint": key})
    if result:
        return result
    for key in ("global", "global_context", "default", "all"):
        selected = value.get(key)
        if isinstance(selected, (list, tuple)):
            return list(deepcopy(selected))
    return []


def _normalize_item(
    value: Any,
    *,
    index: int,
) -> dict[str, Any] | None:
    if isinstance(value, (str, Path)):
        path = str(value)
        role, representation, view_id = _role_from_name(
            Path(path).stem
        )
        return {
            "path": path,
            "role": role or "generic_global_overview",
            "representation": representation or "rgb",
            "view_id": view_id or f"global_overview_{index:02d}",
            "camera_scope": "global",
        }
    if not isinstance(value, dict):
        return None
    path = value.get("path") or value.get("image_path")
    if path is None or not str(path).strip():
        return None
    role = str(value.get("role") or "").strip()
    representation = str(
        value.get("representation") or ""
    ).strip()
    view_id = str(value.get("view_id") or "").strip()
    if not role:
        inferred = _role_from_name(
            str(value.get("role_hint") or Path(str(path)).stem)
        )
        role = inferred[0] or "generic_global_overview"
        representation = representation or inferred[1] or "rgb"
        view_id = view_id or inferred[2] or f"global_overview_{index:02d}"
    result = {
        key: deepcopy(raw)
        for key, raw in value.items()
        if key
        in {
            "object_ids",
            "target_ids",
            "identity_overlay",
            "identity_legend",
        }
    }
    result.update(
        {
            "path": str(path),
            "role": role,
            "representation": representation or "rgb",
            "view_id": view_id or f"grouping_view_{index:02d}",
            "camera_scope": "global",
        }
    )
    return result


def _role_from_name(
    value: str,
) -> tuple[str | None, str | None, str | None]:
    name = str(value).strip().lower()
    if "identity" in name or "object_id" in name:
        return (
            "global_identity_overlay",
            "identity_map",
            "global_identity",
        )
    if "top" in name or "overhead" in name:
        return "global_top_rgb", "rgb", "global_top"
    if (
        "perspective" in name
        or "oblique" in name
        or name in {"global", "overview"}
    ):
        return (
            "global_perspective_rgb",
            "rgb",
            "global_perspective",
        )
    return None, None, None


def _identity_legend(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("grouping identity legend must be a JSON object")
    result = {
        str(alias): str(object_id)
        for alias, object_id in value.items()
        if str(alias).strip() and str(object_id).strip()
    }
    if len(result) != len(value):
        raise ValueError(
            "grouping identity legend requires non-empty alias and object ID"
        )
    return result


def _deduplicate_by_path(
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        path = str(item["path"])
        if path in seen:
            continue
        seen.add(path)
        result.append(item)
    return result
