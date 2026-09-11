"""Shared validation for optional object-identity visual grounding.

Identity overlays are evidence infrastructure, not metric evidence and not a
source of semantic decisions.  Discovery components may consume one trusted
overlay plus a complete alias-to-object-ID legend so repeated categories remain
unambiguous.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


def validate_identity_evidence(
    *,
    image_path: Any,
    legend: Any,
    expected_object_ids: Iterable[str],
    label: str,
) -> dict[str, Any]:
    """Validate an optional all-or-nothing identity overlay contract."""

    path_supplied = image_path is not None and bool(str(image_path).strip())
    legend_supplied = legend is not None and legend != {}
    if not path_supplied and not legend_supplied:
        return {
            "identity_image_path": None,
            "identity_legend": {},
            "identity_grounding": "unavailable",
        }
    if path_supplied != legend_supplied:
        raise ValueError(
            f"{label} identity_image_path and identity_legend must be "
            "provided together"
        )
    path = Path(str(image_path)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"{label} identity overlay does not exist: {path}"
        )
    if not isinstance(legend, dict) or not legend:
        raise ValueError(f"{label} identity_legend must be a non-empty object")
    normalized: dict[str, str] = {}
    for raw_alias, raw_object_id in legend.items():
        alias = str(raw_alias).strip()
        object_id = str(raw_object_id).strip()
        if not alias or not object_id or alias in normalized:
            raise ValueError(
                f"{label} identity_legend requires unique non-empty aliases "
                "and object IDs"
            )
        normalized[alias] = object_id
    values = list(normalized.values())
    if len(values) != len(set(values)):
        raise ValueError(
            f"{label} identity_legend must map one alias to each object ID"
        )
    expected = {
        str(object_id).strip()
        for object_id in expected_object_ids
        if str(object_id).strip()
    }
    if set(values) != expected:
        raise ValueError(
            f"{label} identity_legend must cover exactly the input object IDs"
        )
    return {
        "identity_image_path": str(path),
        "identity_legend": normalized,
        "identity_grounding": "trusted_overlay_and_complete_legend",
    }
