"""LayoutGPT CSS-style/parsed output to canonical scene conversion."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmark.adapters.common.artifacts import read_json_source
from benchmark.adapters.common.assets import (
    AssetProvider,
    asset_fields,
    resolve_asset_record,
)
from benchmark.adapters.common.geometry import (
    build_scene,
    canonical_room,
    finite_float,
    shift_boundary_to_origin,
    shift_center,
)
from benchmark.scene_io.validate import ArtifactValidationError


CSS_OBJECT = re.compile(r"([^\n{}]+)\s*\{([^{}]+)\}")
CSS_FIELD = re.compile(r"([A-Za-z_]+)\s*:\s*(-?[0-9]+(?:\.[0-9]+)?)")


def convert_layout_gpt(
    source_path: Path,
    generation_input: dict,
    config: dict,
    provider: AssetProvider | None,
) -> dict:
    payload, resolved_path = read_json_source(source_path)
    record = _select_record(payload, config)
    object_list = _extract_object_list(record)
    source_boundary, scene_height, _ = canonical_room(generation_input)
    boundary, origin_shift = shift_boundary_to_origin(source_boundary)
    unit, scale = _unit_scale(record, source_boundary, config)
    counts: Counter[str] = Counter()
    objects: list[dict[str, Any]] = []
    asset_ids = config.get("asset_ids")
    asset_ids = asset_ids if isinstance(asset_ids, Mapping) else {}
    default_source = str(config.get("asset_source_db") or "layoutgpt")

    for index, item in enumerate(object_list):
        category, attributes = _object_entry(item, index)
        counts[category] += 1
        object_id = f"{_slug(category)}_{counts[category]}"
        size = [
            _field(attributes, "length", index) * scale,
            _field(attributes, "width", index) * scale,
            _field(attributes, "height", index) * scale,
        ]
        center = [
            _field(attributes, "left", index) * scale,
            _field(attributes, "top", index) * scale,
            _field(attributes, "depth", index) * scale,
        ]
        yaw = finite_float(
            attributes.get("orientation", attributes.get("rotation", 0.0)),
            f"LayoutGPT object_list[{index}].orientation",
        )
        asset_key_value = asset_ids.get(object_id)
        if asset_key_value is None:
            category_assets = asset_ids.get(category)
            if isinstance(category_assets, Sequence) and not isinstance(
                category_assets, (str, bytes)
            ):
                category_assets = list(category_assets)
                asset_key_value = (
                    category_assets[counts[category] - 1]
                    if counts[category] <= len(category_assets)
                    else None
                )
            elif category_assets is not None:
                asset_key_value = category_assets
        asset_key = str(asset_key_value) if asset_key_value is not None else None
        native_record = (
            attributes.get("asset")
            if isinstance(attributes.get("asset"), Mapping)
            else {}
        )
        record_asset = resolve_asset_record(
            provider,
            asset_key=asset_key,
            source_db=default_source,
            category=category,
            description=category,
            size=size,
            hint=attributes,
            native_record=native_record,
        )
        if not record_asset.get("asset_key"):
            record_asset["asset_key"] = f"layoutgpt_proxy:{object_id}"
            record_asset["source_db"] = "layoutgpt_layout"
        fields = asset_fields(
            object_id=object_id,
            target_size=size,
            record=record_asset,
            fallback_category=category,
            fallback_description=category.replace("_", " "),
            config=config,
        )
        metadata = dict(fields["metadata"])
        metadata.update({"native_category": category, "native_attributes": dict(attributes)})
        objects.append(
            {
                "id": object_id,
                **{key: value for key, value in fields.items() if key != "metadata"},
                "size": size,
                "center": shift_center(center, origin_shift),
                "rotation": [0.0, 0.0, yaw],
                "metadata": metadata,
            }
        )

    return build_scene(
        generation_input,
        adapter_name="layout_gpt",
        native_schema="layoutgpt_3d_output_v1",
        boundary=boundary,
        scene_height=scene_height,
        objects=objects,
        coordinate_conversion={
            "source": "layoutgpt",
            "source_axes": "left_x_top_y_depth_z",
            "source_unit": unit,
            "meters_per_source_unit": scale,
            "position_semantics": "bbox_center",
            "origin_shift": origin_shift,
        },
        extra_metadata={
            "source_artifact": resolved_path.as_posix(),
            "query_id": record.get("query_id"),
            "iteration": record.get("iter"),
        },
    )


def _select_record(payload: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(payload, Mapping):
        return dict(payload)
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise ArtifactValidationError("LayoutGPT output must be an object or list of records")
    records = [dict(item) for item in payload if isinstance(item, Mapping)]
    query_id = config.get("query_id")
    if query_id is not None:
        records = [item for item in records if str(item.get("query_id")) == str(query_id)]
    iteration = config.get("iteration")
    if iteration is not None:
        records = [item for item in records if int(item.get("iter", -1)) == int(iteration)]
    if not records:
        raise ArtifactValidationError("LayoutGPT output selector matched no record")
    if len(records) > 1:
        scene_index = config.get("scene_index")
        if scene_index is None:
            raise ArtifactValidationError(
                "LayoutGPT output contains multiple records; set query_id/iteration or scene_index"
            )
        try:
            return records[int(scene_index)]
        except (IndexError, TypeError, ValueError) as exc:
            raise ArtifactValidationError("LayoutGPT scene_index is out of range") from exc
    return records[0]


def _extract_object_list(record: Mapping[str, Any]) -> list[Any]:
    value = record.get("object_list") or record.get("objects")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    raw_layout = record.get("layout") or record.get("response")
    if isinstance(raw_layout, str):
        parsed = []
        for match in CSS_OBJECT.finditer(raw_layout):
            fields = {key.lower(): float(number) for key, number in CSS_FIELD.findall(match.group(2))}
            parsed.append([match.group(1).strip(), fields])
        if parsed:
            return parsed
    raise ArtifactValidationError("LayoutGPT record requires object_list or parseable layout text")


def _object_entry(value: Any, index: int) -> tuple[str, dict[str, Any]]:
    if isinstance(value, Mapping):
        category = str(value.get("category") or value.get("label") or value.get("name") or "").strip()
        attributes = value.get("attributes") if isinstance(value.get("attributes"), Mapping) else value
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
        category = str(value[0]).strip()
        attributes = value[1]
    else:
        raise ArtifactValidationError(f"LayoutGPT object_list[{index}] has an unsupported shape")
    if not category or not isinstance(attributes, Mapping):
        raise ArtifactValidationError(f"LayoutGPT object_list[{index}] requires category and attributes")
    return category, dict(attributes)


def _unit_scale(
    record: Mapping[str, Any],
    boundary: Sequence[Sequence[float]],
    config: Mapping[str, Any],
) -> tuple[str, float]:
    unit = str(config.get("unit") or record.get("unit") or "").strip().lower()
    prompt = str(record.get("prompt") or "").lower()
    if not unit:
        unit = "px" if "px" in prompt else "m"
    if unit in {"m", "meter", "meters"}:
        return "meter", 1.0
    if unit not in {"px", "pixel", "pixels"}:
        raise ArtifactValidationError(f"unsupported LayoutGPT unit {unit!r}")
    explicit = config.get("meters_per_pixel")
    if explicit is not None:
        scale = finite_float(explicit, "adapter_config.meters_per_pixel")
    else:
        xs = [float(point[0]) for point in boundary]
        ys = [float(point[1]) for point in boundary]
        minimum_extent = min(max(xs) - min(xs), max(ys) - min(ys))
        pixels = finite_float(config.get("normalization_pixels", 256.0), "adapter_config.normalization_pixels")
        scale = minimum_extent / pixels
    if scale <= 0.0:
        raise ArtifactValidationError("LayoutGPT meters-per-pixel scale must be positive")
    return "pixel", scale


def _field(attributes: Mapping[str, Any], key: str, index: int) -> float:
    value = finite_float(attributes.get(key), f"LayoutGPT object_list[{index}].{key}")
    if key in {"length", "width", "height"} and value <= 0.0:
        raise ArtifactValidationError(f"LayoutGPT object_list[{index}].{key} must be positive")
    return value


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug or "object"


__all__ = ["convert_layout_gpt"]
