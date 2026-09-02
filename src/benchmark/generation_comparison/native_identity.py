"""Read persisted native asset/slot identities without canonical conversion."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmark.adapters.scene_weaver.converter import discover_layout_iterations
from benchmark.scene_io.validate import ArtifactValidationError
from benchmark.utils.io import read_json


NATIVE_SELECTION_SCHEMA_VERSION = "comparison_native_asset_selection_v1"


def inspect_native_asset_selections(
    *,
    adapter_name: str,
    native_artifact: str | Path,
    execution_metadata: Mapping[str, Any],
    adapter_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source = Path(native_artifact)
    if adapter_name == "scene_weaver":
        iterations = []
        for iteration, path in sorted(discover_layout_iterations(source).items()):
            payload = read_json(path)
            iterations.append(
                {
                    "iteration": iteration,
                    "artifact": path.resolve().as_posix(),
                    "objects": _scene_weaver_objects(
                        payload,
                        _auxiliary_mapping(execution_metadata, "asset_bindings"),
                    ),
                }
            )
        return _result(adapter_name, iterations)

    payload, path = _primary_json(source, adapter_name)
    if adapter_name == "layout_gpt":
        objects = _layout_gpt_objects(
            payload,
            _auxiliary_mapping(execution_metadata, "asset_ids"),
            adapter_config or {},
        )
    elif adapter_name == "direct_layout":
        objects = _direct_layout_objects(payload)
    elif adapter_name == "layout_vlm":
        native_input_path = execution_metadata.get("native_input_path")
        native_input = read_json(native_input_path) if native_input_path else {}
        objects = _layout_vlm_objects(payload, native_input)
    elif adapter_name == "respace":
        objects = _respace_objects(payload)
    else:
        raise ArtifactValidationError(
            f"no native selection inspector for adapter {adapter_name!r}"
        )
    return _result(
        adapter_name,
        [{"iteration": None, "artifact": path.resolve().as_posix(), "objects": objects}],
    )


def selected_iteration_objects(
    inspection: Mapping[str, Any],
    *,
    selected_iteration: int | None,
) -> list[dict[str, Any]]:
    iterations = inspection.get("iterations")
    if not isinstance(iterations, Sequence):
        return []
    if selected_iteration is None and len(iterations) == 1:
        row = iterations[0]
        return list(row.get("objects") or []) if isinstance(row, Mapping) else []
    for row in iterations:
        if isinstance(row, Mapping) and row.get("iteration") == selected_iteration:
            return list(row.get("objects") or [])
    return []


def _result(adapter_name: str, iterations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": NATIVE_SELECTION_SCHEMA_VERSION,
        "adapter": adapter_name,
        "iterations": iterations,
    }


def _layout_gpt_objects(
    payload: Any,
    bindings: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    record = _select_layout_gpt_record(payload, config)
    if not isinstance(record, Mapping):
        raise ArtifactValidationError("LayoutGPT native artifact lacks a layout record")
    objects = record.get("object_list") or record.get("objects")
    if not isinstance(objects, Sequence) or isinstance(objects, (str, bytes)):
        raise ArtifactValidationError("LayoutGPT native artifact lacks object_list")
    counts: Counter[str] = Counter()
    result = []
    for index, item in enumerate(objects):
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) == 2:
            category, attributes = str(item[0]), item[1]
        elif isinstance(item, Mapping):
            category = str(item.get("category") or item.get("name") or "")
            attributes = item
        else:
            raise ArtifactValidationError(
                f"LayoutGPT native object_list[{index}] is malformed"
            )
        if not isinstance(attributes, Mapping):
            raise ArtifactValidationError(
                f"LayoutGPT native object_list[{index}] attributes are malformed"
            )
        counts[category] += 1
        native_id = f"{_slug(category)}_{counts[category]}"
        asset_value = bindings.get(native_id)
        if asset_value is None:
            category_bindings = bindings.get(category)
            if isinstance(category_bindings, Sequence) and not isinstance(
                category_bindings, (str, bytes)
            ):
                position = counts[category] - 1
                asset_value = (
                    category_bindings[position]
                    if position < len(category_bindings)
                    else None
                )
            elif category_bindings is not None:
                asset_value = category_bindings
        asset = attributes.get("asset")
        asset = asset if isinstance(asset, Mapping) else {}
        asset_id = _asset_id(asset_value) or _asset_id(asset)
        result.append(
            {
                "native_object_id": native_id,
                "asset_id": asset_id,
                "category": category,
                "selection_source": (
                    "preserved_asset_ids_sidecar" if _asset_id(asset_value) else "native_object"
                ),
            }
        )
    return result


def _select_layout_gpt_record(
    payload: Any,
    config: Mapping[str, Any],
) -> Any:
    if not isinstance(payload, list):
        return payload
    records = [item for item in payload if isinstance(item, Mapping)]
    if config.get("query_id") is not None:
        records = [
            item
            for item in records
            if str(item.get("query_id")) == str(config["query_id"])
        ]
    if config.get("iteration") is not None:
        records = [
            item
            for item in records
            if int(item.get("iter", -1)) == int(config["iteration"])
        ]
    if len(records) == 1:
        return records[0]
    scene_index = config.get("scene_index")
    if scene_index is not None:
        try:
            return records[int(scene_index)]
        except (IndexError, TypeError, ValueError) as exc:
            raise ArtifactValidationError(
                "LayoutGPT native selection scene_index is out of range"
            ) from exc
    raise ArtifactValidationError(
        "LayoutGPT native selection audit requires the same selector used by conversion"
    )


def _direct_layout_objects(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, Mapping):
        payload = payload.get("objects") or payload.get("layout") or payload.get("placements")
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise ArtifactValidationError("DirectLayout native artifact lacks placements")
    result = []
    for index, raw in enumerate(payload):
        if not isinstance(raw, Mapping):
            raise ArtifactValidationError(
                f"DirectLayout native placement[{index}] must be an object"
            )
        native_id = str(
            raw.get("new_object_id")
            or raw.get("object_id")
            or raw.get("id")
            or f"object_{index}"
        )
        result.append(
            {
                "native_object_id": native_id,
                "asset_id": str(raw.get("asset_id") or raw.get("jid") or native_id),
                "category": raw.get("category"),
                "selection_source": "native_object",
            }
        )
    return result


def _layout_vlm_objects(payload: Any, native_input: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, Mapping):
        raise ArtifactValidationError("LayoutVLM native artifact must be an object")
    layout = payload.get("layout") if isinstance(payload.get("layout"), Mapping) else payload
    scene_config = payload.get("scene_config")
    if not isinstance(scene_config, Mapping):
        scene_config = native_input if isinstance(native_input, Mapping) else {}
    assets = scene_config.get("assets") if isinstance(scene_config, Mapping) else {}
    assets = assets if isinstance(assets, Mapping) else {}
    result = []
    for native_id in layout:
        asset = assets.get(native_id)
        asset = asset if isinstance(asset, Mapping) else {}
        result.append(
            {
                "native_object_id": str(native_id),
                "asset_id": str(
                    asset.get("uid")
                    or asset.get("asset_id")
                    or _strip_instance_suffix(str(native_id))
                ),
                "category": asset.get("category"),
                "selection_source": "native_scene_config_asset_table",
            }
        )
    return result


def _respace_objects(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, Mapping):
        raise ArtifactValidationError("ReSpace native artifact must be an object")
    scene = payload.get("scene") if isinstance(payload.get("scene"), Mapping) else payload
    objects = scene.get("objects") if isinstance(scene, Mapping) else None
    if not isinstance(objects, Sequence) or isinstance(objects, (str, bytes)):
        raise ArtifactValidationError("ReSpace native artifact lacks scene.objects")
    result = []
    for index, raw in enumerate(objects):
        if not isinstance(raw, Mapping):
            raise ArtifactValidationError(f"ReSpace objects[{index}] must be an object")
        native_id = str(
            raw.get("id") or raw.get("instance_id") or raw.get("uid") or f"object_{index}"
        )
        result.append(
            {
                "native_object_id": native_id,
                "asset_id": _text(
                    raw.get("sampled_asset_jid"), raw.get("sampled_jid"), raw.get("jid")
                ),
                "category": raw.get("category") or raw.get("class") or raw.get("label"),
                "selection_source": "native_ssr",
            }
        )
    return result


def _scene_weaver_objects(payload: Any, bindings: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(payload, Mapping) or not isinstance(payload.get("objects"), Mapping):
        raise ArtifactValidationError("SceneWeaver layout lacks objects mapping")
    result = []
    for native_id, raw in payload["objects"].items():
        if not isinstance(raw, Mapping):
            raise ArtifactValidationError(
                f"SceneWeaver object {native_id!r} must be an object"
            )
        binding = bindings.get(native_id)
        binding = binding if isinstance(binding, Mapping) else {}
        asset_id = _text(
            raw.get("asset_id"),
            raw.get("jid"),
            binding.get("asset_key"),
            binding.get("asset_id"),
        )
        result.append(
            {
                "native_object_id": str(native_id),
                "asset_id": asset_id,
                "category": raw.get("category") or binding.get("category"),
                "selection_source": (
                    "native_object"
                    if raw.get("asset_id") or raw.get("jid")
                    else "preserved_asset_bindings_sidecar"
                ),
            }
        )
    return result


def _primary_json(source: Path, adapter_name: str) -> tuple[Any, Path]:
    if source.is_file():
        return read_json(source), source
    names = {
        "layout_gpt": ("layoutgpt.json",),
        "direct_layout": ("layout.json", "output_layout.json", "direct.json"),
        "layout_vlm": ("layout.json",),
        "respace": ("scene.json", "ssr.json"),
    }[adapter_name]
    for name in names:
        candidate = source / name
        if candidate.is_file():
            return read_json(candidate), candidate
    files = sorted(source.glob("*.json"))
    if len(files) == 1:
        return read_json(files[0]), files[0]
    raise ArtifactValidationError(
        f"cannot select one native JSON for {adapter_name}: {source}"
    )


def _auxiliary_mapping(metadata: Mapping[str, Any], name: str) -> dict[str, Any]:
    auxiliary = metadata.get("preserved_auxiliary_artifacts")
    auxiliary = auxiliary if isinstance(auxiliary, Mapping) else {}
    entry = auxiliary.get(name)
    entry = entry if isinstance(entry, Mapping) else {}
    path = entry.get("path")
    if not path:
        return {}
    value = read_json(path)
    if isinstance(value, Mapping) and isinstance(value.get(name), Mapping):
        value = value[name]
    return dict(value) if isinstance(value, Mapping) else {}


def _asset_id(value: Any) -> str | None:
    if isinstance(value, Mapping):
        return _text(
            value.get("asset_key"),
            value.get("asset_id"),
            value.get("jid"),
            value.get("uid"),
        )
    return _text(value)


def _text(*values: Any) -> str | None:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return normalized or "object"


def _strip_instance_suffix(value: str) -> str:
    return re.sub(r"[-_]\d+$", "", value)


__all__ = [
    "NATIVE_SELECTION_SCHEMA_VERSION",
    "inspect_native_asset_selections",
    "selected_iteration_objects",
]
