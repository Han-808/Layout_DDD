"""Final-submission contracts for shared-database Agent generation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
from typing import Any, Mapping, Protocol

from jsonschema import Draft202012Validator

from benchmark.resources import runtime_resource_path
from benchmark.scene_generation.non_rectangular_multi_room.contracts import (
    ASSET_SELECTION_SCHEMA_VERSION,
    validate_global_placement,
    validate_stage_a_artifacts,
)


AGENT_SUBMISSION_SCHEMA_VERSION = "non_rectangular_agent_submission_v1"
AGENT_SUBMISSION_SCHEMA_PATH = runtime_resource_path(
    "schemas/non_rectangular_agent/submission_v1.schema.json"
)
AGENT_ASSET_BINDING_POLICY = "agent_selected_from_frozen_shared_database_v1"


class AgentSubmissionError(ValueError):
    """Raised when an Agent result cannot become an evaluator-compatible scene."""


class SharedAssetCatalog(Protocol):
    snapshot_id: str

    def resolve(self, asset_id: str) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class ValidatedAgentSubmission:
    submission: Mapping[str, Any]
    object_plan: Mapping[str, Any]
    asset_selection: Mapping[str, Any]
    global_placement: Mapping[str, Any]
    plan_validation: Mapping[str, Any]

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "non_rectangular_agent_submission_validation_v1",
            "valid": True,
            "layout_id": str(self.object_plan["layout_id"]),
            "planned_instance_count": int(
                self.plan_validation["planned_instance_count"]
            ),
            "room_count": len(self.object_plan["rooms"]),
            "asset_binding_count": sum(
                len(room["objects"])
                for room in self.asset_selection["rooms"]
            ),
        }


def validate_agent_submission(
    value: Mapping[str, Any],
    *,
    room_layout: Mapping[str, Any],
    room_program: Mapping[str, Any],
    asset_catalog: SharedAssetCatalog,
) -> ValidatedAgentSubmission:
    """Validate the final plan, Agent-selected assets, and exact placement."""

    if not isinstance(value, Mapping):
        raise AgentSubmissionError("Agent submission must be a JSON object")
    _validate_schema(value)
    layout_id = str(room_layout.get("layout_id") or "")
    if str(value.get("layout_id") or "") != layout_id:
        raise AgentSubmissionError("Agent submission layout_id mismatch")
    object_plan = value["object_plan"]
    global_placement = value["global_placement"]
    if not isinstance(object_plan, Mapping) or not isinstance(
        global_placement, Mapping
    ):
        raise AgentSubmissionError("plan and placement must be JSON objects")
    if object_plan.get("schema_version") != (
        "non_rectangular_multi_room_object_plan_v2"
    ):
        raise AgentSubmissionError("Agent track requires object-plan contract v2")
    if object_plan.get("layout_id") != layout_id:
        raise AgentSubmissionError("object_plan layout_id mismatch")
    plan_validation = validate_stage_a_artifacts(
        room_layout=room_layout,
        room_program=room_program,
        object_plan=object_plan,
        expected_plan_contract_version="v2",
    )
    if plan_validation["terminal_status"] != "ready":
        raise AgentSubmissionError(
            str(plan_validation.get("failure_reason") or "object plan is not ready")
        )
    asset_selection = build_verified_asset_selection(
        layout_id=layout_id,
        object_plan=object_plan,
        asset_bindings=value["asset_bindings"],
        asset_catalog=asset_catalog,
    )
    try:
        placement = validate_global_placement(
            global_placement,
            object_plan=object_plan,
            asset_selection=asset_selection,
        )
    except Exception as exc:
        raise AgentSubmissionError(str(exc)) from exc
    return ValidatedAgentSubmission(
        submission=deepcopy(dict(value)),
        object_plan=deepcopy(dict(object_plan)),
        asset_selection=asset_selection,
        global_placement=placement,
        plan_validation=plan_validation,
    )


def build_verified_asset_selection(
    *,
    layout_id: str,
    object_plan: Mapping[str, Any],
    asset_bindings: Any,
    asset_catalog: SharedAssetCatalog,
) -> dict[str, Any]:
    """Resolve Agent-authored asset IDs against evaluator-owned DB metadata."""

    if not isinstance(asset_bindings, list):
        raise AgentSubmissionError("asset_bindings must be an array")
    expected: list[tuple[str, str, Mapping[str, Any]]] = []
    for room in object_plan["rooms"]:
        room_id = str(room["room_id"])
        for planned in room["objects"]:
            expected.append((room_id, str(planned["id"]), planned))
    if len(asset_bindings) != len(expected):
        raise AgentSubmissionError("asset binding count differs from object-plan slots")

    normalized: dict[tuple[str, str], str] = {}
    actual_order: list[tuple[str, str]] = []
    for index, raw in enumerate(asset_bindings):
        if not isinstance(raw, Mapping):
            raise AgentSubmissionError(f"asset_bindings[{index}] must be an object")
        if set(raw) != {"room_id", "slot_id", "asset_id"}:
            raise AgentSubmissionError(
                f"asset_bindings[{index}] keys differ from the fixed contract"
            )
        room_id = str(raw.get("room_id") or "")
        slot_id = str(raw.get("slot_id") or "")
        asset_id = str(raw.get("asset_id") or "")
        key = (room_id, slot_id)
        if not all((*key, asset_id)) or key in normalized:
            raise AgentSubmissionError(
                "asset binding identity must be non-empty and unique"
            )
        normalized[key] = asset_id
        actual_order.append(key)
    expected_order = [(room_id, slot_id) for room_id, slot_id, _ in expected]
    if actual_order != expected_order:
        raise AgentSubmissionError(
            "asset bindings must follow object-plan room/slot order exactly"
        )

    room_rows: list[dict[str, Any]] = []
    for room in object_plan["rooms"]:
        room_id = str(room["room_id"])
        objects: list[dict[str, Any]] = []
        for planned in room["objects"]:
            slot_id = str(planned["id"])
            asset_id = normalized[(room_id, slot_id)]
            try:
                raw_asset = asset_catalog.resolve(asset_id)
            except Exception as exc:
                raise AgentSubmissionError(
                    f"asset {asset_id!r} is unavailable in the frozen shared DB"
                ) from exc
            selected = _normalize_catalog_asset(
                raw_asset,
                requested_asset_id=asset_id,
                snapshot_id=str(asset_catalog.snapshot_id),
            )
            objects.append(
                {
                    "slot_id": slot_id,
                    "retrieval_slot_id": f"{room_id}::{slot_id}",
                    "planned_object": deepcopy(dict(planned)),
                    "selected_asset": selected,
                    "retrieval_query": {
                        "description": str(planned["retrieval_query"]),
                        "category": None,
                        "size_constraint": None,
                        "top_k": None,
                    },
                    "selection_provenance": {
                        "policy": AGENT_ASSET_BINDING_POLICY,
                        "catalog_snapshot_id": str(asset_catalog.snapshot_id),
                    },
                }
            )
        room_rows.append({"room_id": room_id, "objects": objects})
    return {
        "schema_version": ASSET_SELECTION_SCHEMA_VERSION,
        "layout_id": layout_id,
        "binding_policy": AGENT_ASSET_BINDING_POLICY,
        "catalog_snapshot_id": str(asset_catalog.snapshot_id),
        "rooms": room_rows,
    }


def _normalize_catalog_asset(
    value: Mapping[str, Any],
    *,
    requested_asset_id: str,
    snapshot_id: str,
) -> dict[str, Any]:
    observed_id = str(value.get("jid") or value.get("asset_id") or "")
    if observed_id != requested_asset_id:
        raise AgentSubmissionError("catalog returned a different asset identity")
    size_raw = value.get("size") or value.get("canonical_bbox_size_m")
    size = _positive_vec3(size_raw, path=f"asset[{requested_asset_id}].size")
    center = _finite_vec3(
        value.get("bbox_center_local")
        or value.get("canonical_bbox_center_m")
        or [0.0, 0.0, 0.0],
        path=f"asset[{requested_asset_id}].bbox_center_local",
    )
    category = _first_text(value.get("category"), value.get("retrieval_category"))
    description = _first_text(
        value.get("description"), value.get("desc"), value.get("short_desc"), category
    )
    short_desc = _first_text(value.get("short_desc"), description)
    if not category or not description:
        raise AgentSubmissionError(
            f"asset {requested_asset_id!r} lacks authoritative semantic metadata"
        )
    return {
        "jid": requested_asset_id,
        "category": category,
        "desc": description,
        "short_desc": short_desc,
        "size": size,
        "asset_ref": {
            "source_db": "imaginarium",
            "asset_key": requested_asset_id,
            "catalog_snapshot_id": snapshot_id,
        },
        "asset_proxy": {
            "type": "canonical_catalog_bbox",
            "bbox_center_local": center,
            "bbox_size": size,
        },
        "metadata": {
            "catalog_facing_contract_version": "imaginarium_catalog_facing_v1",
            "default_directed_functional_side": "local_neg_y",
            "selection_policy": AGENT_ASSET_BINDING_POLICY,
        },
    }


def _validate_schema(value: Mapping[str, Any]) -> None:
    try:
        schema = json.loads(AGENT_SUBMISSION_SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AgentSubmissionError("cannot load packaged Agent submission schema") from exc
    errors = sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    )
    if errors:
        error = errors[0]
        path = ".".join(str(part) for part in error.absolute_path) or "<root>"
        raise AgentSubmissionError(
            f"Agent submission schema failed at {path}: {error.message}"
        )


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _finite_vec3(value: Any, *, path: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise AgentSubmissionError(f"{path} must be a 3-vector")
    output: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise AgentSubmissionError(f"{path} must contain numbers")
        number = float(item)
        if not math.isfinite(number):
            raise AgentSubmissionError(f"{path} must contain finite numbers")
        output.append(number)
    return output


def _positive_vec3(value: Any, *, path: str) -> list[float]:
    output = _finite_vec3(value, path=path)
    if any(item <= 0.0 for item in output):
        raise AgentSubmissionError(f"{path} must be positive")
    return output


__all__ = [
    "AGENT_ASSET_BINDING_POLICY",
    "AGENT_SUBMISSION_SCHEMA_PATH",
    "AGENT_SUBMISSION_SCHEMA_VERSION",
    "AgentSubmissionError",
    "SharedAssetCatalog",
    "ValidatedAgentSubmission",
    "build_verified_asset_selection",
    "validate_agent_submission",
]
