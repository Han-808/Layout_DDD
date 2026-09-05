"""Version public briefs against already approved fixed assets, never reselect.

This is an explicit preparation-time input revision, not output conversion or
scene repair. The resulting public object plan is shared by all generators and
the unchanged post-hoc evaluator. Old specs/prepared runs are never rewritten.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from benchmark.generation_comparison.identity import canonical_json_sha256
from benchmark.scene_io.validate import ArtifactValidationError, validate_object_plan
from benchmark.utils.io import write_json


def revise_frozen_public_brief(
    source: Mapping[str, Any], recipe: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply a pinned, reviewed text recipe; geometry/inventory stay identical."""
    if (
        recipe.get("schema_version") != "frozen_public_brief_revision_v1"
        or recipe.get("source_spec_sha256") != canonical_json_sha256(source)
        or recipe.get("approval") != "user_approved_actual_frozen_asset_descriptions"
        or recipe.get("object_text_policy") != "project_existing_frozen_slot_category_description"
    ):
        raise ArtifactValidationError("public brief recipe/source/approval mismatch")
    allowed_recipe = {
        "schema_version", "revision_id", "source_spec_sha256", "approval",
        "object_text_policy", "common_constraints", "case_edits", "rationale",
    }
    if set(recipe) - allowed_recipe or not isinstance(recipe.get("revision_id"), str):
        raise ArtifactValidationError("unsupported public brief recipe fields")
    original = deepcopy(dict(source))
    result = deepcopy(original)
    edits = recipe.get("case_edits", {})
    if not isinstance(edits, Mapping) or set(edits) - {case["case_id"] for case in result["cases"]}:
        raise ArtifactValidationError("public brief recipe references unknown cases")
    common = recipe.get("common_constraints", [])
    if not isinstance(common, list) or any(not isinstance(value, str) or not value.strip() for value in common):
        raise ArtifactValidationError("public brief common constraints must be text")
    changes: list[dict[str, Any]] = []

    def replace(target: dict[str, Any], field: str, value: Any, path: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ArtifactValidationError(f"public brief revision requires text: {path}")
        if target.get(field) != value:
            changes.append({"path": path, "before": target.get(field), "after": value})
            target[field] = value

    for case in result["cases"]:
        case_id = case["case_id"]
        plan = case["object_plan"]
        frozen = {obj["slot_id"]: obj for obj in case["objects"]}
        if len(frozen) != len(case["objects"]) or len(plan["objects"]) != len(frozen):
            raise ArtifactValidationError("public brief frozen slot count mismatch")
        plan_objects = {obj["id"]: obj for obj in plan["objects"]}
        if len(plan_objects) != len(frozen) or set(plan_objects) != set(frozen):
            raise ArtifactValidationError("public brief slot IDs must match approved inventory")
        for slot_id, obj in plan_objects.items():
            # Take factual text only from the approved frozen input, not a
            # category guess from the slot name and not a semantic DB lookup.
            for field in ("category", "description"):
                replace(obj, field, frozen[slot_id][field], f"{case_id}/objects/{slot_id}/{field}")
        case_edit = edits.get(case_id, {})
        if not isinstance(case_edit, Mapping) or set(case_edit) - {
            "instruction", "scene_type", "zones", "constraints", "roles",
        }:
            raise ArtifactValidationError(f"unsupported public brief edit: {case_id}")
        for field, plan_field in (("instruction", "scene_description"), ("scene_type", "scene_type")):
            if field in case_edit:
                replace(case, field, case_edit[field], f"{case_id}/{field}")
                replace(plan, plan_field, case_edit[field], f"{case_id}/object_plan/{plan_field}")
        zones = {zone["id"]: zone for zone in plan.get("zones", [])}
        for zone_id, text in case_edit.get("zones", {}).items():
            if zone_id not in zones:
                raise ArtifactValidationError(f"unknown public brief zone: {case_id}/{zone_id}")
            replace(zones[zone_id], "description", text, f"{case_id}/zones/{zone_id}")
        for slot_id, role in case_edit.get("roles", {}).items():
            if slot_id not in plan_objects:
                raise ArtifactValidationError(f"unknown public brief role slot: {case_id}/{slot_id}")
            replace(plan_objects[slot_id], "role", role, f"{case_id}/objects/{slot_id}/role")
        for slot_id, obj in plan_objects.items():
            metadata = obj.get("metadata", {})
            # These duplicate the current public request. Preserve their old
            # values in the separate audit, not as contradictory model input.
            # source_* identity/curation fields remain unchanged.
            for field, source_field in (
                ("requested_category", "category"),
                ("requested_description", "description"),
                ("intended_role", "role"),
            ):
                if field in metadata:
                    replace(metadata, field, obj[source_field], f"{case_id}/objects/{slot_id}/metadata/{field}")
        constraints = plan["global_constraints"]
        for index_text, text in case_edit.get("constraints", {}).items():
            if not str(index_text).isdigit() or not 0 <= int(index_text) < len(constraints):
                raise ArtifactValidationError(f"unknown public constraint: {case_id}/{index_text}")
            index = int(index_text)
            holder = {"text": constraints[index]}
            replace(holder, "text", text, f"{case_id}/constraints/{index}")
            constraints[index] = holder["text"]
        for text in common:
            changes.append({"path": f"{case_id}/constraints/{len(constraints)}", "before": None, "after": text})
            constraints.append(text)
        plan.setdefault("metadata", {})["public_brief_revision"] = recipe["revision_id"]
        validate_object_plan(plan)

    # These are immutable scientific controls, irrespective of any recipe text.
    frozen_fields = ("catalog", "asset_curation", "asset_selection_status", "methods", "generation", "evaluator")
    if any(result.get(key) != original.get(key) for key in frozen_fields):
        raise ArtifactValidationError("public brief revision changed frozen controls")
    for old, new in zip(original["cases"], result["cases"]):
        for field in ("case_id", "room", "objects", "seed", "source_provenance"):
            if old.get(field) != new.get(field):
                raise ArtifactValidationError(f"public brief revision changed frozen {field}")
    audit = {
        "schema_version": "frozen_public_brief_audit_v1",
        "revision_id": recipe["revision_id"], "approval": recipe["approval"],
        "source_spec_sha256": canonical_json_sha256(original),
        "revised_spec_sha256": canonical_json_sha256(result),
        "recipe_sha256": canonical_json_sha256(recipe),
        "case_count": len(result["cases"]),
        "slot_count": sum(len(case["objects"]) for case in result["cases"]),
        "asset_count": len(result["catalog"]["assets"]),
        "frozen_geometry_inventory_bindings_unchanged": True,
        "evaluator_policy_unchanged": True,
        "same_public_plan_for_generation_and_evaluation": True,
        "scene_layout_answers_added": False,
        "model_calls": 0, "changes": changes,
        "limitations": [
            "asset text is approved catalog metadata, not new visual certification",
            "unchanged support relations and clearances may still be hard or infeasible",
            "revised public briefs are a new input treatment, not identical to original SceneBoard prompts",
        ],
    }
    return result, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    destination = args.out_dir.expanduser().resolve()
    if destination.exists():
        raise FileExistsError("public brief output requires a fresh directory")
    source_bytes = args.spec.read_bytes()
    recipe_bytes = args.recipe.read_bytes()
    revised, audit = revise_frozen_public_brief(json.loads(source_bytes), json.loads(recipe_bytes))
    audit["source_spec_file_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    audit["recipe_file_sha256"] = hashlib.sha256(recipe_bytes).hexdigest()
    write_json(destination / "spec.json", revised)
    write_json(destination / "public_brief_audit.json", audit)
    print(json.dumps({key: value for key, value in audit.items() if key != "changes"}, indent=2))


if __name__ == "__main__":
    main()
