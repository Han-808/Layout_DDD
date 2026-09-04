"""Content-addressed loading for the approved complicated FloorPlan suite."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from benchmark.non_rectangular import validate_room_layout, validate_room_program


SUITE_SCHEMA_VERSION = "complicated_floorplan_agent_suite_v1"
EXPECTED_TRACK_ID = "complicated_floorplan_agent_track_v1"
EXPECTED_TOTALS = {
    "scene_count": 10,
    "room_count": 42,
    "wall_segment_count": 314,
}


class AgentFloorPlanSuiteError(ValueError):
    """Raised when the approved suite has drifted or is incomplete."""


@dataclass(frozen=True, slots=True)
class AgentFloorPlanCase:
    scene_id: str
    room_count: int
    wall_segment_count: int
    room_layout_path: Path
    room_program_path: Path
    room_layout: Mapping[str, Any]
    room_program: Mapping[str, Any]
    room_layout_sha256: str
    room_program_sha256: str

    def public_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "room_count": self.room_count,
            "wall_segment_count": self.wall_segment_count,
            "room_layout_sha256": self.room_layout_sha256,
            "room_program_sha256": self.room_program_sha256,
            "target_total_instances": dict(
                self.room_program["target_total_instances"]
            ),
        }


@dataclass(frozen=True, slots=True)
class AgentFloorPlanSuite:
    root: Path
    proposal_manifest_sha256: str
    layout_manifest_sha256: str
    program_manifest_sha256: str
    cases: tuple[AgentFloorPlanCase, ...]
    target_total_instances: Mapping[str, int]

    @property
    def scene_order(self) -> tuple[str, ...]:
        return tuple(case.scene_id for case in self.cases)

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SUITE_SCHEMA_VERSION,
            "track_id": EXPECTED_TRACK_ID,
            "track_type": "agent_only",
            "floorplan_policy": "customized_fixed_cohort",
            "asset_access_mode": "shared_database",
            "scene_count": len(self.cases),
            "room_count": sum(case.room_count for case in self.cases),
            "wall_segment_count": sum(
                case.wall_segment_count for case in self.cases
            ),
            "scene_order": list(self.scene_order),
            "aggregate_target_total_instances": dict(
                self.target_total_instances
            ),
            "proposal_manifest_sha256": self.proposal_manifest_sha256,
            "layout_manifest_sha256": self.layout_manifest_sha256,
            "program_manifest_sha256": self.program_manifest_sha256,
        }


def load_agent_floorplan_suite(root: str | Path) -> AgentFloorPlanSuite:
    """Load and verify every immutable layout/program input without network use."""

    suite_root = Path(root).expanduser().resolve()
    if not suite_root.is_dir() or suite_root.is_symlink():
        raise AgentFloorPlanSuiteError("suite root must be a real directory")
    proposal_path = suite_root / "proposal_manifest.json"
    layout_manifest_path = suite_root / "scaled_layouts/manifest_v1.json"
    program_manifest_path = suite_root / "scaled_programs/manifest_v1.json"
    proposal = _load_json(proposal_path)
    layouts = _load_json(layout_manifest_path)
    programs = _load_json(program_manifest_path)
    for label, value in (
        ("proposal", proposal),
        ("layout", layouts),
        ("program", programs),
    ):
        _verify_self_hash(value, label=label)

    if proposal.get("status") != "human_approved":
        raise AgentFloorPlanSuiteError("suite lacks recorded human approval")
    if proposal.get("generation_authorized") is not True:
        raise AgentFloorPlanSuiteError("suite is not authorized for generation")
    design = proposal.get("experimental_design")
    if not isinstance(design, Mapping):
        raise AgentFloorPlanSuiteError("suite lacks experimental_design")
    expected_design = {
        "track_id": EXPECTED_TRACK_ID,
        "track_type": "agent_only",
        "comparison_unit": "agent_system",
        "model_only_arm_in_this_suite": False,
        "floorplan_policy": "customized_fixed_cohort",
    }
    for field, expected in expected_design.items():
        if design.get(field) != expected:
            raise AgentFloorPlanSuiteError(
                f"experimental_design.{field} drifted"
            )
    asset_access = design.get("asset_access")
    if not isinstance(asset_access, Mapping) or (
        asset_access.get("mode") != "shared_database"
        or asset_access.get("per_scene_assets_prefrozen") is not False
        or asset_access.get("uniform_access_contract_required") is not True
        or asset_access.get("database_snapshot_required_before_launch") is not True
    ):
        raise AgentFloorPlanSuiteError("shared-database policy drifted")

    artifact_links = proposal.get("artifacts")
    if not isinstance(artifact_links, Mapping):
        raise AgentFloorPlanSuiteError("proposal lacks artifact links")
    if artifact_links.get("scaled_layout_manifest", {}).get("sha256") != _sha256_file(
        layout_manifest_path
    ):
        raise AgentFloorPlanSuiteError("linked layout manifest hash drifted")
    if artifact_links.get("scaled_program_manifest", {}).get("sha256") != _sha256_file(
        program_manifest_path
    ):
        raise AgentFloorPlanSuiteError("linked program manifest hash drifted")

    scene_order = _text_list(
        proposal.get("selection", {}).get("scene_order"),
        label="proposal scene_order",
    )
    if len(scene_order) != EXPECTED_TOTALS["scene_count"] or len(set(scene_order)) != len(
        scene_order
    ):
        raise AgentFloorPlanSuiteError("approved scene_order must contain 10 unique scenes")
    if layouts.get("selection", {}).get("scene_order") != scene_order:
        raise AgentFloorPlanSuiteError("layout scene order differs from proposal")
    if programs.get("scene_order") != scene_order:
        raise AgentFloorPlanSuiteError("program scene order differs from proposal")
    layout_totals = layouts.get("totals")
    if not isinstance(layout_totals, Mapping):
        raise AgentFloorPlanSuiteError("layout manifest lacks totals")
    for field, expected in EXPECTED_TOTALS.items():
        if layout_totals.get(field) != expected:
            raise AgentFloorPlanSuiteError(f"layout total {field} drifted")

    layout_rows = _scene_rows(layouts, label="layout manifest")
    program_rows = _scene_rows(programs, label="program manifest")
    if list(layout_rows) != scene_order or list(program_rows) != scene_order:
        raise AgentFloorPlanSuiteError("cohort manifest rows differ from scene_order")

    cases: list[AgentFloorPlanCase] = []
    aggregate_min = 0
    aggregate_max = 0
    for scene_id in scene_order:
        layout_path = suite_root / "scaled_layouts" / scene_id / "room_layout.json"
        program_path = suite_root / "scaled_programs" / scene_id / "room_program.json"
        _require_child_file(suite_root, layout_path, label="room layout")
        _require_child_file(suite_root, program_path, label="room program")
        layout_sha = _sha256_file(layout_path)
        program_sha = _sha256_file(program_path)
        if layout_rows[scene_id].get("room_layout_sha256") != layout_sha:
            raise AgentFloorPlanSuiteError(f"{scene_id} room layout hash drifted")
        if program_rows[scene_id].get("room_program_sha256") != program_sha:
            raise AgentFloorPlanSuiteError(f"{scene_id} room program hash drifted")
        room_layout = _load_json(layout_path)
        room_program = _load_json(program_path)
        layout_report = validate_room_layout(room_layout)
        program_report = validate_room_program(room_program)
        if layout_report["layout_id"] != program_report["layout_id"]:
            raise AgentFloorPlanSuiteError(f"{scene_id} layout/program ID mismatch")
        if layout_report["room_count"] != program_report["program_count"]:
            raise AgentFloorPlanSuiteError(
                f"{scene_id} room/program cardinality mismatch"
            )
        if layout_report["layout_id"] != scene_id:
            raise AgentFloorPlanSuiteError(f"{scene_id} layout identity drifted")
        target = program_report["target_total_instances"]
        aggregate_min += int(target["min"])
        aggregate_max += int(target["max"])
        cases.append(
            AgentFloorPlanCase(
                scene_id=scene_id,
                room_count=int(layout_report["room_count"]),
                wall_segment_count=int(layout_report["wall_segment_count"]),
                room_layout_path=layout_path,
                room_program_path=program_path,
                room_layout=room_layout,
                room_program=room_program,
                room_layout_sha256=layout_sha,
                room_program_sha256=program_sha,
            )
        )

    target = proposal.get("object_count", {}).get(
        "aggregate_target_total_instances"
    )
    expected_target = {"min": aggregate_min, "max": aggregate_max}
    if target != expected_target or programs.get("totals", {}).get(
        "target_total_instances"
    ) != expected_target:
        raise AgentFloorPlanSuiteError("aggregate object-count target drifted")
    if expected_target != {"min": 719, "max": 891}:
        raise AgentFloorPlanSuiteError("approved aggregate count range drifted")

    return AgentFloorPlanSuite(
        root=suite_root,
        proposal_manifest_sha256=_sha256_file(proposal_path),
        layout_manifest_sha256=_sha256_file(layout_manifest_path),
        program_manifest_sha256=_sha256_file(program_manifest_path),
        cases=tuple(cases),
        target_total_instances=expected_target,
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AgentFloorPlanSuiteError(
            f"cannot load JSON artifact {path.name}: {type(exc).__name__}"
        ) from exc
    if not isinstance(value, dict):
        raise AgentFloorPlanSuiteError(f"JSON artifact {path.name} must be an object")
    return value


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_self_hash(value: Mapping[str, Any], *, label: str) -> None:
    observed = value.get("manifest_sha256")
    if not isinstance(observed, str):
        raise AgentFloorPlanSuiteError(f"{label} manifest lacks self hash")
    payload = dict(value)
    payload.pop("manifest_sha256", None)
    encoded = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != observed:
        raise AgentFloorPlanSuiteError(f"{label} manifest self hash drifted")


def _scene_rows(
    manifest: Mapping[str, Any], *, label: str
) -> dict[str, Mapping[str, Any]]:
    rows = manifest.get("scenes")
    if not isinstance(rows, list):
        raise AgentFloorPlanSuiteError(f"{label} scenes must be an array")
    output: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise AgentFloorPlanSuiteError(f"{label} scene row must be an object")
        scene_id = str(row.get("scene_id") or "")
        if not scene_id or scene_id in output:
            raise AgentFloorPlanSuiteError(f"{label} scene identity is invalid")
        output[scene_id] = row
    return output


def _text_list(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() or item != item.strip()
        for item in value
    ):
        raise AgentFloorPlanSuiteError(f"{label} must be trimmed strings")
    return list(value)


def _require_child_file(root: Path, path: Path, *, label: str) -> None:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise AgentFloorPlanSuiteError(f"{label} escapes suite root") from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise AgentFloorPlanSuiteError(f"{label} must be a real file")


__all__ = [
    "AgentFloorPlanCase",
    "AgentFloorPlanSuite",
    "AgentFloorPlanSuiteError",
    "EXPECTED_TRACK_ID",
    "SUITE_SCHEMA_VERSION",
    "load_agent_floorplan_suite",
]
