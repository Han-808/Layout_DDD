"""Write-once and hash-verified artifact layout for multi-room generation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


class MultiRoomArtifactError(RuntimeError):
    """Raised when write-once or resume artifact invariants fail."""


@dataclass(frozen=True, slots=True)
class MultiRoomArtifactLayout:
    output_root: Path
    layout_id: str

    def __post_init__(self) -> None:
        root = Path(self.output_root).expanduser().resolve()
        object.__setattr__(self, "output_root", root)
        if not isinstance(self.layout_id, str) or not self.layout_id:
            raise MultiRoomArtifactError("layout_id must be non-empty")
        if Path(self.layout_id).name != self.layout_id:
            raise MultiRoomArtifactError("layout_id cannot contain a path")

    @property
    def run_manifest_path(self) -> Path:
        return self.output_root / "run_manifest.json"

    @property
    def execution_policy_path(self) -> Path:
        return self.output_root / "execution_policy.json"

    @property
    def summary_path(self) -> Path:
        return self.output_root / "summary.json"

    @property
    def layout_root(self) -> Path:
        return self.output_root / self.layout_id

    @property
    def floor_plan_path(self) -> Path:
        return self.layout_root / "floor_plan.json"

    @property
    def floor_plan_validation_path(self) -> Path:
        return self.layout_root / "floor_plan_validation.json"

    @property
    def rooms_root(self) -> Path:
        return self.layout_root / "rooms"

    def room_root(self, room_key: str) -> Path:
        if (
            not isinstance(room_key, str)
            or not room_key
            or Path(room_key).name != room_key
        ):
            raise MultiRoomArtifactError(f"invalid room key: {room_key!r}")
        return self.rooms_root / room_key

    def room_result_path(self, room_key: str) -> Path:
        return self.room_root(room_key) / "room_result.json"

    @property
    def compiled_architecture_path(self) -> Path:
        return self.layout_root / "compiled_architecture.json"

    @property
    def global_scene_path(self) -> Path:
        return self.layout_root / "assembled_multi_room_scene.json"

    @property
    def evaluation_rooms_root(self) -> Path:
        return self.layout_root / "evaluation_rooms"

    def room_projection_path(self, room_key: str) -> Path:
        return self.evaluation_rooms_root / room_key / "canonical_scene.json"

    @property
    def evaluation_index_path(self) -> Path:
        return self.layout_root / "room_evaluation_index.json"

    @property
    def assembly_manifest_path(self) -> Path:
        return self.layout_root / "assembly_manifest.json"

    def relative_to_layout(self, path: Path) -> str:
        try:
            relative = path.resolve().relative_to(self.layout_root.resolve())
        except ValueError as exc:
            raise MultiRoomArtifactError("artifact path escapes layout root") from exc
        if ".." in relative.parts:
            raise MultiRoomArtifactError("artifact path contains traversal")
        return relative.as_posix()

    def require_fresh(self) -> None:
        if self.output_root.exists():
            raise FileExistsError(
                f"refusing to overwrite existing output: {self.output_root}"
            )

    def initialize_directories(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=False)
        self.layout_root.mkdir(parents=False, exist_ok=False)
        self.rooms_root.mkdir(parents=False, exist_ok=False)

    def verify_resume_root(
        self,
        *,
        expected_run_schema: str,
        expected_fingerprint: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not self.output_root.is_dir() or self.output_root.is_symlink():
            raise MultiRoomArtifactError("resume requires a regular output directory")
        if self.summary_path.exists():
            raise MultiRoomArtifactError("terminal multi-room runs cannot be resumed")
        if not self.layout_root.is_dir() or not self.rooms_root.is_dir():
            raise MultiRoomArtifactError("resume layout directories are incomplete")
        manifest = _load_json(self.run_manifest_path)
        if manifest.get("schema_version") != expected_run_schema:
            raise MultiRoomArtifactError("resume run-manifest schema mismatch")
        if manifest.get("input_fingerprint") != dict(expected_fingerprint):
            raise MultiRoomArtifactError("resume input fingerprint mismatch")
        if not self.execution_policy_path.is_file():
            raise MultiRoomArtifactError("resume execution policy is missing")
        if not self.floor_plan_path.is_file() or not self.floor_plan_validation_path.is_file():
            raise MultiRoomArtifactError("resume floor-plan artifacts are missing")
        return manifest


def gate_artifact_start(
    layout: MultiRoomArtifactLayout,
    *,
    resume: bool,
    expected_run_schema: str,
    expected_fingerprint: Mapping[str, Any],
    floor_plan_source_sha256: str,
    floor_plan_validation: Mapping[str, Any],
    sha256_file: Any,
    expected_room_ids: tuple[str, ...] | list[str] | None = None,
    room_result_schema: str | None = None,
    expected_route_binding: Mapping[str, Any] | None = None,
    expected_room_results: Mapping[str, Mapping[str, Any]] | None = None,
) -> int:
    """Reject stale/terminal output before any external runtime is built."""

    if not resume:
        layout.require_fresh()
        return 0
    manifest = layout.verify_resume_root(
        expected_run_schema=expected_run_schema,
        expected_fingerprint=expected_fingerprint,
    )
    expected_execution_hash = manifest.get("execution_policy_sha256")
    if (
        not isinstance(expected_execution_hash, str)
        or len(expected_execution_hash) != 64
        or sha256_file(layout.execution_policy_path) != expected_execution_hash
    ):
        raise MultiRoomArtifactError("resume execution policy hash mismatch")
    if sha256_file(layout.floor_plan_path) != floor_plan_source_sha256:
        raise MultiRoomArtifactError("resume floor-plan source hash mismatch")
    if _load_json(layout.floor_plan_validation_path) != dict(
        floor_plan_validation
    ):
        raise MultiRoomArtifactError(
            "resume floor-plan validation identity mismatch"
        )
    if expected_route_binding is not None:
        execution = _load_json(layout.execution_policy_path)
        provenance = execution.get("run_provenance")
        observed = (
            provenance.get("route_binding")
            if isinstance(provenance, dict)
            else None
        )
        if observed != dict(expected_route_binding):
            raise MultiRoomArtifactError("resume route-binding identity mismatch")
    if expected_room_ids is not None:
        if not isinstance(room_result_schema, str) or not room_result_schema:
            raise MultiRoomArtifactError("resume room-result schema is required")
        expected_keys = {
            f"room_{index:03d}": room_id
            for index, room_id in enumerate(expected_room_ids)
        }
        observed_entries = list(layout.rooms_root.iterdir())
        observed_keys = {entry.name for entry in observed_entries}
        if not observed_keys <= set(expected_keys):
            raise MultiRoomArtifactError("resume contains an unexpected room artifact")
        for entry in observed_entries:
            if not entry.is_dir() or entry.is_symlink():
                raise MultiRoomArtifactError("resume room artifact must be a directory")
            expected_identity = (
                dict(expected_room_results.get(entry.name, {}))
                if expected_room_results is not None
                else None
            )
            if expected_identity is not None:
                expected_identity["execution_policy_sha256"] = (
                    expected_execution_hash
                )
            terminal = load_terminal_room_result(
                entry,
                expected_schema=room_result_schema,
                expected_room_id=expected_keys[entry.name],
                expected_room_key=entry.name,
                sha256_file=sha256_file,
                expected_identity=expected_identity,
            )
            if terminal is None:
                raise MultiRoomArtifactError("resume room artifact is nonterminal")
        return len(observed_entries)
    return 0


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MultiRoomArtifactError(
            f"cannot load artifact {path.name}: {type(exc).__name__}"
        ) from exc
    if not isinstance(value, dict):
        raise MultiRoomArtifactError(f"artifact {path.name} must be an object")
    return value


def load_terminal_room_result(
    room_root: Path,
    *,
    expected_schema: str,
    expected_room_id: str,
    expected_room_key: str,
    sha256_file: Any,
    expected_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return a verified terminal room result, or fail on a partial directory."""

    if not room_root.exists():
        return None
    if not room_root.is_dir() or room_root.is_symlink():
        raise MultiRoomArtifactError(f"room artifact is not a regular directory: {room_root}")
    result_path = room_root / "room_result.json"
    if not result_path.is_file() or result_path.is_symlink():
        raise MultiRoomArtifactError(
            f"existing nonterminal room cannot be resent: {expected_room_key}"
        )
    result = _load_json(result_path)
    if result.get("schema_version") != expected_schema:
        raise MultiRoomArtifactError("resume room-result schema mismatch")
    if result.get("room_id") != expected_room_id or result.get("room_key") != expected_room_key:
        raise MultiRoomArtifactError("resume room identity mismatch")
    allowed_statuses = {
        "stage_a_failed",
        "stage_a_schema_invalid",
        "retrieval_failed",
        "stage_c_failed",
        "placement_schema_invalid",
        "complete",
    }
    status = result.get("status")
    if status not in allowed_statuses:
        raise MultiRoomArtifactError("resume room terminal status is invalid")
    if result.get("eligible_for_room_projection") is not (status == "complete"):
        raise MultiRoomArtifactError("resume room eligibility/status mismatch")
    if expected_identity is not None:
        room_brief = expected_identity.get("room_brief")
        fields = {
            key: value
            for key, value in expected_identity.items()
            if key != "room_brief"
        }
        for key, expected in fields.items():
            if result.get(key) != expected:
                raise MultiRoomArtifactError(
                    f"resume room immutable identity mismatch: {key}"
                )
        if not isinstance(room_brief, Mapping):
            raise MultiRoomArtifactError("resume expected room brief is missing")
        fixed = _load_json(room_root / "fixed_instruction.json")
        if fixed.get("room_brief") != dict(room_brief):
            raise MultiRoomArtifactError("resume fixed room brief identity mismatch")
        expected_run_identity = {
            key: fields[key]
            for key in (
                "run_input_fingerprint_sha256",
                "execution_policy_sha256",
                "source_manifest_sha256",
            )
            if key in fields
        }
        if fixed.get("run_identity") != expected_run_identity:
            raise MultiRoomArtifactError("resume fixed run identity mismatch")
    hashes = result.get("artifact_hashes")
    if not isinstance(hashes, dict):
        raise MultiRoomArtifactError("resume room result has no artifact hashes")
    if any(
        not isinstance(relative, str) or not isinstance(expected_hash, str)
        for relative, expected_hash in hashes.items()
    ):
        raise MultiRoomArtifactError("resume room artifact hash entry is invalid")
    actual_hashes = artifact_hashes(room_root, sha256_file=sha256_file)
    if hashes != actual_hashes:
        raise MultiRoomArtifactError(
            f"resume room artifact hash set mismatch: {expected_room_key}"
        )
    artifact_names = set(actual_hashes)
    required_by_status = {
        "stage_a_failed": {"fixed_instruction.json", "one_shot_audit.json"},
        "stage_a_schema_invalid": {
            "fixed_instruction.json",
            "object_plan_first_emission.json",
            "object_plan_validation.json",
            "one_shot_audit.json",
        },
        "retrieval_failed": {
            "fixed_instruction.json",
            "object_plan.json",
            "retrieval_requests.json",
            "retrieval_failure.json",
            "one_shot_audit.json",
        },
        "stage_c_failed": {
            "fixed_instruction.json",
            "object_plan.json",
            "retrieval_results.json",
            "generation_input.json",
            "one_shot_audit.json",
        },
        "placement_schema_invalid": {
            "fixed_instruction.json",
            "object_plan.json",
            "retrieval_results.json",
            "generation_input.json",
            "catalog_placement_first_emission.json",
            "placement_validation.json",
            "one_shot_audit.json",
        },
        "complete": {
            "fixed_instruction.json",
            "object_plan.json",
            "retrieval_results.json",
            "asset_selection.json",
            "generation_input.json",
            "catalog_placement_first_emission.json",
            "catalog_placement_v1.json",
            "placement_validation.json",
            "one_shot_audit.json",
        },
    }
    missing = required_by_status[status] - artifact_names
    if missing:
        raise MultiRoomArtifactError(
            f"resume room terminal artifacts disagree with status: missing={sorted(missing)}"
        )
    forbidden_by_status = {
        "stage_a_failed": {"object_plan_first_emission.json", "object_plan.json"},
        "stage_a_schema_invalid": {"object_plan.json"},
        "retrieval_failed": {"retrieval_results.json"},
        "stage_c_failed": {"catalog_placement_first_emission.json"},
        "placement_schema_invalid": {"catalog_placement_v1.json"},
        "complete": set(),
    }
    forbidden = forbidden_by_status[status] & artifact_names
    if forbidden:
        raise MultiRoomArtifactError(
            "resume room terminal artifacts disagree with status: "
            f"forbidden={sorted(forbidden)}"
        )
    if status == "complete":
        if result.get("reason_code") is not None or result.get("error_type") is not None:
            raise MultiRoomArtifactError("complete room exposes a failure reason")
    elif not isinstance(result.get("reason_code"), str) or not result["reason_code"]:
        raise MultiRoomArtifactError("failed room lacks a terminal reason code")
    return result


def artifact_hashes(room_root: Path, *, sha256_file: Any) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(room_root.rglob("*")):
        if path.is_symlink():
            raise MultiRoomArtifactError("room artifacts may not be symlinks")
        if not path.is_file() or path.name == "room_result.json":
            continue
        hashes[path.relative_to(room_root).as_posix()] = sha256_file(path)
    return hashes
