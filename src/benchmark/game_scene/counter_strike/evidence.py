"""Load one immutable Counter-Strike browser-capture evidence bank.

The loader consumes an existing ``render_manifest.json``.  It never opens a
browser, selects views, calls a model, or computes a metric.  Its trust boundary
matches :class:`benchmark.rendering.browser.FrozenBrowserCaptureRenderer`: every
declared capture artifact must remain under the original capture directory,
exist, and match its frozen SHA-256 digest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark.rendering.browser import (
    BROWSER_RENDER_BACKEND,
    CONTROLLED_CAMERA_APPEARANCE_FIDELITY,
)

from .loader import CounterStrikeBenchmarkConfig


GLOBAL_EVIDENCE_ROLE = "global_controlled"
REGIONAL_EVIDENCE_ROLE = "style_local_fallback"
EXPECTED_GLOBAL_IDS = ("global_oblique_00", "global_oblique_01")
EXPECTED_REGIONAL_IDS = (
    "style_region_00",
    "style_region_01",
    "style_region_02",
    "style_region_03",
)


class CounterStrikeEvidenceError(ValueError):
    """Raised when a frozen browser capture cannot be trusted as CS evidence."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(
            f"Counter-Strike frozen evidence invalid [{code}]: {message}"
        )


@dataclass(frozen=True)
class CounterStrikeEvidenceDescriptor:
    """One ordered, hash-bound image from the frozen capture."""

    id: str
    role: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class CounterStrikeFrozenEvidence:
    """The complete fixed six-image evidence bank for one capture."""

    capture_dir: Path
    manifest_path: Path
    manifest_sha256: str
    global_views: tuple[CounterStrikeEvidenceDescriptor, ...]
    regional_views: tuple[CounterStrikeEvidenceDescriptor, ...]

    @property
    def ordered(self) -> tuple[CounterStrikeEvidenceDescriptor, ...]:
        """Global context first, followed by the regional fallback bank."""

        return self.global_views + self.regional_views


def load_counter_strike_frozen_evidence(
    capture_dir: str | Path,
    *,
    benchmark_config: CounterStrikeBenchmarkConfig,
) -> CounterStrikeFrozenEvidence:
    """Validate and expose the fixed browser evidence in manifest order."""

    if not isinstance(benchmark_config, CounterStrikeBenchmarkConfig):
        raise CounterStrikeEvidenceError(
            "benchmark_config_unvalidated",
            "benchmark_config must come from "
            "load_counter_strike_benchmark_config",
        )
    expected_global, expected_regional = _expected_budgets(benchmark_config)

    root = Path(capture_dir).expanduser().resolve()
    if not root.is_dir():
        raise CounterStrikeEvidenceError(
            "capture_dir_missing",
            f"capture directory does not exist: {root}",
        )
    manifest_path = root / "render_manifest.json"
    if not manifest_path.is_file():
        raise CounterStrikeEvidenceError(
            "manifest_missing",
            f"render_manifest.json does not exist under {root}",
        )
    manifest = _read_manifest(manifest_path)
    if manifest.get("backend") != BROWSER_RENDER_BACKEND:
        raise CounterStrikeEvidenceError(
            "backend_mismatch",
            f"expected backend {BROWSER_RENDER_BACKEND!r}",
        )

    artifact_hashes = _validate_capture_artifacts(manifest, capture_dir=root)
    _validate_exported_scene(manifest, capture_dir=root, artifact_hashes=artifact_hashes)
    controlled = _controlled_camera_manifest(manifest)

    global_views = _load_view_bank(
        manifest.get("views"),
        capture_dir=root,
        artifact_hashes=artifact_hashes,
        expected_count=expected_global,
        expected_ids=EXPECTED_GLOBAL_IDS,
        role=GLOBAL_EVIDENCE_ROLE,
        expected_scope="global",
    )
    fallback = controlled.get("style_local_fallback")
    if not isinstance(fallback, dict) or fallback.get("status") != "ready":
        raise CounterStrikeEvidenceError(
            "regional_bank_unavailable",
            "controlled_camera.style_local_fallback must have status 'ready'",
        )
    visual = benchmark_config.raw["visual_evidence"]
    if fallback.get("enabled") is not True:
        raise CounterStrikeEvidenceError(
            "regional_bank_unavailable",
            "controlled_camera.style_local_fallback must be enabled",
        )
    if fallback.get("view_family") != visual["regional_view_family"]:
        raise CounterStrikeEvidenceError(
            "view_family_mismatch",
            "regional view family does not match the frozen benchmark config",
        )
    if fallback.get("image_budget") != expected_regional:
        raise CounterStrikeEvidenceError(
            "evidence_budget_mismatch",
            "regional manifest image_budget does not match the frozen "
            f"benchmark config ({expected_regional})",
        )
    regional_views = _load_view_bank(
        fallback.get("views"),
        capture_dir=root,
        artifact_hashes=artifact_hashes,
        expected_count=expected_regional,
        expected_ids=EXPECTED_REGIONAL_IDS,
        role=REGIONAL_EVIDENCE_ROLE,
        expected_scope="object_local",
    )

    if controlled.get("view_family") != visual["global_view_family"]:
        raise CounterStrikeEvidenceError(
            "view_family_mismatch",
            "global view family does not match the frozen benchmark config",
        )
    if controlled.get("image_budget") != expected_global:
        raise CounterStrikeEvidenceError(
            "evidence_budget_mismatch",
            "global manifest image_budget does not match the frozen "
            f"benchmark config ({expected_global})",
        )
    return CounterStrikeFrozenEvidence(
        capture_dir=root,
        manifest_path=manifest_path,
        manifest_sha256=_sha256(manifest_path),
        global_views=global_views,
        regional_views=regional_views,
    )


def _expected_budgets(
    config: CounterStrikeBenchmarkConfig,
) -> tuple[int, int]:
    visual = config.raw["visual_evidence"]
    global_budget = int(visual["global_image_budget"])
    regional_budget = int(visual["regional_image_budget"])
    if global_budget != len(EXPECTED_GLOBAL_IDS) or regional_budget != len(
        EXPECTED_REGIONAL_IDS
    ):
        raise CounterStrikeEvidenceError(
            "unsupported_benchmark_evidence_budget",
            "counter_strike_static_spatial_benchmark_v1 requires exactly "
            "2 global and 4 regional views",
        )
    return global_budget, regional_budget


def _controlled_camera_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    controlled = manifest.get("controlled_camera")
    if not isinstance(controlled, dict):
        raise CounterStrikeEvidenceError(
            "controlled_camera_missing",
            "render manifest has no controlled_camera object",
        )
    if controlled.get("enabled") is not True or controlled.get("status") != "ready":
        raise CounterStrikeEvidenceError(
            "controlled_camera_unavailable",
            "controlled camera must be enabled with status 'ready'",
        )
    if controlled.get("appearance_fidelity") != CONTROLLED_CAMERA_APPEARANCE_FIDELITY:
        raise CounterStrikeEvidenceError(
            "appearance_fidelity_mismatch",
            "controlled camera did not preserve original-runtime appearance",
        )
    return controlled


def _load_view_bank(
    value: Any,
    *,
    capture_dir: Path,
    artifact_hashes: dict[Path, str],
    expected_count: int,
    expected_ids: tuple[str, ...],
    role: str,
    expected_scope: str,
) -> tuple[CounterStrikeEvidenceDescriptor, ...]:
    if not isinstance(value, list):
        raise CounterStrikeEvidenceError(
            "evidence_bank_invalid",
            f"{role} evidence bank must be a JSON array",
        )
    if len(value) != expected_count:
        raise CounterStrikeEvidenceError(
            "evidence_budget_mismatch",
            f"{role} evidence requires exactly {expected_count} views, "
            f"got {len(value)}",
        )
    descriptors: list[CounterStrikeEvidenceDescriptor] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise CounterStrikeEvidenceError(
                "evidence_view_invalid",
                f"{role}[{index}] must be a JSON object",
            )
        view_id = str(item.get("id") or "").strip()
        if not view_id:
            raise CounterStrikeEvidenceError(
                "evidence_view_invalid",
                f"{role}[{index}].id must be a non-empty string",
            )
        if view_id in seen_ids:
            raise CounterStrikeEvidenceError(
                "duplicate_evidence_id",
                f"{role} repeats view id {view_id!r}",
            )
        seen_ids.add(view_id)
        if item.get("scope") != expected_scope:
            raise CounterStrikeEvidenceError(
                "evidence_role_mismatch",
                f"{role}[{index}] must have scope {expected_scope!r}",
            )
        if role == REGIONAL_EVIDENCE_ROLE and item.get("role") != role:
            raise CounterStrikeEvidenceError(
                "evidence_role_mismatch",
                f"{role}[{index}] must declare role {role!r}",
            )
        if item.get("appearance_fidelity") != CONTROLLED_CAMERA_APPEARANCE_FIDELITY:
            raise CounterStrikeEvidenceError(
                "appearance_fidelity_mismatch",
                f"{role}[{index}] is not an original-runtime controlled view",
            )
        path = _resolve_under_capture(
            item.get("path"),
            capture_dir=capture_dir,
            code="evidence_path_outside_root",
            label=f"{role}[{index}].path",
        )
        if not path.is_file():
            raise CounterStrikeEvidenceError(
                "evidence_file_missing",
                f"{role}[{index}] does not exist: {path}",
            )
        expected_hash = artifact_hashes.get(path)
        if expected_hash is None:
            raise CounterStrikeEvidenceError(
                "evidence_path_unhashed",
                f"{role}[{index}] is absent from capture_artifacts: {path}",
            )
        descriptors.append(
            CounterStrikeEvidenceDescriptor(
                id=view_id,
                role=role,
                path=path,
                sha256=expected_hash,
            )
        )
    actual_ids = tuple(item.id for item in descriptors)
    if actual_ids != expected_ids:
        raise CounterStrikeEvidenceError(
            "evidence_order_mismatch",
            f"{role} IDs must be ordered as {list(expected_ids)}, "
            f"got {list(actual_ids)}",
        )
    return tuple(descriptors)


def _validate_capture_artifacts(
    manifest: dict[str, Any],
    *,
    capture_dir: Path,
) -> dict[Path, str]:
    raw_hashes = manifest.get("capture_artifacts")
    if not isinstance(raw_hashes, dict) or not raw_hashes:
        raise CounterStrikeEvidenceError(
            "artifact_hashes_missing",
            "render manifest has no capture_artifacts hashes",
        )
    verified: dict[Path, str] = {}
    for raw_path, raw_digest in raw_hashes.items():
        path = _resolve_under_capture(
            raw_path,
            capture_dir=capture_dir,
            code="artifact_path_outside_root",
            label=f"capture_artifacts[{raw_path!r}]",
        )
        if path in verified:
            raise CounterStrikeEvidenceError(
                "duplicate_artifact_path",
                f"multiple capture_artifacts keys resolve to {path}",
            )
        digest = str(raw_digest)
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise CounterStrikeEvidenceError(
                "artifact_hash_invalid",
                f"capture artifact has no lowercase SHA-256 digest: {raw_path!r}",
            )
        if not path.is_file():
            raise CounterStrikeEvidenceError(
                "artifact_missing",
                f"frozen capture artifact does not exist: {path}",
            )
        actual = _sha256(path)
        if actual != digest:
            raise CounterStrikeEvidenceError(
                "artifact_hash_mismatch",
                f"hash mismatch for {path}: expected {digest}, got {actual}",
            )
        verified[path] = digest
    return verified


def _validate_exported_scene(
    manifest: dict[str, Any],
    *,
    capture_dir: Path,
    artifact_hashes: dict[Path, str],
) -> None:
    path = _resolve_under_capture(
        manifest.get("exported_scene"),
        capture_dir=capture_dir,
        code="exported_scene_outside_root",
        label="exported_scene",
    )
    if not path.is_file():
        raise CounterStrikeEvidenceError(
            "exported_scene_missing",
            f"frozen capture has no exported canonical scene: {path}",
        )
    if path not in artifact_hashes:
        raise CounterStrikeEvidenceError(
            "exported_scene_unhashed",
            "exported canonical scene is absent from capture_artifacts",
        )
    try:
        with path.open("r", encoding="utf-8") as handle:
            scene = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CounterStrikeEvidenceError(
            "exported_scene_invalid",
            f"could not parse exported canonical scene: {exc}",
        ) from exc
    if not isinstance(scene, dict) or scene.get("schema_version") != "canonical_scene_v1":
        raise CounterStrikeEvidenceError(
            "exported_scene_invalid",
            "exported scene must be a canonical_scene_v1 JSON object",
        )


def _resolve_under_capture(
    raw_path: Any,
    *,
    capture_dir: Path,
    code: str,
    label: str,
) -> Path:
    text = str(raw_path or "").strip()
    if not text:
        raise CounterStrikeEvidenceError(
            "artifact_path_invalid",
            f"{label} must be a non-empty path",
        )
    candidate = Path(text).expanduser()
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (capture_dir / candidate).resolve()
    )
    try:
        resolved.relative_to(capture_dir)
    except ValueError as exc:
        raise CounterStrikeEvidenceError(
            code,
            f"{label} escapes the capture directory: {resolved}",
        ) from exc
    return resolved


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CounterStrikeEvidenceError(
            "manifest_parse_error",
            f"could not parse {path}: {exc}",
        ) from exc
    if not isinstance(value, dict):
        raise CounterStrikeEvidenceError(
            "manifest_invalid",
            "render_manifest.json must contain a JSON object",
        )
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
