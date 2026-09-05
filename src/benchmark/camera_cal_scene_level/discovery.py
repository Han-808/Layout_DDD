"""Case discovery and evidence-path helpers for camera-cal evaluation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterable

from benchmark.case_ids import is_safe_case_id, validate_case_id
from benchmark.camera_cal_scene_level.io import read_json as _read_json


# Keep this alias at module scope so direct users can monkeypatch the I/O
# operation.  The implementation resolves it at call time rather than taking
# an import-time snapshot.
read_json = _read_json


def case_paths(
    case_root: Path,
    manifest: dict[str, Any],
) -> dict[str, Path]:
    paths = manifest.get("paths")
    paths = paths if isinstance(paths, dict) else {}
    evidence = paths.get("evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    return {
        "scene": case_root
        / str(paths.get("canonical_scene") or "scene/canonical_scene.json"),
        "blend": case_root
        / str(paths.get("blend") or "prepared/evaluation.blend"),
        "annotation": case_root
        / str(paths.get("annotation") or "annotation.json"),
        "perspective": case_root
        / str(
            evidence.get("perspective")
            or "evidence/standardized_perspective.png"
        ),
        "top": case_root
        / str(evidence.get("top") or "evidence/standardized_top.png"),
        "identity": case_root
        / str(
            evidence.get("identity")
            or "evidence/standardized_identity_map.png"
        ),
        "render_manifest": case_root
        / "evidence"
        / "prepared_render_manifest.json",
        "collision_geometry": (
            case_root / "evidence" / "collision_geometry_manifest.json"
        ),
    }


def _discover_cases_impl(
    dataset_root: Path,
    *,
    case_ids: Iterable[str] = (),
    max_cases: int | None = None,
    read_json_fn: Callable[[Path], dict[str, Any]] | None = None,
    case_paths_fn: Callable[[Path, dict[str, Any]], dict[str, Path]] | None = None,
) -> list[dict[str, Any]]:
    """Implementation seam allowing the compatibility façade to inject live
    runner globals.
    """

    reader = read_json if read_json_fn is None else read_json_fn
    paths_for_case = case_paths if case_paths_fn is None else case_paths_fn

    root = dataset_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"camera-cal dataset does not exist: {root}")
    materialization_state = root / "materialization_state.json"
    if materialization_state.exists() or materialization_state.is_symlink():
        if materialization_state.is_symlink() or not materialization_state.is_file():
            raise ValueError("dataset materialization state must be a regular file")
        state = reader(materialization_state)
        if (
            state.get("schema_version")
            != "multi_room_evaluation_build_state_v1"
            or state.get("status") != "finalized"
        ):
            raise ValueError("evaluation dataset materialization is not finalized")
    selected_ids = [
        validate_case_id(value, field="camera-cal case ID")
        for value in case_ids
    ]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("camera-cal case IDs must be unique")

    discovered: dict[str, dict[str, Any]] = {}
    for case_root in sorted(root.iterdir()):
        if not case_root.is_dir() or not is_safe_case_id(case_root.name):
            continue
        manifest_path = case_root / "case_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = reader(manifest_path)
        if manifest.get("status") != "ready":
            continue
        case_id = validate_case_id(
            manifest.get("case_id") or case_root.name,
            field=f"{manifest_path} case_id",
        )
        if case_id != case_root.name:
            raise ValueError(
                f"ready camera-cal case ID differs from its directory: {case_root.name!r}"
            )
        required = paths_for_case(case_root, manifest)
        missing = [
            name
            for name, path in required.items()
            if name != "render_manifest" and not path.is_file()
        ]
        if missing:
            continue
        if case_id in discovered:
            raise ValueError(f"duplicate ready camera-cal case ID: {case_id!r}")
        discovered[case_id] = {
            "case_id": case_id,
            "case_root": str(case_root),
            "scene_type": manifest.get("scene_type"),
            "object_count": manifest.get("object_count"),
            "semantic_content_fingerprint": manifest.get(
                "semantic_content_fingerprint"
            ),
        }
    if selected_ids:
        missing_ids = [
            case_id for case_id in selected_ids if case_id not in discovered
        ]
        if missing_ids:
            raise ValueError(f"requested cases are not ready: {missing_ids}")
        cases = [discovered[case_id] for case_id in selected_ids]
    else:
        cases = [discovered[case_id] for case_id in sorted(discovered)]
    if max_cases is not None:
        cases = cases[:max_cases]
    if not cases:
        raise ValueError("no ready camera-cal cases were selected")
    return cases


def discover_cases(
    dataset_root: Path,
    *,
    case_ids: Iterable[str] = (),
    max_cases: int | None = None,
) -> list[dict[str, Any]]:
    """Discover ready camera-cal cases using the historical contract."""

    return _discover_cases_impl(
        dataset_root,
        case_ids=case_ids,
        max_cases=max_cases,
    )


def _endpoint_preflight_image_impl(
    case: dict[str, Any],
    *,
    read_json_fn: Callable[[Path], dict[str, Any]] | None = None,
    case_paths_fn: Callable[[Path, dict[str, Any]], dict[str, Path]] | None = None,
) -> Path:
    reader = read_json if read_json_fn is None else read_json_fn
    paths_for_case = case_paths if case_paths_fn is None else case_paths_fn
    source_root = Path(str(case["case_root"])).expanduser().resolve()
    manifest = reader(source_root / "case_manifest.json")
    image_path = paths_for_case(source_root, manifest)["perspective"]
    if not image_path.is_file():
        raise FileNotFoundError(
            f"endpoint preflight image does not exist: {image_path}"
        )
    return image_path


def _endpoint_preflight_image(case: dict[str, Any]) -> Path:
    return _endpoint_preflight_image_impl(case)


def _identity_legend_from_manifest_impl(
    path: Path,
    *,
    read_json_fn: Callable[[Path], dict[str, Any]] | None = None,
) -> dict[str, str]:
    reader = read_json if read_json_fn is None else read_json_fn
    if not path.is_file():
        return {}
    manifest = reader(path)
    legend = manifest.get("identity_legend")
    if not isinstance(legend, dict):
        return {}
    return {
        str(alias): str(object_id)
        for alias, object_id in legend.items()
        if str(alias).strip() and str(object_id).strip()
    }


def identity_legend_from_manifest(path: Path) -> dict[str, str]:
    return _identity_legend_from_manifest_impl(path)


def grouping_evidence_packet(
    *,
    paths: dict[str, Path],
    identity_legend: dict[str, str],
) -> list[dict[str, Any]]:
    return [
        {
            "path": str(paths["perspective"].resolve()),
            "role": "global_perspective_rgb",
            "representation": "rgb",
            "view_id": "global_perspective",
            "camera_scope": "global",
        },
        {
            "path": str(paths["top"].resolve()),
            "role": "global_top_rgb",
            "representation": "rgb",
            "view_id": "global_top",
            "camera_scope": "global",
        },
        {
            "path": str(paths["identity"].resolve()),
            "role": "global_identity_overlay",
            "representation": "identity_map",
            "view_id": "global_identity",
            "camera_scope": "global",
            "identity_overlay": True,
            "identity_legend": deepcopy(identity_legend),
        },
    ]


__all__ = [
    "case_paths",
    "discover_cases",
    "grouping_evidence_packet",
    "identity_legend_from_manifest",
]
