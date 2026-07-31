from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

from benchmark.visual_judge.interfaces import (
    EvidenceGateRequest,
    EvidenceGateResult,
)


EVIDENCE_GATE_VERSION = "deterministic_evidence_gate_v3"
EVIDENCE_GATE_SCOPE = "input_integrity_only"

_CORRUPT_RENDER_STATUSES = {
    "corrupt",
    "corrupted",
    "failed",
    "failure",
    "error",
    "invalid",
}


class DeterministicEvidenceGate:
    """Validate that an evidence packet is safe to pass to a Judge.

    The gate deliberately does not decide whether the packet is sufficient for
    a metric. Visibility, coverage, view composition, and other
    metric-specific questions belong to the Judge and acquisition planner.
    """

    backend = "deterministic"

    def __init__(
        self,
        *,
        allow_path_only_compatibility: bool = False,
        metric_requirements: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        if not isinstance(allow_path_only_compatibility, bool):
            raise TypeError(
                "EvidenceGate allow_path_only_compatibility must be boolean"
            )
        if allow_path_only_compatibility:
            raise ValueError(
                "EvidenceGate input-integrity checks cannot be bypassed"
            )
        if metric_requirements not in (None, {}):
            raise ValueError(
                "EvidenceGate no longer accepts metric-specific readiness "
                "requirements; metric sufficiency belongs to Judge"
            )

    def check(self, request: EvidenceGateRequest) -> EvidenceGateResult:
        if not isinstance(request, EvidenceGateRequest):
            raise TypeError("EvidenceGate requires an EvidenceGateRequest")

        items = list(deepcopy(request.visual_evidence))
        deficiencies: list[dict[str, str]] = []
        checks = [
            "evidence_packet_present",
            "evidence_paths_present",
            "evidence_files_exist",
            "evidence_files_nonempty",
            "evidence_images_decodable",
            "evidence_images_nonblank",
            "explicit_render_status_valid",
        ]
        manifest_matches = 0
        manifest_failure: str | None = None

        if request.manifest_path:
            checks.append("referenced_manifest_valid")
            manifest_matches, manifest_failure = _validate_manifest(
                items,
                request.manifest_path,
            )
            if manifest_failure is not None:
                deficiencies.append(
                    _deficiency(manifest_failure, "manifest")
                )

        if not items:
            deficiencies.append(
                _deficiency("visual_evidence_missing", "rerender")
            )

        decoded_count = 0
        blank_count = 0
        for item in items:
            status_deficiency = _explicit_status_deficiency(item)
            if status_deficiency is not None:
                deficiencies.append(status_deficiency)

            path = _evidence_path(item)
            if path is None:
                deficiencies.append(
                    _deficiency("evidence_path_missing", "rerender")
                )
                continue
            path = path.expanduser()
            if not path.is_file():
                deficiencies.append(
                    _deficiency("evidence_file_missing", "rerender")
                )
                continue
            try:
                if path.stat().st_size <= 0:
                    deficiencies.append(
                        _deficiency("empty_render_file", "rerender")
                    )
                    continue
            except OSError:
                deficiencies.append(
                    _deficiency("evidence_file_unreadable", "rerender")
                )
                continue

            image_status = _inspect_image(path)
            if image_status == "undecodable":
                deficiencies.append(
                    _deficiency("undecodable_render", "rerender")
                )
            elif image_status == "blank":
                decoded_count += 1
                blank_count += 1
                deficiencies.append(
                    _deficiency("blank_render", "rerender")
                )
            else:
                decoded_count += 1

        deficiencies = _dedupe(deficiencies)
        provenance = {
            "schema_version": EVIDENCE_GATE_VERSION,
            "scope": EVIDENCE_GATE_SCOPE,
            "metric_sufficiency_owner": "judge",
            "camera_request_owner": "judge_evidence_request",
            "checks_applied": checks,
            "semantic_checks_applied": [],
            "evidence_item_count": len(items),
            "decoded_image_count": decoded_count,
            "blank_image_count": blank_count,
            "manifest_metadata_count": manifest_matches,
            "manifest_status": (
                manifest_failure
                or (
                    "valid"
                    if request.manifest_path is not None
                    else "not_provided"
                )
            ),
        }

        if deficiencies:
            return EvidenceGateResult(
                ready=False,
                camera_repairable=False,
                reason_codes=tuple(
                    item["code"] for item in deficiencies
                ),
                deficiencies=tuple(deepcopy(deficiencies)),
                backend=self.backend,
                provenance=provenance,
            )
        return EvidenceGateResult(
            ready=True,
            camera_repairable=False,
            reason_codes=("evidence_ready",),
            deficiencies=(),
            backend=self.backend,
            provenance=provenance,
        )


def _inspect_image(path: Path) -> str:
    """Return valid, blank, or undecodable without metric-specific thresholds."""

    try:
        with Image.open(path) as opened:
            opened.load()
            image = ImageOps.exif_transpose(opened)
            if image.width <= 0 or image.height <= 0:
                return "undecodable"

            if "A" in image.getbands():
                rgba = image.convert("RGBA")
                alpha = rgba.getchannel("A")
                if alpha.getbbox() is None:
                    return "blank"
                flattened = Image.new("RGB", rgba.size, (255, 255, 255))
                flattened.paste(rgba, mask=alpha)
                rgb = flattened
            else:
                rgb = image.convert("RGB")

            # A single-colour frame contains no visual evidence regardless of
            # which clear/background colour the renderer used.
            extrema = rgb.getextrema()
            if all(low == high for low, high in extrema):
                return "blank"
            return "valid"
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        return "undecodable"


def _explicit_status_deficiency(
    item: Any,
) -> dict[str, str] | None:
    if not isinstance(item, dict):
        return None
    # Auxiliary visibility/mask analysis is not image integrity. A failed
    # semantic-analysis status must not block a valid decodable render.
    status = str(item.get("render_status") or "").strip().lower()
    if status == "blank":
        return _deficiency("blank_render", "rerender")
    if status in _CORRUPT_RENDER_STATUSES:
        return _deficiency("corrupt_render_evidence", "rerender")
    return None


def _validate_manifest(
    items: list[Any],
    manifest_path: str,
) -> tuple[int, str | None]:
    path = Path(manifest_path).expanduser()
    if not path.is_file():
        return 0, "evidence_manifest_missing"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        return 0, "evidence_manifest_unreadable"
    except json.JSONDecodeError:
        return 0, "evidence_manifest_invalid"
    if not isinstance(manifest, dict):
        return 0, "evidence_manifest_invalid"

    manifest_items: list[dict[str, Any]] = []
    found_evidence_list = False
    for key in ("render_evidence_items", "visual_evidence", "views"):
        if key not in manifest:
            continue
        found_evidence_list = True
        values = manifest[key]
        if not isinstance(values, list) or any(
            not isinstance(item, dict) for item in values
        ):
            return 0, "evidence_manifest_invalid"
        manifest_items.extend(values)
    if not found_evidence_list or not manifest_items:
        return 0, "evidence_manifest_evidence_items_missing"

    manifest_paths = {
        str(path)
        for item in manifest_items
        if (path := _evidence_path(item)) is not None
    }
    item_paths = [
        str(path)
        for item in items
        if (path := _evidence_path(item)) is not None
    ]
    matches = sum(path in manifest_paths for path in item_paths)
    if items and matches == 0:
        return 0, "evidence_manifest_evidence_mismatch"
    return matches, None


def _evidence_path(item: Any) -> Path | None:
    if isinstance(item, dict):
        value = item.get("path") or item.get("image_path")
    else:
        value = item
    if value is None or not str(value).strip():
        return None
    return Path(str(value))


def _deficiency(code: str, repairability: str) -> dict[str, str]:
    return {"code": str(code), "repairability": str(repairability)}


def _dedupe(items: list[dict[str, str]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        key = (item["code"], item["repairability"])
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result
