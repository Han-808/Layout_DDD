"""Optional post-hoc audit-graph artifact projection for camera-cal runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark.visual_judge.graphs import export_case_audit_graphs

from benchmark.camera_cal_scene_level.progress import ProgressReporter


def maybe_export_audit_graphs(
    *,
    enabled: bool,
    case_id: str,
    case_out: Path,
    grouping_report: dict[str, Any],
    scene_quality_report: dict[str, Any],
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    if not enabled:
        return {
            "enabled": False,
            "status": "disabled",
            "decision_authority": "none",
        }
    result = export_case_audit_graphs(
        case_id=case_id,
        grouping_report=grouping_report,
        scene_quality_report=scene_quality_report,
        output_dir=case_out / "audit_graphs",
    )
    if progress is not None:
        progress.emit(
            "audit_graph_export_completed",
            case_id=case_id,
            status=result["status"],
            relation_candidate_count=(
                (result.get("relation_candidate_graph") or {}).get(
                    "candidate_count"
                )
            ),
            evaluation_query_graph_count=len(
                result.get("evaluation_query_graphs") or []
            ),
            decision_authority="none",
        )
    return {
        "enabled": True,
        "status": result["status"],
        "schema_version": result["schema_version"],
        "decision_authority": "none",
        "manifest_path": str(
            (case_out / "audit_graphs" / "manifest.json").resolve()
        ),
        "relation_candidate_count": (
            (result.get("relation_candidate_graph") or {}).get(
                "candidate_count"
            )
        ),
        "evaluation_query_graph_count": len(
            result.get("evaluation_query_graphs") or []
        ),
        "error_type": result.get("error_type"),
        "error": result.get("error"),
    }


__all__ = ["maybe_export_audit_graphs"]
