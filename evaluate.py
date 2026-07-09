from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.evaluator import evaluate_generic_validity, evaluate_oar, evaluate_oor
from benchmark.scene_io.normalize import normalize_scene
from benchmark.utils.io import read_json, write_json


def run_evaluate(
    *,
    scene: dict,
    out: str | Path,
    eval_oor: bool = False,
    eval_oar: bool = False,
    eval_generic_validity: bool = False,
    asset_csv: str | Path | None = None,
    asset_root: str | Path | None = None,
    enrich_assets: bool = False,
) -> dict:
    if not eval_oor and not eval_oar and not eval_generic_validity:
        eval_generic_validity = True
    normalized_scene = normalize_scene(scene, asset_csv=asset_csv, asset_root=asset_root, enrich_assets=enrich_assets)
    reports: dict[str, dict] = {}
    notes = []
    if eval_oor:
        reports["oor"] = evaluate_oor(normalized_scene)
    if eval_oar:
        reports["oar"] = evaluate_oar(normalized_scene)
    if eval_generic_validity:
        reports["generic_validity"] = evaluate_generic_validity(normalized_scene)
    active_scores = [float(report.get("overall_score", 0.0)) for report in reports.values() if isinstance(report, dict)]
    overall_score = 0.0 if not active_scores else sum(active_scores) / float(len(active_scores))
    report = {
        "scene_id": normalized_scene.get("scene_id"),
        "request_id": normalized_scene.get("request_id"),
        "evaluator_version": "scene_harness_evaluator_v0",
        "overall_score": float(overall_score),
        "reports": reports,
        "notes": notes,
    }
    out_path = Path(out)
    if out_path.suffix.lower() != ".json":
        out_path = out_path / "evaluation_report.json"
    write_json(out_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate canonical generated_scene.json.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--eval-oor", action="store_true")
    parser.add_argument("--eval-oar", action="store_true")
    parser.add_argument("--eval-generic-validity", action="store_true")
    parser.add_argument("--asset-csv", default=None)
    parser.add_argument("--asset-root", default=None)
    parser.add_argument("--enrich-assets", action="store_true")
    args = parser.parse_args()

    report = run_evaluate(
        scene=read_json(_path_arg(args.scene)),
        out=_path_arg(args.out),
        eval_oor=args.eval_oor,
        eval_oar=args.eval_oar,
        eval_generic_validity=args.eval_generic_validity,
        asset_csv=_path_arg(args.asset_csv) if args.asset_csv else None,
        asset_root=_path_arg(args.asset_root) if args.asset_root else None,
        enrich_assets=args.enrich_assets,
    )
    print(f"overall_score: {report['overall_score']}")
    print(f"evaluators: {', '.join(report['reports'].keys())}")


def _path_arg(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


if __name__ == "__main__":
    main()
