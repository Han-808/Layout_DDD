from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.evaluator import evaluate_generic_validity
from benchmark.evaluator.generic_validity.asset_resolver import enrich_scene_assets
from benchmark.utils.io import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate deterministic generic scene validity.")
    parser.add_argument("--scene", required=True, help="Generated scene JSON.")
    parser.add_argument("--config", default=None, help="Optional JSON config override.")
    parser.add_argument("--asset-csv", default=None, help="Optional asset_info.csv used for asset metadata enrichment.")
    parser.add_argument("--asset-root", default=None, help="Optional asset database root used for metadata/URI resolution.")
    parser.add_argument("--enrich-assets", action="store_true", help="Resolve scene objects against asset CSV/root before evaluation.")
    parser.add_argument("--write-enriched-scene", default=None, help="Optional path to write the enriched scene JSON.")
    parser.add_argument("--out", required=True, help="Output report JSON path.")
    args = parser.parse_args()

    scene = read_json(_path_arg(args.scene))
    enrichment_report = None
    if args.enrich_assets:
        scene, enrichment_report = enrich_scene_assets(
            scene,
            asset_csv_path=_path_arg(args.asset_csv) if args.asset_csv else None,
            asset_root=_path_arg(args.asset_root) if args.asset_root else None,
        )
        if args.write_enriched_scene:
            write_json(_path_arg(args.write_enriched_scene), scene)
    config = read_json(_path_arg(args.config)) if args.config else None
    if config is not None and not isinstance(config, dict):
        parser.error("--config must point to a JSON object.")
    report = evaluate_generic_validity(scene, config=config)
    if enrichment_report is not None:
        report["asset_enrichment"] = enrichment_report
    out_path = write_json(_path_arg(args.out), report)
    print(f"overall_score: {report['overall_score']}")
    for metric in ["collision", "oob", "navigability", "accessibility", "support"]:
        result = report.get("metrics", {}).get(metric)
        if isinstance(result, dict):
            print(f"{metric}_score: {result.get('score')}")
    print(f"report: {out_path}")


def _path_arg(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


if __name__ == "__main__":
    main()
