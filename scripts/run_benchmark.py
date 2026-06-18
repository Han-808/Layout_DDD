from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.pipeline import load_pipeline_resources, run_benchmark_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a folder of explicit 3D layout benchmark cases.")
    parser.add_argument("--cases", required=True, help="Directory containing bm_instance JSON files.")
    parser.add_argument("--model", default="mock", help="Model name from configs/model_config.yaml.")
    parser.add_argument("--max_repair_iterations", type=int, default=None)
    parser.add_argument("--out", required=True, help="Output directory for benchmark artifacts.")
    parser.add_argument("--mock_behavior", default=None, help="Optional override for mock behavior.")
    args = parser.parse_args()

    resources = load_pipeline_resources(PROJECT_ROOT)
    _, json_path, csv_path = run_benchmark_pipeline(
        cases_dir=Path(args.cases),
        out_dir=Path(args.out),
        model_name=args.model,
        resources=resources,
        max_repair_iterations=args.max_repair_iterations,
        mock_behavior=args.mock_behavior,
    )
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
