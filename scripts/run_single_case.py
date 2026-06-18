from __future__ import annotations

import argparse
import http.server
import sys
from functools import partial
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.pipeline import copy_viewer_assets, load_pipeline_resources, run_case_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one explicit 3D layout benchmark case.")
    parser.add_argument("--case", required=True, help="Path to bm_instance JSON.")
    parser.add_argument("--model", default="mock", help="Model name from configs/model_config.yaml.")
    parser.add_argument("--max_repair_iterations", type=int, default=None)
    parser.add_argument("--out", required=True, help="Output directory for intermediate artifacts.")
    parser.add_argument("--mock_behavior", default=None, help="Optional override for mock behavior.")
    parser.add_argument("--no_viewer_assets", action="store_true", help="Do not copy static viewer files into --out.")
    parser.add_argument("--serve", action="store_true", help="Serve --out with a local HTTP server after the run.")
    parser.add_argument("--port", type=int, default=8000, help="Port for --serve.")
    args = parser.parse_args()

    resources = load_pipeline_resources(PROJECT_ROOT)
    state = run_case_pipeline(
        case_path=Path(args.case),
        out_dir=Path(args.out),
        model_name=args.model,
        resources=resources,
        max_repair_iterations=args.max_repair_iterations,
        mock_behavior=args.mock_behavior,
    )
    if not args.no_viewer_assets and state.get("viewer_scene_path"):
        copy_viewer_assets(args.out, PROJECT_ROOT)
    print(state["per_case_result_path"])
    if not args.no_viewer_assets and state.get("viewer_scene_path"):
        print(f"viewer: http://127.0.0.1:{args.port}/")
    if args.serve:
        _serve(Path(args.out), args.port)


def _serve(out_dir: Path, port: int) -> None:
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(out_dir))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"Serving http://127.0.0.1:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
