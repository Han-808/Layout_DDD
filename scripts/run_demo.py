from __future__ import annotations

import argparse
import http.server
import sys
from functools import partial
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.pipeline import copy_viewer_assets, load_pipeline_resources, run_case_pipeline
from benchmark.utils.io import ensure_dir, read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal 3D layout benchmark visualization demo.")
    parser.add_argument(
        "--case",
        default=str(PROJECT_ROOT / "data" / "benchmark_cases" / "hssd_small_room_full" / "102344115_structured_basic.json"),
        help="Path to a benchmark case JSON.",
    )
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "outputs" / "demo"),
        help="Output directory. Static viewer files are copied here.",
    )
    parser.add_argument("--port", type=int, default=8000, help="Port for --serve.")
    parser.add_argument("--serve", action="store_true", help="Start a local http.server after writing files.")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out)
    bm_instance = read_json(args.case)
    input_case_path = write_json(out_dir / "bm_instance.json", bm_instance)
    resources = load_pipeline_resources(PROJECT_ROOT)
    run_case_pipeline(
        case_path=input_case_path,
        out_dir=out_dir,
        model_name="mock",
        resources=resources,
        max_repair_iterations=1,
        mock_behavior="colliding_then_repair",
    )
    copy_viewer_assets(out_dir, PROJECT_ROOT)

    print(f"Demo written to: {out_dir}")
    print(f"Open after starting a local server:")
    print(f"  cd {out_dir}")
    print(f"  python -m http.server {args.port}")
    print(f"  # Windows launcher alternative: py -m http.server {args.port}")
    print(f"  http://localhost:{args.port}")

    if args.serve:
        _serve(out_dir, args.port)


def _serve(out_dir: Path, port: int) -> None:
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(out_dir))
    server = http.server.ThreadingHTTPServer(("localhost", port), handler)
    print(f"Serving http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
