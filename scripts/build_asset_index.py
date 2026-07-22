"""Build the semantic asset index used by the retriever (offline, one-time).

Summary:
    Embeds Imaginarium asset metadata into a searchable index so the
    fixed-asset-assisted task can retrieve candidate assets.

Input:
    - ``--asset-csv`` (``asset_info.csv``) and ``--asset-root`` (asset database).
    - ``--embedding-model`` (default Qwen3-Embedding).

Output:
    - ``--out`` index prefix -> a ``.json`` metadata file and a ``.npy``
      embeddings file; prints the counts and written paths.

Function:
    Thin CLI wrapper over ``build_asset_index_from_asset_info(...)``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.assets.retriever import build_asset_index_from_asset_info


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the semantic Imaginarium asset index used by the retriever.")
    parser.add_argument("--asset-csv", required=True)
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--out", required=True, help="Index prefix; writes matching .json and .npy files.")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    args = parser.parse_args()

    asset_csv = Path(args.asset_csv).expanduser().resolve()
    asset_root = Path(args.asset_root).expanduser().resolve()
    output = Path(args.out).expanduser().resolve()
    if not asset_csv.is_file():
        parser.error(f"asset CSV does not exist: {asset_csv}")
    if not asset_root.is_dir():
        parser.error(f"asset root does not exist: {asset_root}")

    index = build_asset_index_from_asset_info(
        asset_info_csv_path=asset_csv,
        asset_dir=asset_root,
        output_path=output,
        embedding_model=args.embedding_model,
        show_progress=True,
    )
    print(f"assets_indexed: {len(index)}")
    print(f"metadata: {output.with_suffix('.json')}")
    print(f"embeddings: {output.with_suffix('.npy')}")


if __name__ == "__main__":
    main()
