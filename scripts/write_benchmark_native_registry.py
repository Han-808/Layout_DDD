from __future__ import annotations

import argparse
from pathlib import Path

from benchmark.materialization import (
    NativeRegistryAuthority,
    write_benchmark_native_registry,
)
from benchmark.utils.io import read_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark placement-tool step: seal a registered rigid native "
            "Blender placement registry."
        )
    )
    parser.add_argument("--source-blend", required=True)
    parser.add_argument("--instances-json", required=True)
    parser.add_argument("--case-bundle-manifest-sha256", required=True)
    parser.add_argument("--catalog-snapshot-id", required=True)
    parser.add_argument("--authority-key-file", required=True)
    parser.add_argument("--authority-key-id", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    instances_value = read_json(
        Path(args.instances_json).expanduser().resolve()
    )
    if isinstance(instances_value, dict):
        instances_value = instances_value.get("instances")
    if not isinstance(instances_value, list):
        parser.error("--instances-json must contain a list or {instances: [...]}")
    secret = Path(args.authority_key_file).expanduser().resolve().read_bytes()
    authority = NativeRegistryAuthority.from_secret(
        key_id=args.authority_key_id,
        secret=secret,
    )
    output = write_benchmark_native_registry(
        args.out,
        authority=authority,
        source_blend_path=args.source_blend,
        case_bundle_manifest_sha256=args.case_bundle_manifest_sha256,
        catalog_snapshot_id=args.catalog_snapshot_id,
        instances=instances_value,
    )
    print(output.as_posix())


if __name__ == "__main__":
    main()
