from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from benchmark.evaluator.generic_validity.asset_resolver import resolve_asset_metadata


def load_asset_csv(asset_csv_path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not asset_csv_path:
        return {}
    path = Path(asset_csv_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Asset CSV not found: {path}")
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            jid = str(row.get("name_en") or "").strip()
            if jid:
                rows[jid] = dict(row)
    return rows


def enrich_object_with_asset_metadata(
    obj: dict,
    *,
    asset_csv_path: str | Path | None = None,
    asset_root: str | Path | None = None,
) -> dict:
    return resolve_asset_metadata(obj, asset_csv_path=asset_csv_path, asset_root=asset_root)


def resolve_asset_metadata_for_jid(
    jid: str,
    *,
    asset_csv_path: str | Path | None = None,
    asset_root: str | Path | None = None,
) -> dict:
    return resolve_asset_metadata({"id": jid, "jid": jid}, asset_csv_path=asset_csv_path, asset_root=asset_root)
