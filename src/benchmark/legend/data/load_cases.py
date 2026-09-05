from __future__ import annotations

from pathlib import Path
from typing import Iterable

from benchmark.utils.io import read_json


def load_case(path: str | Path) -> dict:
    return read_json(path)


def iter_case_paths(cases_dir: str | Path) -> Iterable[Path]:
    root = Path(cases_dir)
    yield from sorted(root.glob("*.json"))


def load_cases(cases_dir: str | Path) -> list[dict]:
    return [load_case(path) for path in iter_case_paths(cases_dir)]
