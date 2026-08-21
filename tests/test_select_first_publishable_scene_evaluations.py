from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))
SPEC = importlib.util.spec_from_file_location(
    "select_first_publishable_scene_evaluations",
    SCRIPTS_ROOT / "select_first_publishable_scene_evaluations.py",
)
assert SPEC is not None and SPEC.loader is not None
selector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selector)


def _write_attempt(
    root: Path,
    case_id: str,
    *,
    score: float,
    publishable: bool,
) -> None:
    case_dir = root / "cases" / case_id
    case_dir.mkdir(parents=True)
    (case_dir / "case_run_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "final_decision_status": "resolved",
                "l1_engineering_failure": False,
            }
        ),
        encoding="utf-8",
    )
    (case_dir / "evaluation_report.json").write_text(
        json.dumps(
            {
                "evaluation_status": "complete" if publishable else "incomplete",
                "benchmark_score_status": (
                    "complete" if publishable else "partial_coverage"
                ),
                "benchmark_score_100": score,
            }
        ),
        encoding="utf-8",
    )


def test_first_publishable_attempt_is_chronological_not_highest_score(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    retry = tmp_path / "retry"
    _write_attempt(baseline, "S100", score=40.0, publishable=True)
    _write_attempt(retry, "S100", score=99.0, publishable=True)

    selections, pending = selector.first_publishable_attempts(
        attempt_roots=(baseline, retry),
        case_ids=("S100",),
    )

    assert pending == []
    assert selections["S100"] == baseline / "cases" / "S100"


def test_partial_coverage_remains_pending_until_publishable_retry(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    retry = tmp_path / "retry"
    _write_attempt(baseline, "S105", score=95.0, publishable=False)

    selections, pending = selector.first_publishable_attempts(
        attempt_roots=(baseline,),
        case_ids=("S105",),
    )
    assert selections == {}
    assert pending == ["S105"]

    _write_attempt(retry, "S105", score=70.0, publishable=True)
    selections, pending = selector.first_publishable_attempts(
        attempt_roots=(baseline, retry),
        case_ids=("S105",),
    )
    assert pending == []
    assert selections["S105"] == retry / "cases" / "S105"
