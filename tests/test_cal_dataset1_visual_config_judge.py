from pathlib import Path

from scripts.judge_cal_dataset1_visual_config import ARMS, _compose_visual_items


def _item(tmp_path: Path, arm: str, kind: str, index: int, view_id: str) -> dict:
    path = tmp_path / arm / f"{kind}_{index:02d}_{view_id}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"png")
    return {
        "path": str(path),
        "sha256": "unused",
        "role": "metric_local_rgb" if kind == "rgb" else "metric_local_highlight",
        "view_id": view_id,
    }


def _comparison(tmp_path: Path) -> dict:
    global_items = [
        _item(tmp_path, "global", "rgb", 0, "global_top"),
        _item(tmp_path, "global", "highlight", 0, "global_top"),
        _item(tmp_path, "global", "rgb", 1, "global_perspective"),
        _item(tmp_path, "global", "highlight", 1, "global_perspective"),
    ]
    local_items = [
        _item(tmp_path, "local", "rgb", 0, "local_a"),
        _item(tmp_path, "local", "highlight", 0, "local_a"),
        _item(tmp_path, "local", "rgb", 1, "local_b"),
        _item(tmp_path, "local", "highlight", 1, "local_b"),
    ]
    return {
        "case_id": "case-a",
        "metric": "support",
        "event_id": "obj_000",
        "arms": {
            "fixed_global_highlight": {"items": global_items},
            "metric_local_highlight": {"items": local_items},
        },
    }


def test_visual_config_replay_excludes_vlm_selector() -> None:
    assert len(ARMS) == 9
    assert "vlm_select_from_candidates" not in ARMS


def test_visual_config_bundles_preserve_expected_budgets_and_order(tmp_path: Path) -> None:
    comparison = _comparison(tmp_path)
    expected = {
        "fixed_global": (2, ["global_top", "global_perspective"]),
        "fixed_global_highlight": (
            4,
            ["global_top", "global_top", "global_perspective", "global_perspective"],
        ),
        "presence_local_raw": (2, ["local_a", "local_b"]),
        "presence_local_raw_highlight": (
            4,
            ["local_a", "local_a", "local_b", "local_b"],
        ),
        "presence_global_local_raw": (3, ["global_top", "local_a", "local_b"]),
        "deterministic_metric_local": (
            5,
            ["global_top", "local_a", "local_a", "local_b", "local_b"],
        ),
        "order_local_first_full": (
            5,
            ["local_a", "local_a", "local_b", "local_b", "global_top"],
        ),
        "budget_global_first_compact": (
            3,
            ["global_top", "local_a", "local_a"],
        ),
        "budget_local_first_compact": (
            3,
            ["local_a", "local_a", "global_top"],
        ),
    }
    for arm, (budget, view_ids) in expected.items():
        items, factors = _compose_visual_items(comparison, arm)
        assert len(items) == budget
        assert [item["view_id"] for item in items] == view_ids
        assert factors["max_local_views"] in {0, 1, 2}
