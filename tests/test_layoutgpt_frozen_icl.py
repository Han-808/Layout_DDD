import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmark.generation_comparison.layoutgpt_icl import prepare_layoutgpt_frozen_icl
from benchmark.scene_io.validate import ArtifactValidationError
from benchmark.utils.io import read_json, write_json


def _fixture(tmp_path, monkeypatch):
    repo = tmp_path / "upstream"
    root = tmp_path / "training"
    repo.mkdir()
    # The top-level assertion must never run. Only the pinned formatter is
    # loaded, preserving its text rather than rebuilding it in the benchmark.
    source = repo / "run_layoutgpt_3d.py"
    source.write_text(
        "raise AssertionError('whole upstream CLI imported')\n"
        "def load_room_boxes(prefix, id, stats, unit):\n"
        "    assert unit == 'm' and args.normalize is False\n"
        "    with open(op.join(prefix, id, 'boxes.npz')) as f:\n"
        "        native = f.read()\n"
        "    return 'Condition: ' + args.room, native, {}\n"
    )
    monkeypatch.setattr(
        "benchmark.generation_comparison.layoutgpt_icl.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout="a" * 40),
    )
    recipe = {
        "schema_version": "layoutgpt_released_icl_recipe_v1",
        "upstream_repo": "https://example.invalid/fake-upstream",
        "upstream_commit": "a" * 40,
        "released_loader_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "unit": "m", "normalize": False, "hidden_evaluator_data_used": False,
        "selection_policy": "first_four_in_each_pinned_rect_train_order_bedroom_then_livingroom",
        "example_count": 8, "files": [], "selections": [],
    }
    for room in ("bedroom", "livingroom"):
        ids = [f"{room}_{index}" for index in range(4)]
        split = write_json(repo / f"dataset/3D/{room}_splits.json",
                           {"rect_train": ids, "test": ["test_scene"], "val": ["val_scene"]})
        recipe["selections"].append({
            "room": room, "selected_ids": ids, "source_split": "rect_train",
            "split_sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
        })
        paths = [write_json(root / f"{room}/dataset_stats.txt", {"object_types": ["chair"]})]
        for name in ids:
            path = root / room / name / "boxes.npz"
            path.parent.mkdir(parents=True)
            path.write_text(f"Layout: {name} unmodified released text")
            paths.append(path)
        for path in paths:
            recipe["files"].append({"local_path": path.relative_to(root).as_posix(),
                                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    recipe_path = write_json(tmp_path / "recipe.json", recipe)
    return {"recipe": recipe_path, "repo_path": repo, "training_root": root,
            "out_dir": tmp_path / "snapshot"}


def test_frozen_icl_is_deterministic_and_uses_only_released_training_text(tmp_path, monkeypatch):
    kwargs = _fixture(tmp_path, monkeypatch)
    first = prepare_layoutgpt_frozen_icl(**kwargs)
    second = prepare_layoutgpt_frozen_icl(**{**kwargs, "out_dir": tmp_path / "snapshot2"})
    assert first["messages_sha256"] == second["messages_sha256"]
    messages = read_json(first["messages"])
    assert len(messages) == 16
    assert [row["role"] for row in messages] == ["user", "assistant"] * 8
    assert messages[1]["content"] == "Layout: bedroom_0 unmodified released text"
    assert first["audit"]["model_calls"] == 0
    assert first["audit"]["benchmark_target_inputs_used"] is False
    with pytest.raises(FileExistsError, match="fresh directory"):
        prepare_layoutgpt_frozen_icl(**kwargs)


@pytest.mark.parametrize("target", ["loader", "split", "stats", "boxes"])
def test_frozen_icl_rejects_actual_source_drift(tmp_path, monkeypatch, target):
    kwargs = _fixture(tmp_path, monkeypatch)
    path = {
        "loader": kwargs["repo_path"] / "run_layoutgpt_3d.py",
        "split": kwargs["repo_path"] / "dataset/3D/bedroom_splits.json",
        "stats": kwargs["training_root"] / "bedroom/dataset_stats.txt",
        "boxes": kwargs["training_root"] / "bedroom/bedroom_0/boxes.npz",
    }[target]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ArtifactValidationError, match="hash mismatch"):
        prepare_layoutgpt_frozen_icl(**kwargs)
    assert not kwargs["out_dir"].exists()


@pytest.mark.parametrize("violation", ["commit", "normalize", "pixel_unit", "held_out", "selected_order", "extra_source", "escape"])
def test_frozen_icl_rejects_unsupported_or_leaking_recipe(tmp_path, monkeypatch, violation):
    kwargs = _fixture(tmp_path, monkeypatch)
    recipe = read_json(kwargs["recipe"])
    if violation == "commit":
        recipe["upstream_commit"] = "b" * 40
    elif violation == "normalize":
        recipe["normalize"] = True
    elif violation == "pixel_unit":
        recipe["unit"] = "px"
    elif violation in {"held_out", "selected_order"}:
        recipe["selections"][0]["selected_ids"][0] = "test_scene" if violation == "held_out" else "bedroom_1"
    elif violation == "extra_source":
        path = write_json(kwargs["training_root"] / "private_evaluator_report.json", {"score": 1})
        recipe["files"].append({"local_path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    else:
        recipe["files"][0]["local_path"] = "../../outside.txt"
    write_json(kwargs["recipe"], recipe)
    with pytest.raises(ArtifactValidationError):
        prepare_layoutgpt_frozen_icl(**kwargs)
    assert not kwargs["out_dir"].exists()
