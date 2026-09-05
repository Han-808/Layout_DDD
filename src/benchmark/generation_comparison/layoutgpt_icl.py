"""Freeze public LayoutGPT training demonstrations without running generation.

Only the pinned release's load_room_boxes function is loaded. Importing its
whole CLI would initialize a tokenizer and model clients. No generated scene,
benchmark target, evaluator annotation or score is accepted by this builder.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os.path
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any

import numpy as np

from benchmark.generation_comparison.identity import canonical_json_sha256
from benchmark.scene_io.validate import ArtifactValidationError
from benchmark.utils.io import read_json, write_json


def _checked_file(root: Path, name: str, expected_hash: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ArtifactValidationError(f"ICL source missing/outside declared root: {name}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise ArtifactValidationError(f"ICL source hash mismatch: {name}")
    return path


def prepare_layoutgpt_frozen_icl(
    *, recipe: str | Path, repo_path: str | Path,
    training_root: str | Path, out_dir: str | Path,
) -> dict[str, Any]:
    """Use the released metric formatter, with frozen, target-independent IDs."""
    recipe_path = Path(recipe).expanduser().resolve()
    recipe_bytes = recipe_path.read_bytes()
    config = json.loads(recipe_bytes)
    recipe_hash = hashlib.sha256(recipe_bytes).hexdigest()
    repo = Path(repo_path).expanduser().resolve()
    data_root = Path(training_root).expanduser().resolve()
    destination = Path(out_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError("ICL output requires a fresh directory")
    if (config.get("schema_version") != "layoutgpt_released_icl_recipe_v1"
            or config.get("unit") != "m" or config.get("normalize") is not False
            or config.get("hidden_evaluator_data_used") is not False
            or config.get("selection_policy") !=
            "first_four_in_each_pinned_rect_train_order_bedroom_then_livingroom"):
        raise ArtifactValidationError("unsupported ICL recipe semantics")
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True).stdout.strip()
    if commit != config["upstream_commit"]:
        raise ArtifactValidationError("ICL upstream commit mismatch")
    source_path = _checked_file(repo, "run_layoutgpt_3d.py", config["released_loader_sha256"])
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == "load_room_boxes"]
    if len(definitions) != 1:
        raise ArtifactValidationError("pinned release has no unique load_room_boxes")
    # Exact selected release function, not a reimplementation or whole-CLI run.
    function_code = compile(ast.Module(body=definitions, type_ignores=[]), str(source_path), "exec")
    namespace: dict[str, Any] = {"np": np, "op": os.path, "args": SimpleNamespace(normalize=False)}
    exec(function_code, namespace)

    source_files: dict[Path, str] = {source_path: config["released_loader_sha256"], recipe_path: recipe_hash}
    pinned = {row["local_path"]: row["sha256"] for row in config["files"]}
    if len(pinned) != len(config["files"]):
        raise ArtifactValidationError("duplicate ICL source file pins")
    for name, digest in pinned.items():
        source_files[_checked_file(data_root, name, digest)] = digest
    selections = config["selections"]
    if [row["room"] for row in selections] != ["bedroom", "livingroom"]:
        raise ArtifactValidationError("ICL recipe must contain bedroom then livingroom")
    messages = []
    examples = []
    required_data_files = set()
    for selection in selections:
        room = selection["room"]
        split_path = _checked_file(repo, f"dataset/3D/{room}_splits.json", selection["split_sha256"])
        source_files[split_path] = selection["split_sha256"]
        split = read_json(split_path)
        selected_ids = selection["selected_ids"]
        if (selection.get("source_split") != "rect_train" or len(selected_ids) != 4
                or selected_ids != split["rect_train"][:4]
                or len(set(selected_ids)) != 4
                or set(selected_ids) & (set(split["test"]) | set(split["val"]))):
            raise ArtifactValidationError("ICL selection differs from frozen training-only policy")
        stats_name = f"{room}/dataset_stats.txt"
        required_data_files.add(stats_name)
        if stats_name not in pinned:
            raise ArtifactValidationError("ICL dataset stats are not pinned")
        stats = read_json(data_root / stats_name)
        namespace["args"].room = room
        for room_id in selected_ids:
            name = f"{room}/{room_id}/boxes.npz"
            required_data_files.add(name)
            if name not in pinned:
                raise ArtifactValidationError("ICL training boxes are not pinned")
            condition, layout, _ = namespace["load_room_boxes"](
                str(data_root / room), room_id, stats, "m",
            )
            if not isinstance(condition, str) or not isinstance(layout, str) or not layout.strip():
                raise ArtifactValidationError("released ICL formatter returned malformed text")
            messages.extend([{"role": "user", "content": condition},
                             {"role": "assistant", "content": layout}])
            examples.append({"room": room, "source_id": room_id,
                             "boxes_sha256": pinned[name],
                             "released_condition_sha256": hashlib.sha256(condition.encode()).hexdigest(),
                             "released_layout_sha256": hashlib.sha256(layout.encode()).hexdigest()})
    if required_data_files != set(pinned) or len(examples) != config["example_count"]:
        raise ArtifactValidationError("ICL source inventory mismatch")
    for path, digest in source_files.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ArtifactValidationError("ICL input changed during preparation")
    destination.mkdir(parents=True)
    snapshot = write_json(destination / "messages.json", messages)
    metadata = {
        "schema_version": "layoutgpt_frozen_icl_manifest_v1",
        "status": "PREPARED_FROM_RELEASED_TRAINING_DATA",
        "workflow_label": "adapted_frozen_icl_layoutgpt",
        "upstream_repo": config["upstream_repo"], "upstream_commit": commit,
        "released_loader_sha256": config["released_loader_sha256"],
        "recipe_sha256": recipe_hash, "selection_policy": config["selection_policy"],
        "example_count": len(examples), "examples": examples,
        "messages": snapshot.as_posix(),
        "messages_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        "messages_semantic_sha256": canonical_json_sha256(messages),
        "source_coordinate_contract": {
            "native_axes": "x_y_up_z", "css_axes": "left_x_top_z_depth_y",
            "native_sizes": "half_extents_m", "css_sizes": "full_dimensions_m",
            "origin": "minimum_floor_corner_via_released_centroid_translation",
            "normalize": False, "unit": "m",
            "rotation": "released_round_radians_to_degrees_unchanged",
            "formatting": "released_two_decimal_metric_CSS_unchanged",
            "canonical_front": "not_provided_by_CSS_examples",
        },
        "audit": {
            "training_only": True, "held_out_upstream_test_or_validation_used": False,
            "benchmark_target_inputs_used": False, "hidden_evaluator_data_used": False,
            "released_llm_predictions_used": False, "model_calls": 0,
            "source_files_verified_unchanged": True,
            "limitations": ["not_native_k_similar_retrieval", "training_rooms_only_bedroom_livingroom",
                            "examples_do_not_define_canonical_front_for_frozen_imaginarium_assets"],
        },
    }
    path = write_json(destination / "icl_manifest.json", metadata)
    return {**metadata, "manifest_path": path.as_posix()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--repo-path", required=True)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--out-dir", required=True)
    print(json.dumps(prepare_layoutgpt_frozen_icl(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
