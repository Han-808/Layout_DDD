from __future__ import annotations

import json
import random
from typing import Any


DUMMY_METRIC_KEYS = ["physical_validity", "text_fidelity", "scene_plausibility", "game_usability", "overall"]


def evaluate_scene(
    generated_scene: dict | str,
    *,
    instruction: str | None = None,
    seed: int = 0,
    dummy: bool = True,
) -> dict:
    """Evaluate a generated scene JSON with deterministic dummy placeholder metrics."""

    scene, parse_issue = _coerce_scene(generated_scene)
    if parse_issue:
        return {
            "evaluator_version": "dummy_v0",
            "dummy": bool(dummy),
            "parse_valid": 0,
            "schema_valid": 0,
            "scene_id": None,
            "instruction": instruction,
            "metrics": {key: 0 for key in DUMMY_METRIC_KEYS},
            "issues": [parse_issue],
            "notes": ["Metrics are dummy random 0/1 placeholders. Real evaluator TBD."],
        }
    issues = _schema_issues(scene)
    rng = random.Random(int(seed))
    metrics = {key: rng.randint(0, 1) for key in DUMMY_METRIC_KEYS}
    return {
        "evaluator_version": "dummy_v0",
        "dummy": bool(dummy),
        "parse_valid": 1,
        "schema_valid": 0 if issues else 1,
        "scene_id": scene.get("scene_id") or scene.get("id") or "scene",
        "instruction": instruction,
        "metrics": metrics,
        "issues": issues,
        "notes": ["Metrics are dummy random 0/1 placeholders. Real evaluator TBD."],
    }


def _coerce_scene(value: dict | str) -> tuple[dict[str, Any], str]:
    if isinstance(value, dict):
        return value, ""
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            return {}, f"generated scene JSON could not be parsed: {exc.msg}"
        if isinstance(parsed, dict):
            return parsed, ""
    return {}, "generated scene must be a JSON object"


def _schema_issues(scene: dict[str, Any]) -> list[str]:
    issues = []
    if not scene.get("scene_id"):
        issues.append("scene is missing scene_id")
    if not isinstance(scene.get("assets"), list) and not isinstance(scene.get("objects"), list):
        issues.append("scene should contain assets or objects")
    return issues
