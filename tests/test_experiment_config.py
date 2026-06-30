from __future__ import annotations

import pytest

from benchmark.experiments import experiment_model_overrides, pick_value, resolve_experiment


def test_resolve_experiment_and_overrides_are_independent() -> None:
    config = {
        "experiments": {
            "hssd": {
                "case": "data/case.json",
                "model": "qwen3vl_sglang_32b",
                "out": "outputs/hssd",
                "model_overrides": {"max_tokens": 12000, "timeout_seconds": 1800},
            }
        }
    }

    experiment = resolve_experiment(config, "hssd")

    assert pick_value(None, experiment, "case") == "data/case.json"
    assert pick_value("override.json", experiment, "case") == "override.json"
    assert pick_value(None, experiment, "model", "mock") == "qwen3vl_sglang_32b"
    assert experiment_model_overrides(experiment) == {"max_tokens": 12000, "timeout_seconds": 1800}


def test_unknown_experiment_reports_available_names() -> None:
    with pytest.raises(ValueError, match="Available experiments: mock"):
        resolve_experiment({"experiments": {"mock": {}}}, "missing")
