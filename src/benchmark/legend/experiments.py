from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark.utils.io import load_yaml


def load_experiment_config(project_root: str | Path) -> dict:
    config = load_yaml(Path(project_root) / "configs" / "experiment_config.yaml", default={})
    return config if isinstance(config, dict) else {}


def resolve_experiment(experiment_config: dict, name: str | None) -> dict:
    if not name:
        return {}
    experiments = experiment_config.get("experiments", {})
    if not isinstance(experiments, dict) or name not in experiments:
        available = sorted(experiments) if isinstance(experiments, dict) else []
        suffix = f" Available experiments: {', '.join(available)}" if available else ""
        raise ValueError(f"Unknown experiment '{name}'.{suffix}")
    selected = experiments[name]
    if not isinstance(selected, dict):
        raise ValueError(f"Experiment '{name}' must be a mapping.")
    return dict(selected)


def pick_value(cli_value: Any, experiment: dict, key: str, default: Any = None) -> Any:
    return cli_value if cli_value is not None else experiment.get(key, default)


def experiment_model_overrides(experiment: dict) -> dict | None:
    overrides = experiment.get("model_overrides")
    return dict(overrides) if isinstance(overrides, dict) else None
