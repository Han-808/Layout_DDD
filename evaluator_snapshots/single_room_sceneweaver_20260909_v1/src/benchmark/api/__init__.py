"""Stable package APIs for generation, adaptation, and evaluation.

Exports are lazy so executing a submodule with ``python -m`` does not preload
that same module through the package initializer.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "run_evaluate",
    "run_generate",
    "run_generate_from_natural_language",
    "TrustedCaseBundle",
    "MaterializationResult",
    "prepare_submission",
    "evaluate_prepared_submission",
    "evaluate_artifact_submission",
    "evaluate_submission",
    "load_case_bundle",
]

_EXPORTS = {
    "run_evaluate": ("benchmark.api.evaluation", "run_evaluate"),
    "run_generate": ("benchmark.api.generation", "run_generate"),
    "run_generate_from_natural_language": (
        "benchmark.api.generation",
        "run_generate_from_natural_language",
    ),
    "TrustedCaseBundle": ("benchmark.api.submission", "TrustedCaseBundle"),
    "MaterializationResult": ("benchmark.api.submission", "MaterializationResult"),
    "prepare_submission": ("benchmark.api.submission", "prepare_submission"),
    "evaluate_prepared_submission": (
        "benchmark.api.submission",
        "evaluate_prepared_submission",
    ),
    "evaluate_artifact_submission": (
        "benchmark.api.submission",
        "evaluate_artifact_submission",
    ),
    "evaluate_submission": ("benchmark.api.submission", "evaluate_submission"),
    "load_case_bundle": ("benchmark.api.submission", "load_case_bundle"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
