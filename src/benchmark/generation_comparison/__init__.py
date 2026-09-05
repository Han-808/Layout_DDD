"""Controlled cross-method generation comparison contracts."""

from importlib import import_module

from benchmark.generation_comparison.catalog import (
    CanonicalAssetCatalog,
    load_asset_catalog,
)
from benchmark.generation_comparison.eligibility import check_method_eligibility
from benchmark.generation_comparison.materializers import materialize_method_catalog
from benchmark.generation_comparison.protocol import (
    FROZEN_ASSETS,
    NATIVE,
    SHARED_DB,
    ComparisonProtocol,
    load_comparison_protocol,
)
from benchmark.generation_comparison.validation import validate_comparison_run


__all__ = [
    "CanonicalAssetCatalog",
    "ComparisonProtocol",
    "ComparisonRunError",
    "FROZEN_ASSETS",
    "NATIVE",
    "SHARED_DB",
    "check_method_eligibility",
    "load_asset_catalog",
    "load_comparison_protocol",
    "materialize_method_catalog",
    "prepare_controlled_pilot",
    "run_controlled_generation",
    "run_prepared_pilot",
    "validate_comparison_run",
]


def __getattr__(name: str):
    if name in {"ComparisonRunError", "run_controlled_generation"}:
        return getattr(
            import_module("benchmark.generation_comparison.execution"),
            name,
        )
    if name in {"prepare_controlled_pilot", "run_prepared_pilot"}:
        return getattr(
            import_module("benchmark.generation_comparison.pilot"),
            name,
        )
    raise AttributeError(name)
