"""Metric computation and aggregation."""

from benchmark.legend.metrics.aggregate import aggregate_case_results
from benchmark.legend.metrics.metrics import compute_case_metrics

__all__ = ["aggregate_case_results", "compute_case_metrics"]
