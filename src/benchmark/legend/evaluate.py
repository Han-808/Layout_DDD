from __future__ import annotations

from benchmark.workflow.evaluate import evaluate_layout_v0, evaluate_layout_vlm_as_judge_v1


def legend_evaluate_layout_vlm_as_judge_v1(**kwargs):
    return evaluate_layout_vlm_as_judge_v1(**kwargs)


def legend_evaluate_layout_v0(**kwargs):
    return evaluate_layout_v0(**kwargs)
