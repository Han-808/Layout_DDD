from __future__ import annotations


def judge_visual_plausibility(*args, **kwargs) -> dict:
    """Compatibility placeholder for old visual_judge imports.

    The active VLM-as-judge implementation now lives in
    benchmark.workflow.vlm_judge and is wired through the main workflow
    evaluator.
    """
    raise NotImplementedError("Use benchmark.workflow.vlm_judge for the active VLM-as-judge implementation.")
