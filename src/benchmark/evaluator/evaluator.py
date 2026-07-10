from __future__ import annotations

from benchmark.evaluator.OAR.evaluator import DEFAULT_OAR_CONFIG, evaluate_oar
from benchmark.evaluator.OOR.evaluator import DEFAULT_OOR_CONFIG, DETERMINISTIC_ONLY, evaluate_oor
from benchmark.evaluator.generic_validity.evaluator import DEFAULT_GENERIC_VALIDITY_CONFIG, evaluate_generic_validity, evaluate_scene_validity


class LayoutEvaluator:
    """Retired legacy evaluator shim.

    The current repo supports evaluation through root evaluate.py and the
    deterministic evaluator functions above. The old BM/layout evaluator is not
    part of the current harness surface.
    """

    def __init__(self, *_args, **_kwargs) -> None:
        raise RuntimeError("LayoutEvaluator is retired. Use evaluate.py with canonical generated_scene.json.")


__all__ = [
    "DEFAULT_GENERIC_VALIDITY_CONFIG",
    "DEFAULT_OAR_CONFIG",
    "DEFAULT_OOR_CONFIG",
    "DETERMINISTIC_ONLY",
    "LayoutEvaluator",
    "evaluate_generic_validity",
    "evaluate_oar",
    "evaluate_oor",
    "evaluate_scene_validity",
]
