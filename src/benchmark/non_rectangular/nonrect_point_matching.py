"""Order-independent, bijective vertex matching with an absolute tolerance.

Pure Python so the exact implementation can be tested without Blender. Keep
this module importable by the standalone read-only Blender inspector too.
"""
from __future__ import annotations

import math
from typing import Sequence


POINT_MATCHING_REVISION = "nonrect_bijective_coordinate_tolerance_v1"


def point_multiset_close(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
    *,
    tolerance: float,
) -> bool:
    """Require a perfect matching within tolerance on every coordinate.

    Quantized buckets reject arbitrarily small errors at rounding boundaries.
    Independent nearest-neighbor checks instead allow multiple vertices to reuse
    the same counterpart. A bipartite perfect matching avoids both problems.
    """
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("point tolerance must be finite and positive")
    if len(left) != len(right):
        return False
    if any(len(point) != 3 or not all(math.isfinite(value) for value in point)
           for points in (left, right) for point in points):
        return False
    neighbors = [
        [j for j, expected in enumerate(right)
         if all(math.isclose(a, b, rel_tol=0.0, abs_tol=tolerance)
                for a, b in zip(observed, expected))]
        for observed in left
    ]
    owners = [-1] * len(right)

    def augment(vertex: int, seen: set[int]) -> bool:
        for candidate in neighbors[vertex]:
            if candidate in seen:
                continue
            seen.add(candidate)
            if owners[candidate] == -1 or augment(owners[candidate], seen):
                owners[candidate] = vertex
                return True
        return False

    return all(augment(i, set()) for i in range(len(left)))
