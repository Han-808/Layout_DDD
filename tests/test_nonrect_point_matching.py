from itertools import permutations
import math
import random
import struct

import pytest

from benchmark.non_rectangular.nonrect_point_matching import point_multiset_close


TOLERANCE = 1e-5


def close(left, right):
    return point_multiset_close(left, right, tolerance=TOLERANCE)


def test_actual_wall_height_float32_rounding_regression():
    height = 2.403444993020895
    observed = struct.unpack("f", struct.pack("f", height))[0]
    assert abs(height - observed) < TOLERANCE
    assert round(height / TOLERANCE) != round(observed / TOLERANCE)
    assert close([(0, 0, height)], [(0, 0, observed)])


@pytest.mark.parametrize("sign", [-1, 1])
def test_bucket_boundary_accepts_small_errors_on_both_sides(sign):
    assert close([(sign * 0.4999 * TOLERANCE, 0, 0)],
                 [(sign * 0.5001 * TOLERANCE, 0, 0)])


def test_inclusive_tolerance_and_real_displacement_rejection():
    assert close([(0, 0, 0)], [(TOLERANCE, -TOLERANCE, TOLERANCE)])
    assert not close([(0, 0, 0)], [(1.01 * TOLERANCE, 0, 0)])
    assert not close([(1e6, 0, 0)], [(1e6 + 0.001, 0, 0)])


def test_order_does_not_matter_but_cardinality_and_multiplicity_do():
    a, b = (0, 0, 0), (1, 0, 0)
    assert close([a, b, a], [b, a, a])
    assert not close([a, a], [a, b])
    assert not close([a, b], [a])
    assert close([], [])


def test_augmenting_path_handles_ambiguous_candidates_without_greedy_rejection():
    # First observed vertex can match either expected vertex; second can match only A.
    assert close([(0.5 * TOLERANCE, 0, 0), (-0.5 * TOLERANCE, 0, 0)],
                 [(0, 0, 0), (1.4 * TOLERANCE, 0, 0)])


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_coordinates_fail_closed(value):
    assert not close([(value, 0, 0)], [(value, 0, 0)])


def test_non_3d_points_fail_closed():
    assert not close([(0, 0)], [(0, 0)])
    assert not close([(0, 0, 0, 0)], [(0, 0, 0, 0)])


@pytest.mark.parametrize("value", [0, -1, math.nan, math.inf])
def test_invalid_tolerance_is_rejected(value):
    with pytest.raises(ValueError):
        point_multiset_close([], [], tolerance=value)


def test_small_multisets_match_brute_force_permutations():
    rng = random.Random(7264)
    for _ in range(100):
        left = [(rng.uniform(-2, 2) * TOLERANCE, 0, 0) for _ in range(4)]
        right = [(rng.uniform(-2, 2) * TOLERANCE, 0, 0) for _ in range(4)]
        expected = any(all(abs(a[0] - b[0]) <= TOLERANCE for a, b in zip(left, order))
                       for order in permutations(right))
        assert close(left, right) == expected
