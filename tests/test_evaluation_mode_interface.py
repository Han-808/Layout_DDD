from __future__ import annotations

import inspect
from typing import Any

import pytest

from benchmark.api import evaluation


def test_evaluation_mode_is_an_optional_keyword_interface() -> None:
    parameter = inspect.signature(evaluation.run_evaluate).parameters[
        "evaluation_mode"
    ]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None


@pytest.mark.parametrize("include_none", [False, True])
def test_default_evaluation_mode_preserves_existing_dispatch_and_kwargs(
    monkeypatch: pytest.MonkeyPatch,
    include_none: bool,
) -> None:
    expected = {"route": "canonical"}
    captured: dict[str, Any] = {}

    def fake_canonical(**kwargs: Any) -> dict:
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(evaluation, "_run_canonical_evaluate", fake_canonical)
    monkeypatch.setattr(evaluation, "is_legacy_game_profile", lambda profile: False)

    kwargs: dict[str, Any] = {
        "scene": {"request_id": "existing"},
        "out": "report.json",
    }
    if include_none:
        kwargs["evaluation_mode"] = None

    result = evaluation.run_evaluate(**kwargs)

    assert result is expected
    assert captured == {
        "scene": {"request_id": "existing"},
        "out": "report.json",
    }


def test_selected_evaluation_mode_stops_before_existing_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(
            f"existing evaluator was called with args={args!r}, kwargs={kwargs!r}"
        )

    monkeypatch.setattr(evaluation, "is_legacy_game_profile", forbidden)
    monkeypatch.setattr(evaluation, "_run_canonical_evaluate", forbidden)
    monkeypatch.setattr(evaluation, "_run_legacy_game_evaluate", forbidden)

    with pytest.raises(
        NotImplementedError,
        match="evaluation_mode 'non_rectangular_multi_room' is not implemented",
    ):
        evaluation.run_evaluate(
            evaluation_mode="non_rectangular_multi_room",
            scene={"future": "format_not_selected"},
            out="report.json",
        )
