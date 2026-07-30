from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.api.submission import CaseBundleError, _validate_camera_evidence
from benchmark.visual_judge.active_fallback import (
    ConditionalActiveCameraEvidenceProvider,
    InsufficientVisualEvidenceError,
)
from benchmark.visual_judge.evidence_sufficiency import (
    INSUFFICIENT,
    SUFFICIENT,
    UNKNOWN,
    assess_visual_evidence_sufficiency,
)


def _image(tmp_path: Path, name: str) -> str:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / name
    path.write_bytes(b"evidence")
    return path.as_posix()


def _oob_items(tmp_path: Path, *, target_fraction: float | None) -> list[dict]:
    visibility = (
        {
            "target_pixel_fractions": {"obj_001": target_fraction},
            "region_pixel_fractions": {"architecture_plane": 0.02},
            "measured": True,
        }
        if target_fraction is not None
        else None
    )
    return [
        {
            "path": _image(tmp_path, "global.png"),
            "role": "metric_highlighted_global",
            "view_id": "global_top",
        },
        {
            "path": _image(tmp_path, "local.png"),
            "role": "metric_local_highlight",
            "view_id": "oob_local_00",
            "target_ids": ["obj_001"],
            "color_legend": [
                {"id": "obj_001", "role": "primary_target"},
                {"id": "east_wall", "role": "architecture_plane"},
            ],
            "visibility": visibility,
        },
    ]


def test_deterministic_sufficiency_gate_distinguishes_clear_failure_from_unknown(
    tmp_path: Path,
) -> None:
    sufficient = assess_visual_evidence_sufficiency(
        "oob",
        _oob_items(tmp_path / "sufficient", target_fraction=0.02),
    )
    insufficient = assess_visual_evidence_sufficiency(
        "oob",
        _oob_items(tmp_path / "insufficient", target_fraction=0.0),
    )
    unknown = assess_visual_evidence_sufficiency(
        "oob",
        _oob_items(tmp_path / "unknown", target_fraction=None),
    )

    assert sufficient["status"] == SUFFICIENT
    assert insufficient["status"] == INSUFFICIENT
    assert insufficient["trigger_recommended"] is True
    assert unknown["status"] == UNKNOWN
    assert unknown["trigger_recommended"] is False


class _Provider:
    def __init__(self, items: list[dict]) -> None:
        self.items = items
        self.calls: list[dict] = []
        self.policy_config = {"provider": type(self).__name__}

    def __call__(self, request: dict) -> list[dict]:
        self.calls.append(request)
        return self.items


class _RaisingProvider(_Provider):
    def __call__(self, request: dict) -> list[dict]:
        self.calls.append(request)
        raise RuntimeError("active render failed")


class _UsageProvider(_Provider):
    def __init__(
        self,
        items: list[dict],
        *,
        selector_calls: int,
        camera_actions: int,
        cache_hit: bool = False,
    ) -> None:
        super().__init__(items)
        self.selector_calls = selector_calls
        self.camera_actions = camera_actions
        self.cache_hit = cache_hit
        self.call_count = 0
        self.last_call_usage: dict | None = None

    def __call__(self, request: dict) -> list[dict]:
        items = super().__call__(request)
        self.call_count += 1
        self.last_call_usage = {
            "call_id": self.call_count,
            "metric": str(request.get("metric") or "").strip().lower(),
            "cache_hit": self.cache_hit,
            "evidence_refs": [
                str(
                    item.get("view_id")
                    or item.get("id")
                    or item.get("path")
                )
                for item in items
            ],
            "manifest_path": None,
            "selector_calls": self.selector_calls,
            "camera_actions": self.camera_actions,
        }
        return items


def test_active_camera_runs_only_after_explicit_insufficiency(tmp_path: Path) -> None:
    deterministic = _Provider(
        _oob_items(tmp_path / "base", target_fraction=0.0)
    )
    active = _Provider(_oob_items(tmp_path / "active", target_fraction=0.02))
    provider = ConditionalActiveCameraEvidenceProvider(
        deterministic_provider=deterministic,
        active_provider=active,
        out_dir=tmp_path / "fallback",
        max_views=2,
        max_steps=1,
        shadow_mode=False,
    )

    items = provider(
        {
            "metric": "oob",
            "object_ids": ["obj_001"],
            "event": {"object_id": "obj_001"},
        }
    )

    assert len(deterministic.calls) == 1
    assert len(active.calls) == 1
    assert active.calls[0]["_camera_selection_phase"] == "active_fallback"
    assert active.calls[0]["_camera_evidence_deficiency"]["status"] == INSUFFICIENT
    assert active.calls[0]["_camera_evidence_deficiency"]["camera_repairable"] is True
    assert items == active.items
    assert all("active_camera_fallback" not in item for item in items)
    manifest = next((tmp_path / "fallback").glob("*/active_camera_fallback_manifest.json"))
    assert manifest.is_file()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["active_used"] is True
    assert payload["official_packet_source"] == "active"
    assert payload["official_assessment"] == payload["active_assessment"]


def test_shadow_fallback_aggregates_actual_internal_provider_usage(
    tmp_path: Path,
) -> None:
    deterministic = _UsageProvider(
        _oob_items(tmp_path / "base", target_fraction=0.0),
        selector_calls=0,
        camera_actions=0,
    )
    active = _UsageProvider(
        _oob_items(tmp_path / "active", target_fraction=0.02),
        selector_calls=2,
        camera_actions=1,
    )
    provider = ConditionalActiveCameraEvidenceProvider(
        deterministic_provider=deterministic,
        active_provider=active,
        out_dir=tmp_path / "fallback",
        max_views=2,
        max_steps=1,
        fail_on_exhausted=False,
        shadow_mode=True,
    )

    returned = provider({"metric": "oob", "object_ids": ["obj_001"]})

    assert returned == deterministic.items
    usage = provider.last_call_usage
    assert isinstance(usage["call_id"], str)
    assert usage["metric"] == "oob"
    assert usage["cache_hit"] is False
    assert usage["evidence_refs"] == ["global_top", "oob_local_00"]
    assert usage["selector_calls"] == 2
    assert usage["camera_actions"] == 1
    assert usage["observability"]["internal_calls"]["deterministic"][
        "observed"
    ] is True
    assert usage["observability"]["internal_calls"]["active"][
        "observed"
    ] is True
    manifest_path = Path(usage["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["active_attempted"] is True
    assert manifest["active_used"] is False
    assert manifest["call_usage"] == usage


def test_fallback_reports_cache_hit_only_when_all_invoked_providers_hit(
    tmp_path: Path,
) -> None:
    deterministic = _UsageProvider(
        _oob_items(tmp_path / "base", target_fraction=0.0),
        selector_calls=0,
        camera_actions=0,
        cache_hit=True,
    )
    active = _UsageProvider(
        _oob_items(tmp_path / "active", target_fraction=0.02),
        selector_calls=0,
        camera_actions=0,
        cache_hit=True,
    )
    provider = ConditionalActiveCameraEvidenceProvider(
        deterministic_provider=deterministic,
        active_provider=active,
        out_dir=tmp_path / "fallback",
        max_views=2,
        max_steps=1,
        fail_on_exhausted=False,
        shadow_mode=True,
    )

    provider({"metric": "oob", "object_ids": ["obj_001"]})

    assert provider.last_call_usage["cache_hit"] is True
    assert provider.last_call_usage["selector_calls"] == 0
    assert provider.last_call_usage["camera_actions"] == 0


def test_unknown_evidence_does_not_broadly_trigger_active_camera(tmp_path: Path) -> None:
    deterministic = _Provider(
        _oob_items(tmp_path / "base", target_fraction=None)
    )
    active = _Provider(_oob_items(tmp_path / "active", target_fraction=0.02))
    provider = ConditionalActiveCameraEvidenceProvider(
        deterministic_provider=deterministic,
        active_provider=active,
        out_dir=tmp_path / "fallback",
        max_views=2,
        max_steps=1,
    )

    items = provider({"metric": "oob", "object_ids": ["obj_001"]})

    assert not active.calls
    assert items == deterministic.items
    assert all("active_camera_fallback" not in item for item in items)


def test_sufficient_base_is_not_recorded_as_counterfactual_replacement(
    tmp_path: Path,
) -> None:
    deterministic = _Provider(
        _oob_items(tmp_path / "base", target_fraction=0.02)
    )
    active = _Provider(_oob_items(tmp_path / "active", target_fraction=0.02))
    provider = ConditionalActiveCameraEvidenceProvider(
        deterministic_provider=deterministic,
        active_provider=active,
        out_dir=tmp_path / "fallback",
        max_views=2,
        max_steps=1,
    )

    items = provider({"metric": "oob", "object_ids": ["obj_001"]})

    assert not active.calls
    assert items == deterministic.items
    manifest_path = next(
        (tmp_path / "fallback").glob("*/active_camera_fallback_manifest.json")
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["active_attempted"] is False
    assert manifest["counterfactual_would_replace"] is False
    assert manifest["official_packet_source"] == "deterministic"


def test_active_camera_budget_exhaustion_remains_unresolved(tmp_path: Path) -> None:
    deterministic = _Provider(
        _oob_items(tmp_path / "base", target_fraction=0.0)
    )
    active = _Provider(_oob_items(tmp_path / "active", target_fraction=0.0))
    provider = ConditionalActiveCameraEvidenceProvider(
        deterministic_provider=deterministic,
        active_provider=active,
        out_dir=tmp_path / "fallback",
        max_views=2,
        max_steps=0,
        fail_on_exhausted=True,
        shadow_mode=False,
    )

    with pytest.raises(
        InsufficientVisualEvidenceError,
        match="exhausted without sufficient",
    ):
        provider({"metric": "oob", "object_ids": ["obj_001"]})


def test_non_shadow_nonfatal_exhaustion_keeps_deterministic_packet(
    tmp_path: Path,
) -> None:
    deterministic = _Provider(
        _oob_items(tmp_path / "base", target_fraction=0.0)
    )
    active = _Provider(_oob_items(tmp_path / "active", target_fraction=0.0))
    provider = ConditionalActiveCameraEvidenceProvider(
        deterministic_provider=deterministic,
        active_provider=active,
        out_dir=tmp_path / "fallback",
        max_views=2,
        max_steps=0,
        fail_on_exhausted=False,
        shadow_mode=False,
    )

    items = provider({"metric": "oob", "object_ids": ["obj_001"]})

    assert items == deterministic.items
    manifest_path = next(
        (tmp_path / "fallback").glob("*/active_camera_fallback_manifest.json")
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["active_attempted"] is True
    assert manifest["active_used"] is False
    assert manifest["counterfactual_would_replace"] is False
    assert manifest["official_packet_source"] == "deterministic"
    assert manifest["official_assessment"] == manifest["deterministic_assessment"]
    assert manifest["active_assessment"]["status"] == INSUFFICIENT


def test_non_shadow_nonfatal_active_error_keeps_deterministic_packet(
    tmp_path: Path,
) -> None:
    deterministic = _Provider(
        _oob_items(tmp_path / "base", target_fraction=0.0)
    )
    active = _RaisingProvider([])
    provider = ConditionalActiveCameraEvidenceProvider(
        deterministic_provider=deterministic,
        active_provider=active,
        out_dir=tmp_path / "fallback",
        max_views=2,
        max_steps=1,
        fail_on_exhausted=False,
        shadow_mode=False,
    )

    items = provider({"metric": "oob", "object_ids": ["obj_001"]})

    assert items == deterministic.items
    manifest_path = next(
        (tmp_path / "fallback").glob("*/active_camera_fallback_manifest.json")
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["active_attempted"] is True
    assert manifest["active_used"] is False
    assert manifest["counterfactual_would_replace"] is False
    assert manifest["official_packet_source"] == "deterministic"
    assert manifest["official_assessment"] == manifest["deterministic_assessment"]
    assert manifest["active_assessment"]["reason_codes"] == [
        "active_camera_execution_failed"
    ]
    assert "active render failed" in manifest["active_error"]


def test_case_bundle_rejects_active_fallback_wrapping_query_cov() -> None:
    with pytest.raises(CaseBundleError, match="base policy must be deterministic"):
        _validate_camera_evidence(
            {
                "mode": "query_cov",
                "active_fallback": {"enabled": True, "max_steps": 1},
            }
        )

    config = _validate_camera_evidence(
        {
            "mode": "auto",
            "active_fallback": {
                "enabled": True,
                "max_steps": 2,
                "fail_on_exhausted": True,
            },
        }
    )
    assert config["active_fallback"] == {
        "enabled": True,
        "max_steps": 2,
        "candidate_count": 5,
        "fail_on_exhausted": True,
        "shadow_mode": True,
    }

    custom = _validate_camera_evidence(
        {
            "mode": "auto",
            "max_views": 3,
            "active_fallback": {
                "enabled": True,
                "candidate_count": 7,
            },
        }
    )
    assert custom["active_fallback"]["candidate_count"] == 7

    with pytest.raises(CaseBundleError, match="at least.*max_views"):
        _validate_camera_evidence(
            {
                "mode": "auto",
                "max_views": 3,
                "active_fallback": {
                    "enabled": True,
                    "candidate_count": 2,
                },
            }
        )
