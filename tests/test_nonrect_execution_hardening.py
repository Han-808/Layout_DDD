"""Offline execution hardening regressions: no real API or Blender calls."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from scripts.nonrect_hardening_v1 import VERSION
from scripts.nonrect_hardening_v1 import overlay, safety
from scripts.nonrect_hardening_v1 import results, assurance
from scripts import run_complicated_combined142_hardened as runner


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "Support/worktrees/collision-final-bundle-v1"
if not FROZEN.is_dir():
    # CI exercises the checked-in evaluator with synthetic catalogs, not local
    # campaign data. The publication audit separately verifies core byte identity.
    FROZEN = ROOT


@pytest.mark.parametrize("field,value", [
    ("minimum_free_gib", 0), ("room_timeout_seconds", -1),
    ("worker_timeout_seconds", float("inf")), ("heartbeat_timeout_seconds", float("nan")),
    ("terminate_grace_seconds", True), ("checkpoint_enabled", 1),
])
def test_policy_rejects_unbounded_or_invalid(field, value):
    with pytest.raises(ValueError):
        safety.Policy(**{field: value})


def test_atomic_private_and_replace_failure_preserves_old_value(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    safety.atomic_json(path, {"old": True})
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    def fail(*_):
        raise OSError("disk failure")
    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(OSError):
        safety.atomic_json(path, {"new": True})
    assert safety.read_json(path) == {"old": True}
    assert list(tmp_path.iterdir()) == [path]


def test_storage_guard_has_no_deletion(tmp_path, monkeypatch):
    marker = tmp_path / "preserve.bin"
    marker.write_bytes(b"keep")
    monkeypatch.setattr(safety.shutil, "disk_usage", lambda _: SimpleNamespace(free=10))
    with pytest.raises(safety.StorageGuardError):
        safety.check_storage(tmp_path, 30)
    assert marker.read_bytes() == b"keep"


def test_redaction_keeps_exact_validator_location_not_secrets(monkeypatch):
    monkeypatch.setenv("STANDARD_API_CREDENTIAL", "appid:private_credential_123")
    raw = {"phase": "placement", "check_id": "check42", "field_path": "rows[2].check_id",
           "error_type": "ValueError", "error": "bad check42; appid:private_credential_123",
           "raw_response": "PRIVATE_RAW", "headers": {"Authorization": "Bearer secret"},
           "nested": [{"stop_reason": "camera_constraint_contract_invalid"}],
           "url": "https://example.invalid/path?token=private", "other": "Bearer private_abc"}
    projected = safety.diagnostic_projection(raw)
    text = json.dumps(projected)
    assert "PRIVATE_RAW" not in text and "private" not in text
    assert projected["field_path"] == "rows[2].check_id"
    assert projected["nested"][0]["stop_reason"] == "camera_constraint_contract_invalid"


def test_diagnostic_write_failure_never_changes_primary_exception(tmp_path, monkeypatch, capfd):
    monkeypatch.setattr(safety, "atomic_json", lambda *_: (_ for _ in ()).throw(OSError("secret")))
    assert safety.best_effort_record(tmp_path / "failed.json", {"secret": "PRIVATE"}) is False
    assert capfd.readouterr().err == "execution_diagnostic_write_failed\n"


def test_exception_chain_is_bounded_and_drops_endpoint_body():
    EndpointHTTPError = type("EndpointHTTPError", (RuntimeError,), {})
    try:
        try:
            raise EndpointHTTPError("HTTP 503: PRIVATE_RESPONSE_BODY")
        except Exception as exc:
            raise ValueError("placement_check_results missing check42") from exc
    except Exception as exc:
        value = safety.exception_record(exc)
    assert value["exception_chain"][0]["error_type"] == "ValueError"
    assert value["exception_chain"][0]["frames"][-1]["function"] == "test_exception_chain_is_bounded_and_drops_endpoint_body"
    assert value["exception_chain"][1]["error"] == "HTTP 503"
    assert "PRIVATE_RESPONSE_BODY" not in json.dumps(value)


@pytest.mark.parametrize("fail_status", ["active", "released"])
def test_slot_state_write_failure_cannot_leak_lock(tmp_path, fail_status):
    def write(path, value):
        if value["status"] == fail_status:
            raise OSError("full disk")
        safety.atomic_json(path, value)
    if fail_status == "active":
        with pytest.raises(OSError):
            with safety.room_slot(tmp_path, slots=1, timeout=0.1, work={}, write=write):
                pytest.fail("body must not run")
    else:
        with safety.room_slot(tmp_path, slots=1, timeout=0.1, work={}, write=write):
            pass
    with safety.room_slot(tmp_path, slots=1, timeout=0.1, work={}):
        pass


def test_slot_timeout_and_exception_release(tmp_path):
    with safety.room_slot(tmp_path, slots=1, timeout=0.1, work={}):
        with pytest.raises(safety.AdmissionTimeout):
            with safety.room_slot(tmp_path, slots=1, timeout=0.02, work={}):
                pass
    with pytest.raises(ValueError):
        with safety.room_slot(tmp_path, slots=1, timeout=0.1, work={}):
            raise ValueError("primary")
    with safety.room_slot(tmp_path, slots=1, timeout=0.1, work={}):
        pass


def test_slots_bound_six_concurrent_threads(tmp_path):
    current, maximum = 0, 0
    lock = threading.Lock()
    def work(index):
        nonlocal current, maximum
        with safety.room_slot(tmp_path, slots=2, timeout=2, work={"i": index}):
            with lock:
                current += 1
                maximum = max(maximum, current)
            time.sleep(0.03)
            with lock:
                current -= 1
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(work, range(6)))
    assert maximum == 2 and current == 0


@pytest.mark.parametrize("raw,expected", [
    ({"error_type": "EndpointConnectionError"}, True),
    ({"error_type": "TimeoutError"}, True),
    ({"error_type": "EndpointHTTPError", "error": "HTTP 503 unavailable"}, True),
    ({"error_type": "EndpointHTTPError", "error": "HTTP 401 denied"}, False),
    ({"error_type": "EndpointHTTPError"}, False),
    ({"failure_category": "camera_or_renderer_failure"}, False),
    ({"error_type": "ValueError", "error": "HTTP 503 in a bad field"}, False),
    ({"error_type": "EndpointConfigurationError"}, False),
    ({"error_type": "BlenderRenderError", "error": "contract failed"}, False),
    ({"error_type": "ResponseSchemaRepairError"}, False),
    ({"error_type": "TimeoutError", "stop_reason": "camera_constraint_contract_invalid"}, False),
    ([{"error_type": "TimeoutError"}, {"error_type": "ValueError"}], False),
    ({"adjudication_error": "EndpointConnectionError: failed"}, True),
])
def test_retry_origin_requires_unambiguous_transient_evidence(raw, expected):
    assert overlay.transient_origin(raw) is expected


def context(tmp_path, token="run"):
    heartbeat = safety.Heartbeat(tmp_path / "heartbeat.json", "worker")
    return overlay.RoomContext(tmp_path / "room", "room", heartbeat, safety.Policy(),
                               token, tmp_path / "attempt1")


def test_checkpoint_reuses_same_result_and_accepts_quality_invalid(tmp_path):
    ctx, calls = context(tmp_path), []
    result = {"status": "checked", "score": 0.0, "pairs": [{"final_verdict": "invalid"}]}
    def call():
        calls.append(1)
        return result
    first = overlay.checkpoint_call(ctx, "collision", {"scene": "same"}, call,
                                    lambda r: overlay.valid_metric(r, "l1"))
    ctx.attempt = tmp_path / "attempt2"
    second = overlay.checkpoint_call(ctx, "collision", {"scene": "same"}, call,
                                     lambda r: overlay.valid_metric(r, "l1"))
    assert first == second == result and len(calls) == 1
    assert second is not first
    assert (ctx.attempt / "execution_diagnostics/collision_reuse.json").is_file()


@pytest.mark.parametrize("changed", ["input", "run"])
def test_checkpoint_cannot_cross_input_or_execution_identity(tmp_path, changed):
    ctx, calls = context(tmp_path), []
    def call():
        calls.append(1)
        return {"status": "checked", "score": 1}
    overlay.checkpoint_call(ctx, "support", "one", call, lambda r: True)
    if changed == "run":
        ctx.run_identity = "different"
    overlay.checkpoint_call(ctx, "support", "two" if changed == "input" else "one", call, lambda r: True)
    assert len(calls) == 2


def test_checkpoint_evidence_corruption_refuses_reuse(tmp_path):
    image = tmp_path / "evidence.png"
    image.write_bytes(b"image bytes")
    ctx = context(tmp_path)
    overlay.checkpoint_call(ctx, "collision", "same", lambda: {"evidence": str(image)}, lambda r: True)
    image.write_bytes(b"changed")
    with pytest.raises(safety.IntegrityError):
        overlay.checkpoint_call(ctx, "collision", "same", lambda: pytest.fail("must not resample"), lambda r: True)


def test_incomplete_metric_not_cached_and_failure_diagnostic_keeps_scope(tmp_path):
    ctx, calls = context(tmp_path), []
    raw = {"status": "failed", "score": None, "infrastructure_failures": [
        {"phase": "target_local", "scope_id": "s42", "error_type": "ValueError",
         "error": "placement_check_results must cover every required check exactly once"}]}
    def call():
        calls.append(1)
        return raw
    for _ in range(2):
        assert overlay.checkpoint_call(ctx, "placement", "same", call,
                                       lambda r: overlay.valid_metric(r, "l3_metric")) == raw
    assert len(calls) == 2
    record = safety.read_json(ctx.attempt / "execution_diagnostics/placement.json")
    assert record["result"]["infrastructure_failures"][0]["scope_id"] == "s42"
    assert "exactly once" in record["result"]["infrastructure_failures"][0]["error"]


def test_checkpoint_secret_not_persisted_or_altered(tmp_path, monkeypatch):
    monkeypatch.setenv("STANDARD_API_CREDENTIAL", "private:credential_token")
    ctx = context(tmp_path)
    raw = {"status": "checked", "score": 1, "error": "private:credential_token"}
    assert overlay.checkpoint_call(ctx, "collision", "x", lambda: raw, lambda r: True) is raw
    assert not list(ctx.root.glob("execution_checkpoints/**/*.json"))
    for p in tmp_path.rglob("*.json"):
        assert "private:credential_token" not in p.read_text()


def test_overlay_observes_l3_before_l1_normalization_and_keeps_core_results(tmp_path):
    """Regression for diagnostics previously lost on the room-failure path."""
    original_diagnostics = []
    placement = {"status": "failed", "score": None, "infrastructure_failures": [
        {"phase": "scene_global", "scope_id": "placement42", "error_type": "ValueError",
         "error": "placement check42 missing", "raw_response": "PRIVATE_RESPONSE"}]}
    quality = {"metrics": {"semantic_placement_consistency": placement}}
    collision = {"status": "requires_vlm", "score": None, "pairs": [{
        "requires_vlm": True, "final_verdict": None, "event_key": "collision:a:b",
        "evidence_control": {"stop_reason": "camera_constraint_contract_invalid",
                             "trace": [{"error_type": "ValueError", "error": "unknown observation"}]}}]}
    class Factory:
        def build(self, context):
            return SimpleNamespace(scene_quality_evaluator=lambda *a, **k: quality, evaluate=lambda *a: {},
                _persist_metric_diagnostic=lambda **kw: original_diagnostics.append(kw))
    resilient = SimpleNamespace(_Coordinator=type("Coordinator", (), {}),
        COORDINATOR_REVISION="original", classify_failure=lambda *a, **k: None)
    runtime = SimpleNamespace(DefaultNonRectangularRuntimeFactory=Factory)
    module = SimpleNamespace(NonRectangularRoomMetricIncomplete=type("Incomplete", (RuntimeError,), {}),
        check_collision=lambda *a, **k: collision, check_polygon_oob=lambda *a, **k: {},
        check_support=lambda *a, **k: {})
    ctx = context(tmp_path)
    token = overlay.CURRENT.set(ctx)
    try:
        overlay.install(resilient, runtime, module, output=tmp_path, dataset="fixture",
                        policy=ctx.policy, heartbeat=ctx.heartbeat, code_identity="fixture")
        evaluator = Factory().build(SimpleNamespace(attempt_root=ctx.attempt,
                                    materialization=SimpleNamespace(identity_sha256="material")))
        assert module.check_collision({}) is collision
        assert evaluator.scene_quality_evaluator({}) is quality
        # Both detailed results exist BEFORE the core room normalizer runs.
        p = safety.read_json(ctx.attempt / "execution_diagnostics/semantic_placement_consistency.json")
        assert p["result"]["infrastructure_failures"][0]["error"] == "placement check42 missing"
        c = safety.read_json(ctx.attempt / "execution_diagnostics/collision.json")
        assert c["result"]["pairs"][0]["evidence_control"]["stop_reason"] == "camera_constraint_contract_invalid"
        assert "PRIVATE_RESPONSE" not in json.dumps(p)
        evaluator._persist_metric_diagnostic(unit=None, metric="collision", layer="l1", raw=collision)
        assert original_diagnostics[0]["raw"] is collision
    finally:
        overlay.CURRENT.reset(token)


def terminal_fixture(tmp_path, *, statuses=("complete", "failed_nonretryable")):
    evaluation = tmp_path / "evaluation"
    payload = {"model_order": ["model"]}
    safety.atomic_json(evaluation / "run_manifest.json", {"identity": payload, "identity_sha256": safety.identity(payload)})
    for index, status in enumerate(statuses):
        room = evaluation / "models/model/scenes/scene/rooms" / f"room{index}"
        safety.atomic_json(room / "summary.json", {"status": status, "latest_failure": {
            "stage": "evaluation", "category": "judge_response_contract_failure"}})
        if status == "complete":
            report = room / "report.json"
            safety.atomic_json(report, {"status": "complete", "room_id": room.name,
                "schema_version": "non_rectangular_complete_room_report_v1",
                "metrics": {name: {"metric": name, "status": "complete", "score": 0.0} for name in (
                    "collision", "oob", "support", "scale_consistency", "style_consistency",
                    "object_pairing_consistency", "functional_consistency", "semantic_placement_consistency")}})
            safety.atomic_json(room / "room_report_selected.json", {"room_report_path": str(report),
                                                                      "room_report_sha256": safety.digest(report)})
    complete = statuses.count("complete")
    scene = evaluation / "models/model/scenes/scene"
    safety.atomic_json(scene / "summary.json", {"model": "model", "scene_id": "scene",
        "room_order": [f"room{index}" for index in range(len(statuses))]})
    if complete == len(statuses):
        safety.atomic_json(scene / "evaluation_report.json", {"fixture": True, "terminal_status": "complete"})
    safety.atomic_json(evaluation / "terminal_manifest.json", {
        "schema_version": "non_rectangular_resilient_terminal_manifest_v1",
        "status": "complete" if complete == len(statuses) else "failed",
        "room_count": len(statuses), "complete_room_count": complete,
        "failed_room_count": len(statuses) - complete,
        "nonretryable_scene_failure_count": 0, "excluded_incomplete_scene_count": 0})
    return evaluation


@pytest.mark.parametrize("code,expected", [(2, True), (1, False), (70, False), (-9, False), (0, False)])
def test_partial_exit_protocol_is_not_accept_all_nonzero(tmp_path, code, expected):
    evaluation = terminal_fixture(tmp_path)
    result = safety.worker_outcome(evaluation, code, expected_model="model", expected_rooms=2)
    assert result["continue_lane"] is expected


def test_complete_worker_requires_zero_and_matching_manifest(tmp_path):
    evaluation = terminal_fixture(tmp_path, statuses=("complete",))
    assert safety.worker_outcome(evaluation, 0, expected_model="model", expected_rooms=1)["continue_lane"]
    assert not safety.worker_outcome(evaluation, 2, expected_model="model", expected_rooms=1)["continue_lane"]


@pytest.mark.parametrize("broken", ["missing", "identity", "count", "pending", "report_hash", "preflight"])
def test_terminal_mismatch_refuses_next_dataset(tmp_path, broken):
    evaluation = terminal_fixture(tmp_path, statuses=("complete", "pending") if broken == "pending" else ("complete", "failed_nonretryable"))
    terminal = evaluation / "terminal_manifest.json"
    if broken == "missing":
        terminal.unlink()
    if broken == "identity":
        run = safety.read_json(evaluation / "run_manifest.json")
        run["identity_sha256"] = "bad"
        safety.atomic_json(evaluation / "run_manifest.json", run)
    if broken in {"count", "preflight"}:
        value = safety.read_json(terminal)
        value["room_count" if broken == "count" else "nonretryable_scene_failure_count"] = 9
        safety.atomic_json(terminal, value)
    if broken == "report_hash":
        next(evaluation.glob("models/*/scenes/*/rooms/*/report.json")).write_text("{}")
    assert not safety.worker_outcome(evaluation, 2, expected_model="model", expected_rooms=2)["continue_lane"]


def test_stale_or_failed_worker_receipt_cannot_use_old_terminal(tmp_path):
    terminal_fixture(tmp_path)
    receipt = {"run_id": "old", "execution_plan_sha256": "plan", "cli_completed": True, "returncode": 2}
    safety.atomic_json(tmp_path / "execution_worker_result.json", receipt)
    result = runner.adjudicate_worker(tmp_path, "new", 2, None, plan_sha="plan", model="model", rooms=2)
    assert result["continue_lane"] is False
    receipt.update(run_id="new", cli_completed=False)
    safety.atomic_json(tmp_path / "execution_worker_result.json", receipt)
    assert not runner.adjudicate_worker(tmp_path, "new", 2, None, plan_sha="plan", model="model", rooms=2)["continue_lane"]


@pytest.mark.parametrize("change,expected", [
    ({}, None), ({"time": 0}, "worker_heartbeat_stale"),
    ({"time": 2000}, "invalid_worker_heartbeat"),
    ({"rooms": {"r": {"started_at": -50000}}}, "room_deadline_exceeded"),
])
def test_watchdog_deadlines(change, expected):
    heartbeat = {"time": 1000, "run_id": "run", "pid": 42, **change}
    assert runner.watchdog_reason(heartbeat, now=1000, started=990,
                                  policy=safety.Policy(), run_id="run", pid=42) == expected


def test_watchdog_does_not_accept_fresh_timestamp_from_old_process():
    assert runner.watchdog_reason({"time": 999, "run_id": "old", "pid": 1},
                                  now=1000, started=0, policy=safety.Policy(),
                                  run_id="new", pid=2) == "worker_heartbeat_stale"


def test_old_heartbeat_rooms_cannot_kill_new_worker():
    assert runner.watchdog_reason({"time": 999, "run_id": "old", "pid": 1,
                                   "rooms": {"old_room": {"started_at": -50000}}},
                                  now=1000, started=999, policy=safety.Policy(),
                                  run_id="new", pid=2) is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_nonfinite_heartbeat_rejected(value):
    assert runner.watchdog_reason({"time": value, "run_id": "run", "pid": 1},
                                  now=1000, started=999, policy=safety.Policy(),
                                  run_id="run", pid=1) == "invalid_worker_heartbeat"


@pytest.mark.parametrize("first_code,category,expected", [
    (2, "judge_response_contract_failure", 0), (2, "transport_or_timeout", 1),
    (1, "judge_response_contract_failure", 1), (70, "judge_response_contract_failure", 1)])
def test_full_parent_flow_partial_continues_but_crashes_block(tmp_path, monkeypatch, first_code, category, expected):
    output = tmp_path / "combined_remaining29_plus20_113_execution_hardened_test"
    monkeypatch.setattr(runner, "OUTPUT_PARENT", tmp_path)
    @contextmanager
    def lock(path):
        yield
    aliases = {name: "model" for name in ("kimi", "glm", "sol")}
    legacy = SimpleNamespace(DEFAULT_OUTPUT=tmp_path / "old",
        DATASETS=("remaining29", "plus20"), inputs=SimpleNamespace(DEFAULT_OUTPUT=tmp_path / "old_inputs"),
        previous=SimpleNamespace(DEFAULT_OUTPUT=tmp_path / "old_previous", SOURCE_OUTPUT=tmp_path / "old_source",
            ALIASES=aliases, operator_lock=lock, check_ports=lambda *_: None,
            collect_credentials=lambda *_: {name: "fixture:not_real" for name in aliases}))
    for root in (legacy.DEFAULT_OUTPUT, legacy.previous.DEFAULT_OUTPUT, legacy.previous.SOURCE_OUTPUT,
                 tmp_path / "merged_success_kimi_glm_sol_fc0_parallel_shared_api2_v3_r1"):
        root.mkdir()
    def prepare(*_):
        safety.atomic_json(output / "execution_plan.json", {"test": True})
        safety.atomic_json(output / "combined_plan.json", {"routes": {}})
        return {}
    monkeypatch.setattr(runner, "prepare", prepare)
    monkeypatch.setattr(runner, "expected_rooms", lambda legacy, dataset, *_: 2 if dataset == "remaining29" else 1)
    launched = []
    class Process:
        def __init__(self, command, **kwargs):
            label = command[command.index("--worker") + 1]
            dataset, alias = label.split(":")
            self.code = first_code if dataset == "remaining29" else 0
            self.pid = 10000 + len(launched)
            launched.append(label)
            lane = output / "datasets" / dataset / "lanes" / alias
            evaluation = terminal_fixture(lane, statuses=("complete", "failed_nonretryable") if dataset == "remaining29" else ("complete",))
            if dataset == "remaining29":
                path = evaluation / "models/model/scenes/scene/rooms/room1/summary.json"
                summary = safety.read_json(path)
                summary["latest_failure"]["category"] = category
                safety.atomic_json(path, summary)
            run = safety.read_json(evaluation / "run_manifest.json")
            results.lane_result(evaluation, run_id=command[command.index("--run-id") + 1],
                                expected_rooms=2 if dataset == "remaining29" else 1)
            safety.atomic_json(lane / "execution_worker_result.json", {
                "run_id": command[command.index("--run-id") + 1], "cli_completed": True,
                "execution_plan_sha256": safety.digest(output / "execution_plan.json"),
                "campaign_identity_sha256": run["identity_sha256"], "returncode": self.code})
            receipt = safety.read_json(lane / "execution_worker_result.json")
            safety.atomic_json(lane / "execution_worker_result.json", {**receipt,
                "execution_result_sha256": safety.digest(evaluation / "execution_result.json")})
    monkeypatch.setattr(runner.subprocess, "Popen", Process)
    monkeypatch.setattr(runner, "supervise_worker", lambda process, *_: (process.code, None))
    monkeypatch.setattr(runner, "stop_process", lambda *_: None)
    result = runner.launch(SimpleNamespace(output_root=output, prepare=False, resume=False), legacy, safety.Policy())
    assert result == expected
    state = safety.read_json(output / "execution_state.json")
    assert state["status"] == ("completed_with_skips" if expected == 0 else "failed")
    assert len(launched) == (6 if first_code == 2 else 3)
    for alias in aliases:
        assert state["jobs"][f"plus20:{alias}"]["status"] == ("complete" if first_code == 2 else "not_started")


def test_heartbeat_has_stage_and_stops_thread(tmp_path):
    hb = safety.Heartbeat(tmp_path / "hb.json", "run", interval=0.01)
    with hb:
        hb.update("r", stage="collision", started_at=time.time())
        hb.emit()
        assert safety.read_json(hb.path)["rooms"]["r"]["stage"] == "collision"
    assert not hb.thread.is_alive()


def test_check_mode_is_read_only(tmp_path, monkeypatch):
    calls = []
    legacy = startup_fixture(tmp_path)
    legacy.verify_plan = lambda p: calls.append(p)
    monkeypatch.setattr(runner, "load_legacy", lambda: legacy)
    assert runner.main(["--check", "--output-root", str(tmp_path / "new")]) == 0
    assert calls == [legacy.DEFAULT_OUTPUT]
    assert not (tmp_path / "new").exists()


@pytest.mark.skipif(not (FROZEN / "tests/test_non_rectangular_resilient_runner.py").is_file(), reason="sealed test fixture unavailable")
@pytest.mark.parametrize("origin,limit", [("EndpointConnectionError", 3), ("ValueError", 3),
                                        ("EndpointConnectionError", 1), ("BlenderTimeout", 3),
                                        ("BlenderContract", 3), ("LocalKeyError", 3)])
def test_real_pinned_coordinator_bounded_retry_and_unchanged_success(tmp_path, origin, limit):
    """Fresh process, real frozen coordinator, synthetic assets/mock evaluator."""
    script = r'''
import importlib.util, json, sys
from pathlib import Path
root, frozen, target = map(Path, sys.argv[1:4]); origin = sys.argv[4]; limit = int(sys.argv[5])
sys.path.insert(0, str(root)); sys.path.insert(0, str(frozen / 'src'))
spec = importlib.util.spec_from_file_location('frozen_fixtures', frozen / 'tests/test_non_rectangular_resilient_runner.py')
fixtures = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixtures)
import benchmark.non_rectangular.resilient as r
import benchmark.non_rectangular.runtime as runtime
import benchmark.non_rectangular.evaluator as evaluator
from scripts.nonrect_hardening_v1 import overlay
from scripts.nonrect_hardening_v1.safety import Heartbeat, Policy
model_root, csv_path, asset_root = fixtures._generation_root(target)
def config(name):
 return r.ResilientCampaignConfig.create(model_roots={'model': model_root}, output_root=target/name,
   asset_csv=csv_path, asset_root=asset_root, catalog_snapshot_id='fixture-v1',
   blender_bin=Path(sys.executable), max_workers=2, max_room_attempts=3)
baseline=r.run_resilient_nonrect_campaign(config('baseline'), evaluator_factory=r.NoAPIMockEvaluatorFactory(), materializer_backend=r.NoAPIMockMaterializer())
assert baseline.status=='complete'
original_evaluate = evaluator.CanonicalNonRectangularRoomEvaluator.evaluate
class Factory(r.NoAPIMockEvaluatorFactory):
 def __init__(self): self.calls={}
 def build(self, context):
  n=self.calls.get(context.unit.room_id,0)+1;self.calls[context.unit.room_id]=n
  original=super().build(context)
  if n!=1:return original
  class FailOnce:
   def evaluate(self, unit):
    if origin == 'LocalKeyError': raise KeyError('fixture local evaluator bug')
    cause = {'error_type': origin}
    if origin.startswith('Blender'):
     cause = {'exception_chain': [{'error_type':'BlenderRenderError'},
       {'error_type':'TimeoutExpired' if origin=='BlenderTimeout' else 'ValueError'}]}
    overlay.CURRENT.get().observe('collision', {'pairs':[{'requires_vlm':True,'final_verdict':None, **cause}]})
    raise evaluator.NonRectangularRoomMetricIncomplete('fixture transient',metric_id='collision', failure_category='api_transport_failure')
  return FailOnce()
factory=Factory()
with Heartbeat(target/'heartbeat.json','fixture') as heartbeat:
 overlay.install(r,runtime,evaluator,output=target,dataset='fixture',policy=Policy(minimum_free_gib=0.001,max_repeated_failure_attempts=limit),heartbeat=heartbeat,code_identity='fixture')
 hardened=r.run_resilient_nonrect_campaign(config('hardened'),evaluator_factory=factory,materializer_backend=r.NoAPIMockMaterializer())
recovered = origin in {'EndpointConnectionError','BlenderTimeout'} and limit > 1
assert hardened.status==('complete' if recovered else 'failed')
assert all(n==(2 if recovered else 1) for n in factory.calls.values()),factory.calls
assert evaluator.CanonicalNonRectangularRoomEvaluator.evaluate is original_evaluate
for baseline_report in (target/'baseline').glob('models/*/scenes/*/rooms/*/evaluation_attempts/attempt_001/room_evaluation_report.json'):
 relative=baseline_report.relative_to(target/'baseline')
 hardened_report=target/'hardened'/str(relative).replace('attempt_001','attempt_002')
 if recovered:
  assert json.loads(baseline_report.read_text())==json.loads(hardened_report.read_text())
 else:
  assert not hardened_report.exists()
print('PINNED_COORDINATOR_GOLDEN_OK',len(factory.calls))
'''
    completed = subprocess.run([sys.executable, "-B", "-c", script, str(ROOT), str(FROZEN), str(tmp_path), origin, str(limit)],
                               capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout[-2500:] + completed.stderr[-2500:]
    assert "PINNED_COORDINATOR_GOLDEN_OK" in completed.stdout


@pytest.mark.requires_local_data
@pytest.mark.skipif(not (FROZEN / "scripts/run_non_rectangular_resilient_evaluation.py").is_file(), reason="sealed runtime unavailable")
def test_real_completed_reports_reused_read_only_with_identical_runtime(tmp_path):
    """Verify one historical success per lane; only write temporary pointers."""
    legacy_root = runner.OUTPUT_PARENT / "combined_remaining29_plus20_113_collision_bundle_v1_r1"
    if not (legacy_root / "combined_plan.json").is_file():
        pytest.skip("historical completed campaign unavailable")
    script = r'''
import sys
from pathlib import Path
root, frozen, target = map(Path, sys.argv[1:4])
sys.path.insert(0, str(root)); sys.path.insert(0, str(frozen / 'src'))
from scripts import run_complicated_combined142_hardened as runner
from scripts.nonrect_hardening_v1 import overlay, safety
legacy = runner.load_legacy()
code, plan = legacy.verify_plan(legacy.DEFAULT_OUTPUT)
cli = legacy.previous.load_official_cli(code)
import benchmark.non_rectangular.resilient as r
import benchmark.non_rectangular.runtime as runtime
import benchmark.non_rectangular.evaluator as evaluator
class Base:
 _room_selected_report = r._Coordinator._room_selected_report
 def _room_root(self, bundle, unit):
  return self.output_root / 'models' / bundle.model / 'scenes' / bundle.scene_id / 'rooms' / unit.room_id
r._Coordinator = Base
overlay.install(r, runtime, evaluator, output=target, dataset='remaining29',
 policy=safety.Policy(), heartbeat=safety.Heartbeat(target/'heartbeat.json', 'fixture'),
 code_identity='fixture', reuse_root=legacy.DEFAULT_OUTPUT)
for alias, model in legacy.previous.ALIASES.items():
 source_eval = legacy.DEFAULT_OUTPUT/'datasets/remaining29/lanes'/alias/'evaluation'
 selected = next(source_eval.glob('models/*/scenes/*/rooms/*/room_report_selected.json'))
 source_sha = safety.digest(selected)
 scene_id, room_id = selected.parents[2].name, selected.parent.name
 bundles = r.discover_generation_bundles(((model, legacy.DEFAULT_OUTPUT/'datasets/remaining29/inputs'/model),))
 bundle = next(b for b in bundles if b.scene_id == scene_id)
 unit = next(u for u in bundle.units if u.room_id == room_id)
 proxy = cli._OwnedAPI2GPT56Proxy(port=plan['routes'][alias]['port'],
  launcher=root/'Support/bash/local/run_litellm_gpt56sol_standard_proxy.sh')
 coordinator = r._Coordinator()
 coordinator.evaluator_factory = runtime.DefaultNonRectangularRuntimeFactory(proxy.runtime_config(blender_bin=legacy.previous.GATE))
 coordinator.output_root = target/'datasets/remaining29/lanes'/alias/'evaluation'
 original = safety.read_json(Path(safety.read_json(selected)['room_report_path']))
 assert coordinator._room_selected_report(bundle, unit) == original
 assert coordinator._room_selected_report(bundle, unit) == original
 pointer = safety.read_json(coordinator._room_root(bundle, unit)/'room_report_selected.json')
 assert pointer['reused_read_only'] and pointer['attempt'] == 0
 assert pointer['room_report_path'] == safety.read_json(selected)['room_report_path']
 assert safety.digest(selected) == source_sha
print('HISTORICAL_REUSE_OK', len(legacy.previous.ALIASES))
'''
    completed = subprocess.run([sys.executable, "-B", "-c", script, str(ROOT), str(FROZEN), str(tmp_path)],
                               capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout[-2500:] + completed.stderr[-2500:]
    assert "HISTORICAL_REUSE_OK 3" in completed.stdout


@pytest.mark.parametrize("failure,expected", [
    ({"category": "room_hard_failure", "error_type": "KeyError"}, "hard_failure"),
    ({"category": "evidence_exhaustion_unclosed"}, "hard_failure"),
    ({"category": "judge_response_contract_failure"}, "hard_failure"),
    ({"category": "api_configuration", "error_type": "ValueError"}, "infra_failure"),
    ({"error_type": "EndpointMalformedResponseError"}, "infra_failure"),
    ({"category": "execution_budget_exhausted"}, "infra_failure"),
    ({"category": "execution_identity_drift"}, "infra_failure"),
    ({"category": "unclassified_failure"}, "unresolved"),
])
def test_execution_skip_taxonomy_keeps_infrastructure_distinct(failure, expected):
    assert results.failure_kind(failure) == expected


def test_accounted_skip_returns_result_without_zero_imputation(tmp_path):
    evaluation = terminal_fixture(tmp_path)
    report = evaluation / "models/model/scenes/scene/rooms/room0/report.json"
    before = safety.digest(report)
    value = results.lane_result(evaluation, run_id="fresh", expected_rooms=2)
    assert value["status"] == "completed_with_skips" and value["all_work_accounted"]
    assert value["score_coverage"] == {"complete_rooms": 1, "planned_rooms": 2,
                                        "completed_metrics": 8, "planned_metrics": 16}
    good, skipped = value["scenes"][0]["rooms"]
    assert good["metrics"]["collision"]["score"] == 0.0  # A real invalid score remains a score.
    assert skipped["status"] == "skipped_hard_failure"
    assert all(m["score"] is None and m["status"] == "skipped" for m in skipped["metrics"].values())
    assert value["aggregate_score"] is None and skipped["official_room_report"] is None
    assert safety.digest(report) == before
    assert value["official_score_publication"]["eligible"] is False
    assert value["workflow_alerts"][0]["code"] == "hard_failure_skips_present"


def test_missing_work_stays_in_denominator_and_cannot_be_complete(tmp_path):
    value = results.lane_result(terminal_fixture(tmp_path), run_id="fresh", expected_rooms=4)
    assert value["status"] == "incomplete" and not value["all_work_accounted"]
    assert value["not_started_room_count"] == 2
    assert value["score_coverage"]["planned_metrics"] == 32


def test_worker_abort_cannot_publish_complete_accounting_from_old_scores(tmp_path):
    value = results.lane_result(terminal_fixture(tmp_path, statuses=("complete",)),
        run_id="fresh", expected_rooms=1, execution_failure={"error_type": "IntegrityError"})
    assert value["status"] == "incomplete" and not value["all_work_accounted"]
    assert value["score_coverage"]["complete_rooms"] == 1


@pytest.mark.parametrize("other", ["IncompleteRead", "UnknownError", "BlenderRenderError", "IntegrityError"])
def test_hard_skip_cannot_absorb_mixed_infrastructure_or_unknown_causes(other):
    assert not results.hard_origin([{"error_type": "ValueError"}, {"error_type": other}])


def test_stale_aggregation_skip_does_not_override_success(tmp_path):
    evaluation = terminal_fixture(tmp_path, statuses=("complete",))
    scene = evaluation / "models/model/scenes/scene"
    safety.atomic_json(scene / "execution_aggregation_skip.json", {"old": True})
    value = results.lane_result(evaluation, run_id="fresh", expected_rooms=1)
    assert value["status"] == "complete" and value["aggregation_skip_count"] == 0
    assert "aggregation_failure" not in value["scenes"][0]


def test_accounted_receipt_cannot_hide_infrastructure_failure(tmp_path):
    evaluation = terminal_fixture(tmp_path)
    path = evaluation / "models/model/scenes/scene/rooms/room1/summary.json"
    summary = safety.read_json(path)
    summary["latest_failure"]["category"] = "transport_or_timeout"
    safety.atomic_json(path, summary)
    value = results.lane_result(evaluation, run_id="fresh", expected_rooms=2)
    assert value["status"] == "incomplete" and value["room_counts"]["infra_failure"] == 1
    safety.atomic_json(evaluation / "execution_result.json", {**value, "all_work_accounted": True,
                                                             "status": "completed_with_skips"})
    run = safety.read_json(evaluation / "run_manifest.json")
    safety.atomic_json(tmp_path / "execution_worker_result.json", {"run_id": "fresh", "cli_completed": True,
        "execution_plan_sha256": "plan", "campaign_identity_sha256": run["identity_sha256"], "returncode": 2,
        "execution_result_sha256": safety.digest(evaluation / "execution_result.json")})
    assert not runner.adjudicate_worker(tmp_path, "fresh", 2, None, plan_sha="plan", model="model", rooms=2)["continue_lane"]


@pytest.mark.skipif(not (FROZEN / "tests/test_non_rectangular_resilient_runner.py").is_file(), reason="sealed fixture unavailable")
@pytest.mark.parametrize("mode", ["metric_failures", "aggregation_bug"])
def test_frozen_core_returns_partial_metric_results_and_skips_local_aggregation(tmp_path, mode):
    script = r'''
import importlib.util, json, sys
from pathlib import Path
root, frozen, target = map(Path,sys.argv[1:4]); mode=sys.argv[4]
sys.path.insert(0,str(root));sys.path.insert(0,str(frozen/'src'))
spec=importlib.util.spec_from_file_location('fixture',frozen/'tests/test_non_rectangular_resilient_runner.py')
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
import benchmark.non_rectangular.resilient as r
import benchmark.non_rectangular.runtime as runtime
import benchmark.non_rectangular.evaluator as evaluator
from scripts.nonrect_hardening_v1 import overlay, results, safety
model_root,csv_path,asset_root=fixture._generation_root(target)
original_evaluate=evaluator.CanonicalNonRectangularRoomEvaluator.evaluate
evaluator.check_collision=lambda *a,**kw: ({'status':'requires_vlm','score':None,'pairs':[
 {'requires_vlm':True,'final_verdict':None,'error_type':'ValueError',
  'stop_reason':'camera_constraint_contract_invalid'}]} if mode=='metric_failures' else {'status':'checked','score':0})
evaluator.check_polygon_oob=lambda *a,**kw: {'status':'checked','score':1}
evaluator.check_support=lambda *a,**kw: {'status':'checked','score':1}
def build(self, context):
 raw={name:{'status':'evaluated','score':1} for name in results.METRICS[3:]}
 if mode=='metric_failures': raw['semantic_placement_consistency']={'status':'failed','score':None,
  'infrastructure_failures':[{'error_type':'ValueError','error':'fixture missing check id'}]}
 return evaluator.CanonicalNonRectangularRoomEvaluator(
  runtime_by_room={context.unit.room_id:{'object_grouping_report':{}}},
  scene_quality_config={},scene_quality_evaluator=lambda *a,**kw:{'metrics':raw})
runtime.DefaultNonRectangularRuntimeFactory.build=build
if mode=='aggregation_bug':
 def fail_aggregate(self,bundle):raise KeyError('fixture aggregation bug')
 r._Coordinator._aggregate_scene=fail_aggregate
config=r.ResilientCampaignConfig.create(model_roots={'model':model_root}, output_root=target/'eval',
 asset_csv=csv_path,asset_root=asset_root,catalog_snapshot_id='fixture',blender_bin=Path(sys.executable),max_workers=2)
factory=runtime.DefaultNonRectangularRuntimeFactory({'judge':{'model':'fixture'}})
with safety.Heartbeat(target/'hb','fixture') as hb:
 overlay.install(r,runtime,evaluator,output=target,dataset='fixture',policy=safety.Policy(minimum_free_gib=.001),
  heartbeat=hb,code_identity='fixture')
 core=r.run_resilient_nonrect_campaign(config,evaluator_factory=factory,materializer_backend=r.NoAPIMockMaterializer())
value=results.lane_result(target/'eval',run_id='fixture',expected_rooms=core.room_count)
assert value['status']=='completed_with_skips',value
assert value['all_work_accounted'] and value['aggregate_score'] is None
assert evaluator.CanonicalNonRectangularRoomEvaluator.evaluate is original_evaluate
if mode=='metric_failures':
 assert core.status=='failed'
 assert value['room_counts']['skipped_hard_failure']==core.room_count
 assert value['score_coverage']['completed_metrics']==core.room_count*6,value
 for scene in value['scenes']:
  for room in scene['rooms']:
   assert room['metrics']['collision']['score'] is None
   assert room['metrics']['semantic_placement_consistency']['score'] is None
   assert room['metrics']['support']['score']==1
   assert room['official_room_report'] is None
else:
 assert core.status=='complete'
 assert value['room_counts']['complete']==core.room_count
 assert value['aggregation_skip_count']>0
 assert value['score_coverage']['completed_metrics']==core.room_count*8
print('FROZEN_ACCOUNTING_OK')
'''
    completed = subprocess.run([sys.executable, "-B", "-c", script, str(ROOT), str(FROZEN), str(tmp_path), mode],
                               capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout[-4000:] + completed.stderr[-4000:]
    assert "FROZEN_ACCOUNTING_OK" in completed.stdout


def fresh_receipt(lane, *, code, run_id="fresh"):
    evaluation = lane / "evaluation"
    run = safety.read_json(evaluation / "run_manifest.json")
    safety.atomic_json(lane / "execution_worker_result.json", {
        "run_id": run_id, "cli_completed": True, "returncode": code,
        "execution_plan_sha256": "plan", "campaign_identity_sha256": run["identity_sha256"],
        "execution_result_sha256": safety.digest(evaluation / "execution_result.json")})


def test_repeated_reasons_count_distinct_rooms_not_nested_diagnostics(tmp_path):
    evaluation = terminal_fixture(tmp_path, statuses=("failed_nonretryable", "failed_nonretryable"))
    for root in sorted(evaluation.glob("models/*/scenes/*/rooms/*")):
        summary = safety.read_json(root / "summary.json")
        safety.atomic_json(root / "summary.json", {**summary, "evaluation_attempt_count": 1})
        attempt = root / "evaluation_attempts/attempt_001"
        detail = {"stage": "evaluation", "metric": "semantic_placement_consistency",
                  "reason_code": "placement.check_coverage_mismatch", "error_type": "ValueError",
                  "message": "raw message must not enter the summary"}
        safety.atomic_json(attempt / "execution_diagnostics/failure_reasons.json", {
            "attempt_root": str(attempt), "details": [detail, detail]})
    value = results.lane_result(evaluation, run_id="fresh", expected_rooms=2)
    group, = value["failure_reason_summary"]
    assert group["affected_room_count"] == 2 and group["planned_room_count"] == 2
    assert group["reason_code"] == "placement.check_coverage_mismatch"
    assert group["same_root_cause_confirmed"] is False
    assert "raw message" not in json.dumps(value["failure_reason_summary"])
    repeated = [a for a in value["workflow_alerts"] if a["code"] == "repeated_failure_signature"]
    assert len(repeated) == 1 and repeated[0]["affected_room_count"] == 2
    assert value["official_score_publication"]["eligible"] is False
    assert all(c == {"complete_rooms": 0, "planned_rooms": 2} for c in value["metric_coverage"].values())


@pytest.mark.parametrize("category", ["transport_or_timeout", "http_service_error", "api_protocol_failure",
                                      "execution_budget_exhausted", "camera_or_renderer_failure"])
def test_infrastructure_category_vetoes_generic_python_hard_type(category):
    assert results.failure_kind({"category": category, "error_type": "ValueError"}) == "infra_failure"


@pytest.mark.parametrize("tamper", ["coverage", "duplicate", "metric_id", "publication", "boolean_count", "status"])
def test_parent_reconciles_coverage_and_gate_not_just_worker_hash(tmp_path, tamper):
    evaluation = terminal_fixture(tmp_path)
    value = results.lane_result(evaluation, run_id="fresh", expected_rooms=2)
    if tamper == "coverage": value["score_coverage"]["completed_metrics"] = 16
    if tamper == "duplicate": value["scenes"][0]["rooms"][1] = value["scenes"][0]["rooms"][0]
    if tamper == "metric_id": value["scenes"][0]["rooms"][0]["metrics"]["collision"]["metric"] = "support"
    if tamper == "publication": value["official_score_publication"]["eligible"] = True
    if tamper == "boolean_count": value["score_coverage"]["complete_rooms"] = True
    if tamper == "status": value["status"] = "complete"
    safety.atomic_json(evaluation / "execution_result.json", value)
    fresh_receipt(tmp_path, code=2)
    outcome = runner.adjudicate_worker(tmp_path, "fresh", 2, None, plan_sha="plan", model="model", rooms=2)
    assert outcome["status"] == "fatal" and not outcome["continue_lane"]


def test_original_aggregate_drift_blocks_verified_worker(tmp_path):
    evaluation = terminal_fixture(tmp_path, statuses=("complete",))
    results.lane_result(evaluation, run_id="fresh", expected_rooms=1)
    fresh_receipt(tmp_path, code=0)
    safety.atomic_json(evaluation / "models/model/scenes/scene/evaluation_report.json", {"terminal_status": "complete", "changed": True})
    outcome = runner.adjudicate_worker(tmp_path, "fresh", 0, None, plan_sha="plan", model="model", rooms=1)
    assert outcome["status"] == "fatal" and not outcome["continue_lane"]


def test_self_consistent_envelope_cannot_change_an_original_score(tmp_path):
    evaluation = terminal_fixture(tmp_path, statuses=("complete",))
    value = results.lane_result(evaluation, run_id="fresh", expected_rooms=1)
    value["scenes"][0]["rooms"][0]["metrics"]["collision"]["score"] = 1.0
    assurance.validate(value, results.METRICS)  # Internally valid, but not its source score.
    safety.atomic_json(evaluation / "execution_result.json", value)
    fresh_receipt(tmp_path, code=0)
    outcome = runner.adjudicate_worker(tmp_path, "fresh", 0, None, plan_sha="plan", model="model", rooms=1)
    assert outcome["status"] == "fatal" and not outcome["continue_lane"]


def test_self_consistent_envelope_cannot_relabel_an_infra_room_as_skipped(tmp_path):
    evaluation = terminal_fixture(tmp_path)
    value = results.lane_result(evaluation, run_id="fresh", expected_rooms=2)
    summary_path = evaluation / "models/model/scenes/scene/rooms/room1/summary.json"
    summary = safety.read_json(summary_path)
    summary["latest_failure"]["category"] = "transport_or_timeout"
    safety.atomic_json(summary_path, summary)
    assurance.validate(value, results.METRICS)
    fresh_receipt(tmp_path, code=2)
    outcome = runner.adjudicate_worker(tmp_path, "fresh", 2, None, plan_sha="plan", model="model", rooms=2)
    assert outcome["status"] == "fatal" and not outcome["continue_lane"]


@pytest.mark.parametrize("order", [["room0", "room0"], ["../room0"], ["/room0"], [".."]])
def test_scene_inventory_cannot_duplicate_or_escape_rooms(tmp_path, order):
    evaluation = terminal_fixture(tmp_path)
    scene = evaluation / "models/model/scenes/scene"
    value = safety.read_json(scene / "summary.json")
    safety.atomic_json(scene / "summary.json", {**value, "room_order": order})
    with pytest.raises(safety.IntegrityError):
        results.lane_result(evaluation, run_id="fresh", expected_rooms=2)


def test_supervisor_harvests_before_first_scene_summary_without_faking_success(tmp_path):
    evaluation = terminal_fixture(tmp_path, statuses=("complete", "pending"))
    scene = evaluation / "models/model/scenes/scene"
    (scene / "summary.json").unlink()
    (scene / "rooms/room1/summary.json").unlink()
    (evaluation / "terminal_manifest.json").unlink()
    safety.atomic_json(scene / "preflight.json", {"room_order": ["room0", "room1"]})
    originals = {p: safety.digest(p) for p in evaluation.rglob("*.json")}
    outcome = runner.adjudicate_worker(tmp_path, "new", -9, "room_deadline_exceeded",
        plan_sha="plan", model="model", rooms=2)
    assert outcome["status"] == "interrupted" and not outcome["continue_lane"]
    value = safety.read_json(evaluation / "execution_result.json")
    assert value["status"] == "incomplete" and value["all_work_accounted"] is False
    assert value["run_id"] == "new" and value["score_coverage"]["completed_metrics"] == 8
    assert value["score_coverage"]["planned_metrics"] == 16
    assert value["room_counts"]["complete"] == 1 and value["room_counts"]["unresolved"] == 1
    assert value["scenes"][0]["rooms"][1]["source_status"] == "not_started_or_interrupted"
    assert value["official_score_publication"]["eligible"] is False
    assert all(safety.digest(p) == sha for p, sha in originals.items())
    assert not (evaluation / "terminal_manifest.json").exists()
    assert not (tmp_path / "execution_worker_result.json").exists()


def test_supervisor_interruption_cannot_relabel_old_complete_scores(tmp_path):
    evaluation = terminal_fixture(tmp_path, statuses=("complete",))
    results.lane_result(evaluation, run_id="old", expected_rooms=1)
    fresh_receipt(tmp_path, code=0, run_id="old")
    receipt_sha = safety.digest(tmp_path / "execution_worker_result.json")
    outcome = runner.adjudicate_worker(tmp_path, "new", -9, "worker_deadline_exceeded",
        plan_sha="plan", model="model", rooms=1)
    assert outcome["status"] == "interrupted" and not outcome["continue_lane"]
    value = safety.read_json(evaluation / "execution_result.json")
    assert value["status"] == "incomplete" and value["score_coverage"]["completed_metrics"] == 8
    assert value["official_score_publication"]["eligible"] is False
    assert safety.digest(tmp_path / "execution_worker_result.json") == receipt_sha


def test_supervisor_recovery_failure_is_diagnostic_not_success(tmp_path):
    evaluation = terminal_fixture(tmp_path, statuses=("complete",))
    safety.atomic_json(evaluation / "run_manifest.json", {"identity": {"model_order": ["other"]}, "identity_sha256": "bad"})
    outcome = runner.adjudicate_worker(tmp_path, "new", -9, "worker_deadline_exceeded",
        plan_sha="plan", model="model", rooms=1)
    assert outcome["status"] == "interrupted" and "result_recovery_diagnostic" in outcome
    assert not (evaluation / "execution_result.json").exists()


def test_interrupted_retry_cannot_resurrect_previous_attempt_scores(tmp_path):
    evaluation = terminal_fixture(tmp_path, statuses=("failed_nonretryable",))
    room = evaluation / "models/model/scenes/scene/rooms/room0"
    summary = safety.read_json(room / "summary.json")
    safety.atomic_json(room / "summary.json", {**summary, "evaluation_attempt_count": 1})
    first = room / "evaluation_attempts/attempt_001"
    payload = {"schema_version": results.RESULT_VERSION, "room_id": "room0", "attempt_root": str(first),
               "materialization_identity_sha256": "materialized", "execution_identity": "fixture",
               "metrics": {name: {"metric": name, "status": "complete", "score": 1.0} for name in results.METRICS}}
    safety.atomic_json(first / "attempt_manifest.json", {"materialization_identity_sha256": "materialized"})
    safety.atomic_json(first / "execution_metric_results.json", {**payload, "payload_sha256": safety.identity(payload)})
    (room / "evaluation_attempts/attempt_002").mkdir()
    outcome = runner.adjudicate_worker(tmp_path, "new", -9, "room_deadline_exceeded",
        plan_sha="plan", model="model", rooms=1)
    value = safety.read_json(evaluation / "execution_result.json")
    row, = value["scenes"][0]["rooms"]
    assert row["status"] == "unresolved" and row["source_status"] == "interrupted_after_room_summary"
    assert row["observed_evaluation_attempt_count"] == 2
    assert all(m["score"] is None for m in row["metrics"].values())
    assert value["score_coverage"]["completed_metrics"] == 0
    assert not outcome["continue_lane"] and not value["official_score_publication"]["eligible"]


@pytest.mark.parametrize("mode,exit_code", [("complete", 0), ("skip", 1), ("aggregation", 1), ("drift", 1), ("forged", 1)])
def test_publication_gate_is_read_only_and_requires_original_complete_scores(tmp_path, monkeypatch, capsys, mode, exit_code):
    legacy = SimpleNamespace(DATASETS=("remaining29",), previous=SimpleNamespace(ALIASES={"sol": "model"}))
    monkeypatch.setattr(runner, "verify_execution_plan", lambda *_: None)
    monkeypatch.setattr(runner, "expected_rooms", lambda *_: 1)
    safety.atomic_json(tmp_path / "execution_plan.json", {"test": True})
    lane = tmp_path / "datasets/remaining29/lanes/sol"
    evaluation = terminal_fixture(lane, statuses=("failed_nonretryable",) if mode == "skip" else ("complete",))
    if mode == "aggregation":
        scene = evaluation / "models/model/scenes/scene"
        (scene / "evaluation_report.json").unlink()
        safety.atomic_json(scene / "execution_aggregation_skip.json", {"error_type": "KeyError"})
    value = results.lane_result(evaluation, run_id="fresh", expected_rooms=1)
    code = 2 if mode == "skip" else 0
    fresh_receipt(lane, code=code)
    receipt = safety.read_json(lane / "execution_worker_result.json")
    safety.atomic_json(lane / "execution_worker_result.json", {**receipt,
        "execution_plan_sha256": safety.digest(tmp_path / "execution_plan.json")})
    if mode == "drift":
        safety.atomic_json(evaluation / "models/model/scenes/scene/rooms/room0/report.json", {"broken": True})
    state = {"status": value["status"], "execution_plan_sha256": safety.digest(tmp_path / "execution_plan.json"),
             "jobs": {"remaining29:sol": {"status": value["status"], "run_id": "fresh", "returncode": code}}}
    if mode == "forged":
        value["score_coverage"]["planned_metrics"] = 7
        safety.atomic_json(evaluation / "execution_result.json", value)
    safety.atomic_json(tmp_path / "execution_state.json", state)
    before = {p: (safety.digest(p), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}
    assert runner.check_publication(tmp_path, legacy) == exit_code
    response = json.loads(capsys.readouterr().out)
    assert response["eligible"] is (exit_code == 0) and response["published"] is False
    assert response["read_only"] is True
    after = {p: (safety.digest(p), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after
    if mode == "complete":
        assert value["workflow_alerts"] == []  # A real zero score is publishable.


def test_publication_mode_never_dispatches_prepare_or_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "load_legacy", lambda: "fixture")
    monkeypatch.setattr(runner, "check_publication", lambda output, legacy: 17)
    monkeypatch.setattr(runner, "launch", lambda *_: pytest.fail("read-only gate launched a campaign"))
    assert runner.main(["--check-publication", "--output-root", str(tmp_path)]) == 17


def startup_fixture(tmp_path):
    @contextmanager
    def lock(path):
        with path.open("a") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield handle
    legacy = SimpleNamespace(DEFAULT_OUTPUT=tmp_path / "old_combined", DATASETS=("remaining29", "plus20"),
        inputs=SimpleNamespace(DEFAULT_OUTPUT=tmp_path / "unused_historical_default"),
        previous=SimpleNamespace(DEFAULT_OUTPUT=tmp_path / "old_remaining", SOURCE_OUTPUT=tmp_path / "old_source",
                                 ALIASES={"sol": "model"}, operator_lock=lock))
    for root in (legacy.DEFAULT_OUTPUT, legacy.previous.DEFAULT_OUTPUT, legacy.previous.SOURCE_OUTPUT,
                 tmp_path / "merged_success_kimi_glm_sol_fc0_parallel_shared_api2_v3_r1"):
        root.mkdir()
    return legacy


def test_startup_uses_actual_historical_roots_not_unused_default(tmp_path, monkeypatch):
    legacy = startup_fixture(tmp_path)
    monkeypatch.setattr(runner, "OUTPUT_PARENT", tmp_path)
    calls = []
    monkeypatch.setattr(runner, "prepare", lambda *a: calls.append(a[0]) or {})
    output = tmp_path / "combined_remaining29_plus20_113_execution_hardened_test"
    assert runner.launch(SimpleNamespace(output_root=output, prepare=True, resume=False), legacy, safety.Policy()) == 0
    assert calls == [output]
    assert not legacy.inputs.DEFAULT_OUTPUT.exists()
    assert (tmp_path / "merged_success_kimi_glm_sol_fc0_parallel_shared_api2_v3_r1/.operator.lock").is_file()


def test_check_detects_missing_actual_source_without_creating_output(tmp_path, monkeypatch):
    legacy = startup_fixture(tmp_path)
    legacy.previous.SOURCE_OUTPUT.rmdir()
    legacy.verify_plan = lambda *_: pytest.fail("missing source should fail at startup readiness")
    monkeypatch.setattr(runner, "load_legacy", lambda: legacy)
    output = tmp_path / "new"
    with pytest.raises(runner.StartupPathError) as caught:
        runner.main(["--check", "--output-root", str(output)])
    assert caught.value.path == legacy.previous.SOURCE_OUTPUT
    assert not output.exists()


@pytest.mark.parametrize("layout", ["lanes/sol", "datasets/remaining29/lanes/sol"])
def test_check_and_launch_protect_direct_old_worker_locks(tmp_path, monkeypatch, layout):
    legacy = startup_fixture(tmp_path)
    path = legacy.DEFAULT_OUTPUT / layout / "evaluation/.runner.lock"
    safety.atomic_json(path, {})
    monkeypatch.setattr(runner, "OUTPUT_PARENT", tmp_path)
    monkeypatch.setattr(runner, "prepare", lambda *_: pytest.fail("busy historical worker must prevent staging"))
    output = tmp_path / "combined_remaining29_plus20_113_execution_hardened_test"
    with legacy.previous.operator_lock(path):
        with pytest.raises(BlockingIOError):
            runner.check_protected_locks(legacy)
        with pytest.raises(BlockingIOError):
            runner.launch(SimpleNamespace(output_root=output, prepare=True, resume=False), legacy, safety.Policy())
    assert not (output / "datasets").exists()
    runner.check_protected_locks(legacy)


def test_readiness_does_not_create_historical_locks(tmp_path):
    legacy = startup_fixture(tmp_path)
    before = set(tmp_path.rglob("*"))
    runner.check_protected_locks(legacy)
    assert set(tmp_path.rglob("*")) == before
    assert not legacy.inputs.DEFAULT_OUTPUT.exists()


def test_refusal_context_shows_local_missing_path_not_raw_message(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    path = tmp_path / "missing.json"
    value = runner.refusal_context(FileNotFoundError(2, "private raw exchange must never appear", str(path)))
    assert value["local_path"] == str(path)
    assert "private raw" not in json.dumps(value)
    assert runner.refusal_context(RuntimeError("raw API response text")) == {}
    assert runner.refusal_context(runner.StartupPathError(path))["local_path"] == str(path)
