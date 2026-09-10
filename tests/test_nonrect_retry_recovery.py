"""Offline tests for cumulative budgets and channel recovery; no live services."""
from contextvars import ContextVar
import errno
import http.client
from pathlib import Path
import socket
import ssl
import subprocess
import sys
from types import SimpleNamespace
import urllib.error
import urllib.request

import pytest

from scripts.nonrect_hardening_v1 import overlay, recovery as r, safety


class Clock:
    def __init__(self): self.now = 100.0
    def __call__(self): return self.now
    def advance(self, n): self.now += n


@pytest.mark.parametrize("field,value", [
    ("max_same_request_attempts", 1.5), ("max_request_attempts_per_room", True),
    ("circuit_probe_limit", 0), ("circuit_failure_threshold", float("nan")),
    ("circuit_max_cooldown_seconds", 1),
])
def test_bounded_policy_validation(field, value):
    with pytest.raises(ValueError):
        safety.Policy(**{field: value})


@pytest.mark.parametrize("same", [True, False])
def test_request_budget_persists_across_attempt_and_resume(tmp_path, same):
    policy = safety.Policy(max_same_request_attempts=2, max_request_attempts_per_room=2)
    path, clock = tmp_path / "budget.json", Clock()
    first = r.RoomBudget(path, "scope", policy, clock=clock)
    first.reserve("one")
    clock.advance(3)
    first.finish()
    clock.advance(100)
    second = r.RoomBudget(path, "scope", policy, clock=clock)
    assert second.value["elapsed_seconds"] == 3
    second.reserve("one" if same else "two")
    with pytest.raises(r.RetryBudgetExhausted): second.reserve("one")
    assert safety.read_json(path)["wire_attempts"] == 2
    with pytest.raises(safety.IntegrityError): r.RoomBudget(path, "other", policy)


def test_same_request_cap_cannot_reset_by_changing_attempt(tmp_path):
    policy = safety.Policy(max_same_request_attempts=1)
    budget = r.RoomBudget(tmp_path / "budget", "scope", policy)
    budget.reserve("same")
    budget.finish()
    budget = r.RoomBudget(tmp_path / "budget", "scope", policy)
    with pytest.raises(r.RetryBudgetExhausted): budget.reserve("same")


def test_budget_clock_survives_crash_and_refuses_corrupt_counters(tmp_path):
    path, clock = tmp_path / "budget", Clock()
    policy = safety.Policy(room_timeout_seconds=10)
    r.RoomBudget(path, "scope", policy, clock=clock)
    clock.advance(11)
    resumed = r.RoomBudget(path, "scope", policy, clock=clock)
    with pytest.raises(r.RetryBudgetExhausted): resumed.ensure()
    value = safety.read_json(path)
    value["wire_attempts"] = -1
    safety.atomic_json(path, value)
    with pytest.raises(safety.IntegrityError): r.RoomBudget(path, "scope", policy)


def test_repeated_root_cause_budget_and_no_double_count(tmp_path):
    policy = safety.Policy(max_repeated_failure_attempts=2)
    path = tmp_path / "budget"
    budget = r.RoomBudget(path, "scope", policy)
    budget.failure()
    budget.failed_attempt("same root")
    assert budget.value["infra_failures"] == 1
    budget.finish()
    budget = r.RoomBudget(path, "scope", policy)
    with pytest.raises(r.RetryBudgetExhausted): budget.failed_attempt("same root")
    assert budget.value["infra_failures"] == 2


def test_circuit_opens_single_probe_recovers_and_ignores_stale_success(tmp_path):
    clock = Clock()
    p = safety.Policy(circuit_failure_threshold=2, circuit_cooldown_seconds=1)
    circuit = r.Circuit(tmp_path / "circuit", p, clock=clock)
    first = circuit.acquire()
    stale = circuit.acquire()
    circuit.outcome(first, transient=True)
    circuit.outcome(first, transient=True)
    assert circuit.value["state"] == "open"
    circuit.outcome(stale, transient=False)
    assert circuit.value["state"] == "open"
    clock.advance(1)
    probe = circuit.acquire()
    assert circuit.value["state"] == "half_open"
    def waiting(): raise RuntimeError("second requester must wait")
    with pytest.raises(RuntimeError, match="must wait"):
        circuit.acquire(on_wait=waiting)
    circuit.outcome(probe, transient=False)
    assert circuit.value["state"] == "closed"
    assert circuit.value["probes"] == 0


def test_circuit_probe_exhaustion_survives_resume(tmp_path):
    clock = Clock()
    p = safety.Policy(circuit_failure_threshold=1, circuit_probe_limit=1, circuit_cooldown_seconds=1)
    path = tmp_path / "circuit"
    circuit = r.Circuit(path, p, clock=clock)
    circuit.outcome(circuit.acquire(), transient=True)
    clock.advance(1)
    circuit.outcome(circuit.acquire(), transient=True)
    with pytest.raises(r.ChannelUnavailable): circuit.acquire()
    with pytest.raises(r.ChannelUnavailable): r.Circuit(path, p, clock=clock).ensure()


def test_circuit_crashed_probe_does_not_get_unlimited_recovery(tmp_path):
    clock = Clock()
    p = safety.Policy(circuit_failure_threshold=1, circuit_probe_limit=1, circuit_cooldown_seconds=1)
    path = tmp_path / "circuit"
    circuit = r.Circuit(path, p, clock=clock)
    circuit.outcome(circuit.acquire(), transient=True)
    clock.advance(1)
    circuit.acquire()
    resumed = r.Circuit(path, p, clock=clock)
    clock.advance(1)
    with pytest.raises(r.ChannelUnavailable): resumed.acquire()


def test_outage_deadline_and_cancelled_probe(tmp_path):
    clock = Clock()
    p = safety.Policy(circuit_failure_threshold=1, circuit_cooldown_seconds=1, circuit_outage_timeout_seconds=5)
    circuit = r.Circuit(tmp_path / "circuit", p, clock=clock)
    circuit.outcome(circuit.acquire(), transient=True)
    clock.advance(1)
    circuit.cancel(circuit.acquire())
    assert circuit.value["probes"] == 0
    clock.advance(5)
    with pytest.raises(r.ChannelUnavailable): circuit.ensure()


@pytest.mark.parametrize("exc,expected", [
    (urllib.error.HTTPError("https://invalid", 503, "failure", {}, None), True),
    (urllib.error.HTTPError("https://invalid", 401, "failure", {}, None), False),
    (urllib.error.URLError(ConnectionResetError()), True),
    (urllib.error.URLError(socket.gaierror(socket.EAI_AGAIN, "temporary")), True),
    (urllib.error.URLError(socket.gaierror(socket.EAI_NONAME, "wrong host")), False),
    (urllib.error.URLError(ssl.SSLCertVerificationError()), False),
    (urllib.error.URLError(FileNotFoundError()), False),
    (http.client.IncompleteRead(b"partial"), True),
])
def test_transport_root_classification(exc, expected):
    assert r.transport_transient(exc) is expected


@pytest.mark.parametrize("raw,expected", [
    ({"exception_chain": [{"error_type": "BlenderRenderError"}, {"error_type": "TimeoutExpired"}]}, True),
    ({"exception_chain": [{"error_type": "BlenderRenderError"}, {"error_type": "ValueError"}]}, False),
    ({"error_type": "BlenderRenderError", "error": "renderer failure"}, False),
    ({"error_type": "OSError", "errno": errno.EAGAIN}, True),
    ({"error_type": "OSError", "errno": errno.ENOSPC}, False),
    ({"error_type": "OSError", "errno": errno.ENOMEM}, False),
    ({"error_type": "TimeoutExpired", "stop_reason": "evidence_budget_exhausted"}, False),
    ({"error_type": "ValueError", "cause": {"error_type": "TimeoutExpired"}}, False),
    ([{"error_type": "BlenderRenderError"}, {"error_type": "TimeoutError"}], False),
    ([{}, {"error_type": "TimeoutError"}], False),
    ({"errors": [{"error": "untyped failure"}, {"error_type": "TimeoutError"}]}, False),
])
def test_structured_causes_do_not_promote_contract_failures(raw, expected):
    assert overlay.transient_origin(raw) is expected


@pytest.mark.parametrize("fail_read", [False, True])
def test_wire_hook_counts_actual_attempts_preserves_bytes_and_no_secrets(tmp_path, fail_read):
    calls = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def read(self):
            if fail_read: raise http.client.IncompleteRead(b"secret response")
            return b"unaltered response"
    def open_wire(*args):
        calls.append(args)
        return Response()
    module = SimpleNamespace(_urlopen_no_redirect=open_wire)
    current = ContextVar("request_context", default=None)
    p = safety.Policy(max_same_request_attempts=1)
    ctx = overlay.RoomContext(tmp_path / "room", "room", safety.Heartbeat(tmp_path / "hb", "run"), p, "scope")
    ctx.budget = r.RoomBudget(tmp_path / "budget", "scope", p)
    circuit = r.Circuit(tmp_path / "circuit", p)
    r.install_transport(module, current, circuit)
    token = current.set(ctx)
    request = urllib.request.Request("https://example.invalid/private?token=secret", data=b"secret prompt",
                                     headers={"Authorization": "Bearer secret"})
    try:
        if fail_read:
            with pytest.raises(http.client.IncompleteRead):
                with module._urlopen_no_redirect(request, 3) as response: response.read()
        else:
            with module._urlopen_no_redirect(request, 3) as response:
                assert response.read() == b"unaltered response"
        with pytest.raises(r.RetryBudgetExhausted): module._urlopen_no_redirect(request, 3)
    finally:
        current.reset(token)
    assert len(calls) == 1
    assert ctx.budget.value["infra_failures"] == int(fail_read)
    assert "secret" not in "".join(p.read_text() for p in tmp_path.glob("*") if p.is_file())


def test_checkpoint_cannot_hide_swallowed_execution_budget(tmp_path):
    p = safety.Policy(max_infra_failures_per_room=1)
    ctx = overlay.RoomContext(tmp_path, "room", safety.Heartbeat(tmp_path / "hb", "run"), p, "scope", tmp_path / "attempt")
    ctx.budget = r.RoomBudget(tmp_path / "budget", "scope", p)
    def swallowed():
        ctx.budget.failure()
        return {"status": "checked", "score": 1}
    with pytest.raises(r.RetryBudgetExhausted):
        overlay.checkpoint_call(ctx, "collision", "same", swallowed, lambda r: True)
    assert not list(tmp_path.glob("execution_checkpoints/**/*.json"))


def test_failure_scope_requires_all_failed_events_and_respects_root_stop():
    raw = {"pairs": [{"requires_vlm": True, "final_verdict": None, "error_type": "TimeoutError"},
                     {"requires_vlm": True, "final_verdict": None}]}
    assert not overlay.transient_origin(overlay._failed_scope(raw, "collision"))
    raw["pairs"].pop()
    assert overlay.transient_origin(overlay._failed_scope(raw, "collision"))
    raw["stop_reason"] = "evidence_budget_exhausted"
    assert not overlay.transient_origin(overlay._failed_scope(raw, "collision"))


@pytest.mark.parametrize("raw,expected", [
    ({"exception_chain": [{"error_type": "BlenderRenderError"}, {"error_type": "IntegrityError"}]}, True),
    ({"error_type": "BlenderSourceSceneModifiedError"}, True),
    ({"error_type": "ValueError"}, False),
    ({"raw_response": {"error_type": "IntegrityError"}}, False),
])
def test_identity_guard_survives_exception_wrapping(raw, expected):
    assert overlay.identity_failure_origin(raw) is expected


def test_renderer_exception_chain_survives_core_string_wrapping_without_report_mutation(tmp_path):
    BlenderRenderError = type("BlenderRenderError", (RuntimeError,), {})
    class Renderer:
        def render_camera_views(self):
            try:
                raise subprocess.TimeoutExpired(["fixture-blender"], 1)
            except subprocess.TimeoutExpired as exc:
                raise BlenderRenderError("fixture render deadline") from exc
    overlay.observe_renderer_exceptions(Renderer)
    ctx = overlay.RoomContext(tmp_path, "r", safety.Heartbeat(tmp_path / "hb", "run"), safety.Policy(), "scope")
    token = overlay.CURRENT.set(ctx)
    try:
        with pytest.raises(BlenderRenderError): Renderer().render_camera_views()
    finally:
        overlay.CURRENT.reset(token)
    raw = {"error_type": "BlenderRenderError", "error": "fixture render deadline"}
    assert not overlay.transient_origin(raw)
    enriched = overlay.enrich_renderer_origins(raw, ctx)
    assert overlay.transient_origin(enriched)
    assert "cause" not in raw
    assert not overlay.transient_origin(overlay.enrich_renderer_origins({**raw, "error": "different failure"}, ctx))
    assert list(tmp_path.glob("execution_diagnostics/renderer_origin_*.json"))


def test_sibling_outage_does_not_invalidate_complete_result_but_own_guard_does(tmp_path):
    p = safety.Policy()
    circuit = r.Circuit(tmp_path / "circuit", p)
    ctx = overlay.RoomContext(tmp_path, "r", safety.Heartbeat(tmp_path / "hb", "run"), p, "scope", tmp_path / "attempt")
    ctx.circuit = circuit
    circuit.value.update(state="halted", outage_since=0)
    result = {"status": "checked", "score": 0}
    assert overlay.checkpoint_call(ctx, "collision", "same", lambda: result, lambda r: True) is result
    ctx.guard_error = r.ChannelUnavailable("own failed request")
    with pytest.raises(r.ChannelUnavailable):
        overlay.checkpoint_call(ctx, "collision", "same", lambda: result, lambda r: True)


def test_wire_timeout_clamped_to_remaining_room_and_outage_budgets(tmp_path):
    clock = Clock()
    p = safety.Policy(room_timeout_seconds=15, circuit_outage_timeout_seconds=10,
                      circuit_failure_threshold=1, circuit_cooldown_seconds=1)
    ctx = overlay.RoomContext(tmp_path, "r", safety.Heartbeat(tmp_path / "hb", "run"), p, "scope")
    ctx.budget = r.RoomBudget(tmp_path / "budget", "scope", p, clock=clock)
    circuit = r.Circuit(tmp_path / "circuit", p, clock=clock)
    circuit.outcome(circuit.acquire(), transient=True)
    clock.advance(3)
    calls = []
    def wire(request, timeout):
        calls.append(timeout)
        raise ConnectionResetError()
    module = SimpleNamespace(_urlopen_no_redirect=wire)
    current = ContextVar("ctx", default=ctx)
    r.install_transport(module, current, circuit)
    with pytest.raises(ConnectionResetError):
        module._urlopen_no_redirect(urllib.request.Request("https://example.invalid"), 3000)
    assert calls == [7]


def test_real_frozen_api_inner_retries_obey_shared_wire_budget(tmp_path):
    root = Path(__file__).resolve().parents[1]
    frozen = root / "Support/worktrees/collision-final-bundle-v1"
    if not frozen.is_dir():
        frozen = root  # Checked-in core, mocked transport; no local data needed.
    script = r'''
import sys, urllib.error, urllib.request
from pathlib import Path
root, frozen, target = map(Path, sys.argv[1:])
sys.path.insert(0,str(root));sys.path.insert(0,str(frozen/'src'))
from scripts.nonrect_hardening_v1 import overlay, recovery as r, safety
import benchmark.models.openai_compatible_model as m
from benchmark.non_rectangular.runtime import RecordingOpenAICompatibleModel, APIUsageRecorder
calls=[]
def wire(*a):
 calls.append(1)
 raise urllib.error.URLError(ConnectionResetError('fixture'))
m._urlopen_no_redirect=wire
p=safety.Policy(max_same_request_attempts=2,circuit_failure_threshold=100)
ctx=overlay.RoomContext(target,'room',safety.Heartbeat(target/'hb','run'),p,'scope')
ctx.budget=r.RoomBudget(target/'budget','scope',p)
r.install_transport(m,overlay.CURRENT,r.Circuit(target/'circuit',p))
token=overlay.CURRENT.set(ctx)
model=RecordingOpenAICompatibleModel(recorder=APIUsageRecorder(),exact_max_retries=5,exact_retry_delay_seconds=0,
 name='fixture',endpoint='https://example.invalid/v1',model_id='fixture',require_api_key=False,
 max_retries=5,retry_backoff_seconds=0,min_request_interval_seconds=0)
try:
 model.chat_messages([{'role':'user','content':'fixture'}])
 raise AssertionError('must stop')
except r.RetryBudgetExhausted: pass
finally:overlay.CURRENT.reset(token)
assert len(calls)==2,calls
print('NESTED_WIRE_BUDGET_OK')
'''
    completed = subprocess.run([sys.executable, "-B", "-c", script, str(root), str(frozen), str(tmp_path)],
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout[-2000:] + completed.stderr[-2000:]
    assert "NESTED_WIRE_BUDGET_OK" in completed.stdout
