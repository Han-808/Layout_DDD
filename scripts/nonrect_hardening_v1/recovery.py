"""Bounded execution recovery; no model output repair or metric substitution.

The transport hook observes *each wire attempt*, including frozen API retries.
Only hashes/counters are persisted, never request bodies, URLs, or credentials.
"""
from __future__ import annotations

import hashlib
import http.client
import math
import threading
import time
import urllib.error
from typing import Any

from .safety import (ExecutionGuardError, IntegrityError, Policy, atomic_json, read_json)

RETRY_POLICY_VERSION = "nonrect_infra_recovery_v2"
TRANSIENT_HTTP = {429, 500, 502, 503, 504}


class RetryBudgetExhausted(ExecutionGuardError):
    pass


class ChannelUnavailable(ExecutionGuardError):
    pass


class RoomBudget:
    """Cumulative across room attempts/resume, with a conservative crash clock."""
    def __init__(self, path, scope: str, policy: Policy, *, clock=time.time):
        self.path, self.policy, self.clock = path, policy, clock
        self.lock = threading.RLock()
        self.error = None
        if path.exists():
            self.value = read_json(path)
            if self.value.get("scope") != scope or self.value.get("version") != RETRY_POLICY_VERSION:
                raise IntegrityError("room retry budget identity mismatch")
            v = self.value
            if (any(type(v.get(k)) is not int or v[k] < 0 for k in ("wire_attempts", "infra_failures"))
                    or not isinstance(v.get("requests"), dict) or not isinstance(v.get("failure_origins"), dict)
                    or any(type(n) is not int or n < 0 for k in ("requests", "failure_origins") for n in v[k].values())
                    or sum(v["requests"].values()) != v["wire_attempts"]
                    or not _finite_nonnegative(v.get("elapsed_seconds"))
                    or (v.get("active_since") is not None and not _finite_nonnegative(v["active_since"]))):
                raise IntegrityError("corrupt room retry budget")
        else:
            self.value = {"version": RETRY_POLICY_VERSION, "scope": scope,
                          "wire_attempts": 0, "infra_failures": 0, "requests": {},
                          "elapsed_seconds": 0.0, "active_since": None, "failure_origins": {}}
        self._tick()
        self.value["active_since"] = clock()
        self.initial_failures = self.value["infra_failures"]
        self._save()

    def _tick(self):
        active = self.value["active_since"]
        if active is not None:
            self.value["elapsed_seconds"] += max(0, self.clock() - active)
            self.value["active_since"] = self.clock()

    def _save(self):
        atomic_json(self.path, self.value)

    def ensure(self):
        with self.lock:
            self._tick()
            reason = self.value.get("exhausted")
            if self.value["elapsed_seconds"] >= self.policy.room_timeout_seconds:
                reason = "cumulative_room_deadline"
            if reason:
                self.value["exhausted"] = reason
                self.error = RetryBudgetExhausted(reason)
                self._save()
                raise self.error

    def reserve(self, request_key: str):
        with self.lock:
            self.ensure()
            counts = self.value["requests"]
            reason = None
            if self.value["wire_attempts"] >= self.policy.max_request_attempts_per_room:
                reason = "cumulative_wire_attempt_limit"
            elif counts.get(request_key, 0) >= self.policy.max_same_request_attempts:
                reason = "same_request_attempt_limit"
            elif self.value["infra_failures"] >= self.policy.max_infra_failures_per_room:
                reason = "cumulative_infra_failure_limit"
            if reason:
                self.value["exhausted"] = reason
                self.ensure()
            self.value["wire_attempts"] += 1
            counts[request_key] = counts.get(request_key, 0) + 1
            self._save()  # Reserve durably BEFORE sending, even if later interrupted.

    def remaining_seconds(self):
        with self.lock:
            self.ensure()
            return self.policy.room_timeout_seconds - self.value["elapsed_seconds"]

    def failure(self):
        with self.lock:
            self.value["infra_failures"] += 1
            if self.value["infra_failures"] >= self.policy.max_infra_failures_per_room:
                self.value["exhausted"] = "cumulative_infra_failure_limit"
            self._save()

    def finish(self):
        with self.lock:
            self._tick()
            self.value["active_since"] = None
            self._save()

    def failed_attempt(self, signature: str):
        with self.lock:
            origins = self.value["failure_origins"]
            origins[signature] = origins.get(signature, 0) + 1
            # Wire failures were already counted at the transport boundary.
            if self.value["infra_failures"] == self.initial_failures:
                self.value["infra_failures"] += 1
            if origins[signature] >= self.policy.max_repeated_failure_attempts:
                self.value["exhausted"] = "repeated_failure_attempt_limit"
            if self.value["infra_failures"] >= self.policy.max_infra_failures_per_room:
                self.value["exhausted"] = "cumulative_infra_failure_limit"
            self._save()
            self.ensure()


class Circuit:
    """One shared channel, bounded open/half-open recovery, stale permits ignored.

Only the normal pending request can probe recovery; no extra model call. State
survives worker resume. A crashed half-open probe is treated conservatively as
failed, not as permission to reset the outage or probe budget.
"""
    def __init__(self, path, policy: Policy, *, clock=time.time):
        self.path, self.policy, self.clock = path, policy, clock
        self.condition = threading.Condition()
        if path.exists():
            self.value = read_json(path)
            if self.value.get("version") != RETRY_POLICY_VERSION:
                raise IntegrityError("channel recovery version mismatch")
            v = self.value
            if (v.get("state") not in {"closed", "open", "half_open", "halted"}
                    or any(type(v.get(k)) is not int or v[k] < 0 for k in ("epoch", "consecutive_failures", "probes"))
                    or not _finite_nonnegative(v.get("until"))
                    or (v.get("outage_since") is not None and not _finite_nonnegative(v["outage_since"]))
                    or (v["state"] != "closed" and v.get("outage_since") is None)):
                raise IntegrityError("corrupt channel recovery state")
            if self.value["state"] == "half_open":
                self.value["state"] = "open"
                self.value["epoch"] += 1
                self.value["until"] = clock() + policy.circuit_cooldown_seconds
        else:
            self.value = {"version": RETRY_POLICY_VERSION, "state": "closed", "epoch": 0,
                          "consecutive_failures": 0, "probes": 0, "outage_since": None, "until": 0}

    def _save(self):
        atomic_json(self.path, self.value)

    def _check(self):
        v = self.value
        if v["outage_since"] is not None and self.clock() - v["outage_since"] >= self.policy.circuit_outage_timeout_seconds:
            v["state"] = "halted"
        if v["state"] == "halted":
            self._save()
            raise ChannelUnavailable("channel recovery budget exhausted; requires inspection")

    def ensure(self):
        with self.condition:
            self._check()

    def remaining_seconds(self):
        with self.condition:
            self._check()
            started = self.value["outage_since"]
            return (float("inf") if started is None else
                    self.policy.circuit_outage_timeout_seconds - (self.clock() - started))

    def acquire(self, *, check=lambda: None, on_wait=lambda: None):
        with self.condition:
            while True:
                check()
                self._check()
                v = self.value
                if v["state"] == "closed":
                    return v["epoch"]
                if v["state"] == "open" and self.clock() >= v["until"]:
                    if v["probes"] >= self.policy.circuit_probe_limit:
                        v["state"] = "halted"
                        self._check()
                    v["state"] = "half_open"
                    v["probes"] += 1
                    self._save()
                    return v["epoch"]
                on_wait()
                self.condition.wait(timeout=0.25)

    def outcome(self, permit: int, *, transient: bool):
        with self.condition:
            v = self.value
            if permit != v["epoch"] or v["state"] == "halted":
                return
            if not transient:
                v.update(state="closed", consecutive_failures=0, probes=0, outage_since=None, until=0)
                v["epoch"] += 1
            else:
                v["consecutive_failures"] += 1
                if v["outage_since"] is None:
                    v["outage_since"] = self.clock()
                if v["state"] == "half_open" or v["consecutive_failures"] >= self.policy.circuit_failure_threshold:
                    delay = min(self.policy.circuit_max_cooldown_seconds,
                                self.policy.circuit_cooldown_seconds * 2 ** v["probes"])
                    v.update(state="open", until=self.clock() + delay)
                    v["epoch"] += 1
                    if v["probes"] >= self.policy.circuit_probe_limit:
                        v["state"] = "halted"
            self._save()
            self.condition.notify_all()

    def cancel(self, permit: int):
        with self.condition:
            if permit == self.value["epoch"] and self.value["state"] == "half_open":
                self.value["state"] = "open"
                self.value["probes"] -= 1
                self.value["epoch"] += 1
                self._save()
                self.condition.notify_all()


def transport_transient(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in TRANSIENT_HTTP
    if isinstance(exc, urllib.error.URLError):
        # TLS verification / malformed endpoint errors are not transient.
        import ssl
        if isinstance(exc.reason, (ssl.SSLCertVerificationError, ValueError)):
            return False
        import socket
        import errno
        if isinstance(exc.reason, socket.gaierror):
            return exc.reason.errno == socket.EAI_AGAIN
        return (isinstance(exc.reason, (TimeoutError, ConnectionError))
                or isinstance(exc.reason, OSError) and exc.reason.errno in {
                    errno.EAGAIN, errno.ETIMEDOUT, errno.ECONNRESET, errno.ECONNREFUSED,
                    errno.ECONNABORTED, errno.ENETDOWN, errno.ENETUNREACH, errno.EHOSTUNREACH})
    return isinstance(exc, (TimeoutError, ConnectionError, BrokenPipeError,
                            http.client.IncompleteRead, http.client.RemoteDisconnected))


def _finite_nonnegative(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def install_transport(module, current, circuit: Circuit):
    """Wrap the frozen wire boundary without replacing its request/retry logic."""
    original = module._urlopen_no_redirect

    def guarded(request, timeout):
        ctx = current.get()
        if ctx is None or ctx.budget is None:
            return original(request, timeout)
        budget = ctx.budget
        try:
            permit = circuit.acquire(check=budget.ensure, on_wait=lambda: ctx.heartbeat.update(
                ctx.key, stage="channel_cooldown"))
        except ChannelUnavailable as exc:
            ctx.guard_error = exc
            raise
        fingerprint = hashlib.sha256(request.get_method().encode() + b"\0" +
                                     request.full_url.encode() + b"\0" + (request.data or b"")).hexdigest()
        try:
            budget.reserve(fingerprint)
        except BaseException:
            # Release a claimed probe without issuing a request; retain outage.
            circuit.cancel(permit)
            raise
        ctx.heartbeat.update(ctx.key, stage="api_wire_attempt")
        finished = False

        def outcome(exc=None):
            nonlocal finished
            if finished:
                return
            finished = True
            transient = exc is not None and transport_transient(exc)
            if exc is None or transient or isinstance(exc, urllib.error.HTTPError):
                circuit.outcome(permit, transient=transient)
            else:
                # A local error or rejected TLS certificate does not prove
                # that the endpoint recovered; release only our probe.
                circuit.cancel(permit)
            if transient:
                budget.failure()
                try:
                    circuit.ensure()
                except ChannelUnavailable as guard:
                    ctx.guard_error = guard

        try:
            bounded_timeout = min(timeout, budget.remaining_seconds(), circuit.remaining_seconds())
        except BaseException as exc:
            circuit.cancel(permit)
            if isinstance(exc, ChannelUnavailable):
                ctx.guard_error = exc
            raise
        try:
            response = original(request, bounded_timeout)
        except BaseException as exc:
            outcome(exc)
            raise

        class Response:
            def __enter__(self):
                try:
                    self.inner = response.__enter__()
                    return self
                except BaseException as exc:
                    outcome(exc)
                    raise

            def read(self, *args, **kwargs):
                try:
                    result = self.inner.read(*args, **kwargs)
                except BaseException as exc:
                    outcome(exc)
                    raise
                outcome()
                return result

            def __exit__(self, kind, exc, tb):
                try:
                    return response.__exit__(kind, exc, tb)
                finally:
                    outcome(exc)

            def __getattr__(self, name):
                return getattr(response, name)

        return Response()

    module._urlopen_no_redirect = guarded
