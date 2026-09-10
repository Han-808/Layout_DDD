#!/usr/bin/env python3
"""Execution-only hardening for combined142; never edits the sealed release.

--check is read-only and does not stage inputs, contact APIs or run Blender.
--prepare stages a NEW output identity without evaluating. With neither flag,
run the new campaign; old results are verified/reused read-only, never rewritten.
--check-publication is a read-only original-score completeness check, not a publisher.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import asdict
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.nonrect_hardening_v1 import VERSION
from scripts.nonrect_hardening_v1 import overlay
from scripts.nonrect_hardening_v1 import results, assurance
from scripts.nonrect_hardening_v1.recovery import RETRY_POLICY_VERSION, Circuit, install_transport
from scripts.nonrect_hardening_v1.safety import (
    ExecutionGuardError, Heartbeat, IntegrityError, Policy, StorageGuardError,
    atomic_json, best_effort_record, check_storage, diagnostic_projection, digest, exception_record,
    identity, read_json, stamp, worker_outcome,
)

LEGACY_FILE = ROOT / "Support/artifacts/releases/complicated_eval_combined142_v1/run_combined.py"
OUTPUT_PARENT = ROOT / "Support/artifacts/outputs/complicated_agent_evaluation"
DEFAULT_OUTPUT = OUTPUT_PARENT / "combined_remaining29_plus20_113_execution_hardened_v4_r1"
EVALUATOR_COMMIT = "31869837105d7ef10c3b3e382cb84eeb1f1f1efc"


def load_legacy():
    spec = importlib.util.spec_from_file_location("sealed_combined142", LEGACY_FILE)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def code_files() -> list[Path]:
    return [Path(__file__).resolve(), *sorted((Path(__file__).parent / "nonrect_hardening_v1").glob("*.py")),
            ROOT / "Support/bash/local/run_complicated_combined142_hardened.sh"]


class StartupPathError(IntegrityError):
    def __init__(self, path: Path):
        self.path = path
        super().__init__("required startup path is missing or unsafe")


def protected_lock_paths(legacy) -> list[Path]:
    """Actual source campaigns, including the sealed launcher's FC0 ancestor.

    inputs.DEFAULT_OUTPUT is merely an unused historical CLI default, not a
    source directory. Do not create it or require it to exist to start this run.
    """
    roots = {legacy.DEFAULT_OUTPUT, legacy.previous.DEFAULT_OUTPUT, legacy.previous.SOURCE_OUTPUT,
             legacy.previous.SOURCE_OUTPUT.parent / "merged_success_kimi_glm_sol_fc0_parallel_shared_api2_v3_r1"}
    paths = []
    for root in sorted(roots):
        if root.is_symlink() or not root.is_dir():
            raise StartupPathError(root)
        paths.append(root / ".operator.lock")
        lane_roots = [root / "lanes", *(root / "datasets" / d / "lanes" for d in legacy.DATASETS)]
        for lane_root in lane_roots:
            for alias in legacy.previous.ALIASES:
                evaluation = lane_root / alias / "evaluation"
                if evaluation.is_symlink() or evaluation.exists() and not evaluation.is_dir():
                    raise StartupPathError(evaluation)
                if evaluation.is_dir():
                    paths.append(evaluation / ".runner.lock")
    for path in paths:
        if path.is_symlink() or path.exists() and not path.is_file():
            raise StartupPathError(path)
    return paths


def check_protected_locks(legacy) -> None:
    """Read-only readiness probe. Launch still takes and holds every lock."""
    for path in protected_lock_paths(legacy):
        if not path.exists():
            continue  # The launch may create this lock; the check never does.
        with path.open("r") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def refusal_context(exc: Exception) -> dict:
    """Useful local startup context without dumping exception messages/API data."""
    context = {}
    if isinstance(exc, StartupPathError):
        context["local_path"] = str(exc.path)
    elif isinstance(exc, OSError) and isinstance(exc.filename, str):
        path = Path(exc.filename)
        if path.is_absolute() and path.is_relative_to(ROOT):
            context["local_path"] = str(path)
    tb = exc.__traceback__
    while tb is not None:
        filename = tb.tb_frame.f_code.co_filename
        if Path(filename).is_absolute() and Path(filename).is_relative_to(ROOT):
            context["location"] = {"file": filename, "line": tb.tb_lineno,
                                   "function": tb.tb_frame.f_code.co_name}
        tb = tb.tb_next
    return diagnostic_projection(context)


def verify_execution_plan(output: Path, legacy):
    value = read_json(output / "execution_plan.json")
    if value.get("version") != VERSION or value.get("output_root") != str(output.resolve()):
        raise IntegrityError("execution plan version/output identity mismatch")
    if value.get("evaluator_commit") != EVALUATOR_COMMIT:
        raise IntegrityError("evaluator commit mismatch")
    if value.get("retry_policy_version") != RETRY_POLICY_VERSION:
        raise IntegrityError("retry policy changed; use a new execution output")
    if value.get("result_schema_version") != results.RESULT_VERSION:
        raise IntegrityError("execution result contract changed; use a new output")
    for filename, sha in value["runner_sha256"].items():
        if digest(Path(filename)) != sha:
            raise IntegrityError("execution runner changed; use a new release/output")
    if digest(output / "combined_plan.json") != value["combined_plan_sha256"]:
        raise IntegrityError("sealed input plan changed")
    code, _ = legacy.verify_plan(output)
    # Reuse only the specifically bound old campaign, still verified by its
    # original release; never accept arbitrary user-supplied old report roots.
    if value.get("reuse_root") != str(legacy.DEFAULT_OUTPUT):
        raise IntegrityError("unapproved successful-report reuse root")
    legacy.verify_plan(legacy.DEFAULT_OUTPUT)
    if digest(legacy.DEFAULT_OUTPUT / "combined_plan.json") != value["reuse_plan_sha256"]:
        raise IntegrityError("old campaign plan drift")
    return code, value, Policy(**value["policy"])


def prepare(output: Path, legacy, policy: Policy):
    if output.parent != OUTPUT_PARENT or not output.name.startswith("combined_remaining29_plus20_113_execution_hardened_"):
        raise IntegrityError("use a separate named hardened output under the evaluation directory")
    if output == legacy.DEFAULT_OUTPUT:
        raise IntegrityError("cannot modify the sealed campaign")
    if (output / "execution_plan.json").exists():
        _, plan, existing = verify_execution_plan(output, legacy)
        if existing != policy:
            raise IntegrityError("execution policy drift; use a new output identity")
        return plan
    if output.exists() and any(p.name != ".operator.lock" for p in output.iterdir()):
        raise IntegrityError("fresh hardened output must be empty; partial prepare requires inspection")
    check_storage(output, policy.minimum_free_gib)
    legacy.verify_plan(legacy.DEFAULT_OUTPUT)
    output.mkdir(parents=True, exist_ok=True)
    legacy.prepare(output, argparse.Namespace(max_workers=4, blender_slots=2))
    # The inherited plan describes input/metric identity. This separate overlay
    # plan is authoritative for execution safeguards (legacy says no disk guard).
    plan = {"version": VERSION, "created_at": stamp(), "output_root": str(output),
            "retry_policy_version": RETRY_POLICY_VERSION,
            "result_schema_version": results.RESULT_VERSION,
            "evaluator_commit": EVALUATOR_COMMIT, "policy": asdict(policy),
            "storage_guard_enabled": True, "core_eval_unchanged": True,
            "runner_sha256": {str(p): digest(p) for p in code_files()},
            "combined_plan_sha256": digest(output / "combined_plan.json"),
            "reuse_root": str(legacy.DEFAULT_OUTPUT),
            "reuse_plan_sha256": digest(legacy.DEFAULT_OUTPUT / "combined_plan.json")}
    atomic_json(output / "execution_plan.json", plan)
    return plan


def expected_rooms(legacy, dataset: str, model: str, output: Path) -> int:
    if dataset == "remaining29":
        return sum(target.startswith(model + "/") for target in legacy.selection.read(
            legacy.previous.SELECTION)["targets"])
    return int(read_json(output / "datasets" / dataset / "input_verification.json")["model_totals"][model]["rooms"])


def worker(args, legacy) -> int:
    output = args.output_root.resolve()
    lane = output / "datasets" / args.worker.split(":")[0] / "lanes" / args.worker.split(":")[1]
    result_path = lane / "execution_worker_result.json"
    try:
        code, plan, policy = verify_execution_plan(output, legacy)
        check_storage(output, policy.minimum_free_gib)
        with Heartbeat(lane / "execution_heartbeat.json", args.run_id) as heartbeat:
            circuit = Circuit(lane / "execution_circuit.json", policy)
            circuit.ensure()
            def install(resilient, unused_output, unused_plan, dataset):
                import benchmark.non_rectangular.runtime as runtime
                import benchmark.non_rectangular.evaluator as evaluator_module
                import benchmark.models.openai_compatible_model as transport
                install_transport(transport, overlay.CURRENT, circuit)
                overlay.install(resilient, runtime, evaluator_module, output=output,
                                dataset=dataset, policy=policy, heartbeat=heartbeat,
                                code_identity=identity({"core": EVALUATOR_COMMIT,
                                                        "runner": plan["runner_sha256"]}),
                                reuse_root=Path(plan["reuse_root"]), circuit=circuit)
            # The original CLI/metric construction stays intact. This is its
            # existing scheduling-overlay hook, scoped to this child process.
            legacy.install_room_limit = install
            code = legacy.worker(args)
            circuit.ensure()  # A swallowed channel guard cannot authorize the next dataset.
        run = read_json(lane / "evaluation/run_manifest.json")
        dataset, alias = args.worker.split(":")
        results.lane_result(lane / "evaluation", run_id=args.run_id,
                            expected_rooms=expected_rooms(legacy, dataset, legacy.previous.ALIASES[alias], output))
        atomic_json(result_path, {"version": VERSION, "run_id": args.run_id,
                                  "worker": args.worker, "cli_completed": True,
                                  "campaign_identity_sha256": run["identity_sha256"],
                                  "execution_plan_sha256": digest(output / "execution_plan.json"),
                                  "execution_result_sha256": digest(lane / "evaluation/execution_result.json"),
                                  "returncode": code, "finished_at": stamp()})
        return code
    except Exception as exc:
        # Best effort accounting still returns already acquired results after
        # infrastructure failure; it NEVER creates a successful CLI receipt.
        if (lane / "evaluation/run_manifest.json").is_file():
            try:
                dataset, alias = args.worker.split(":")
                results.lane_result(lane / "evaluation", run_id=args.run_id,
                    expected_rooms=expected_rooms(legacy, dataset, legacy.previous.ALIASES[alias], output),
                    execution_failure=exception_record(exc))
            except Exception:
                pass
        best_effort_record(result_path, {"version": VERSION, "run_id": args.run_id,
                                        "worker": args.worker, "cli_completed": False,
                                        "finished_at": stamp(), **exception_record(exc)})
        return 70


def stop_process(process, grace: float) -> None:
    """Only our newly created process group. All waits are bounded."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    # The leader may have exited while a descendant is still alive.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=max(1.0, grace))


def watchdog_reason(heartbeat: dict, *, now: float, started: float,
                    policy: Policy, run_id: str, pid: int) -> str | None:
    if now - started > policy.worker_timeout_seconds:
        return "worker_deadline_exceeded"
    if heartbeat and (heartbeat.get("run_id") != run_id or heartbeat.get("pid") != pid):
        # A receipt from a previous worker must not supply either its timestamp
        # OR its old room deadlines to the newly started worker.
        heartbeat = {}
    observed = heartbeat.get("time", started)
    if (isinstance(observed, bool) or not isinstance(observed, (int, float))
            or not math.isfinite(observed) or observed > now + 5):
        return "invalid_worker_heartbeat"
    if now - observed > policy.heartbeat_timeout_seconds:
        return "worker_heartbeat_stale"
    if not isinstance(heartbeat.get("rooms", {}), dict):
        return "invalid_worker_heartbeat"
    for room in heartbeat.get("rooms", {}).values():
        if not isinstance(room, dict):
            return "invalid_worker_heartbeat"
        active = room.get("started_at")
        if active is not None and (isinstance(active, bool) or not isinstance(active, (float, int))
                                   or not math.isfinite(active) or active > now + 5):
            return "invalid_worker_heartbeat"
        if isinstance(active, (float, int)) and now - active > policy.room_timeout_seconds:
            return "room_deadline_exceeded"
    return None


def supervise_worker(process, lane: Path, run_id: str, policy: Policy,
                     stopping: threading.Event, storage_root: Path) -> tuple[int, str | None]:
    started, reason = time.time(), None
    while process.poll() is None:
        if stopping.is_set():
            reason = "operator_or_supervisor_stop"
        else:
            try:
                check_storage(storage_root, policy.minimum_free_gib)
                heartbeat_path = lane / "execution_heartbeat.json"
                value = read_json(heartbeat_path) if heartbeat_path.is_file() else {}
                reason = watchdog_reason(value, now=time.time(), started=started,
                                         policy=policy, run_id=run_id, pid=process.pid)
            except StorageGuardError:
                reason = "storage_guard_stop"
                stopping.set()
            except (OSError, ValueError, IntegrityError):
                reason = "unreadable_worker_heartbeat"
        if reason:
            stop_process(process, policy.terminate_grace_seconds)
            break
        stopping.wait(1.0)
    return process.wait(timeout=policy.terminate_grace_seconds), reason


def adjudicate_worker(lane: Path, run_id: str, code: int, reason: str | None,
                      *, plan_sha: str, model: str, rooms: int) -> dict:
    if reason is not None:
        outcome = {"status": "interrupted", "returncode": code, "reason": reason, "continue_lane": False}
        # Called only after the supervisor has terminated this process group.
        # Recover persisted observations, never infer a successful CLI exit.
        try:
            run = read_json(lane / "evaluation/run_manifest.json")
            if run["identity"].get("model_order") != [model]:
                raise IntegrityError("interrupted result belongs to a different model")
            value = results.lane_result(lane / "evaluation", run_id=run_id, expected_rooms=rooms,
                execution_failure={"stage": "supervisor", "category": "execution_interrupted",
                                   "error_type": "WorkerInterrupted", "reason": reason, "returncode": code})
            outcome.update(execution_result_path=str(lane / "evaluation/execution_result.json"),
                           score_coverage=value["score_coverage"], room_outcomes=value["room_counts"],
                           workflow_alerts=value["workflow_alerts"],
                           official_score_publication=value["official_score_publication"])
        except Exception as exc:
            outcome["result_recovery_diagnostic"] = exception_record(exc)
        return outcome
    try:
        receipt = read_json(lane / "execution_worker_result.json")
        if (receipt.get("run_id") != run_id or receipt.get("execution_plan_sha256") != plan_sha
                or receipt.get("cli_completed") is not True or receipt.get("returncode") != code):
            raise IntegrityError("no fresh verified CLI completion receipt")
        outcome = worker_outcome(lane / "evaluation", code, expected_model=model,
                                 expected_rooms=rooms,
                                 expected_run_identity=receipt["campaign_identity_sha256"])
        if not outcome.get("continue_lane"):
            return outcome
        path = lane / "evaluation/execution_result.json"
        if digest(path) != receipt.get("execution_result_sha256"):
            raise IntegrityError("accounted execution result missing or changed")
        value = read_json(path)
        assurance.validate(value, results.METRICS)
        actual_rooms = {str(p.parent.relative_to(lane / "evaluation/models"))
                        for p in (lane / "evaluation/models").glob("*/scenes/*/rooms/*/summary.json")}
        observed_rooms = set()
        for scene in value["scenes"]:
            for room in scene["rooms"]:
                relative = Path(scene["model"]) / "scenes" / scene["scene_id"] / "rooms" / room["room_id"]
                observed_rooms.add(str(relative))
                if results.room_result(lane / "evaluation/models" / relative, persist=False) != room:
                    raise IntegrityError("accounted room differs from its persisted source")
            if scene["official_aggregation_status"] == "complete":
                pointer = scene.get("official_scene_report") or {}
                original = lane / "evaluation/models" / scene["model"] / "scenes" / scene["scene_id"] / "evaluation_report.json"
                if pointer.get("path") != str(original) or original.is_symlink() or digest(original) != pointer.get("sha256"):
                    raise IntegrityError("original scene aggregate changed after accounting")
        if actual_rooms != observed_rooms:
            raise IntegrityError("accounted room inventory differs from persisted summaries")
        if (value.get("schema_version") != results.RESULT_VERSION or value.get("run_id") != run_id
                or value.get("campaign_identity_sha256") != receipt["campaign_identity_sha256"]
                or value.get("planned_room_count") != rooms
                or value.get("room_counts", {}).get("complete") != outcome["complete"]):
            raise IntegrityError("accounted result does not match verified CLI outcome")
        counts = value["room_counts"]
        if (set(counts) != {"complete", "skipped_hard_failure", "infra_failure", "unresolved"}
                or any(type(n) is not int or n < 0 for n in counts.values())
                or sum(counts.values()) != rooms or value.get("not_started_room_count") != 0):
            raise IntegrityError("accounted room counts do not reconcile")
        if value.get("all_work_accounted") is True:
            if (value.get("status") not in {"complete", "completed_with_skips"}
                    or counts["infra_failure"] or counts["unresolved"]
                    or counts["skipped_hard_failure"] and value["status"] != "completed_with_skips"):
                raise IntegrityError("inconsistent accounted execution status")
            outcome["status"] = value["status"]
        elif outcome["status"] == "complete":
            outcome["status"] = "partial_failure"
        return {**outcome, "execution_result_path": str(path), "score_coverage": value["score_coverage"],
                "room_outcomes": value["room_counts"], "workflow_alerts": value["workflow_alerts"],
                "official_score_publication": value["official_score_publication"]}
    except Exception as exc:
        return {"status": "fatal", "returncode": code, "continue_lane": False,
                "diagnostic": exception_record(exc)}


def launch(args, legacy, policy: Policy) -> int:
    output = args.output_root.resolve()
    # Never stage or mutate an arbitrary/old output before checking its identity.
    if output.parent != OUTPUT_PARENT or not output.name.startswith("combined_remaining29_plus20_113_execution_hardened_"):
        raise IntegrityError("invalid hardened output root")
    protected = protected_lock_paths(legacy)
    output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as locks:
        locks.enter_context(legacy.previous.operator_lock(output / ".operator.lock"))
        for path in protected:
            locks.enter_context(legacy.previous.operator_lock(path))
        plan = prepare(output, legacy, policy)
        if args.prepare:
            print("PREPARED: new execution identity only; no evaluation/API/Blender launched")
            return 0
        if not args.resume and any((output / "datasets" / d / "lanes").exists() for d in legacy.DATASETS):
            raise IntegrityError("existing hardened campaign requires exact --resume")
        # The protected operator locks above already reject any active legacy
        # owner. Re-probing a lock we now own would report ourselves as active.
        check_storage(output, policy.minimum_free_gib)
        legacy.previous.check_ports(plan_routes := read_json(output / "combined_plan.json")["routes"])
        credentials = legacy.previous.collect_credentials(plan_routes)
        outcomes = {f"{d}:{a}": {"status": "not_started"}
                    for d in legacy.DATASETS for a in legacy.previous.ALIASES}
        stopping, guard = threading.Event(), threading.Lock()
        plan_sha = digest(output / "execution_plan.json")
        interrupted = False

        def state(status="running"):
            ready = status == "complete" and all(
                item.get("official_score_publication", {}).get("eligible") is True for item in outcomes.values())
            atomic_json(output / "execution_state.json", {
                "version": VERSION, "status": status, "updated_at": stamp(),
                "execution_plan_sha256": plan_sha, "jobs": deepcopy_json(outcomes),
                "official_score_publication": {"policy": assurance.POLICY, "eligible": ready,
                    "scope": "execution_completeness_only_not_quality_or_comparability",
                    "blockers": [] if ready else ["one_or_more_jobs_not_verified_score_complete"]},
            })

        def run_lane(alias):
            for dataset in legacy.DATASETS:
                if stopping.is_set():
                    return
                label, run_id = f"{dataset}:{alias}", str(uuid.uuid4())
                lane = output / "datasets" / dataset / "lanes" / alias
                process = None
                try:
                    check_storage(output, policy.minimum_free_gib)
                    lane.mkdir(parents=True, exist_ok=True)
                    env = {k: v for k, v in os.environ.items()
                           if k not in {"STANDARD_API_CREDENTIAL", "LITELLM_MASTER_KEY"}}
                    env.update(STANDARD_API_CREDENTIAL=credentials[alias], PYTHONDONTWRITEBYTECODE="1")
                    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", label,
                               "--run-id", run_id, "--output-root", str(output)]
                    if args.resume:
                        command += ["--resume"]
                    with (lane / "execution_worker.log").open("ab") as log:
                        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                                   stderr=subprocess.STDOUT, start_new_session=True)
                    with guard:
                        outcomes[label] = {"status": "running", "pid": process.pid, "run_id": run_id}
                        state()
                    code, reason = supervise_worker(process, lane, run_id, policy, stopping, output)
                    result = adjudicate_worker(lane, run_id, code, reason, plan_sha=plan_sha,
                                               model=legacy.previous.ALIASES[alias],
                                               rooms=expected_rooms(legacy, dataset, legacy.previous.ALIASES[alias], output))
                except Exception as exc:
                    result = {"status": "fatal", "continue_lane": False, "diagnostic": exception_record(exc)}
                    if isinstance(exc, StorageGuardError):
                        stopping.set()
                finally:
                    if process is not None:
                        try:
                            # Clean up our group even if only its leader exited.
                            stop_process(process, policy.terminate_grace_seconds)
                        except Exception as exc:
                            result = {"status": "cleanup_failed", "continue_lane": False,
                                      "diagnostic": exception_record(exc)}
                with guard:
                    outcomes[label] = {**result, "run_id": run_id, "finished_at": stamp()}
                    state()
                if not result.get("continue_lane"):
                    with guard:
                        for later in legacy.DATASETS[legacy.DATASETS.index(dataset) + 1:]:
                            outcomes[f"{later}:{alias}"] = {"status": "not_started", "reason": f"blocked_by:{label}"}
                        state()
                    return

        prior = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        def stop(*_):
            nonlocal interrupted
            interrupted = True
            stopping.set()
        for sig in prior:
            signal.signal(sig, stop)
        try:
            state()
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = [pool.submit(run_lane, alias) for alias in legacy.previous.ALIASES]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception:
                        stopping.set()
        finally:
            credentials.clear()
            for sig, handler in prior.items():
                signal.signal(sig, handler)
            complete = all(item["status"] in {"complete", "completed_with_skips"} for item in outcomes.values())
            has_skips = any(item["status"] == "completed_with_skips" for item in outcomes.values())
            state("interrupted" if interrupted else
                  "completed_with_skips" if complete and has_skips else "complete" if complete else "failed")
        return 130 if interrupted else 0 if complete else 1


def deepcopy_json(value):
    return json.loads(json.dumps(value))


def check_publication(output: Path, legacy) -> int:
    """Read-only gate for complete original scores; never publish or repair."""
    verify_execution_plan(output, legacy)
    state = read_json(output / "execution_state.json")
    plan_sha = digest(output / "execution_plan.json")
    expected = {f"{d}:{a}" for d in legacy.DATASETS for a in legacy.previous.ALIASES}
    if state.get("execution_plan_sha256") != plan_sha or set(state.get("jobs", {})) != expected:
        raise IntegrityError("publication check requires the exact planned job inventory")
    blockers = []
    if state.get("status") != "complete":
        blockers.append({"scope": "campaign", "reason": "campaign_not_score_complete"})
    for label in sorted(expected):
        job = state["jobs"][label]
        if job.get("status") != "complete":
            blockers.append({"scope": label, "reason": "job_not_score_complete"})
            continue
        dataset, alias = label.split(":")
        outcome = adjudicate_worker(output / "datasets" / dataset / "lanes" / alias,
            job["run_id"], job["returncode"], None, plan_sha=plan_sha,
            model=legacy.previous.ALIASES[alias],
            rooms=expected_rooms(legacy, dataset, legacy.previous.ALIASES[alias], output))
        gate = outcome.get("official_score_publication", {})
        if outcome["status"] != "complete" or gate.get("eligible") is not True:
            blockers.append({"scope": label, "reason": "unverified_or_incomplete_original_scores",
                             "details": gate.get("blockers", []), "diagnostic": outcome.get("diagnostic")})
    print(json.dumps({"policy": assurance.POLICY, "eligible": not blockers, "blockers": blockers,
                      "read_only": True, "published": False,
                      "scope": "execution_completeness_only_not_quality_or_comparability"}, indent=2))
    return 1 if blockers else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--check-publication", action="store_true",
                       help="Read-only original-score completeness gate; does not publish")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", choices=[f"{d}:{a}" for d in ("remaining29", "plus20") for a in ("kimi", "glm", "sol")])
    parser.add_argument("--run-id")
    for name, default in asdict(Policy()).items():
        if name != "checkpoint_enabled":
            parser.add_argument("--" + name.replace("_", "-"), type=type(default), default=default)
    args = parser.parse_args(argv)
    if args.worker and (not args.run_id or args.check or args.prepare or args.check_publication):
        parser.error("worker requires run-id and cannot use check/prepare")
    policy = Policy(**{k: getattr(args, k) for k in asdict(Policy()) if k != "checkpoint_enabled"})
    legacy = load_legacy()
    if args.worker:
        return worker(args, legacy)
    if args.check_publication:
        return check_publication(args.output_root.resolve(), legacy)
    if args.check:
        check_protected_locks(legacy)
        if (args.output_root / "execution_plan.json").is_file():
            verify_execution_plan(args.output_root.resolve(), legacy)
        else:
            legacy.verify_plan(legacy.DEFAULT_OUTPUT)
            for path in code_files():
                if not path.is_file():
                    raise IntegrityError("hardening source missing")
        print("CHECK_OK: sealed core and protected startup paths/locks verified; no output preparation, API or Blender")
        return 0
    return launch(args, legacy, policy)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Hardened runner refused: {type(exc).__name__}; "
              f"context={json.dumps(refusal_context(exc))}; no raw exception dumped", file=sys.stderr)
        raise SystemExit(70)
