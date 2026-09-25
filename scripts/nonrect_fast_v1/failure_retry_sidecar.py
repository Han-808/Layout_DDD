"""Re-evaluate the three infrastructure failures without touching the running batches.

The running handoff already retired these rooms and will not claim them again.
This process republishes their ready markers, then evaluates only those rooms.
Queue status is written to its own directory. Blender uses the same three slots.
"""
from __future__ import annotations
import argparse
import json
import os
import signal
import sys
import threading
from pathlib import Path

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nonrect_fast_v1.handoff_execution_high36 as high36
from nonrect_fast_v1 import content_policy, derive, run
import nonrect_fast_v1.handoff_execution as handoff
from nonrect_fast_v1.pipeline import HostProbe, Pipeline, atomic_write, canonical_sha, inventory_files, read, sha
from nonrect_fast_v1.resume_capacity import DesktopHostProbe

OWN = high36.handoff.BASE / 'failure_retry_sidecar_20260926'
CASES = (
    'nr.fastv1.hy4.scene_011687.room_003',
    'nr.fastv1.hy4.scene_011838.room_001',
    'nr.fastv1.kimi.scene_011838.room_003',
)
WORKERS = 3


def _save_to_own(self, results, waiting, ready, active, peak_prepare, peak_eval, peak_ready, final=False):
    atomic_write(OWN / 'queue_summary.json', {
        'identity': self.identity, 'limits': high36.asdict(self.limits), 'results': results,
        'waiting': [task['case_id'] for task in waiting],
        'ready': [task['case_id'] for task, _, _ in ready],
        'active': [{'stage': stage, 'case_id': task['case_id']} for stage, task, _, _ in active.values()],
        'peaks': {'prepare': peak_prepare, 'evaluate': peak_eval, 'ready': peak_ready},
        'host': self.governor.last, 'backpressure_reason': self.governor.reason,
        'status': ('interrupted' if self.stop.is_set()
                   else 'paused_with_pending' if final and (waiting or ready or active)
                   else 'finished_with_failures' if final and any(r['status'] not in handoff.TERMINAL for r in results)
                   else 'finished' if final else 'running'),
        'events': self.events})


def republish(case):
    import fcntl
    identity = read(case / 'identity.json')
    digest = canonical_sha(identity)
    state_path = case / 'state.json'
    state = read(state_path)
    if state.get('identity_sha256') != digest:
        raise ValueError('State identity mismatch: ' + case.name)
    if state.get('status') not in {'infrastructure_failure', 'ready'}:
        raise ValueError('Case is not a retired failure: ' + case.name + ' ' + str(state.get('status')))
    if (case / 'consuming.json').exists():
        raise ValueError('Case is still consuming: ' + case.name)
    lock_path = case / 'case.lock'
    descriptor = os.open(lock_path, os.O_RDWR | os.O_NOFOLLOW)
    stream = os.fdopen(descriptor, 'r+')
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        files = inventory_files(case)
        marker = {'schema_version': 'nonrect_atomic_ready_v1', 'identity': identity,
                  'input_receipt_sha256': sha(case / 'input_receipt.json'), 'files': files,
                  'prepared_bytes': sum(value[0] for value in files.values())}
        atomic_write(case / 'ready.json', marker)
        atomic_write(state_path, {'case_id': case.name, 'identity_sha256': digest,
                                  'prepared_bytes': marker['prepared_bytes'], 'status': 'ready'})
    finally:
        stream.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--workers', type=int, default=WORKERS)
    parser.add_argument('--worker-memory-gib', type=float, default=20)
    parser.add_argument('--desktop-memory-reserve-gib', type=float, default=6)
    parser.add_argument('--evaluation-growth-gib', type=float, default=6)
    parser.add_argument('--no-resource-gates', action='store_true')
    args = parser.parse_args(argv)
    if not args.run:
        parser.error('--run is required')
    if args.workers != WORKERS:
        raise ValueError('This sidecar admits exactly three failure rooms')
    os.environ['NONRECT_BLENDER_SLOTS'] = str(high36.BLENDER_SLOTS)
    os.environ['NONRECT_BLENDER_SLOT_DIR'] = str(high36.handoff.CONTROL / 'blender_slots')
    Pipeline.save = _save_to_own
    root = handoff.EXISTING
    manifest = derive.verify(run.DEFAULT_RUNTIME)
    base = run.load('nonrect_handoff_inventory', run.DEFAULT_RUNTIME / 'scripts/nonrect_merged30_v8_api2_sol_catalog/run.py')
    _inventory, pins = base.inventory()
    plan = handoff.verified_plan(root, manifest, pins)
    tasks = [task for task in plan['tasks'] if task['case_id'] in CASES]
    if sorted(task['case_id'] for task in tasks) != sorted(CASES):
        raise ValueError('Failure cases are not all in the saved plan')
    plan = dict(plan)
    plan['tasks'] = tasks
    for case_id in CASES:
        republish(root / 'rooms' / case_id)
    preflight = high36._ready_preflight(root, plan, allow_terminal=True)
    if preflight != {'ready': 3, 'terminal': 0, 'held': 0}:
        raise ValueError('Failure rooms are not exactly three ready cases: %s' % preflight)
    migration = read(handoff.CONTROL / 'migration.json')
    OWN.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    previous = {sig: signal.signal(sig, lambda *unused: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    stream = (OWN / 'runner.lock').open('a+')
    try:
        import fcntl
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        settings = handoff.effective_limits(plan, args)
        atomic_write(OWN / 'status.json', {'status': 'starting', 'pid': os.getpid(), 'rooms': list(CASES)})
        with run.live_environment(run.DEFAULT_RUNTIME, base, OWN, stop) as env:
            env[content_policy.ENV] = plan['identity']['content_fingerprint_validation']
            stages = handoff.MigratedStages(
                run.DEFAULT_RUNTIME, env, settings, migration=migration,
                reserve_gib=args.desktop_memory_reserve_gib, rss_gib=args.worker_memory_gib,
                memory_gates=False, disk_guard=not args.no_resource_gates)
            def no_preparation(*unused):
                raise ValueError('Evaluation cannot regenerate missing inputs')
            pipeline = handoff.SuccessCleanupPipeline(
                root, plan['identity'], settings, no_preparation, stages.evaluate,
                probe=DesktopHostProbe(root, args.desktop_memory_reserve_gib, args.worker_memory_gib),
                stop=stop, prepare_only=False, cleanup=False, retry_failed=False)
            pipeline.governor = handoff.ResourceUngatedGovernor(settings, HostProbe(root))
            atomic_write(OWN / 'status.json', {
                'status': 'evaluating', 'pid': os.getpid(), 'rooms': list(CASES),
                'blender_slots': high36.BLENDER_SLOTS,
                'blender_slot_dir': os.environ['NONRECT_BLENDER_SLOT_DIR']})
            pipeline.execute(plan['tasks'])
        summary = read(OWN / 'queue_summary.json')
        status = {'status': summary['status'], 'pid': os.getpid(), 'results': len(summary['results'])}
        atomic_write(OWN / 'status.json', status)
        print(json.dumps(status), flush=True)
        return 0 if summary['status'] == 'finished' else 2
    finally:
        stream.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())
