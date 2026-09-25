"""Evaluate the four ready Sol rooms without touching the running handoff.

This process does not take the running runner locks and does not write the
running control directory. It only leases the four Sol room directories.
Blender admission uses the same three slot files as the running handoff.
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
from nonrect_fast_v1.pipeline import HostProbe, atomic_write, read
from nonrect_fast_v1.resume_capacity import DesktopHostProbe

OWN = high36.handoff.BASE / 'sol_batch_sidecar_20260926'
WORKERS = 4


def sol_plan(plan):
    tasks = [task for task in plan['tasks'] if high36._keep_task(task)]
    if len(tasks) != 4:
        raise ValueError('Expected exactly four Sol rerun tasks, found %d' % len(tasks))
    scoped = dict(plan)
    scoped['tasks'] = tasks
    return scoped


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
        raise ValueError('This sidecar admits exactly four Sol rooms')
    os.environ['NONRECT_BLENDER_SLOTS'] = str(high36.BLENDER_SLOTS)
    os.environ['NONRECT_BLENDER_SLOT_DIR'] = str(
        high36.handoff.CONTROL / 'blender_slots')
    root = handoff.ADDITIONAL
    manifest = derive.verify(run.DEFAULT_RUNTIME)
    base = run.load('nonrect_handoff_inventory', run.DEFAULT_RUNTIME / 'scripts/nonrect_merged30_v8_api2_sol_catalog/run.py')
    inventory, pins = base.inventory()
    del inventory
    plan = sol_plan(handoff.verified_plan(root, manifest, pins))
    preflight = high36._ready_preflight(root, plan, allow_terminal=True)
    if preflight != {'ready': 4, 'terminal': 0, 'held': 0}:
        raise ValueError('Sol rooms are not exactly four ready cases: %s' % preflight)
    migration = read(handoff.CONTROL / 'migration.json')
    OWN.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    previous = {sig: signal.signal(sig, lambda *unused: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    lock_path = OWN / 'runner.lock'
    stream = lock_path.open('a+')
    try:
        import fcntl
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        settings = handoff.effective_limits(plan, args)
        atomic_write(OWN / 'status.json', {'status': 'starting', 'pid': os.getpid(), 'rooms': [t['case_id'] for t in plan['tasks']]})
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
                'status': 'evaluating', 'pid': os.getpid(),
                'rooms': [t['case_id'] for t in plan['tasks']],
                'blender_slots': high36.BLENDER_SLOTS,
                'blender_slot_dir': os.environ['NONRECT_BLENDER_SLOT_DIR']})
            result = pipeline.execute(plan['tasks'])
        status = {'status': result['status'], 'pid': os.getpid(),
                  'results': len(result['results']), 'ready': len(result['ready']),
                  'waiting': len(result['waiting'])}
        atomic_write(OWN / 'status.json', status)
        print(json.dumps(status), flush=True)
        return 0 if result['status'] == 'finished' else 2
    finally:
        stream.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())
