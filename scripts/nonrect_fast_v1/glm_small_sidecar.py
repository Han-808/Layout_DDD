"""Prepare and evaluate the 14 GLM small rooms in a new directory.

Does not take the running handoff locks and does not write their rooms or
status. Blender, including preparation renders, uses the existing 3 slots.
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import signal
import sys
import threading
import time
from pathlib import Path

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nonrect_fast_v1.handoff_execution_high36 as high36
from nonrect_fast_v1 import content_policy, derive, run
import nonrect_fast_v1.handoff_execution as handoff
from nonrect_fast_v1.pipeline import GIB, HostProbe, atomic_write, read
from nonrect_fast_v1.resume_capacity import DesktopHostProbe

OWN = high36.handoff.BASE / 'glm_small_sidecar_20260926'
GLM_ROOT = Path('/Users/han_mohan/Desktop/Layout_DDD/Support/artifacts/outputs/non_rectangular_generation/glm_api2_resilient_v1/glm')
SCENES = ('scene_011888', 'scene_011687', 'scene_011634', 'scene_011838', 'scene_011760')
VIEW = OWN / 'generation_view'
WORKERS = 14
THRESH = 50


def small_rooms():
    found = []
    for scene in SCENES:
        data = json.loads((GLM_ROOT / scene / 'generated_scene.json').read_text())
        rooms = data['rooms']
        items = rooms.items() if isinstance(rooms, dict) else enumerate(rooms)
        for rid, room in items:
            if not isinstance(room, dict):
                continue
            name = str(rid if isinstance(rid, str) and str(rid).startswith('room_') else room.get('room_id') or room.get('id') or rid)
            objs = room.get('objects')
            count = len(objs) if isinstance(objs, (list, dict)) else None
            if count is None or count >= THRESH:
                continue
            found.append({'scene': scene, 'room': name, 'objects': count})
    if len(found) != 14:
        raise ValueError('Expected 14 GLM small rooms, found %d' % len(found))
    return found


def link_view(rooms):
    for row in rooms:
        scene = GLM_ROOT / row['scene']
        dest = VIEW / 'glm' / row['scene']
        (dest / 'stage_a').mkdir(parents=True, exist_ok=True)
        (dest / 'retrieval').mkdir(parents=True, exist_ok=True)
        mapping = {
            'room_layout.json': scene / 'room_layout.json',
            'room_program.json': scene / 'room_program.json',
            'generated_scene.json': scene / 'generated_scene.json',
            'compiled_architecture.json': scene / 'compiled_architecture.json',
            'stage_a/object_plan.json': scene / 'object_plan.json',
            'retrieval/asset_selection.json': scene / 'asset_selection.json',
        }
        for rel, source in mapping.items():
            if not source.is_file():
                raise ValueError('Missing GLM artifact: ' + str(source))
            target = dest / rel
            if target.is_symlink() or target.exists():
                continue
            target.symlink_to(source)


class GlmStages(run.Stages):
    def prepare(self, task, work, stop):
        started = time.monotonic()
        self.conflict_check(work)
        if (work / 'ready.json').exists() or (work / 'consuming.json').exists():
            raise ValueError('Preparation cannot replace queued/in-use artifacts')
        for name in ('dataset', 'initial_render', '.materialized.building'):
            path = work / name
            if path.is_symlink() or any(p.is_symlink() for p in path.rglob('*') if p.exists()):
                raise ValueError('Cannot recover symlink artifacts')
            if path.exists():
                shutil.rmtree(path)
        blender = str(high36.BLENDER_WRAPPER)
        if (work / 'materialized').exists():
            self.cmd([sys.executable, '-B', '-I', self.catalog / 'verify_materialized.py',
                      '--materialized', work / 'materialized', '--generation-root', VIEW,
                      '--model', task['model'], '--scene', task['scene'], '--room', task['room']],
                     work / 'materialized_recheck.log', stop, work)
        else:
            self.cmd([sys.executable, '-B', '-I', self.catalog / 'materialize_room.py',
                      '--generation-root', VIEW, '--model', task['model'],
                      '--scene', task['scene'], '--room', task['room'], '--dest', work / 'materialized',
                      '--blender-bin', blender, '--timeout-seconds', '1800'],
                     work / 'materialize.log', stop, work)
        materialized_seconds = time.monotonic() - started
        self.cmd([sys.executable, '-B', '-I', self.catalog / 'build_case.py',
                  '--room-dir', work / 'materialized', '--dataset-root', work / 'dataset',
                  '--case-id', task['case_id'], '--dataset-id', derive.VERSION,
                  '--render-dir', work / 'initial_render', '--blender-bin', blender,
                  '--evidence-worker', self.catalog / 'nonrect_evidence_worker.py'],
                 work / 'build_case.log', stop, work)
        self.cmd(self.uniform_args(work), work / 'input_check.json', stop, work)
        if run.disk_bytes(work) > self.limits.preparation_growth_gib * GIB:
            raise OSError('Prepared case exceeds its reserved size; no ready marker published')
        checked = read(work / 'input_check.json')
        if len(checked['cases']) != 1 or checked['cases'][0]['case_id'] != task['case_id']:
            raise ValueError('Prepared input differs from task')
        atomic_write(work / 'input_receipt.json', checked)
        atomic_write(work / 'preparation_timing.json', {
            'materialize_seconds': materialized_seconds,
            'total_prepare_seconds': time.monotonic() - started, 'real_blender': True})
        self.conflict_check(work)


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
        raise ValueError('This sidecar admits the 14 GLM small rooms')
    os.environ['NONRECT_BLENDER_SLOTS'] = str(high36.BLENDER_SLOTS)
    os.environ['NONRECT_BLENDER_SLOT_DIR'] = str(high36.handoff.CONTROL / 'blender_slots')
    rooms = small_rooms()
    OWN.mkdir(parents=True, exist_ok=True)
    link_view(rooms)
    tasks = []
    for row in rooms:
        case_id = 'nr.fastv1.glm.%s.%s' % (row['scene'], row['room'])
        tasks.append({'case_id': case_id, 'model': 'glm', 'scene': row['scene'], 'room': row['room'],
                      'source_case_id': case_id, 'objects': row['objects']})
    atomic_write(OWN / 'selected_rooms.json', {'rooms': tasks})
    prior = read(handoff.EXISTING / 'plan.json')
    plan = {'identity': {'campaign': 'glm_small_sidecar_20260926',
                         'content_fingerprint_validation': 'off', 'evaluation_mode': 'api'},
            'limits': prior['limits'], 'tasks': tasks}
    stop = threading.Event()
    previous = {sig: signal.signal(sig, lambda *unused: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    stream = (OWN / 'runner.lock').open('a+')
    try:
        import fcntl
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        settings = handoff.effective_limits(plan, args)
        runtime = run.DEFAULT_RUNTIME
        base = run.load('glm_small_inventory', runtime / 'scripts/nonrect_merged30_v8_api2_sol_catalog/run.py')
        atomic_write(OWN / 'status.json', {'status': 'starting', 'pid': os.getpid(),
                                            'rooms': [t['case_id'] for t in tasks]})
        with run.live_environment(runtime, base, OWN, stop) as env:
            env['NONRECT_BLENDER_SLOTS'] = str(high36.BLENDER_SLOTS)
            env['NONRECT_BLENDER_SLOT_DIR'] = os.environ['NONRECT_BLENDER_SLOT_DIR']
            env[content_policy.ENV] = 'off'
            stages = GlmStages(runtime, env, settings)
            pipeline = handoff.SuccessCleanupPipeline(
                OWN, plan['identity'], settings, stages.prepare, stages.evaluate,
                probe=DesktopHostProbe(OWN, args.desktop_memory_reserve_gib, args.worker_memory_gib),
                stop=stop, prepare_only=False, cleanup=False, retry_failed=False)
            pipeline.governor = handoff.ResourceUngatedGovernor(settings, HostProbe(OWN))
            atomic_write(OWN / 'status.json', {
                'status': 'evaluating', 'pid': os.getpid(),
                'rooms': [t['case_id'] for t in tasks], 'blender_slots': high36.BLENDER_SLOTS,
                'blender_slot_dir': os.environ['NONRECT_BLENDER_SLOT_DIR']})
            result = pipeline.execute(tasks)
        status = {'status': result['status'], 'pid': os.getpid(), 'results': len(result['results'])}
        atomic_write(OWN / 'status.json', status)
        print(json.dumps(status), flush=True)
        return 0 if result['status'] == 'finished' else 2
    finally:
        stream.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())
