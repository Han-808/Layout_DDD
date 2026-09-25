"""Same handoff runner. Up to 19 rooms may be in evaluation. At most 3 of
them may hold Blender at once; a finished Blender frees its slot for the
next waiting room. Judge reasoning_effort is high and the API
retry gap is 20s. Only the five reused scenes are admitted. Sol keeps its
completed rooms and reruns the four missing ones. Successful rooms keep
their rendered visuals.

Does not edit the pinned runner files or the saved plan identity. The live
process keeps its own imported modules. This process raises the in-memory
worker ceiling, rewrites relay.adapt so room requests send high, and sets
the transport retry gap to 20 seconds.
"""
from __future__ import annotations
import math
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nonrect_fast_v1 import run
from nonrect_fast_v1.pipeline import CaseLease, Limits, Pipeline, canonical_sha, read, report_path, sha, validate_ready
import nonrect_fast_v1.handoff_execution as handoff

WORKERS = 19
BLENDER_SLOTS = 3
BLENDER_WRAPPER = Path(__file__).resolve().parent / 'blender_slot.py'
REASONING_EFFORT = 'high'
RETRY_GAP_SECONDS = 20
SCENES = frozenset({
    'scene_011888', 'scene_011687', 'scene_011634', 'scene_011838', 'scene_011760',
})
SOL_RERUN = frozenset({
    ('sol', 'scene_011687', 'room_000'),
    ('sol', 'scene_011634', 'room_003'),
    ('sol', 'scene_011838', 'room_000'),
    ('sol', 'scene_011760', 'room_000'),
})


def _ready_preflight(root, plan, *, allow_terminal=False):
    """Accept finished and interrupted rooms so a resumed run can start.

    infrastructure_failure keeps its report and is not evaluated again.
    interrupted_evaluation_needs_review is left for the scheduler, which does
    not retry it. Only status ready is admitted to evaluation.
    """
    ready = terminal = held = 0
    for task in plan['tasks']:
        lease = CaseLease(root / 'rooms' / task['case_id'], {'campaign': plan['identity'], 'task': task})
        try:
            state = read(lease.root / 'state.json')
            if state.get('identity_sha256') != canonical_sha(lease.identity):
                raise ValueError('State identity mismatch')
            status = state['status']
            if allow_terminal and status in {'complete', 'not_score_eligible'}:
                report = report_path(lease.root, state['report'])
                if sha(report) != state['report_sha256']:
                    raise ValueError('Terminal report changed')
                replay = read(report_path(lease.root, state['scoring_replay_receipt']))
                if replay.get('status') != 'passed' or replay.get('report_sha256') != state['report_sha256']:
                    raise ValueError('Terminal scoring replay proof changed')
                terminal += 1
            elif allow_terminal and status == 'infrastructure_failure':
                report = report_path(lease.root, state['report'])
                if sha(report) != state['report_sha256']:
                    raise ValueError('Infrastructure-failure report changed')
                terminal += 1
            elif allow_terminal and status == 'interrupted_evaluation_needs_review':
                held += 1
            else:
                if status != 'ready' or (lease.root / 'consuming.json').exists():
                    raise ValueError('Case is not ready; explicit review needed: ' + task['case_id'])
                validate_ready(lease)
                ready += 1
        finally:
            lease.close()
    return {'ready': ready, 'terminal': terminal, 'held': held}


def _keep_task(task):
    if task.get('scene') not in SCENES or task.get('model') not in {'hy4', 'kimi', 'sol'}:
        return False
    if task.get('model') == 'sol':
        return (task.get('model'), task.get('scene'), task.get('room')) in SOL_RERUN
    return True


def _verified_plan(root, runtime_manifest, pins):
    plan = _verified_plan.original(root, runtime_manifest, pins)
    scoped = dict(plan)
    scoped['tasks'] = [task for task in plan['tasks'] if _keep_task(task)]
    return scoped


def _create_handoff(manifest, inventory, pins):
    """Partition check needs the full saved plan; admission uses the scoped plan."""
    handoff.verified_plan = _verified_plan.original
    try:
        return _create_handoff.original(manifest, inventory, pins)
    finally:
        handoff.verified_plan = _verified_plan


def _limits_post_init(self):
    # Saved plans still carry the original 1..12 ceiling. Live admission is capped by WORKERS.
    if not 1 <= self.preparation_workers <= WORKERS or not 1 <= self.evaluation_workers <= WORKERS:
        raise ValueError("Preparation and evaluation worker counts must be 1..%d" % WORKERS)
    if self.ready_capacity < 1 or self.ready_gib < self.preparation_growth_gib:
        raise ValueError("Ready capacity must fit at least one preparation reservation")
    if self.minimum_free_gib < 30:
        raise ValueError("Keep at least 30 GiB host disk headroom")
    for name, value in asdict(self).items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Resource limits must be finite and positive: " + name)


def _effective_limits(plan, args):
    if not 1 <= args.workers <= WORKERS:
        raise ValueError('Worker ceiling must be 1..%d; resource guards still apply' % WORKERS)
    if not 4 <= args.desktop_memory_reserve_gib <= 16 or not 8 <= args.worker_memory_gib <= 20:
        raise ValueError('Desktop reserve must be 4..16 GiB and per-case RSS reservation 8..20 GiB')
    if not 6 <= args.evaluation_growth_gib <= 12:
        raise ValueError('Evaluation growth reservation must be 6..12 GiB')
    return handoff.replace(Limits(**plan['limits']), preparation_workers=args.workers,
        evaluation_workers=args.workers, preparation_growth_gib=6,
        evaluation_growth_gib=args.evaluation_growth_gib,
        preparation_memory_gib=args.worker_memory_gib, evaluation_memory_gib=args.worker_memory_gib,
        reserve_memory_gib=max(8, args.desktop_memory_reserve_gib), ready_gib=max(80, plan['limits']['ready_gib']),
        backpressure_timeout=900)


def _load(name, path):
    module = _load.original(name, path)
    if name == 'nonrect_fast_relay':
        original_adapt = module.adapt

        def adapt(payload):
            result = original_adapt(payload)
            result['reasoning_effort'] = REASONING_EFFORT
            return result

        module.adapt = adapt
    return module


def _save_keep_visuals(self, results, *args, **kwargs):
    Pipeline.save(self, results, *args, **kwargs)


@contextmanager
def _live_environment(runtime, base, output, stop):
    sys.path.insert(0, str(Path(runtime) / 'scripts'))
    import nonrect30_api2_sol_retry_transport as transport
    transport.POLICY['retry_gap_seconds'] = RETRY_GAP_SECONDS
    with _live_environment.original(runtime, base, output, stop) as env:
        yield env


_verified_plan.original = handoff.verified_plan
_create_handoff.original = handoff.create_handoff
_load.original = run.load
_live_environment.original = run.live_environment
Limits.__post_init__ = _limits_post_init
handoff.verified_plan = _verified_plan
handoff.create_handoff = _create_handoff
handoff.effective_limits = _effective_limits
handoff.ready_preflight = _ready_preflight
handoff.SuccessCleanupPipeline.save = _save_keep_visuals
run.load = _load
run.live_environment = _live_environment
run.BLENDER = BLENDER_WRAPPER
os.environ['NONRECT_BLENDER_SLOTS'] = str(BLENDER_SLOTS)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--workers' not in argv:
        argv = ['--workers', str(WORKERS), *argv]
    return handoff.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
