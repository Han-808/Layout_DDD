"""Read-only projections of S100 artifacts; no evaluator/model/Blender run.

Output is a diagnostic receipt only. Historical verdicts, reports and scores
are never rewritten. jq projects large reports before Python loads them.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from benchmark.camera_cal_scene_level.uniform import digest
from benchmark.evaluator.scene_quality.consistency_acceptance_v2 import summarize_metric, terminalize_scope
from benchmark.evaluator.scene_quality.group_scoped import _combine_functional_check_episodes
from benchmark.visual_judge.render_views import _blank_view_ids
from benchmark.scene_generation.feedback_loop.frozen_report import compact_report, normalized_feedback_scores, validate_acceptance
from benchmark.camera_cal_scene_level.persisted_scoring import case_scoring_summary


def project(path, expression):
    result = subprocess.run(['jq', '-c', expression, str(path)], text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


SCOPE_FILTER = '''
def j: if type != "object" then {} else
    {verdict,terminal_state,evidence_resolution,failure,functional_check_results,placement_check_results,defects,reason,confidence,evidence_status}
    | with_entries(select(.value != null)) end;
def s: {group_id,relation_id,target_id,status,score,terminal_state,functional_check_episode_id,vlm_invoked,
    evidence_resolution,infrastructure_failure,reason,judgement:(.judgement|j),
    functional_probe_evidence:{required_checks:.functional_probe_evidence.required_checks},
    check_episodes:(if has("check_episodes") then [.check_episodes[]|s] else null end)}
    | with_entries(select(.value != null));
.metrics | with_entries(.value |= {
    status,score,reason,resolution_coverage,judgement_coverage_projection,
    group_results:[.group_results[]?|s],
    functional_check_ledger,placement_check_ledger,
    judgement:{scene_global_judgement:(.judgement.scene_global_judgement|j),
      residual_global_placement_judgement:(.judgement.residual_global_placement_judgement|j)}
})
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(args.case_dir.resolve()):
        raise ValueError('A new diagnostic destination outside the historical case is required')
    case = args.case_dir
    scene_path = case / 'scene_quality_report.json'
    metrics = project(scene_path, SCOPE_FILTER)
    placement = metrics['semantic_placement_consistency']
    old_placement = placement['resolution_coverage']
    projected = summarize_metric(placement, old_placement['original_planned_scope_ids'])
    assert projected['accepted_count'] == projected['planned_count'] == 12
    assert old_placement['accepted_count'] == 11
    functional = metrics['functional_consistency']
    groups = []
    for group in functional['group_results']:
        episodes = group['check_episodes']
        checks = [check for episode in episodes
                  for check in episode['functional_probe_evidence']['required_checks']]
        packet = {'group': {'group_id': group['group_id']},
                  'functional_probe_evidence': {'required_checks': checks}}
        restored = _combine_functional_check_episodes(packets=[packet], episode_results=episodes)[0]
        terminalize_scope(restored, phase='group_local:' + group['group_id'])
        groups.append(restored)
    assert all(group['status'] == 'evaluated' for group in groups)
    # Only reconstruct the pre-terminal aggregate from existing model episodes.
    # Do not rewrite the old failed metric's status or invent a new metric score.
    diagnostic_functional = {**functional, 'group_results': groups}
    new_functional = summarize_metric(diagnostic_functional,
        functional['resolution_coverage']['original_planned_scope_ids'])
    assert new_functional['accepted_count'] == new_functional['planned_count'] == 29

    blank_path = case / 'l3_functional_probes/functional_consistency__reading_bookcase__49cf91b4ac/previews/step_00/collision_overlay_manifest.json'
    manifest = json.loads(blank_path.read_text())
    blanks = _blank_view_ids(manifest)
    surviving = [v['id'] for v in manifest['views'] if v['id'] not in blanks]
    assert len(surviving) == 3 and manifest['views'][0]['id'] in blanks

    # Retained original uniform projections already include every metric's
    # coverage/score. Use these for failure serialization, NOT repaired scores.
    l1 = project(case / 'l1_report.json', '{metrics:(.metrics|with_entries(.value |= {status,score,judgement_coverage_projection,scoring,pairs,objects})),scoring}')
    raw_path = case / 'evaluation_report.json'
    # Stream only fixed top-level scalars and the acceptance receipt out of the
    # 1.1GB envelope; do not materialize its duplicate layer/Judge traces.
    header_rows = subprocess.run(['jq', '--stream', '-c',
        'select(length==2 and ((.[0]|length)==1 or (.[0][0]=="runner_outcome" and .[0][1]=="uniform_score_acceptance")))',
        str(raw_path)], capture_output=True, text=True, check=True)
    header = {}
    for line in header_rows.stdout.splitlines():
        path, value = json.loads(line)
        node = header
        for component in path[:-1]:
            if not isinstance(component, str):
                break
            node = node.setdefault(component, {})
        else:
            node[path[-1]] = value
    scoring = case_scoring_summary(case_id='S100', case_manifest=json.loads((case/'case_run_manifest.json').read_text()),
        l1_report=l1, l3_report={'metrics': metrics})
    view, audit = normalized_feedback_scores(scoring)
    assert view['feedback_score'] is None
    assert set(audit['unavailable_metrics']) == {'functional_consistency','semantic_placement_consistency'}
    header.update(layer_reports={'l1_physical_plausibility':l1}, reports={'scene_quality': {'metrics': metrics}},
        judgement_coverage_summary=scoring['judgement_coverage_summary'])
    # Normalize the streamed empty-list receipt (jq emits it as a leaf), and
    # preserve its actual infrastructure failure metric list from the stream.
    acceptance = header.get('runner_outcome', {}).get('uniform_score_acceptance', {})
    failures = acceptance.get('infrastructure_failure_metrics')
    if isinstance(failures, dict):
        acceptance['infrastructure_failure_metrics'] = [failures[k] for k in sorted(failures)]
    compact = compact_report(header, scoring_summary=scoring)
    assert compact['feedback_scores']['feedback_score'] is None
    try:
        validate_acceptance(compact)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Historical failed report was incorrectly accepted')
    json.dumps(compact, allow_nan=False)
    receipt = {
        'status':'offline_real_artifact_replay_passed',
        'source_hashes':{str(p):digest(p) for p in (scene_path, raw_path, blank_path)},
        'preview':{'blank_camera_ids':[v['id'] for v in manifest['views'] if v['id'] in blanks],
                   'surviving_camera_ids':surviving, 'live_rerender_performed':False},
        'placement':{'old_accepted':11,'new_scope_accepted':12,'planned':12},
        'functional':{'old_accepted':functional['resolution_coverage']['accepted_count'],
                      'new_scope_accepted':29,'planned':29,'accepted_groups':len(groups),
                      'episode_count':sum(len(g['check_episodes']) for g in groups)},
        'feedback':{'serialization':'passed','feedback_score':None,'acceptance':'rejected_as_expected',
                    'unavailable_metrics':audit['unavailable_metrics'],
                    'original_benchmark_score':header.get('benchmark_score'),
                    'original_benchmark_score_status':header.get('benchmark_score_status')},
        'historical_files_changed':False,'scores_republished':False,'model_calls':0,'blender_calls':0,
        'limitations':['Replays stored episodes and metadata only; no new model decision or render.',
                       'Coverage accounting repair is not live-canary acceptance or a new published score.']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()
