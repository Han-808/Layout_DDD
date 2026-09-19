"""Replay the actual S100 Function failure, with explicitly synthetic endings.

No network/render service is used. The saved JudgeResult and readiness request
are real; raw initial transport is reconstructed and final verdicts are synthetic
because the failed run never reached terminal Judge. Images are placeholders.
These outputs are regression receipts, never publishable evaluation results.
"""
import argparse
import ast
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from PIL import Image
from benchmark.evaluator.scene_quality.group_scoped import _evaluate_group_scoped_judgements_batched, _combine_functional_check_episodes
from benchmark.evaluator.scene_quality.global_group_first import _aggregate_global_and_group_results, _apply_functional_acquisition_budget_status
from benchmark.evaluator.scene_quality.interfaces import _judge_request, _call_scene_quality_judge, _normalize_judgement, _apply_prompt_exemptions
from benchmark.evaluator.scene_quality.functional_checks import apply_functional_check_judgements
from benchmark.evaluator.scene_quality.functional_ownership import build_functional_ownership_ledger
from benchmark.evaluator.scene_quality.consistency_acceptance_v2 import finish_metric
from benchmark.evaluator.judgement_coverage import apply_report
from benchmark.camera_cal_scene_level.persisted_scoring import case_scoring_summary
from benchmark.scene_generation.feedback_loop.frozen_report import compact_report, validate_acceptance, VIEW_VERSION
from benchmark.visual_judge import OpenAICompatibleVLMJudge
from benchmark.visual_judge.adapters.legacy_judge import ControlledVLMJudge
from benchmark.visual_judge.acquisition_outcome import AcquisitionExhausted
from benchmark.visual_judge.control_config import resolve_vlm_evaluation_control
from benchmark.visual_judge.evidence_resolution import FALLBACK_POLICY, evidence_policy_scope
from benchmark.visual_judge.best_effort_terminal import POLICY as TERMINAL_POLICY
from benchmark.visual_judge.group_scope import GroupCameraScope

METRIC = 'functional_consistency'


class ReplaySelector:
    preserve_configured_adapter = True
    supports_evidence_readiness = True
    backend = 'offline_stored_readiness'
    def __init__(self, readiness):
        self.readiness = readiness
        self.requests, self.readiness_requests = [], []
        self.last_call_usage = {'vlm_call_count': 0}
    def select(self, request):
        self.requests.append(request)
        return {'selected_view_ids': [request.candidate_views[0]['id']], 'action': None,
                'reason': 'Offline transport fixture, not a real camera decision.'}
    def review_evidence_readiness(self, request):
        self.readiness_requests.append(request)
        return deepcopy(self.readiness)


class ReplayRenderer:
    def __init__(self, images, ending):
        self.images, self.ending, self.requests = images, ending, []
    def render(self, request):
        self.requests.append(request)
        if len(self.requests) == 1:
            return {'visual_evidence': self.images, 'merge_policy': 'append',
                    'next_candidate_views': [{'id': 'offline-view-2'}], 'replaces_candidate_views': True}
        if self.ending == 'renderer_fault':
            raise ConnectionError('Synthetic renderer service failure')
        raise AcquisitionExhausted('trusted_candidate_bank_empty')


class ReplayModel:
    model_id = 'offline-functional-regression'
    endpoint = 'offline:no-network'
    last_request_metadata = {}
    def __init__(self, initial, check, ending):
        self.initial, self.check, self.ending = initial, check, ending
        self.calls, self.contexts = [], []
    def chat_messages(self, messages, **kwargs):
        self.calls.append(deepcopy(messages))
        context = json.loads(messages[1]['content'][0]['text'].split('\n', 1)[1])
        self.contexts.append(context)
        assert len(self.calls) <= 2, 'Unexpected schema repair or unbounded retry'
        row = {'check_id': self.check['check_id'], 'target_ids': self.check['target_ids'],
               'observation_status': 'missing', 'conclusion': 'unresolved', 'reason': self.initial['reason']}
        if len(self.calls) == 1:
            return json.dumps({'evidence_status': 'insufficient', 'verdict': 'ambiguous',
                'confidence': self.initial['confidence'], 'reason': self.initial['reason'],
                'defects': [], 'missing_evidence': self.initial['evidence_request']['missing_observations'],
                'evidence_request': self.initial['evidence_request'], 'functional_check_results': [row]})
        if self.ending == 'terminal_fault':
            raise ConnectionError('Synthetic terminal model service failure')
        assert self.ending in {'valid', 'invalid'}
        row.update(observation_status='inferred_under_budget', conclusion=self.ending,
                   reason='Synthetic terminal decision for control/score regression only.')
        return json.dumps({'evidence_status': 'sufficient', 'verdict': self.ending,
            'confidence': .6, 'reason': row['reason'], 'missing_evidence': [], 'evidence_request': None,
            'functional_check_results': [row], 'defects': [] if self.ending == 'valid' else [{
                'check_refs': [self.check['check_id']], 'scope': 'reachability',
                'relation': 'relative_use_geometry', 'category': 'functional_correspondence_failure',
                'attribution_mode': 'minimum_repair_set',
                'target_ids': self.check['target_ids'], 'severity': 'impaired', 'reason': row['reason']} ]})


def campaign_acceptor(directory):
    path = directory/'feedback_s100fix_v1/campaign.py'
    fn = next(n for n in ast.parse(path.read_text()).body
              if isinstance(n, ast.FunctionDef) and n.name == 'accepted_report')
    namespace = {'CheckpointError': type('CheckpointError', (RuntimeError,), {}),
        'IncompleteEvaluation': type('IncompleteEvaluation', (RuntimeError,), {}),
        'VIEW_VERSION': VIEW_VERSION, 'RELEASE_ID': 'offline-functional-replay',
        'MANIFEST_SHA': 'offline-fixture', 'validate_acceptance': validate_acceptance}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), namespace)
    return SimpleNamespace(**namespace)


def replay(fixture, directory, *, ending='valid', campaign=None, expect_repaired=True):
    # macOS /var aliases /private/var; keep fixture artifact identities stable.
    directory = directory.resolve()
    expected = expect_repaired and ending in {'valid', 'invalid'}
    l1, l3, manifest = deepcopy(fixture['l1']), deepcopy(fixture['l3']), deepcopy(fixture['case_manifest'])
    original = deepcopy(l3['metrics'][METRIC])
    p, failed = deepcopy(original), deepcopy(fixture['failed_episode'])
    audit = fixture['controller_audit']
    context = audit['judge_request']['context']
    check = context['required_functional_checks'][0]
    initial = next(r['result'] for r in audit['trace'] if r['stage'] == 'judge')
    readiness = deepcopy(next(r['result'] for r in audit['trace'] if r['stage'] == 'functional_evidence_readiness'))
    if ending == 'unknown_observation':
        readiness['missing_observations'] = ['interaction_region_visible_typo']
    images = []
    for index in range(5):
        path = directory/f'offline-placeholder-{index}.png'
        picture = Image.new('RGB', (32,32), (40+index*20, 140, 200))
        picture.paste((230, 30, 60), (index, 0, 16+index, 32))
        picture.save(path)
        images.append(str(path))
    selector, renderer = ReplaySelector(readiness), ReplayRenderer(images[3:], ending)
    model = ReplayModel(initial, check, ending)
    control = replace(resolve_vlm_evaluation_control({}),
        evidence_resolution_policy=FALLBACK_POLICY, camera_acquisition_policy='vlm_only')
    wrapper = ControlledVLMJudge(OpenAICompatibleVLMJudge(model, max_context_chars=200000,
        evidence_resolution_policy=FALLBACK_POLICY, terminal_evidence_policy=TERMINAL_POLICY),
        control=control, camera_selector=selector, vlm_camera_selector=selector, evidence_renderer=renderer)
    scope_data = failed['group_scope']
    scope = GroupCameraScope(**{k:v for k,v in scope_data.items() if k in GroupCameraScope.__dataclass_fields__},
        target_bounds_min=tuple(scope_data['target_bounds']['min']),
        target_bounds_max=tuple(scope_data['target_bounds']['max']))
    group = next(g for g in context['object_groups'] if g['group_id'] == failed['group_id'])
    scene = audit['judge_request']['scene_context']
    probe = deepcopy(fixture['failed_probe_evidence'])
    packet = {'group': group, 'group_scope': scope, 'paths': images[:3],
        'resolution': {'scope_satisfied': True, 'available_paths': images[:3]},
        'functional_probe_evidence': probe, 'functional_check_episode_id': check['check_id'],
        'functional_check_granularity': 'per_check'}
    def builder(**kwargs):
        request = _judge_request(**kwargs)
        request['candidate_views'] = [{'id': 'offline-view-1'}]
        return request
    with evidence_policy_scope({'evidence_resolution_policy': FALLBACK_POLICY}):
        output = _evaluate_group_scoped_judgements_batched(base={'evidence_request': {}}, metric_name=METRIC, scene=scene,
            prompt=context.get('prompt'), packets=[packet], vlm_judge=wrapper,
            authorized_deviations=context.get('authorized_deviations') or [], visual_style_spec=None,
            build_judge_request=builder, call_judge=_call_scene_quality_judge,
            apply_prompt_exemptions=_apply_prompt_exemptions, normalize_judgement=_normalize_judgement,
            evidence_phase=context['evidence_phase'], decision_mode=context['decision_mode'])
        repaired = output['group_results'][0]
        if expected:
            assert repaired['status'] == 'evaluated', (repaired.get('judgement'),
                [r for a in wrapper.audit_records for r in a.get('audit', {}).get('trace', [])
                 if r.get('status') == 'failed' or r.get('error') or r.get('stage') == 'terminal_choice_policy'])
            assert len(model.calls) == 2
            assert len(renderer.requests) == 2 and len(selector.readiness_requests) == 1, (
                len(renderer.requests), len(selector.readiness_requests),
                [(r.get('stage'), r.get('status'), r.get('trigger_stop_reason'), r.get('error'))
                 for a in wrapper.audit_records for r in a.get('audit', {}).get('trace', [])])
            resolution = repaired['judgement']['evidence_resolution']
            assert set(resolution['images_used']) == set(images), resolution
            assert resolution['model_judgement_completed'] and not resolution['visual_observation_complete']
            assert resolution['terminal_evidence_policy'] == TERMINAL_POLICY
            assert model.contexts[-1]['required_functional_checks'][0]['check_id'] == check['check_id']
            assert model.contexts[-1].get('functional_probe_evidence'), 'Structured facts must reach final Judge'
            measured = model.contexts[-1]['functional_measurements']['check_measurements']
            assert any(row['check_id'] == check['check_id'] for row in measured), 'Retain scoped measured facts'
        else:
            assert repaired['status'] == 'failed', repaired
            assert len(model.calls) == (2 if ending == 'terminal_fault' and expect_repaired else 1)
        groups, combine_packets, episodes = [], [], []
        for stored_group in p['group_results']:
            rows = stored_group.get('check_episodes') or [stored_group]
            group_checks = [c for c in p['functional_check_ledger']['checks']
                            if c.get('owning_group_id') == stored_group['group_id'] and c.get('owner_stage') == 'group_local']
            combine_packets.append({'group': {'group_id': stored_group['group_id']},
                                    'functional_probe_evidence': {'required_checks': group_checks}})
            episodes.extend(repaired if e.get('functional_check_episode_id') == check['check_id'] else deepcopy(e)
                            for e in rows)
        groups = _combine_functional_check_episodes(packets=combine_packets, episode_results=episodes)
        # Rebuild from source component records, not from stale aggregate outputs.
        for key in ('infrastructure_failures','judgement','coverage','component_degradations',
                    'judgement_coverage_projection','resolution_coverage','final_defect_claims',
                    'final_object_findings','score','reason','terminal_state','status'):
            p.pop(key, None)
        p['group_results'] = groups
        p['functional_check_ledger'], p['functional_check_coverage'] = apply_functional_check_judgements(
            p['functional_check_ledger'], relation_results=p['cross_group_relation_results'], group_results=groups)
        if expected:
            rebuilt = {c['check_id']: c for c in p['functional_check_ledger']['checks']}
            assert rebuilt[check['check_id']]['check_conclusion'] == ending
            for g in original['group_results']:
                for e in g.get('check_episodes') or []:
                    if e.get('functional_check_episode_id') == check['check_id']:
                        continue
                    for row in (e.get('judgement') or {}).get('functional_check_results') or []:
                        assert rebuilt[row['check_id']]['result_row'] == row
        prior_rows = {c['check_id']: c for c in original['functional_check_ledger']['checks']}
        for c in p['functional_check_ledger']['checks']:
            prior = prior_rows[c['check_id']]
            if prior.get('check_conclusion') in {'valid','invalid'}:
                assert c.get('result_row') == prior.get('result_row'), c['check_id']
        ids = manifest['canonical_object_denominator']['ordered_object_ids']
        global_record = p['global_discovery']
        result = _aggregate_global_and_group_results(p, metric_name=METRIC, global_record=global_record,
            global_outcome=_normalize_judgement(global_record, metric_name=METRIC, valid_object_ids=set(ids)),
            scene_claims=p['global_scene_claims'], relation_claims=p['cross_group_relation_claims'],
            relation_results=p['cross_group_relation_results'], relation_phase_complete=True,
            group_results=groups, group_phase_required=True,
            group_phase_complete=all(g['status']=='evaluated' for g in groups),
            functional_check_phase_complete=p['functional_check_coverage']['complete'],
            placement_check_phase_complete=True)
        result = _apply_functional_acquisition_budget_status(result, metric_name=METRIC)
        resolved = result['status'] == 'evaluated'
        result['functional_ownership_ledger'] = build_functional_ownership_ledger(scene_object_ids=ids,
            global_record=global_record if resolved else None,
            relation_results=p['cross_group_relation_results'] if resolved else [],
            group_results=groups if resolved else [], functional_check_ledger=p['functional_check_ledger'] if resolved else None)
        result = finish_metric(result, original['resolution_coverage']['original_planned_scope_ids'])
    l3['metrics'][METRIC] = result
    report = apply_report({'report_schema_version':'scene_evaluation_report_v2', 'evaluation_status':'completed',
        'layer_reports':{'l1_physical_plausibility':l1,'l3_scene_quality':l3}, 'reports':{'scene_quality':l3},
        'canonical_object_denominator':manifest['canonical_object_denominator'], 'scoring_reliability':{}})
    summary = report['judgement_coverage_summary']
    assert summary['eligible'] == expected, summary
    for metric, projection in summary['metric_projections'].items():
        if metric != METRIC:
            source = fixture['l1']['metrics'] if metric in fixture['l1']['metrics'] else fixture['l3']['metrics']
            assert projection['observed_score'] == source[metric]['judgement_coverage_projection']['observed_score'], metric
    report['runner_outcome'] = {'uniform_score_acceptance':{k:summary[k] for k in (
        'schema_version','eligible','status','judgement_coverage_fraction',
        'minimum_judgement_coverage','infrastructure_failure_metrics')}}
    scoring = case_scoring_summary(case_id='S100',case_manifest=manifest,l1_report=l1,l3_report=l3)
    compact = compact_report(report, scoring_summary=scoring)
    if expected:
        validate_acceptance(compact)
        assert compact['feedback_scores']['feedback_score'] is not None
        assert scoring['combined_score_100'] == report['benchmark_score_100']
    else:
        assert compact['feedback_scores']['feedback_score'] is None
    runner = 'not_requested'
    if campaign:
        compact['source_report'] = {'release':'offline-functional-replay','release_manifest_sha256':'offline-fixture'}
        try:
            campaign.accepted_report(compact)
        except campaign.IncompleteEvaluation:
            assert not expected
            runner = 'rejected'
        else:
            assert expected
            runner = 'accepted_offline_binding'
    projection = summary['metric_projections'][METRIC]
    return {'case':'S100', 'ending':ending, 'synthetic_terminal_verdict':expected, 'live_model_calls':0,
        'mock_model_calls':len(model.calls), 'renderer_calls':len(renderer.requests),
        'eligible':summary['eligible'], 'status':summary['status'], 'coverage':summary['judgement_coverage_fraction'],
        'function_score':projection['observed_score'], 'function_accepted':projection['accepted_count'],
        'function_planned':projection['planned_count'], 'feedback_score':compact['feedback_scores']['feedback_score'],
        'other_seven_metric_scores_unchanged':True, 'resolved_check_rows_unchanged':True, 'runner':runner,
        'all_metric_scores':{k:v['observed_score'] for k,v in summary['metric_projections'].items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--runner-scripts', type=Path)
    parser.add_argument('--expect-failure', action='store_true')
    args = parser.parse_args()
    campaign = campaign_acceptor(args.runner_scripts) if args.runner_scripts else None
    fixture = json.loads((args.fixtures/'S100.json').read_text())
    results = []
    endings = ('valid',) if args.expect_failure else ('valid','invalid','renderer_fault','unknown_observation','terminal_fault')
    for ending in endings:
        with tempfile.TemporaryDirectory(prefix='functional-replay-') as temp:
            results.append(replay(fixture, Path(temp), ending=ending, campaign=campaign, expect_repaired=not args.expect_failure))
    if not args.expect_failure:
        assert results[0]['function_score'] > results[1]['function_score']
        assert results[0]['feedback_score'] > results[1]['feedback_score']
    args.output.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__ == '__main__':
    main()
