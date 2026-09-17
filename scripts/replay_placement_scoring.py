"""Offline real-candidate replay through Judge, aggregation, scoring and feedback.

Only transport and image bytes are fixtures. No historical report is rewritten,
no remote inference or rendering is performed, and these are not published scores.
"""
import argparse
import ast
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
from types import FunctionType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from benchmark.evaluator.scene_quality.global_group_first import _evaluate_global_scope, _aggregate_global_and_group_results
from benchmark.evaluator.scene_quality.interfaces import _normalize_judgement, _judge_request, _apply_prompt_exemptions, _call_scene_quality_judge
from benchmark.evaluator.scene_quality.placement_checks import apply_placement_check_judgements
from benchmark.evaluator.scene_quality.consistency_acceptance_v2 import finish_metric, summarize_metric
from benchmark.evaluator.judgement_coverage import apply_report
from benchmark.camera_cal_scene_level.persisted_scoring import case_scoring_summary
from benchmark.scene_generation.feedback_loop.frozen_report import compact_report, validate_acceptance
from benchmark.visual_judge import OpenAICompatibleVLMJudge
from benchmark.visual_judge.adapters.legacy_judge import ControlledVLMJudge
from benchmark.visual_judge.control_config import resolve_vlm_evaluation_control
from benchmark.visual_judge.evidence_resolution import FALLBACK_POLICY, evidence_policy_scope
from benchmark.visual_judge.best_effort_terminal import POLICY as TERMINAL_POLICY
from PIL import Image

METRIC = 'semantic_placement_consistency'

class StoredCandidate:
    model_id = 'offline-real-semantic-candidate'
    endpoint = 'offline:no-network'
    last_request_metadata = {}
    def __init__(self, value):
        self.value, self.calls = value, 0
    def chat_messages(self, messages, **kwargs):
        self.calls += 1
        return json.dumps(self.value)

def replay(fixture, directory, *, campaign_module=None, expect_repaired=True):
    manifest = deepcopy(fixture['case_manifest'])
    l1, l3 = deepcopy(fixture['l1']), deepcopy(fixture['l3'])
    original = deepcopy(l3['metrics'][METRIC])
    p = deepcopy(original)
    context = p['residual_global_placement_context']
    ids = manifest['canonical_object_denominator']['ordered_object_ids']
    scene = {'objects': [{'id': o['object_id'], 'category': o['category']} for o in context['object_inventory']],
             'scene_type': context['scene_program']['scene_type']}
    picture = directory/'offline-placeholder.png'
    pixels = Image.new('RGB', (32,32), (80, 140, 200))
    pixels.paste((230, 30, 60), (0, 0, 16, 32))
    pixels.save(picture)
    model = StoredCandidate(fixture['residual_candidates'][0]['candidate'])
    judge = ControlledVLMJudge(OpenAICompatibleVLMJudge(model, max_context_chars=200000,
        evidence_resolution_policy=FALLBACK_POLICY, terminal_evidence_policy=TERMINAL_POLICY),
        control=replace(resolve_vlm_evaluation_control({}), evidence_resolution_policy=FALLBACK_POLICY))
    # Reconstruct pre-aggregation input from persisted component records. Stale
    # aggregate outputs are NOT inputs to a fresh metric aggregation.
    for key in ('infrastructure_failures', 'judgement', 'coverage', 'component_degradations',
                'judgement_coverage_projection', 'resolution_coverage', 'final_defect_claims',
                'final_object_findings', 'score', 'reason', 'terminal_state', 'status'):
        p.pop(key, None)
    p['initial_acquisition_resolution'] = {'scope_satisfied': True, 'available_paths': [str(picture)]}
    with evidence_policy_scope({'evidence_resolution_policy': FALLBACK_POLICY}):
        residual, outcome, audit = _evaluate_global_scope(base=p, metric_name=METRIC,
            scene=scene, object_ids=ids, groups=context['groups'], global_evidence=[str(picture)],
            functional_probe_packet=None, vlm_judge=judge, prompt=None, visual_style_spec=None,
            authorized_deviations=p.get('authorized_deviations') or [], build_judge_request=_judge_request,
            call_judge=_call_scene_quality_judge,
            apply_prompt_exemptions=_apply_prompt_exemptions, normalize_judgement=_normalize_judgement,
            camera_acquisition_ledger={}, forbidden_cross_group_target_sets=[], required_placement_checks=[],
            functional_ownership_ledger=p['functional_ownership_ledger'],
            evidence_phase='residual_global_placement_review', placement_residual_context=context,
            update_placement_ledger=False)
        if expect_repaired:
            assert outcome['status'] == 'evaluated', {k: residual.get(k) for k in ('error_type','error','failure')}
            assert residual['verdict'] == fixture['residual_candidates'][0]['candidate']['verdict'] == 'invalid'
            assert model.calls == 1, 'Identity normalization must not cause paid schema retries'
        p['residual_global_placement_review'] = residual
        p['residual_global_placement_phase']['status'] = 'complete' if outcome['status']=='evaluated' else 'infrastructure_failure'
        global_record = p['global_discovery']
        p['placement_check_ledger'], p['placement_check_coverage'] = apply_placement_check_judgements(
            p['placement_check_ledger'], global_record=global_record, group_results=p['group_results'],
            target_results=p.get('target_scope_results') or [], residual_records=[residual],
            handoff_records=[r['judgement'] for r in p.get('placement_global_handoff_reviews') or []])
        prior_checks = {c['check_id']:c for c in original['placement_check_ledger']['checks']}
        for check in p['placement_check_ledger']['checks']:
            prior = prior_checks[check['check_id']]
            for field in ('judge_result_ref','result_row','evidence_resolution','check_conclusion'):
                assert check.get(field)==prior.get(field), (check['check_id'],field)
        result = _aggregate_global_and_group_results(p, metric_name=METRIC, global_record=global_record,
            global_outcome=_normalize_judgement(global_record, metric_name=METRIC, valid_object_ids=set(ids)),
            scene_claims=p.get('global_scene_claims') or [], relation_claims=p.get('cross_group_relation_claims') or [],
            relation_results=p.get('cross_group_relation_results') or [],
            relation_phase_complete=original['coverage']['cross_group_relation_phase_complete'],
            group_results=p['group_results'], group_phase_required=True,
            group_phase_complete=original['coverage']['group_phase_complete'],
            functional_check_phase_complete=original['coverage']['functional_check_phase_complete'],
            placement_check_phase_complete=p['placement_check_coverage']['complete'],
            target_results=p.get('target_scope_results') or [],
            target_phase_complete=original['coverage']['target_scope_phase_complete'],
            residual_global_record=residual, residual_global_outcome=outcome,
            residual_phase_required=True, residual_phase_complete=outcome['status']=='evaluated')
        result = finish_metric(result, original['resolution_coverage']['original_planned_scope_ids'])
    l3['metrics'][METRIC] = result
    report = {'report_schema_version': 'scene_evaluation_report_v2', 'evaluation_status': 'completed', 'layer_reports': {
        'l1_physical_plausibility': l1, 'l3_scene_quality': l3}, 'reports': {'scene_quality': l3},
        'canonical_object_denominator': manifest['canonical_object_denominator'], 'scoring_reliability': {}}
    report = apply_report(report)
    summary = report['judgement_coverage_summary']
    for metric, projection in summary['metric_projections'].items():
        if metric == METRIC:
            continue
        source = fixture['l1']['metrics'] if metric in fixture['l1']['metrics'] else fixture['l3']['metrics']
        assert projection['observed_score']==source[metric]['judgement_coverage_projection']['observed_score'], metric
    report['runner_outcome'] = {'uniform_score_acceptance': {k:summary[k] for k in (
        'schema_version','eligible','status','judgement_coverage_fraction',
        'minimum_judgement_coverage','infrastructure_failure_metrics')}}
    scoring = case_scoring_summary(case_id=fixture['case_id'], case_manifest=manifest, l1_report=l1, l3_report=l3)
    compact = compact_report(report, scoring_summary=scoring)
    if expect_repaired:
        validate_acceptance(compact)
        assert compact['feedback_scores']['feedback_score'] is not None
        assert result['judgement']['defects'], 'Typed invalid findings must survive'
        assert not any(d.get('placement_component')=='residual_global_review' for d in result['judgement']['defects'])
    else:
        assert summary['eligible'] is False
    runner = 'not_requested'
    if campaign_module:
        # Execute the ACTUAL runner function body with an explicit offline
        # identity binding. Never claim that new scores came from the old release.
        globals_ = {**campaign_module.accepted_report.__globals__, 'RELEASE_ID': 'offline-replay',
                    'MANIFEST_SHA': 'offline-fixture', 'validate_acceptance': validate_acceptance}
        accepted = FunctionType(campaign_module.accepted_report.__code__, globals_)
        compact['source_report'] = {'release':'offline-replay', 'release_manifest_sha256':'offline-fixture'}
        try:
            accepted(compact)
        except campaign_module.IncompleteEvaluation:
            assert not expect_repaired
            runner = 'rejected'
        else:
            assert expect_repaired
            runner = 'accepted_offline_binding'
    return {'case':fixture['case_id'], 'mock_model_calls':model.calls, 'live_model_calls':0,
        'status':summary['status'], 'coverage':summary['judgement_coverage_fraction'],
        'score':summary['score'], 'feedback_score':compact['feedback_scores']['feedback_score'],
        'placement_score':summary['metric_projections'][METRIC]['observed_score'],
        'placement_planned':summary['metric_projections'][METRIC]['planned_count'],
        'placement_accepted':summary['metric_projections'][METRIC]['accepted_count'],
        'runner':runner, 'persisted_score_matches':scoring['combined_score_100']==report['benchmark_score_100'],
        'other_metric_scores_unchanged':True, 'typed_ledger_receipts_unchanged':True,
        'all_metric_scores':{k:p['observed_score'] for k,p in summary['metric_projections'].items()},
        'typed_defect_ids':sorted({d['check_id'] for d in result['judgement']['defects']})}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runner-scripts', type=Path)
    parser.add_argument('--expect-failure', action='store_true')
    args = parser.parse_args()
    campaign = None
    if args.runner_scripts:
        # Load only the actual acceptance function; importing campaign would
        # require the mutable scheduler, absent from an evaluator-only release.
        path = args.runner_scripts/'feedback_s100fix_v1/campaign.py'
        tree = ast.parse(path.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='accepted_report')
        from benchmark.scene_generation.feedback_loop.frozen_report import VIEW_VERSION
        namespace = {'CheckpointError': type('CheckpointError', (RuntimeError,), {}),
                     'IncompleteEvaluation': type('IncompleteEvaluation', (RuntimeError,), {}),
                     'VIEW_VERSION': VIEW_VERSION}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), namespace)
        campaign = SimpleNamespace(**namespace)
    receipts = []
    for case in ('S100', 'S101', 'S102'):
        with tempfile.TemporaryDirectory(prefix='placement-replay-') as tmp:
            receipts.append(replay(json.loads((args.fixtures/(case+'.json')).read_text()), Path(tmp),
                                   campaign_module=campaign, expect_repaired=not args.expect_failure))
    args.output.write_text(json.dumps(receipts, indent=2)+'\n')
    print(json.dumps(receipts, indent=2))

if __name__ == '__main__':
    main()
