"""Replay S101 ownership failure plus explicitly synthetic new model decisions.

The first TWO replies are real preserved candidates. A third reply never existed
in the failed run: every third answer here is a synthetic contract test. Images
are placeholders. No live calls; these are NOT publishable evaluation results.
"""
import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile

from replay_placement_scoring import (
    METRIC, Image, ControlledVLMJudge, OpenAICompatibleVLMJudge,
    resolve_vlm_evaluation_control, FALLBACK_POLICY, TERMINAL_POLICY,
    evidence_policy_scope, _judge_request, _call_scene_quality_judge,
    _apply_prompt_exemptions, _normalize_judgement,
    apply_placement_check_judgements, _aggregate_global_and_group_results,
    finish_metric, apply_report, case_scoring_summary, compact_report, validate_acceptance,
)
from replay_functional_scoring import campaign_acceptor
from benchmark.evaluator.scene_quality.group_scoped import _evaluate_group_scoped_judgements_batched
from benchmark.visual_judge.group_scope import GroupCameraScope


class StoredModel:
    model_id = 'offline-real-placement-reference-candidates'
    endpoint = 'offline:no-network'
    last_request_metadata = {}

    def __init__(self, candidates):
        self.candidates, self.calls = candidates, 0

    def chat_messages(self, messages, **kwargs):
        assert self.calls < len(self.candidates), 'Unbounded repair retry'
        value = self.candidates[self.calls]
        self.calls += 1
        if isinstance(value, Exception):
            raise value
        return json.dumps(value)


def replay(fixture, directory, *, ending='valid', expect_repaired=True, campaign=None):
    expected = expect_repaired and ending in {'valid','invalid'}
    directory = directory.resolve()
    l1, l3, manifest = deepcopy(fixture['l1']), deepcopy(fixture['l3']), deepcopy(fixture['case_manifest'])
    original = deepcopy(l3['metrics'][METRIC])
    p, failed = deepcopy(original), deepcopy(fixture['failed_group'])
    candidates = [deepcopy(a['candidate']) for a in fixture['model_candidates']]
    from benchmark.evaluator.scene_quality.placement_checks import placement_check_id
    proposal = candidates[0]['judge_originated_placement_results'][0]
    cid = placement_check_id(proposal['check_type'],proposal['subject_id'],proposal['context_ids'])
    conclusion = 'invalid' if ending=='invalid' else 'valid'
    final = {'evidence_status':'sufficient','verdict':conclusion,'confidence':0.7,
        'reason':'SYNTHETIC offline model decision; not a real room judgement.',
        'missing_evidence':[],'evidence_request':None,'defects':[],
        'placement_check_results':[{'check_id':cid,'subject_id':proposal['subject_id'],
            'context_ids':proposal['context_ids'],'observation_status':'inferred_under_budget',
            'conclusion':conclusion,'reason':'SYNTHETIC contract test under the original metric rubric.'}]}
    if conclusion=='invalid':
        final['defects'] = [{'scope':'semantically_inappropriate_support_surface','target_ids':[proposal['subject_id']],
            'check_id':cid,'check_type':'support_and_height','relation':'support_and_height',
            'reason':'SYNTHETIC independent Placement finding.','severity':'implausible',
            'category':'semantic_surface_mismatch','attribution_mode':'unary'}]
    if ending=='delete_pending':
        final['placement_check_results']=[]
    elif ending=='false_exclusion':
        final['placement_check_results'][0].update(conclusion='excluded_function_owned',
            function_event_ref=proposal['function_event_ref'],same_physical_event=True)
    elif ending=='alter_independent':
        final['placement_check_results'].append({**deepcopy(candidates[0]['placement_check_results'][0]),
                                                'conclusion':'valid'})
    elif ending=='service_failure':
        final=ConnectionError('Synthetic ownership-decision transport failure')
    else:
        assert ending in {'valid','invalid'}
    candidates.append(final)
    model = StoredModel(candidates)
    judge = ControlledVLMJudge(OpenAICompatibleVLMJudge(model, max_context_chars=200000,
        evidence_resolution_policy=FALLBACK_POLICY, terminal_evidence_policy=TERMINAL_POLICY),
        control=replace(resolve_vlm_evaluation_control({}), evidence_resolution_policy=FALLBACK_POLICY))
    images = []
    for index in range(len(failed['evidence_paths'])):
        path = directory/f'offline-placeholder-{index}.png'
        picture = Image.new('RGB', (32, 32), (60+index*20, 140, 200))
        picture.paste((230, 30, 60), (index, 0, 16+index, 32))
        picture.save(path)
        images.append(str(path))
    scope_data = failed['group_scope']
    scope = GroupCameraScope(**{k:v for k,v in scope_data.items() if k in GroupCameraScope.__dataclass_fields__},
        target_bounds_min=tuple(scope_data['target_bounds']['min']),
        target_bounds_max=tuple(scope_data['target_bounds']['max']))
    group = next(g for g in p['placement_scene_groups'] if g['group_id'] == failed['group_id'])
    packet = {'group':group, 'group_scope':scope, 'paths':images,
        'resolution':{'scope_satisfied':True, 'available_paths':images},
        **{k:deepcopy(failed.get(k)) for k in ('placement_discovery','required_placement_checks',
                                             'functional_ownership_ledger','routed_candidate_claims')}}
    for key in ('infrastructure_failures','judgement','coverage','component_degradations',
                'judgement_coverage_projection','resolution_coverage','final_defect_claims',
                'final_object_findings','score','reason','terminal_state','status'):
        p.pop(key, None)
    with evidence_policy_scope({'evidence_resolution_policy':FALLBACK_POLICY}):
        output = _evaluate_group_scoped_judgements_batched(base=p, metric_name=METRIC,
            scene=fixture['scene'], prompt=None, packets=[packet], vlm_judge=judge,
            authorized_deviations=p.get('authorized_deviations') or [], visual_style_spec=None,
            build_judge_request=_judge_request, call_judge=_call_scene_quality_judge,
            apply_prompt_exemptions=_apply_prompt_exemptions, normalize_judgement=_normalize_judgement)
        repaired = output['group_results'][0]
        assert model.calls == (3 if expect_repaired else 2)
        assert (repaired['status'] == 'evaluated') == expected, repaired.get('judgement')
        rows = (repaired.get('judgement') or {}).get('placement_check_results') or []
        if expected:
            assert len(rows)==2
            cabinet=candidates[0]['placement_check_results'][0]
            assert next(r for r in rows if r['subject_id']=='upper_kitchen_cabinet_1') == cabinet
            assert next(r for r in rows if r['check_id']==cid)['conclusion']==conclusion
            assert not [r for r in rows if r['conclusion']=='excluded_function_owned']
        # The real stage replaces base.group_results with this packet batch.
        # Restore untouched original batches when assembling the whole metric.
        p['group_results'] = [repaired if g['group_id']==failed['group_id'] else deepcopy(g)
                              for g in original['group_results']]
        global_record, residual = p['global_discovery'], p['residual_global_placement_review']
        p['placement_check_ledger'], p['placement_check_coverage'] = apply_placement_check_judgements(
            p['placement_check_ledger'], global_record=global_record, group_results=p['group_results'],
            target_results=p.get('target_scope_results') or [], residual_records=[residual],
            handoff_records=[r['judgement'] for r in p.get('placement_global_handoff_reviews') or []])
        prior = {c['check_id']:c for c in original['placement_check_ledger']['checks']}
        rebuilt = {c['check_id']:c for c in p['placement_check_ledger']['checks']}
        for key, old in prior.items():
            if expected and key in {c['check_id'] for c in failed['required_placement_checks']}:
                assert rebuilt[key]['check_conclusion']=='invalid'
                continue
            for field in ('judge_result_ref','result_row','evidence_resolution','check_conclusion'):
                assert rebuilt[key].get(field) == old.get(field), (key,field)
        assert len(rebuilt) == len(prior)+(1 if expected else 0)
        if expected:
            assert rebuilt[cid]['check_conclusion']==conclusion
            assert rebuilt[cid]['observation_status']=='inferred_under_budget'
        ids = manifest['canonical_object_denominator']['ordered_object_ids']
        cov = original['coverage']
        result = _aggregate_global_and_group_results(p, metric_name=METRIC, global_record=global_record,
            global_outcome=_normalize_judgement(global_record, metric_name=METRIC, valid_object_ids=set(ids)),
            scene_claims=p.get('global_scene_claims') or [], relation_claims=p.get('cross_group_relation_claims') or [],
            relation_results=p.get('cross_group_relation_results') or [],
            relation_phase_complete=cov['cross_group_relation_phase_complete'],
            group_results=p['group_results'], group_phase_required=True,
            group_phase_complete=all(g['status']=='evaluated' for g in p['group_results']),
            functional_check_phase_complete=cov['functional_check_phase_complete'],
            placement_check_phase_complete=p['placement_check_coverage']['complete'],
            target_results=p.get('target_scope_results') or [], target_phase_complete=cov['target_scope_phase_complete'],
            residual_global_record=residual,
            residual_global_outcome=_normalize_judgement(residual, metric_name=METRIC, valid_object_ids=set(ids)),
            residual_phase_required=True, residual_phase_complete=cov['residual_global_placement_phase_complete'])
        result = finish_metric(result, original['resolution_coverage']['original_planned_scope_ids'])
    l3['metrics'][METRIC] = result
    assert l3['metrics']['functional_consistency']==fixture['l3']['metrics']['functional_consistency']
    report = apply_report({'report_schema_version':'scene_evaluation_report_v2','evaluation_status':'completed',
        'layer_reports':{'l1_physical_plausibility':l1,'l3_scene_quality':l3}, 'reports':{'scene_quality':l3},
        'canonical_object_denominator':manifest['canonical_object_denominator'],'scoring_reliability':{}})
    summary = report['judgement_coverage_summary']
    assert summary['eligible'] == expected, summary
    for metric, projection in summary['metric_projections'].items():
        if metric != METRIC:
            source = fixture['l1']['metrics'] if metric in fixture['l1']['metrics'] else fixture['l3']['metrics']
            assert projection['observed_score'] == source[metric]['judgement_coverage_projection']['observed_score'], metric
    report['runner_outcome'] = {'uniform_score_acceptance':{k:summary[k] for k in (
        'schema_version','eligible','status','judgement_coverage_fraction',
        'minimum_judgement_coverage','infrastructure_failure_metrics')}}
    scoring = case_scoring_summary(case_id=fixture['case_id'], case_manifest=manifest,l1_report=l1,l3_report=l3)
    compact = compact_report(report, scoring_summary=scoring)
    if expected:
        validate_acceptance(compact)
        assert compact['feedback_scores']['feedback_score'] is not None
        assert scoring['combined_score_100'] == report['benchmark_score_100']
        defects = result['judgement']['defects']
        assert any('upper_kitchen_cabinet_1' in d['target_ids'] and d['severity']=='atypical' for d in defects)
        assert any('microwave_oven' in d['target_ids'] for d in defects)==(conclusion=='invalid')

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
    function_unchanged = all(l3['metrics']['functional_consistency'].get(k)==
        fixture['l3']['metrics']['functional_consistency'].get(k) for k in (
            'judgement','functional_check_ledger','functional_check_coverage','group_results',
            'functional_ownership_ledger','final_defect_claims','final_object_findings','resolution_coverage'))
    assert function_unchanged
    scoring_events=(projection.get('observed_scoring') or {}).get('events') or []
    microwave_events=[e for e in scoring_events if 'microwave_oven' in e.get('scoring_target_ids',[])]
    if expected:
        assert bool(microwave_events)==(conclusion=='invalid')
        assert all(e['burden']>0 for e in microwave_events)
    return {'case':fixture['case_id'], 'ending':ending, 'stored_real_candidates':2, 'third_candidate':'synthetic_not_historical',
        'third_candidate_executed':model.calls==3,
        'model_calls':model.calls, 'live_model_calls':0, 'eligible':summary['eligible'], 'status':summary['status'],
        'coverage':summary['judgement_coverage_fraction'],'placement_score':projection['observed_score'],
        'placement_planned':projection['planned_count'],'placement_accepted':projection['accepted_count'],
        'feedback_score':compact['feedback_scores']['feedback_score'],'runner':runner,
        'other_seven_scores_unchanged':True,'prior_resolved_typed_rows_unchanged':True,
        'original_cabinet_invalid_preserved':expected,
        'function_checks_unchanged':function_unchanged,
        'microwave_scoring_events':[{k:e.get(k) for k in ('event_id','burden','scoring_target_ids',
            'placement_component','placement_component_weight')} for e in microwave_events],
        'component_scoring':{name:{k:v.get(k) for k in ('score','burden_total_b_m','p_max',
            'prevalence_deduction','worst_event_floor_deduction','metric_deduction')}
            for name,v in (projection.get('observed_scoring') or {}).get('placement_components',{}).items()},
        'pending_check_retained':expected, 'pending_check_id':cid,
        'ownership_decision_retry':((repaired.get('judgement') or {}).get('request_metadata') or {}).get(
            'response_schema_validation',{}).get('ownership_decision_retry',{}),
        'all_metric_scores':{k:v['observed_score'] for k,v in summary['metric_projections'].items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runner-scripts', type=Path)
    parser.add_argument('--expect-failure', action='store_true')
    args = parser.parse_args()
    fixture = json.loads((args.fixtures/'S101_diagnostics_R1.json').read_text())
    campaign = campaign_acceptor(args.runner_scripts) if args.runner_scripts else None
    results = []
    for ending in (('valid',) if args.expect_failure else (
            'valid','invalid','delete_pending','false_exclusion','alter_independent','service_failure')):
        with tempfile.TemporaryDirectory(prefix='placement-reference-replay-') as temp:
            results.append(replay(fixture, Path(temp), ending=ending, campaign=campaign, expect_repaired=not args.expect_failure))
    args.output.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__ == '__main__':
    main()
