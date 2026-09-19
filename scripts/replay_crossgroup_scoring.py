"""Replay attempt04 acquisition through all scoring layers, without live calls.

The failed run never called this relation Judge: ALL model responses below are
synthetic regression data, not historical decisions or publishable scores.
"""
import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import replay_functional_scoring as shared
from PIL import Image
from benchmark.evaluator.scene_quality.cross_group_relations import (
    _cross_group_relation_episode_specs, _evaluate_cross_group_relation_scopes,
    reconcile_directional_relation_conflicts)
from benchmark.evaluator.scene_quality.claim_identity import claim_record
from benchmark.visual_judge.evidence_gap_v2 import classify_failure
from benchmark.visual_judge.acquisition_outcome import AcquisitionExhausted

METRIC = shared.METRIC
SUCCESS = {'valid','invalid','not_scheduled','no_images','pair_available','partial_packet'}


class TerminalModel:
    model_id = 'synthetic-crossgroup-regression'
    endpoint = 'offline:no-network'
    last_request_metadata = {}
    def __init__(self, ending):
        self.ending, self.contexts = ending, []
    def chat_messages(self, messages, **kwargs):
        context = json.loads(messages[1]['content'][0]['text'].split('\n',1)[1])
        self.contexts.append(context)
        assert len(self.contexts) <= 2, 'No unbounded Judge loop'
        if self.ending == 'terminal_fault':
            raise ConnectionError('Synthetic terminal service failure')
        invalid = self.ending == 'invalid'
        check = context['required_functional_checks'][0]
        reason = 'Synthetic best-supported relation decision for offline regression only.'
        row = {'check_id':check['check_id'], 'target_ids':check['target_ids'],
            'conclusion':'invalid' if invalid else 'valid', 'observation_status':'inferred_under_budget', 'reason':reason}
        return json.dumps({'evidence_status':'sufficient','verdict':row['conclusion'],
            'confidence':.6,'reason':reason,'missing_evidence':[],'evidence_request':None,
            'functional_check_results':[row], 'defects': [] if not invalid else [{
                'check_refs':[check['check_id']], 'scope':'facing_and_interaction_direction',
                'target_ids':['television'], 'relation':'directional_correspondence',
                'category':'functional_correspondence_failure', 'attribution_mode':'responsible_endpoint',
                'severity':'impaired','reason':reason}]})


def replay(fixture, directory, *, ending='valid', campaign=None, expect_repaired=True):
    directory = directory.resolve()
    expected = expect_repaired and (ending in SUCCESS or ending == 'evidence_gap')
    picture = directory/'offline-global.png'
    pixels = Image.new('RGB',(32,32),(30,150,230))
    pixels.paste((220,50,90),(0,0,16,32))
    pixels.save(picture)
    global_evidence = [] if ending in {'no_images','evidence_gap'} else [str(picture)]
    if ending == 'missing_file':
        global_evidence = [str(directory/'missing.png')]
    l1, l3, manifest = deepcopy(fixture['l1']), deepcopy(fixture['l3']), deepcopy(fixture['case_manifest'])
    original = deepcopy(l3['metrics'][METRIC])
    p, acquisition = deepcopy(original), deepcopy(fixture['acquisition_audit'])
    # A legitimate consistency recheck reuses endpoint images from earlier
    # group episodes. Materialize only their fixture references, preserving
    # each distinct artifact identity instead of reading the old run directory.
    endpoint_images = {}
    for group_result in p['group_results']:
        for episode in [group_result, *(group_result.get('check_episodes') or [])]:
            paths = []
            for reference in episode.get('evidence_paths') or []:
                if reference not in endpoint_images:
                    path = directory/f'offline-endpoint-{len(endpoint_images)}.png'
                    pixels.save(path)
                    endpoint_images[reference] = str(path)
                paths.append(endpoint_images[reference])
            episode['evidence_paths'] = paths
    context = fixture['controller_audit']['judge_request']['context']
    scene = fixture['controller_audit']['judge_request']['scene_context']
    groups = context['object_groups']
    if ending == 'not_scheduled':
        acquisition['probe_results'] = [r for r in acquisition['probe_results'] if r.get('route_scope') != 'cross_group']
    if ending == 'evidence_gap':
        acquisition.pop('functional_measurement_bank',None)
    for probe in acquisition['probe_results']:
        if ending == 'evidence_gap':
            probe.pop('functional_measurements',None)
        if probe.get('route_scope') != 'cross_group':
            continue
        if ending == 'provider_fault':
            failure = classify_failure(ConnectionError('Synthetic camera service failure'),phase='acquisition')
            probe.update(failure=failure,error_type='ConnectionError')
            probe['acquisition_outcome']['failure'] = deepcopy(failure)
        elif ending == 'unknown_failure':
            probe.pop('failure',None)
            probe.pop('acquisition_outcome',None)
        elif ending == 'malformed_certificate':
            probe['acquisition_outcome']['failure']['recoverable_acquisition'] = 'yes'
        elif ending in {'pair_available','partial_packet'}:
            probe.update(status='available',evidence_paths=[str(picture)],error_type=None,error=None)
            if ending == 'pair_available':
                probe.pop('failure',None)
                probe['acquisition_outcome'] = {'schema_version':'acquisition_outcome_v1','packet_incomplete':False}
    model = TerminalModel(ending)
    wrapper = shared.ControlledVLMJudge(shared.OpenAICompatibleVLMJudge(model,max_context_chars=200000,
        evidence_resolution_policy=shared.FALLBACK_POLICY,terminal_evidence_policy=shared.TERMINAL_POLICY),
        control=replace(shared.resolve_vlm_evaluation_control({}), evidence_resolution_policy=shared.FALLBACK_POLICY))
    common = dict(metric_name=METRIC,scene=scene,global_evidence=global_evidence,vlm_judge=wrapper,
        prompt=context.get('prompt'),visual_style_spec=None,authorized_deviations=context.get('authorized_deviations') or [],
        build_judge_request=shared._judge_request,call_judge=shared._call_scene_quality_judge,
        apply_prompt_exemptions=shared._apply_prompt_exemptions,normalize_judgement=shared._normalize_judgement)
    with shared.evidence_policy_scope({'evidence_resolution_policy':shared.FALLBACK_POLICY}):
        specs = _cross_group_relation_episode_specs(acquisition_audit=acquisition,groups=groups,global_paths=global_evidence)
        assert len(specs)==1 and specs[0]['required_check_ids']==['functional_check_029']
        relations = _evaluate_cross_group_relation_scopes(specs=specs,**common)
        if ending == 'evidence_gap' and expect_repaired:
            assert relations[0]['status']=='not_evaluable',relations[0].get('judgement')
            assert relations[0]['score'] is None and not model.contexts
        elif expected:
            assert relations[0]['status']=='evaluated', relations[0].get('judgement')
            resolution = relations[0]['judgement']['evidence_resolution']
            assert set(resolution['images_used'])==set(global_evidence)
            assert resolution['model_judgement_completed'] and resolution['accepted']
            if ending not in {'pair_available','partial_packet'}:
                assert resolution['terminal_evidence_policy']==shared.TERMINAL_POLICY
                assert not resolution['visual_observation_complete']
            assert model.contexts[0]['required_functional_checks'][0]['check_id']=='functional_check_029'
            assert any(r['check_id']=='functional_check_029' for r in model.contexts[0]['functional_measurements']['check_measurements']), (
                ending, model.contexts[0]['functional_measurements'])
            assert (model.contexts[0].get('budget_exhaustion_finalization') or {}).get('trigger_stop_reason') != 'pair_specific_evidence_unavailable'
        else:
            assert relations[0]['status']=='failed', relations[0]
            assert len(model.contexts)==(1 if ending=='terminal_fault' and expect_repaired else 0)
            if expect_repaired and ending in {'provider_fault','unknown_failure','malformed_certificate'}:
                assert relations[0]['judgement']['failure']['phase']=='acquisition'
                assert relations[0]['vlm_invoked'] is False
        # Real reconciliation can perform one bounded second Judge call; it
        # must carry the same acquisition classification and retained evidence.
        relations, reconciliation = reconcile_directional_relation_conflicts(
            specs=specs,relation_results=relations,group_results=p['group_results'],**common)
        p['cross_group_relation_results'] = relations
        p['functional_consistency_reconciliation'] = reconciliation
        p['cross_group_relation_claims'] = [claim_record(METRIC,defect,
            source_phase='cross_group_relation_review:'+record['relation_id'],claim_status='final')
            for record in relations if record.get('score')==0.
            for defect in (record.get('judgement') or {}).get('defects') or []]
        p['functional_check_ledger'],p['functional_check_coverage'] = shared.apply_functional_check_judgements(
            p['functional_check_ledger'],relation_results=relations,group_results=p['group_results'])
        if ending=='invalid' and expected:
            assert next(c for c in p['functional_check_ledger']['checks'] if c['check_id']=='functional_check_029')['check_conclusion']=='invalid'
            assert any('functional_check_029' in d.get('check_refs',[]) for r in relations
                       for d in r['judgement']['defects'])
        prior={c['check_id']:c for c in original['functional_check_ledger']['checks']}
        for check in p['functional_check_ledger']['checks']:
            if check['check_id']!='functional_check_029':
                assert check.get('result_row')==prior[check['check_id']].get('result_row'),check['check_id']
        for key in ('infrastructure_failures','judgement','coverage','component_degradations',
                    'judgement_coverage_projection','resolution_coverage','final_defect_claims',
                    'final_object_findings','score','reason','terminal_state','status'):
            p.pop(key,None)
        ids=manifest['canonical_object_denominator']['ordered_object_ids']
        global_record=p['global_discovery']
        result=shared._aggregate_global_and_group_results(p,metric_name=METRIC,global_record=global_record,
            global_outcome=shared._normalize_judgement(global_record,metric_name=METRIC,valid_object_ids=set(ids)),
            scene_claims=p['global_scene_claims'],relation_claims=p['cross_group_relation_claims'],
            relation_results=relations,relation_phase_complete=all(r['status']=='evaluated' for r in relations),
            group_results=p['group_results'],group_phase_required=True,group_phase_complete=True,
            functional_check_phase_complete=p['functional_check_coverage']['complete'],placement_check_phase_complete=True)
        result=shared._apply_functional_acquisition_budget_status(result,metric_name=METRIC)
        resolved=result['status']=='evaluated'
        result['functional_ownership_ledger']=shared.build_functional_ownership_ledger(scene_object_ids=ids,
            global_record=global_record if resolved else None,relation_results=relations if resolved else [],
            group_results=p['group_results'] if resolved else [],functional_check_ledger=p['functional_check_ledger'] if resolved else None)
        result=shared.finish_metric(result,original['resolution_coverage']['original_planned_scope_ids'])
    l3['metrics'][METRIC]=result
    report=shared.apply_report({'report_schema_version':'scene_evaluation_report_v2','evaluation_status':'completed',
        'layer_reports':{'l1_physical_plausibility':l1,'l3_scene_quality':l3},'reports':{'scene_quality':l3},
        'canonical_object_denominator':manifest['canonical_object_denominator'],'scoring_reliability':{}})
    summary=report['judgement_coverage_summary']
    assert summary['eligible']==expected, {
        'ending':ending,'status':summary['status'],'failures':result.get('infrastructure_failures'),
        'relations':[{k:r.get(k) for k in ('status','reason','judgement')} for r in relations]}
    for metric,projection in summary['metric_projections'].items():
        if metric!=METRIC:
            source=fixture['l1']['metrics'] if metric in fixture['l1']['metrics'] else fixture['l3']['metrics']
            assert projection['observed_score']==source[metric]['judgement_coverage_projection']['observed_score'],metric
    report['runner_outcome']={'uniform_score_acceptance':{k:summary[k] for k in (
        'schema_version','eligible','status','judgement_coverage_fraction','minimum_judgement_coverage','infrastructure_failure_metrics')}}
    scoring=shared.case_scoring_summary(case_id='S100',case_manifest=manifest,l1_report=l1,l3_report=l3)
    compact=shared.compact_report(report,scoring_summary=scoring)
    if expected:
        shared.validate_acceptance(compact)
        assert compact['feedback_scores']['feedback_score'] is not None
        assert scoring['combined_score_100']==report['benchmark_score_100']
    else:
        assert compact['feedback_scores']['feedback_score'] is None
    runner='not_requested'
    if campaign:
        compact['source_report']={'release':'offline-functional-replay','release_manifest_sha256':'offline-fixture'}
        try: campaign.accepted_report(compact)
        except campaign.IncompleteEvaluation:
            assert not expected
            runner='rejected'
        else:
            assert expected
            runner='accepted_offline_binding'
    projection=summary['metric_projections'][METRIC]
    return {'case':'S100','ending':ending,'synthetic_model_responses':True,'live_model_calls':0,
        'mock_model_calls':len(model.contexts),'eligible':summary['eligible'],'status':summary['status'],
        'coverage':summary['judgement_coverage_fraction'],'function_score':projection['observed_score'],
        'function_accepted':projection['accepted_count'],'function_planned':projection['planned_count'],
        'feedback_score':compact['feedback_scores']['feedback_score'],'runner':runner,
        'other_seven_scores_unchanged':True,'other_typed_rows_unchanged':True,
        'reconciliation':reconciliation.get('status'),
        'all_metric_scores':{k:v['observed_score'] for k,v in summary['metric_projections'].items()}}


ENDINGS=('valid','invalid','not_scheduled','no_images','pair_available','partial_packet','evidence_gap',
         'provider_fault','unknown_failure','malformed_certificate','terminal_fault','missing_file')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--runner-scripts',type=Path)
    parser.add_argument('--expect-failure',action='store_true')
    args=parser.parse_args()
    fixture=json.loads((args.fixtures/'S100.json').read_text())
    campaign=shared.campaign_acceptor(args.runner_scripts) if args.runner_scripts else None
    results=[]
    for ending in ('valid',) if args.expect_failure else ENDINGS:
        with tempfile.TemporaryDirectory(prefix='crossgroup-replay-') as temp:
            results.append(replay(fixture,Path(temp),ending=ending,campaign=campaign,expect_repaired=not args.expect_failure))
    if not args.expect_failure:
        # Existing per-object max burden can already charge this endpoint;
        # an additional invalid check must survive but need not double-charge.
        assert results[0]['function_score']>=results[1]['function_score']
        assert results[0]['feedback_score']>=results[1]['feedback_score']
    args.output.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__': main()
