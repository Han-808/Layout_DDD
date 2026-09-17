"""Typed acquisition results survive Function scheduling and terminal review."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

from benchmark.visual_judge.acquisition_outcome import AcquisitionExhausted, AcquisitionOutcome, recorded_acquisition_audit
from benchmark.visual_judge.evidence_gap_v2 import classify_failure
from benchmark.visual_judge.evidence_resolution import FALLBACK_POLICY, evidence_policy_scope
from benchmark.evaluator.scene_quality.cross_group_relations import _merge_relation_probe_results, _cross_group_relation_episode_specs


def probe(state):
    row = {'route_scope':'cross_group','kind':'functional_correspondence',
           'target_ids':['television'],'related_target_ids':['sofa'],
           'relation_predicates':['directional_correspondence'],
           'observation_kinds':['directional_correspondence'],
           'status':'failed','evidence_paths':[]}
    if state=='available':
        row.update(status='available',evidence_paths=['pair.png'])
    elif state=='not_scheduled':
        row.update(status='not_scheduled')
    elif state=='unknown':
        row.update(error_type='AcquisitionExhausted',error='trusted_candidate_bank_empty')
    else:
        error = AcquisitionExhausted('bounded') if state=='gap' else ConnectionError('service down')
        row['failure'] = classify_failure(error,phase='acquisition')
        row['acquisition_outcome'] = {'schema_version':'acquisition_outcome_v1',
                                    'failure':deepcopy(row['failure']),'packet_incomplete':True}
    return row


@pytest.mark.parametrize('state',['gap','not_scheduled','available','hard','unknown'])
def test_probe_result_requires_positive_exhaustion_certificate(state):
    audit=recorded_acquisition_audit(probe(state))
    failure=audit.get('failure')
    if state=='available': assert failure is None
    else: assert failure['recoverable_acquisition'] == (state in {'gap','not_scheduled'})
    if state in {'hard','unknown'}:
        with pytest.raises(Exception): AcquisitionOutcome([],audit).raise_if_failed()
    else: AcquisitionOutcome([],audit).raise_if_failed()


@pytest.mark.parametrize('first',['gap','not_scheduled','available','hard','unknown'])
@pytest.mark.parametrize('second',['gap','not_scheduled','available','hard','unknown'])
def test_retry_merge_never_loses_hard_failure_or_stale_gap(first,second):
    with evidence_policy_scope({'evidence_resolution_policy':FALLBACK_POLICY}):
        merged=_merge_relation_probe_results(probe(first),probe(second))
        audit=recorded_acquisition_audit(merged)
    hard=bool({first,second}&{'hard','unknown'})
    failure=audit.get('failure')
    if hard:
        assert failure and not failure['recoverable_acquisition']
    elif 'available' in {first,second}:
        assert failure is None and merged['evidence_paths']==['pair.png']
    else:
        assert failure['recoverable_acquisition']


@pytest.mark.parametrize('broken',[
    'not-an-object', {}, {'phase':'acquisition','failure_category':'model_service_failure','recoverable_acquisition':True},
    {'phase':'acquisition','failure_category':'evidence_unavailable','recoverable_acquisition':'true'},
    {'phase':'judge','failure_category':'evidence_unavailable','recoverable_acquisition':True}])
def test_malformed_certificate_cannot_turn_failed_acquisition_into_a_pass(broken):
    row=probe('gap')
    row['acquisition_outcome']['failure']=broken
    assert recorded_acquisition_audit(row)['failure']['recoverable_acquisition'] is False


@pytest.mark.parametrize('state',['gap','not_scheduled','available','hard','unknown'])
def test_schedule_and_judge_packet_keep_the_same_classification(state):
    row=probe(state)
    audit={'probe_results':[] if state=='not_scheduled' else [row],
        'functional_discovery':{'cross_group_correspondences':[{
            'discovery_id':'pair','target_ids':['television','sofa'],
            'predicate':'directional_correspondence','observation_kinds':['directional_correspondence'],
            'observation_goal':'Inspect ordinary viewing correspondence.'}]}}
    with evidence_policy_scope({'evidence_resolution_policy':FALLBACK_POLICY}):
        specs=_cross_group_relation_episode_specs(acquisition_audit=audit,
            groups=[{'group_id':'media','object_ids':['television']},{'group_id':'seating','object_ids':['sofa']}],
            global_paths=['global.png'])
    assert len(specs)==1
    spec=specs[0]
    assert spec['acquisition_outcome']==spec['judge_packet']['acquisition_outcome']
    assert spec['acquisition_outcome'].get('failure')==recorded_acquisition_audit(row).get('failure')


@pytest.mark.parametrize('route',['group_local','cross_group'])
@pytest.mark.parametrize('fault',['gap','service'])
def test_public_function_prejudgement_preserves_failure_boundary(tmp_path,route,fault):
    from PIL import Image
    from test_evidence_adaptive_judgement import Model, complete_required_rows
    from test_scene_quality_interfaces import _scene
    from test_r13_best_effort_terminal import judge
    from test_postrun_policy_boundary import v2_control
    from benchmark.visual_judge.adapters.legacy_judge import ControlledVLMJudge
    from benchmark.evaluator.scene_quality import evaluate_scene_quality_interfaces, SCENE_QUALITY_INTERFACE_METRICS
    path=tmp_path/'global.png'
    pixels=Image.new('RGB',(16,16),'white')
    pixels.putpixel((0,0),(0,0,0))
    pixels.save(path)
    scene=_scene()
    ids=[o['id'] for o in scene['objects']]
    groups=([{'group_id':'work','object_ids':ids}] if route=='group_local' else
            [{'group_id':f'g{i}','object_ids':[object_id]} for i,object_id in enumerate(ids)])
    class Planner:
        def discover_functional_evidence(self,req):
            relation={'discovery_id':'use_pair','target_ids':ids,'predicate':'relative_use_geometry',
                      'observation_kinds':['relative_use_geometry'],'observation_goal':'Inspect ordinary joint use.'}
            return {'schema_version':'functional_discovery_v3','inspected_object_ids':ids,
                'directed_surface_targets':[{
                    'discovery_id':'chair_surface','target_id':ids[0],
                    'owning_group_id':'work','surface_roles':['seating_side'],
                    'need_clearance':False,'observation_goal':'Inspect ordinary seating side.'
                }] if route=='group_local' else [],
                'within_group_correspondences':[dict(relation,owning_group_id='work')] if route=='group_local' else [],
                'cross_group_correspondences':[dict(relation,group_ids=['g0','g1'])] if route=='cross_group' else [],
                'approach_clearance_targets':[],'boundary_sensitive_targets':[],'unusual_unconfirmed':[],
                'reason':'Offline fixture','provenance':{}}
    calls=[]
    def provider(req):
        calls.append(req)
        if fault=='service': raise ConnectionError('Synthetic provider failure')
        raise AcquisitionExhausted('bounded')
    model=Model(complete_required_rows)
    wrapper=ControlledVLMJudge(judge(model),control=v2_control())
    report=evaluate_scene_quality_interfaces(scene,
        config={'enabled':True,'metrics':{m:{'enabled':m=='functional_consistency'} for m in SCENE_QUALITY_INTERFACE_METRICS}},
        object_grouping_report={'object_groups':groups},render_evidence={'global':[str(path)]},
        camera_evidence_provider=lambda req:[str(path)],functional_evidence_planner=Planner(),
        functional_probe_evidence_provider=provider,vlm_judge=wrapper,
        metric_applicability={'functional_consistency':{'applicability':'relevant'}})
    metric=report['metrics']['functional_consistency']
    assert calls, metric.get('reason')
    if fault=='service':
        assert metric['status']=='failed' and not model.calls, (metric['status'],len(model.calls))
        assert metric['failure']['failure_category']=='model_service_failure'
        assert metric['functional_probe_acquisition']['failure']==metric['failure']
    else:
        assert metric['status']=='evaluated', {k:metric.get(k) for k in ('reason','judgement','infrastructure_failures')}
        assert model.calls


@pytest.mark.requires_local_data
@pytest.mark.parametrize('ending',[
    'valid','invalid','not_scheduled','no_images','pair_available','partial_packet','evidence_gap',
    'provider_fault','unknown_failure','malformed_certificate','terminal_fault','missing_file'])
def test_actual_crossgroup_failure_through_full_scoring(tmp_path,ending):
    directory=os.environ.get('CROSSGROUP_REPLAY_FIXTURES')
    if not directory:
        pytest.skip('Set CROSSGROUP_REPLAY_FIXTURES to preserved attempt04 inputs')
    scripts=Path(__file__).resolve().parents[1]/'scripts'
    sys.path.insert(0,str(scripts))
    try:
        spec=importlib.util.spec_from_file_location('crossgroup_offline_replay',scripts/'replay_crossgroup_scoring.py')
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result=module.replay(json.loads((Path(directory)/'S100.json').read_text()),tmp_path,ending=ending)
    finally:
        sys.path.remove(str(scripts))
    assert result['eligible']==(ending in module.SUCCESS or ending=='evidence_gap')
    assert result['other_seven_scores_unchanged'] and result['other_typed_rows_unchanged']
