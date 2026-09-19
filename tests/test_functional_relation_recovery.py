"""No paid model/render calls: failed relation inventories cannot certify absence."""
from copy import deepcopy
import json
import os
from pathlib import Path

from PIL import Image
import pytest

from benchmark.evaluator.judgement_coverage import metric_projection
from benchmark.evaluator.scene_quality.consistency_acceptance_v2 import finish_metric
from benchmark.evaluator.scene_quality.functional_acquisition import build_functional_acquisition_plan
from benchmark.evaluator.scene_quality.functional_probe import acquire_functional_probe_evidence
from benchmark.evaluator.scene_quality.terminal import recoverable_validation_failure
from benchmark.visual_judge.evidence_gap_v2 import classify_failure
from benchmark.visual_judge.evidence_resolution import FALLBACK_POLICY, evidence_policy_scope
from benchmark.visual_judge.functional_discovery import discover_openai_compatible_functional_evidence
from benchmark.visual_judge.functional_relation_recovery import (
    FunctionalRelationInventoryError, complete_relation_inventory, relation_inventory_issues,
    unresolved_inventory_units,
)
from benchmark.visual_judge.openai_camera_selector import OpenAICompatibleCameraSelector


OBJECTS = [{"id":"chair","category":"chair"},{"id":"table","category":"table"}]
ROW = {"target_ids":["chair","table"],"predicate":"directional_correspondence",
    "dependency":"required","counterpart_mode":"shared","ordinary_mobility":"fixed",
    "observation_goal":"Inspect assigned joint-use facing."}


class Model:
    max_tokens = 2048
    response_format_json = True

    def __init__(self, answers):
        self.answers=list(answers); self.calls=[]; self.last_request_metadata={}

    def chat_messages(self, messages, **kwargs):
        self.calls.append({"messages":deepcopy(messages),**kwargs})
        answer=self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        raw=json.dumps(answer) if isinstance(answer,dict) else answer
        self.last_request_metadata={"finish_reason":"length" if not raw else "stop",
            "content_chars":len(raw),"max_tokens":kwargs["max_tokens"]}
        return raw


def inventory(rows=None):
    return {"considered_object_ids":["chair","table"],"relations":[deepcopy(ROW)] if rows is None else rows,
        "reason":"Completed all-object joint-use inspection."}


def affordance():
    return {"objects":[{"object_id":o["id"],"directionality":"directed" if o["id"]=="chair" else "non_directed",
        "surface_roles":["seating_side"] if o["id"]=="chair" else [],"need_clearance":False,
        "boundary_review_state":"routine","review_state":"routine",
        "observation_goal":"Inspect ordinary use.","boundary_observation_goal":""} for o in OBJECTS],
        "reason":"All objects inspected."}


def request(tmp_path):
    image=tmp_path/'global.png'
    pixels=Image.new('RGB',(16,16),'white')
    pixels.putpixel((0,0),(0,0,0))
    pixels.save(image)
    return {"metric":"functional_consistency","scene_id":"test","global_image_path":str(image),
        "objects":deepcopy(OBJECTS),"groups":[{"group_id":"dining","object_ids":["chair","table"]}]}


@pytest.mark.parametrize('rows',[[],[ROW]])
def test_valid_initial_empty_or_nonempty_is_unchanged(tmp_path,rows):
    model=Model([affordance(),inventory(deepcopy(rows))])
    discovery=discover_openai_compatible_functional_evidence(model=model,request=request(tmp_path))
    assert len(model.calls)==2 and not relation_inventory_issues(discovery)
    assert discovery['coverage']['complete']


@pytest.mark.parametrize('repair',['',inventory()])
def test_empty_initial_cannot_erase_real_relations_from_repair_or_new_inventory(tmp_path,repair):
    model=Model([affordance(),'',repair,inventory()])
    req=request(tmp_path)
    discovery=discover_openai_compatible_functional_evidence(model=model,request=req)
    assert len(model.calls)==4
    assert model.calls[-1]['messages'][:-1]==model.calls[1]['messages']
    assert model.calls[-1]['max_tokens']==16384
    assert model.calls[-1]['call_type'].endswith('.inventory_recovery')
    audit=discovery['provenance']['calls']['relations']['schema_validation']
    assert audit['inventory_recovery']['complete'] and audit['attempt_count']==3
    assert audit['recovered']==bool(repair)  # Old repair audit is not rewritten.
    assert not relation_inventory_issues(discovery)
    assert discovery['coverage']['complete']
    assert len(discovery['within_group_correspondences'])==1
    plan=build_functional_acquisition_plan(discovery,max_probe_units=32,groups=req['groups'])
    assert any(c['check_type']=='directional_correspondence' for c in plan['functional_check_ledger']['checks'])


@pytest.mark.parametrize('answer',['',{},inventory([])])
def test_new_complete_no_relations_requires_real_valid_model_inventory(tmp_path,answer):
    model=Model([affordance(),'','',answer])
    if answer==inventory([]):
        result=discover_openai_compatible_functional_evidence(model=model,request=request(tmp_path))
        assert result['coverage']['complete'] and not relation_inventory_issues(result)
    else:
        with pytest.raises(FunctionalRelationInventoryError) as error:
            discover_openai_compatible_functional_evidence(model=model,request=request(tmp_path))
        assert not recoverable_validation_failure(error.value)
        assert classify_failure(error.value,phase='discovery')['failure_category']=='judge_response_failure'
    assert len(model.calls)==4


@pytest.mark.parametrize('failure,category',[
    (ConnectionError('offline'), 'model_service_failure'),
    (RuntimeError('program bug'), 'implementation_or_input_failure'),
    (FileNotFoundError('missing'), 'input_integrity_failure'),
])
def test_third_service_program_or_input_failure_never_becomes_empty_inventory(tmp_path,failure,category):
    model=Model([affordance(),'','',failure])
    with pytest.raises(FunctionalRelationInventoryError) as caught:
        discover_openai_compatible_functional_evidence(model=model,request=request(tmp_path))
    assert not recoverable_validation_failure(caught.value)
    assert not classify_failure(caught.value,phase='discovery')['recoverable_acquisition']
    assert classify_failure(caught.value,phase='discovery')['failure_category'] == category
    assert caught.value.__cause__ is failure and len(model.calls)==4


@pytest.mark.parametrize('mutation',['delete','change_role','change_predicate','valid'])
def test_third_call_preserves_retained_and_pending_atoms(mutation):
    pending={**ROW,'predicate':'relative_use_geometry'}
    salvage={'consideration_contract_valid':False,'dropped_relation_anchors':[
        {'target_ids':['chair','table'],'predicate':'relative_use_geometry'}],'dropped_relation_count':1}
    relations={'relations':[deepcopy(ROW)],'item_salvage':salvage}
    answer=inventory([deepcopy(ROW),deepcopy(pending)])
    if mutation=='delete': answer['relations'].pop()
    if mutation=='change_role': answer['relations'][0]['target_ids'].reverse()
    if mutation=='change_predicate': answer['relations'][0]['dependency']='contextual'
    model=Model([answer]); audit={'attempts':[{},{}],'salvage':deepcopy(salvage)}
    kwargs=dict(model=model,normalized={'objects':deepcopy(OBJECTS)},messages=[{'role':'user','content':'same evidence'}],
        initial_metadata={},relations=relations,audit=audit,response_format_json=True)
    if mutation=='valid':
        result,new_audit=complete_relation_inventory(**kwargs)
        assert result['relations']==answer['relations'] and new_audit['inventory_recovery']['complete']
    else:
        with pytest.raises(FunctionalRelationInventoryError): complete_relation_inventory(**kwargs)
    assert len(model.calls)==1


def broken_discovery():
    return {'coverage':{'relations':{'consideration_contract_valid':False,
        'anchored_relation_count':0,'accepted_relation_count':0}},
        'provenance':{'calls':{'relations':{'finish_reason':'length'}}}}


def test_hidden_discovery_gap_cannot_enter_uniform_projection():
    report={'functional_discovery':broken_discovery(),'resolution_coverage':{
        'planned_ids':['scope'],'units':[{'unit_id':'scope','accepted':True}]}}
    with pytest.raises(ValueError,match='unresolved relation discovery'):
        metric_projection('functional_consistency',report,['chair','table'])


def test_failure_obligations_retained_in_final_coverage_no_valid_default():
    er={'accepted':True,'model_judgement_completed':True,'decision_source':'model','visual_observation_complete':False}
    report={'status':'evaluated','score':1.,'functional_discovery':broken_discovery(),
        'global_discovery':{'verdict':'valid','defects':[],'evidence_resolution':er}}
    report=finish_metric(report,['scene_global'])
    required=unresolved_inventory_units(report)
    assert required and all(u in report['resolution_coverage']['units'] for u in required)
    assert report['status']=='failed' and report['score'] is None
    result=metric_projection('functional_consistency',report,['chair','table'])
    assert result['infrastructure_failure'] and result['observed_score'] is None
    assert result['judgement_fraction']<1


def test_public_acquisition_does_not_run_judges_after_inventory_failure(tmp_path):
    req=request(tmp_path)
    model=Model([affordance(),'','',''])
    selector=OpenAICompatibleCameraSelector(model)
    scene={'scene_id':'test','scene_type':'room','objects':[{
        **o,'center':[0,0,1],'size':[1,1,1],'rotation':[0,0,0]} for o in OBJECTS]}
    with evidence_policy_scope({'evidence_resolution_policy':FALLBACK_POLICY}):
        paths,audit=acquire_functional_probe_evidence(planner=selector,provider=None,scene=scene,
            global_image_path=req['global_image_path'],groups=req['groups'])
    assert paths==[] and audit['status']=='failed'
    assert audit['failure']['failure_category']=='judge_response_failure'
    assert audit['fallback']['base_group_and_global_judges_continue'] is False
    assert audit['response_schema_validation']['inventory_recovery']['complete'] is False


@pytest.mark.parametrize('broken',[
    {'considered_object_ids':['chair','table'],'reason':'Missing relations field'},
    {'considered_object_ids':['chair','table'],'relations':None,'reason':'Invalid list'},
])
def test_correct_considered_ids_do_not_hide_malformed_relation_envelope(tmp_path,broken):
    model=Model([affordance(),broken,broken,inventory()])
    result=discover_openai_compatible_functional_evidence(model=model,request=request(tmp_path))
    assert len(model.calls)==4 and not relation_inventory_issues(result)
    assert result['provenance']['calls']['relations']['schema_validation']['inventory_recovery']['complete']


def test_public_metric_preserves_inventory_failure_before_any_judge(tmp_path):
    from test_evidence_adaptive_judgement import Model as JudgeModel, complete_required_rows
    from test_scene_quality_interfaces import _scene
    from test_r13_best_effort_terminal import judge
    from test_postrun_policy_boundary import v2_control
    from benchmark.visual_judge.adapters.legacy_judge import ControlledVLMJudge
    from benchmark.evaluator.scene_quality import evaluate_scene_quality_interfaces, SCENE_QUALITY_INTERFACE_METRICS
    req=request(tmp_path); scene=_scene(); ids=[o['id'] for o in scene['objects']]
    class Planner:
        def discover_functional_evidence(self, request):
            raise FunctionalRelationInventoryError('Unresolved complete inventory',schema_audit={
                'attempt_count':3,'attempts':[{}, {}, {}], 'inventory_recovery':{'complete':False}})
    model=JudgeModel(complete_required_rows)
    report=evaluate_scene_quality_interfaces(scene,
        config={'enabled':True,'metrics':{m:{'enabled':m=='functional_consistency'} for m in SCENE_QUALITY_INTERFACE_METRICS}},
        object_grouping_report={'object_groups':[{'group_id':'work','object_ids':ids}]},
        render_evidence={'global':[req['global_image_path']]},
        camera_evidence_provider=lambda r:[req['global_image_path']],functional_evidence_planner=Planner(),
        functional_probe_evidence_provider=lambda r:[req['global_image_path']],
        vlm_judge=ControlledVLMJudge(judge(model),control=v2_control()),
        metric_applicability={'functional_consistency':{'applicability':'relevant'}})
    function=report['metrics']['functional_consistency']
    assert function['status']=='failed' and not model.calls
    assert function['failure']['failure_category']=='judge_response_failure'
    assert function['resolution_coverage']['failure_count']>0


def test_legacy_score_grounding_cannot_ignore_unresolved_discovery_when_typed_checks_exist():
    from benchmark.evaluator.scene_quality.global_group_first import _score_grounding_coverage
    result=_score_grounding_coverage(global_scope={},relation_results=[],group_results=[],target_results=[],
        functional_discovery=broken_discovery(),functional_probe=None,placement_discovery=None,
        functional_check_coverage={'required_check_count':24,'grounded_check_count':24},placement_check_coverage=None)
    assert result['fraction']<1 and not result['complete']
    assert any(u['unit_type']=='unresolved_discovery_contract' for u in result['judge_units'])


def test_preserved_s100_fixture_exposes_gap_without_mutating_it():
    path=os.environ.get('RELATION_DISCOVERY_FIXTURE')
    if not path: pytest.skip('Independent preserved real-input fixture not configured')
    fixture=json.loads(Path(path).read_text())
    before=deepcopy(fixture)
    assert relation_inventory_issues(fixture['discovery'])
    report={'functional_discovery':fixture['discovery'],'resolution_coverage':fixture['resolution_coverage']}
    with pytest.raises(ValueError,match='unresolved relation discovery'):
        metric_projection('functional_consistency',report,['chair','table'])
    assert fixture==before


@pytest.mark.parametrize('route', ['group_local', 'cross_group'])
@pytest.mark.parametrize('ending', ['valid', 'invalid'])
def test_recovered_relation_reaches_real_judge_and_burden(tmp_path, route, ending):
    from test_evidence_adaptive_judgement import Model as JudgeModel, complete_required_rows
    from test_scene_quality_interfaces import _scene
    from test_r13_best_effort_terminal import judge
    from test_postrun_policy_boundary import v2_control
    from benchmark.visual_judge.acquisition_outcome import AcquisitionExhausted
    from benchmark.visual_judge.adapters.legacy_judge import ControlledVLMJudge
    from benchmark.evaluator.scene_quality import evaluate_scene_quality_interfaces, SCENE_QUALITY_INTERFACE_METRICS

    req = request(tmp_path)
    scene = _scene()
    for obj, identity in zip(scene['objects'], ['chair', 'table']):
        obj['id'] = identity
    groups = (req['groups'] if route == 'group_local' else [
        {'group_id': 'g0', 'object_ids': ['chair']},
        {'group_id': 'g1', 'object_ids': ['table']}])
    aff = affordance()
    for row in aff['objects']:
        row.update(directionality='non_directed', surface_roles=[])
    relation = {**ROW, 'predicate': 'relative_use_geometry'}
    planner_model = Model([aff, '', '', inventory([relation])])
    seen_checks = []

    def respond(messages):
        result = complete_required_rows(messages)
        context = json.loads(messages[1]['content'][0]['text'].split('\n', 1)[1])
        required = context.get('required_functional_checks') or []
        for check in required:
            if check['check_type'] != 'relative_use_geometry':
                continue
            seen_checks.append(deepcopy(check))
            row = next(r for r in result['functional_check_results'] if r['check_id'] == check['check_id'])
            row['conclusion'] = ending
            if ending == 'invalid':
                result['verdict'] = 'invalid'
                result['defects'].append({
                    'scope': 'facing_and_interaction_direction', 'target_ids': ['chair'],
                    'relation': 'relative_use_geometry', 'category': 'functional_correspondence_failure',
                    'severity': 'impaired', 'attribution_mode': 'responsible_endpoint',
                    'check_refs': [check['check_id']], 'reason': 'Synthetic joint-use defect.'})
        return result

    probes = []

    def provider(probe):
        probes.append(deepcopy(probe))
        raise AcquisitionExhausted('bounded')

    judge_model = JudgeModel(respond)
    result = evaluate_scene_quality_interfaces(scene,
        config={'enabled': True, 'metrics': {m: {'enabled': m == 'functional_consistency'}
            for m in SCENE_QUALITY_INTERFACE_METRICS}},
        object_grouping_report={'object_groups': groups}, render_evidence={'global': [req['global_image_path']]},
        camera_evidence_provider=lambda r: [req['global_image_path']],
        functional_evidence_planner=OpenAICompatibleCameraSelector(planner_model),
        functional_probe_evidence_provider=provider,
        vlm_judge=ControlledVLMJudge(judge(judge_model), control=v2_control()),
        metric_applicability={'functional_consistency': {'applicability': 'relevant'}})
    metric = result['metrics']['functional_consistency']
    assert metric['status'] == 'evaluated', json.dumps({k: metric.get(k) for k in (
        'reason', 'infrastructure_failures', 'global_discovery', 'functional_check_ledger')}, default=str)
    assert len(planner_model.calls) == 4 and seen_checks
    if route == 'cross_group':
        assert probes
    checks = [c for c in metric['functional_check_ledger']['checks'] if c['check_type'] == 'relative_use_geometry']
    assert len(checks) == 1 and checks[0]['check_conclusion'] == ending
    assert checks[0]['result_row']['observation_status'] == 'inferred_under_budget'
    projection = metric_projection('functional_consistency', metric, ['chair', 'table'])
    assert not projection['infrastructure_failure'] and projection['judgement_fraction'] == 1
    assert projection['observed_score'] == (1 if ending == 'valid' else .6)
    assert not relation_inventory_issues(metric['functional_discovery'])


def test_preserved_empty_raw_candidates_require_new_real_inventory(tmp_path):
    path = os.environ.get('RELATION_DISCOVERY_FIXTURE')
    if not path:
        pytest.skip('Independent preserved real-input fixture not configured')
    fixture = json.loads(Path(path).read_text())
    attempts = fixture['discovery']['provenance']['calls']['relations']['schema_validation']['attempts']
    assert len(attempts) == 2 and all(a['raw_response'] == '' for a in attempts)
    _replay_preserved_discovery(fixture, tmp_path)


@pytest.mark.parametrize('trial', ['shared_trial_00', 'none_trial_01', 'none_trial_02', 'diagnostics_trial_01'])
def test_preserved_valid_schema_repair_does_not_erase_unanchored_relations(tmp_path, trial):
    directory = os.environ.get('RELATION_DISCOVERY_FIXTURES')
    if not directory:
        pytest.skip('Independent preserved real-input fixtures not configured')
    fixture = json.loads((Path(directory) / ('batch_01_S100_' + trial + '.json')).read_text())
    schema = fixture['discovery']['provenance']['calls']['relations']['schema_validation']
    assert schema['recovered'] and schema['salvage']['rejected_relation_count'] > 0
    assert relation_inventory_issues(fixture['discovery'])
    _replay_preserved_discovery(fixture, tmp_path)


def _replay_preserved_discovery(fixture, tmp_path):
    discovery = fixture['discovery']
    context = discovery['provenance']['calls']['relations']['input_contract']['structured_context']
    ids = [o['id'] for o in context['object_list']]
    # Affordance is reconstructed from the actual normalized ledger; the third
    # inventory and image pixels are SYNTHETIC, not a new real S100 judgement.
    aff = {'objects': deepcopy(discovery['object_affordance_ledger']), 'reason': 'Reconstructed fixture ledger.'}
    relation = {**ROW, 'target_ids': ['dining_chair_east_north', 'dining_table']}
    final = {'considered_object_ids': ids, 'relations': [relation], 'reason': 'Synthetic complete review.'}
    attempts = discovery['provenance']['calls']['relations']['schema_validation']['attempts']
    assert len(attempts) == 2
    model = Model([aff, *[a['raw_response'] for a in attempts], final])
    req = request(tmp_path)
    req.update(scene_id=context['scene_id'], objects=context['object_list'], groups=context['trusted_group_partition'])
    result = discover_openai_compatible_functional_evidence(model=model, request=req)
    assert len(model.calls) == 4 and model.calls[-1]['messages'][:-1] == model.calls[1]['messages']
    assert not relation_inventory_issues(result) and result['coverage']['complete']
    plan = build_functional_acquisition_plan(result, max_probe_units=32, groups=req['groups'])
    assert any(c['check_type'] == 'directional_correspondence' and set(c['target_ids']) == set(relation['target_ids'])
        for c in plan['functional_check_ledger']['checks'])


def test_valid_json_with_length_finish_is_not_complete_inventory(tmp_path):
    class TruncatedModel(Model):
        def chat_messages(self, messages, **kwargs):
            raw = super().chat_messages(messages, **kwargs)
            if kwargs['call_type'].endswith('.inventory_recovery'):
                self.last_request_metadata['finish_reason'] = 'length'
            return raw
    model = TruncatedModel([affordance(), '', '', inventory()])
    with pytest.raises(FunctionalRelationInventoryError):
        discover_openai_compatible_functional_evidence(model=model, request=request(tmp_path))
    assert len(model.calls) == 4
