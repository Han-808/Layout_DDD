"""Unbound Function exclusions require a new model decision, never a default."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

from benchmark.evaluator.scene_quality.placement_checks import placement_check_id
from benchmark.visual_judge.contracts import ResponseSchemaRepairError
from benchmark.visual_judge.evidence_resolution import FALLBACK_POLICY, ADAPTIVE_POLICY
from test_evidence_adaptive_judgement import Model
from test_placement_audit_contract import answer, check, judge_request, ownership, picture
from test_r13_best_effort_terminal import judge


def inputs(kind='proposal', *, independent=True):
    req = judge_request()
    req.update(evidence_resolution_policy=FALLBACK_POLICY)
    other = {**req['objects'][0], 'id':'other'}
    req['objects'].append(other)
    req['scene_summary']['objects'].append(other)
    req['selected_object_ids'] = ['chair','other']
    events = [{**ownership()[0], 'affected_object_ids':['other'],
               'causal_object_ids':['other'], 'scoring_target_ids':['other']}]
    req['functional_ownership_ledger'].update(events=events, event_count=1)
    initial = answer('excluded_function_owned')
    row = initial['placement_check_results'][0]
    if kind == 'proposal':
        row.pop('check_id')
        row.update(proposal_id='new-chair', check_type='scene_zone', severity='atypical',
                   observation_goal='Inspect location under the original Placement rubric.')
        initial['judge_originated_placement_results'] = [row]
        initial['placement_check_results'] = []
        req['required_placement_checks'] = []
        pending_id = placement_check_id('scene_zone','chair',[])
    else:
        pending_id = 'zone'
    if independent:
        req['required_placement_checks'].append({**check(), 'check_id':'independent', 'subject_id':'other'})
        other_answer = answer('invalid')
        other_row = other_answer['placement_check_results'][0]
        other_row.update(check_id='independent',subject_id='other')
        initial['placement_check_results'].append(other_row)
        other_defect = other_answer['defects'][0]
        other_defect.update(check_id='independent',target_ids=['other'])
        initial.update(verdict='invalid',defects=[other_defect])
    second = deepcopy(initial)
    if kind == 'proposal':
        second['judge_originated_placement_results'] = []
    else:
        second['placement_check_results'] = [r for r in second['placement_check_results'] if r['check_id']!='zone']
    return req, initial, second, pending_id


def final_answer(check_id, conclusion='valid'):
    final = answer(conclusion)
    row = final['placement_check_results'][0]
    row.update(check_id=check_id, observation_status='inferred_under_budget')
    for defect in final['defects']:
        defect['check_id'] = check_id
    return final


@pytest.mark.parametrize('kind',['proposal','required'])
@pytest.mark.parametrize('independent',[False,True])
@pytest.mark.parametrize('conclusion',['valid','invalid'])
def test_new_decision_retains_pending_and_independent_obligations(picture,kind,independent,conclusion):
    req,first,second,cid = inputs(kind,independent=independent)
    req['render_evidence'] = [picture]
    frozen = deepcopy(first)
    model = Model([first,second,final_answer(cid,conclusion)])
    result = judge(model)._adjudicate_scene_quality_raw(req)
    assert first == frozen
    assert len(model.calls)==3
    assert model.calls[2][:-1] == model.calls[0]  # Identical original evidence packet.
    assert 'ONE NEW, same-evidence Placement judgement' in model.calls[2][-1]['content']
    rows = {r['check_id']:r for r in result['placement_check_results']}
    assert set(rows)=={cid,*(['independent'] if independent else [])}
    assert rows[cid] == final_answer(cid,conclusion)['placement_check_results'][0]
    if independent:
        assert rows['independent']==first['placement_check_results'][-1]
        assert result['defects'][0]==first['defects'][0]
    assert result['verdict']==('invalid' if independent or conclusion=='invalid' else 'valid')
    audit=result['request_metadata']['response_schema_validation']
    assert audit['attempt_count']==3 and audit['repair_retry_count']==1 and audit['decision_retry_count']==1
    assert audit['recovered'] and 'function_reference_repairs' not in audit
    assert audit['ownership_decision_retry']['pending_check_ids']==[cid]
    assert result['evidence_resolution']['decision_source']=='model'
    if kind=='proposal':
        registration=result['judge_originated_placement_check_registrations'][0]
        assert registration['check_id']==cid and registration['check_conclusion']==conclusion


@pytest.mark.parametrize('mutation',['delete','add','duplicate','retarget','context','exclusion','missing',
    'unresolved','independent','defect','prose_only','unknown','bad_confidence','empty_reason',
    'mismatched_envelope','service'])
def test_atomic_retry_stays_strict_and_bounded(picture,mutation):
    req,first,second,cid=inputs()
    req['render_evidence']=[picture]
    final=final_answer(cid)
    row=final['placement_check_results'][0]
    if mutation=='delete': final['placement_check_results']=[]
    elif mutation=='add': final['placement_check_results'].append({**row,'check_id':'invented'})
    elif mutation=='duplicate': final['placement_check_results'].append(deepcopy(row))
    elif mutation=='retarget': row['subject_id']='other'
    elif mutation=='context': row['context_ids']=['other']
    elif mutation=='exclusion': row.update(conclusion='excluded_function_owned',function_event_ref='function-1',same_physical_event=True)
    elif mutation=='missing': row['observation_status']='missing'
    elif mutation=='unresolved': row['conclusion']='unresolved'
    elif mutation=='independent': final['placement_check_results'].append(deepcopy(first['placement_check_results'][0]))
    elif mutation=='defect': row['conclusion']='invalid'; final['verdict']='invalid'
    elif mutation=='prose_only': final.pop('placement_check_results')
    elif mutation=='unknown': final['judge_originated_placement_results']=[]
    elif mutation=='bad_confidence': final['confidence']=float('nan')
    elif mutation=='empty_reason': final['reason']=''
    elif mutation=='mismatched_envelope': final['verdict']='invalid'
    elif mutation=='service': final=ConnectionError('offline new decision service failure')
    model=Model([first,second,final])
    with pytest.raises(ResponseSchemaRepairError) as caught:
        judge(model)._adjudicate_scene_quality_raw(req)
    assert len(model.calls)==3
    assert not caught.value.schema_audit['recovered']
    assert caught.value.schema_audit['decision_retry_count']==1
    assert caught.value.schema_audit['attempts'][-1]['failure_kind']==('transport' if mutation=='service' else 'validation')


@pytest.mark.parametrize('blocker',['legal_binding','missing_ledger','initial_transport','repair_transport','unrelated_initial_fault','legacy_policy'])
def test_no_new_decision_for_ineligible_failure(picture,blocker):
    req,first,second,cid=inputs()
    req['render_evidence']=[picture]
    if blocker=='legal_binding': req['functional_ownership_ledger']['events']=ownership()
    elif blocker=='missing_ledger': req.pop('functional_ownership_ledger')
    elif blocker=='initial_transport': first=ConnectionError('offline initial')
    elif blocker=='repair_transport': second=ConnectionError('offline schema repair')
    elif blocker=='unrelated_initial_fault': first['placement_check_results'][0]['subject_id']='invented'
    elif blocker=='legacy_policy': req['evidence_resolution_policy']=ADAPTIVE_POLICY
    if blocker=='legal_binding':
        # Keep the first reference invalid, but give the model a legal alternative.
        first['judge_originated_placement_results'][0]['function_event_ref']='unknown'
    model=Model([first,second,final_answer(cid)])
    with pytest.raises(Exception): judge(model)._adjudicate_scene_quality_raw(req)
    assert len(model.calls)==(1 if blocker=='initial_transport' else 2)


def test_multiple_pending_decisions_preserve_a_legal_independent_exclusion(picture):
    req,first,second,cid=inputs()
    req['render_evidence']=[picture]
    third={**req['objects'][0],'id':'third'}
    req['objects'].append(third)
    req['scene_summary']['objects'].append(third)
    req['selected_object_ids'].append('third')
    first['judge_originated_placement_results'].append({
        **first['judge_originated_placement_results'][0],'subject_id':'third','proposal_id':'third-zone'})
    legal=first['placement_check_results'][0]
    legal.update(conclusion='excluded_function_owned',function_event_ref='function-1',same_physical_event=True)
    first.update(verdict='valid',defects=[])
    second=deepcopy(first)
    second['judge_originated_placement_results']=[]
    final=final_answer(cid)
    third_id=placement_check_id('scene_zone','third',[])
    final['placement_check_results'].append({**final['placement_check_results'][0],
        'check_id':third_id,'subject_id':'third'})
    model=Model([first,second,final])
    result=judge(model)._adjudicate_scene_quality_raw(req)
    rows={r['check_id']:r for r in result['placement_check_results']}
    assert len(model.calls)==3 and set(rows)=={cid,third_id,'independent'}
    assert rows['independent']==legal
    assert result['verdict']=='valid' and not result['defects']
    assert len(result['judge_originated_placement_check_registrations'])==2


@pytest.mark.requires_local_data
@pytest.mark.parametrize('ending',['valid','invalid','delete_pending','false_exclusion',
                                   'alter_independent','service_failure'])
def test_real_s101_failure_through_full_scoring(tmp_path,ending):
    directory=os.environ.get('PLACEMENT_OWNERSHIP_FIXTURES')
    if not directory: pytest.skip('Set PLACEMENT_OWNERSHIP_FIXTURES to preserved S101 C-R1 inputs')
    scripts=Path(__file__).resolve().parents[1]/'scripts'
    sys.path.insert(0,str(scripts))
    try:
        spec=importlib.util.spec_from_file_location('placement_ownership_replay',scripts/'replay_placement_ownership_scoring.py')
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result=module.replay(json.loads((Path(directory)/'S101_diagnostics_R1.json').read_text()),tmp_path,ending=ending)
    finally:
        sys.path.remove(str(scripts))
    assert result['eligible']==(ending in {'valid','invalid'})
    assert result['model_calls']==3 and result['live_model_calls']==0
    assert result['function_checks_unchanged'] and result['other_seven_scores_unchanged']
