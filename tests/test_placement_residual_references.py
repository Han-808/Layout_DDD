"""Exact residual claim ownership and failure-accounting regressions."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path

import pytest

from benchmark.evaluator.scene_quality.placement_residual_references import resolve_typed_references
from benchmark.evaluator.scene_quality.consistency_acceptance_v2 import summarize_metric, finish_metric
from benchmark.evaluator.scene_quality.global_group_first import _validate_and_tag_residual_placement_result
from benchmark.evaluator.scene_quality.placement_checks import canonicalize_placement_defect_linkage, validate_placement_check_results


def context():
    return {'typed_checks': [{'check_id':'a', 'subject_id':'sofa', 'context_ids':['table'],
                             'check_type':'contextual_anchor', 'conclusion':'invalid'}],
            'typed_defects': [{'check_id':'a', 'target_ids':['sofa'], 'check_type':'contextual_anchor'}],
            'groups':[]}


def observation():
    return {'verdict':'invalid', 'evidence_status':'sufficient', 'confidence':.8, 'reason':'Observed separation',
            'defects':[{'check_id':'a', 'target_ids':['sofa'], 'check_type':'contextual_anchor',
                        'reason':'Observed separation', 'severity':'atypical'}],
            'group_global_observations':[], 'evidence_request':None, 'missing_evidence':[]}


def accepted():
    return {'verdict':'valid', 'terminal_state':'evaluated', 'evidence_resolution': {
        'policy':'evidence_consistency_fallback_v2',
        'accepted':True, 'model_judgement_completed':True, 'visual_observation_complete':True}}


def test_exact_reference_preserves_invalid_without_new_registration():
    raw = observation()
    linked, checks = resolve_typed_references(raw, context())
    assert linked['verdict'] == 'invalid' and linked['defects'] == raw['defects']
    assert len(checks) == 1 and checks[0]['check_id'] == 'a'
    assert linked['placement_check_results'][0]['conclusion'] == 'invalid'
    assert 'placement_check_results' not in raw
    assert resolve_typed_references(linked, context()) == (linked, checks)
    projected = _validate_and_tag_residual_placement_result(linked, residual_context=context())
    assert projected['verdict'] == 'invalid'
    assert projected['placement_check_results'] == []
    assert len(projected['repeated_typed_check_observations']) == 1
    assert projected['defects'][0]['placement_component'] == 'typed'


@pytest.mark.parametrize('mutation', ['subject','context','type','prior_subject','prior_type','unconfirmed','ambiguous','insufficient'])
def test_identity_or_decision_conflict_never_becomes_a_pass(mutation):
    value, ctx = observation(), context()
    if mutation=='subject': value['defects'][0]['target_ids']=['chair']
    if mutation=='context': value['defects'][0]['context_ids']=['other']
    if mutation=='type': value['defects'][0]['check_type']='scene_zone'
    if mutation=='prior_subject': ctx['typed_defects'][0]['target_ids']=['chair']
    if mutation=='prior_type': ctx['typed_defects'][0]['check_type']='scene_zone'
    if mutation=='unconfirmed': ctx['typed_checks'][0]['conclusion']='valid'
    if mutation=='ambiguous': value['verdict']='ambiguous'
    if mutation=='insufficient': value['evidence_status']='insufficient'
    with pytest.raises(ValueError): resolve_typed_references(value, ctx)


def test_unknown_and_subject_overlap_do_not_borrow_typed_identity():
    value = observation()
    value['defects'][0]['check_id']='new'
    result, checks = resolve_typed_references(value, context())
    assert checks == [] and result == value
    # A new exact ID on the same object remains a distinct claim.
    value['defects'][0]['context_ids']=['other']
    assert resolve_typed_references(value, context())[1] == []


def test_existing_conflicting_row_is_never_overwritten_by_linkage():
    value = observation()
    value['placement_check_results']=[{'check_id':'a','subject_id':'other','context_ids':['table'],
        'observation_status':'observed','conclusion':'invalid','reason':'Wrong subject'}]
    linked, checks = resolve_typed_references(value,context())
    linked = canonicalize_placement_defect_linkage(linked,required_checks=checks)
    with pytest.raises(ValueError): validate_placement_check_results(linked,required_checks=checks)


def test_novel_residual_on_same_object_survives_and_cannot_self_mark_duplicate():
    value = observation()
    value['defects'].append({'check_id':'new','target_ids':['sofa'],'context_ids':['other'],
        'check_type':'contextual_anchor','scope':'implausible_local_context','reason':'Another relation',
        'severity':'atypical','residual_repeats_typed_claim':True})
    linked, _ = resolve_typed_references(value,context())
    result = _validate_and_tag_residual_placement_result(linked,residual_context=context())
    assert result['verdict']=='invalid'
    assert result['defects'][0]['residual_repeats_typed_claim'] is True
    assert result['defects'][1]['placement_component']=='residual_global_review'
    assert 'residual_repeats_typed_claim' not in result['defects'][1]


@pytest.mark.parametrize('state', ['hard_failure','gap','missing','accepted'])
def test_required_residual_is_never_omitted_from_denominator(state):
    report = {'status':'evaluated','score':1., 'global_discovery':accepted(),
              'residual_global_placement_phase':{'required':True}}
    if state=='accepted': report['residual_global_placement_review']=accepted()
    elif state!='missing': report['residual_global_placement_review']={
        'terminal_state':'infrastructure_failure' if state=='hard_failure' else 'evidence_gap',
        'failure':{'failure_category':'judge_response_failure' if state=='hard_failure' else 'evidence_gap'}}
    result = summarize_metric(report, ['scene_global'])
    assert result['planned_count']==2
    assert result['accepted_count']==(2 if state=='accepted' else 1)
    assert result['failure_count']==int(state in {'hard_failure','missing'})


def test_top_level_infrastructure_failure_is_not_relabelled_as_evidence_gap():
    report = {'status':'failed', 'score':None, 'global_discovery':accepted(),
              'infrastructure_failures':[{'phase':'residual','failure_kind':'engineering_failure'}]}
    result = finish_metric(report,['scene_global'])
    assert result['status']=='failed' and result['terminal_state']=='infrastructure_failure'


@pytest.mark.requires_local_data
@pytest.mark.parametrize('case', ['S100','S101','S102'])
def test_independent_real_fixture_full_score_chain(tmp_path, case):
    directory = os.environ.get('PLACEMENT_REPLAY_FIXTURES')
    if not directory:
        pytest.skip('Set PLACEMENT_REPLAY_FIXTURES to the independently sealed sanitized fixture directory')
    source = Path(__file__).resolve().parents[1]/'scripts/replay_placement_scoring.py'
    spec = importlib.util.spec_from_file_location('offline_placement_replay',source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.replay(json.loads((Path(directory)/(case+'.json')).read_text()),tmp_path)
    assert result['status']=='complete'
    assert result['placement_accepted']==result['placement_planned']
    assert result['placement_score']==pytest.approx(.92 if case=='S100' else .8)
    assert result['persisted_score_matches'] and result['live_model_calls']==0
