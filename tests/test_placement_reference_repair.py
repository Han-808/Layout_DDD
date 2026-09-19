"""Invalid ownership bindings are repairable; substantive decisions stay locked."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

from benchmark.evaluator.scene_quality.placement_checks import (
    canonicalize_placement_defect_linkage, normalize_judge_originated_placement_results,
    validate_placement_check_results,
)
from benchmark.visual_judge.contracts import ResponseSchemaRepairError
from benchmark.visual_judge.response_repair import repair_canonical_response_schema_once
from benchmark.visual_judge.evidence_resolution import FALLBACK_POLICY
from test_evidence_adaptive_judgement import Model
from test_placement_audit_contract import answer, check, ownership, judge_request, picture
from test_r13_best_effort_terminal import judge


def events():
    return [*ownership(), {**ownership()[0], 'event_id':'function-2'},
            {**ownership()[0], 'event_id':'wrong', 'affected_object_ids':['other'],
             'causal_object_ids':['other'], 'scoring_target_ids':['other']}]


def candidate(kind, reference='function-1'):
    value = answer('excluded_function_owned')
    row = value['placement_check_results'][0]
    row['function_event_ref'] = reference
    if kind == 'proposal':
        row.pop('check_id')
        row.update(proposal_id='chair-zone', check_type='scene_zone', severity='atypical',
                   observation_goal='Inspect ordinary zone placement.')
        value['placement_check_results'] = []
        value['judge_originated_placement_results'] = [row]
    return value


def row(value, kind):
    return value['placement_check_results' if kind=='required' else 'judge_originated_placement_results'][0]


def validate(value, kind):
    checks = [check()]
    if kind == 'proposal':
        value, checks = normalize_judge_originated_placement_results(value,
            known_ids={'chair','other'}, groups=[{'group_id':'g','object_ids':['chair','other']}],
            existing_checks=[], expected_owner_stage='scene_global')
    value = canonicalize_placement_defect_linkage(value, required_checks=checks)
    validate_placement_check_results(value, required_checks=checks, function_events=events())
    return value


def repair(first, second, kind, *, enabled=True, terminal=False):
    model = Model([first, second])
    result, audit = repair_canonical_response_schema_once(model=model,messages=[],
        response_format_json=False,call_type='offline',judge_label='Placement',
        validator=lambda v:validate(v,kind),function_events=events() if enabled else None,
        force_binary_choice=terminal,preserve_terminal_semantics=True)
    assert len(model.calls)==2
    return result,audit


@pytest.mark.parametrize('kind',['required','proposal'])
@pytest.mark.parametrize('reference',['wrong','unknown',''])
@pytest.mark.parametrize('terminal',[False,True])
def test_invalid_binding_can_be_corrected_without_changing_decision(kind,reference,terminal):
    first, second = candidate(kind,reference), candidate(kind)
    result,audit = repair(first,second,kind,terminal=terminal)
    assert audit['recovered'] and len(audit['function_reference_repairs'])==1
    assert audit['function_reference_repairs'][0]['original_ref']==reference
    assert audit['function_reference_repairs'][0]['repaired_ref']=='function-1'
    assert result['placement_check_results'][0]['conclusion']=='excluded_function_owned'
    assert not result['defects']


@pytest.mark.parametrize('kind',['required','proposal'])
@pytest.mark.parametrize('mutation',['subject','context','conclusion','same_event','observed','verdict','new_atom','delete_atom'])
def test_reference_repair_cannot_change_locked_semantics(kind,mutation):
    first,second = candidate(kind,'wrong'), candidate(kind)
    r=row(second,kind)
    if mutation=='subject': r['subject_id']='other'
    elif mutation=='context': r['context_ids']=['other']
    elif mutation=='conclusion': r['conclusion']='valid'
    elif mutation=='same_event': r['same_physical_event']=False
    elif mutation=='observed': r['observation_status']='inferred_under_budget'
    elif mutation=='verdict': second['verdict']='invalid'
    else:
        key='placement_check_results' if kind=='required' else 'judge_originated_placement_results'
        if mutation=='new_atom': second[key].append(deepcopy(r))
        else: second[key]=[]
    with pytest.raises(ResponseSchemaRepairError): repair(first,second,kind)


@pytest.mark.parametrize('kind',['required','proposal'])
def test_already_valid_reference_cannot_switch_to_another_valid_event(kind):
    first,second = candidate(kind),candidate(kind,'function-2')
    row(first,kind)['unexpected']=True  # Trigger repair without invalidating the binding.
    with pytest.raises(ResponseSchemaRepairError,match='locked semantic'):
        repair(first,second,kind)


@pytest.mark.parametrize('kind',['required','proposal'])
@pytest.mark.parametrize('second',['wrong','unknown',''])
def test_strict_validator_still_rejects_unfixed_or_invalid_replacement(kind,second):
    with pytest.raises(ResponseSchemaRepairError):
        repair(candidate(kind,'wrong'),candidate(kind,second),kind)


@pytest.mark.parametrize('kind',['required','proposal'])
def test_legacy_repair_without_trusted_event_context_keeps_old_lock(kind):
    with pytest.raises(ResponseSchemaRepairError,match='locked semantic'):
        repair(candidate(kind,'wrong'),candidate(kind),kind,enabled=False)


def test_ambiguous_duplicate_proposal_identity_does_not_unlock_a_valid_reference():
    first=candidate('proposal','wrong')
    first['judge_originated_placement_results'].append(row(candidate('proposal'),'proposal'))
    second=deepcopy(first)
    second['judge_originated_placement_results'][0]['function_event_ref']='function-1'
    with pytest.raises(ResponseSchemaRepairError,match='locked semantic'):
        repair(first,second,'proposal')


def test_proposal_severity_cannot_change_during_reference_repair():
    second=candidate('proposal')
    row(second,'proposal')['severity']='material_contextual_mismatch'
    with pytest.raises(ResponseSchemaRepairError,match='locked semantic'):
        repair(candidate('proposal','wrong'),second,'proposal')


@pytest.mark.parametrize('kind',['required','proposal'])
def test_reference_repair_service_failure_does_not_become_a_verdict(kind):
    with pytest.raises(ResponseSchemaRepairError) as caught:
        repair(candidate(kind,'wrong'),ConnectionError('offline'),kind)
    assert caught.value.schema_audit['attempts'][1]['failure_kind']=='transport'


@pytest.mark.parametrize('terminal',[False,True])
def test_public_v2_judge_passes_trusted_ledger_to_bounded_repair(picture,terminal):
    req=judge_request()
    other={**req['objects'][0], 'id':'other'}
    req['objects'].append(other)
    req['scene_summary']['objects'].append(other)
    req.update(render_evidence=[picture],evidence_resolution_policy=FALLBACK_POLICY,adaptive_terminal=terminal)
    req['functional_ownership_ledger'].update(events=events(),event_count=len(events()))
    model=Model([candidate('required','wrong'),candidate('required')])
    result=judge(model)._adjudicate_scene_quality_raw(req)
    assert len(model.calls)==2
    assert result['placement_check_results'][0]['function_event_ref']=='function-1'
    assert result['request_metadata']['response_schema_validation']['function_reference_repairs']


@pytest.mark.requires_local_data
def test_preserved_actual_reference_error_through_full_scoring(tmp_path):
    directory=os.environ.get('PLACEMENT_REFERENCE_FIXTURES')
    if not directory: pytest.skip('Set PLACEMENT_REFERENCE_FIXTURES to preserved C-R1 attempt01 inputs')
    scripts=Path(__file__).resolve().parents[1]/'scripts'
    sys.path.insert(0,str(scripts))
    try:
        spec=importlib.util.spec_from_file_location('placement_reference_replay',scripts/'replay_placement_reference_scoring.py')
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result=module.replay(json.loads((Path(directory)/'S100_diagnostics_R1.json').read_text()),tmp_path)
    finally:
        sys.path.remove(str(scripts))
    assert result['eligible'] and result['model_calls']==2
