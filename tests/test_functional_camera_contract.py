"""Function observations must reach acquisition without changing verdict authority."""
import importlib.util
import json
import os
from pathlib import Path

import pytest

from benchmark.evaluator.scene_quality import functional_checks as checks
from benchmark.visual_judge.camera_dsl import camera_constraints_from_judge_request
from benchmark.visual_judge.interfaces.judge import EvidenceRequest


RECIPES = [
    *checks._RELATION_CHECK_OBSERVATIONS.values(),
    *checks._RELATION_ACQUISITION_OBSERVATIONS.values(),
    checks._ARCHITECTURE_ORIENTATION_OBSERVATIONS,
    checks._CLEARANCE_OBSERVATIONS,
    checks._OPERATING_CLEARANCE_OBSERVATIONS,
    checks._LOCAL_CONFIRMATION_OBSERVATIONS,
]


def plan(observations, metric='functional_consistency', targets=('table', 'sofa')):
    return camera_constraints_from_judge_request(
        EvidenceRequest(target_ids=targets, missing_observations=tuple(observations),
                        view_goal='Show the side table and sofa seated-reach region.'),
        metric=metric, known_target_ids=('table', 'sofa'))


@pytest.mark.parametrize('observations', RECIPES)
def test_every_function_recipe_and_individual_missing_token_plans(observations):
    for subset in (observations, *((word,) for word in observations)):
        result = plan(subset)
        assert result.target_ids == ('table', 'sofa')
        assert tuple(result.metadata['source_missing_observations']) == subset
        assert result.view_goal == 'Show the side table and sofa seated-reach region.'


def test_relative_use_does_not_invent_direction_or_a_verdict():
    result = plan(checks.functional_relation_required_observations('relative_use_geometry'))
    assert result.required_observations == (
        'target_visible', 'joint_visibility', 'group_context_visible', 'limited_local_context')
    assert not {'front_back_disambiguated', 'interaction_side_visible'} & set(result.required_observations)
    assert not {'verdict', 'score', 'conclusion'} & result.to_dict().keys()


@pytest.mark.parametrize('metric', ['collision', 'support', 'semantic_placement_consistency', 'style_consistency'])
@pytest.mark.parametrize('word', ['relative_layout_visible', 'interaction_region_visible'])
def test_function_aliases_do_not_widen_other_metric_contracts(metric, word):
    with pytest.raises(ValueError, match='unknown camera observation'):
        plan([word], metric=metric)


def test_unknown_observation_and_target_remain_hard_contract_errors():
    with pytest.raises(ValueError, match='unknown camera observation'):
        plan(['interaction_region_visible_typo'])
    with pytest.raises(ValueError, match='unknown target IDs'):
        plan(['interaction_region_visible'], targets=('invented',))


def test_relative_use_readiness_enters_real_controller_acquisition():
    from test_vlm_control_loop import (
        _functional_request, _Judge, _ReadinessSelector, _Renderer, _Gate,
        _gate_result, _selection, _valid_result)
    from benchmark.visual_judge.control_loop import VLMEvaluationController
    calls = []
    req = _functional_request()
    check = req.context['functional_probe_evidence']['required_checks'][0]
    check.update(check_type='relative_use_geometry', predicate='relative_use_geometry', target_ids=['a','b'],
                 required_observations=list(checks.functional_relation_required_observations('relative_use_geometry')))
    first = {'status':'need_more_evidence', 'confidence':.4, 'reason':'Inspect relative use', 'defects':[],
             'evidence_request':{'target_ids':['a','b'], 'missing_observations':['joint_visibility'],
                                 'view_goal':'Show the pair'}}
    selector = _ReadinessSelector(_selection(), calls, [
        {'outcome':'acquire', 'reason':'Inspect seated reach', 'supporting_evidence_refs':[],
         'visible_observations':['target_visible','joint_visibility','relative_layout_visible'],
         'missing_observations':['interaction_region_visible'], 'view_goal':'Show the interaction region'},
        {'outcome':'pass', 'reason':'All requested observations available', 'supporting_evidence_refs':[],
         'visible_observations':check['required_observations'], 'missing_observations':[], 'view_goal':''}])
    class Renderer(_Renderer):
        def render(self, request):
            result = super().render(request)
            result['visual_evidence'] = [f'repair-{len(self.requests)}.png']
            return result
    renderer = Renderer({'merge_policy':'append'},calls)
    judge = _Judge([first, _valid_result()], calls)
    controller = VLMEvaluationController(judge=judge,camera_selector=selector,renderer=renderer,
        evidence_gate=_Gate([_gate_result(ready=True)],calls))
    result = controller.run(req,candidate_views=({'id':'view-1'},),allowed_actions=('orbit',))
    assert result.status == 'valid'
    assert len(renderer.requests) == 2 and len(selector.readiness_requests) == 2
    assert list(judge.requests[-1].visual_evidence) == ['initial.png','repair-1.png','repair-2.png']
    constraint = next(r['camera_constraints'] for r in result.audit['trace']
                      if r['stage']=='acquisition_planner' and r['evidence_request']['missing_observations']==['interaction_region_visible'])
    assert constraint['required_observations'] == ['joint_visibility','limited_local_context']


@pytest.mark.requires_local_data
@pytest.mark.parametrize('ending', ['valid', 'invalid', 'renderer_fault', 'unknown_observation', 'terminal_fault'])
def test_independent_s100_full_score_chain(tmp_path, ending):
    directory = os.environ.get('FUNCTIONAL_REPLAY_FIXTURES')
    if not directory:
        pytest.skip('Set FUNCTIONAL_REPLAY_FIXTURES to preserved S100 fixture directory')
    source = Path(__file__).resolve().parents[1]/'scripts/replay_functional_scoring.py'
    spec = importlib.util.spec_from_file_location('offline_functional_replay', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fixture = json.loads((Path(directory)/'S100.json').read_text())
    result = module.replay(fixture, tmp_path, ending=ending)
    assert result['eligible'] == (ending in {'valid', 'invalid'})
    assert result['live_model_calls'] == 0
