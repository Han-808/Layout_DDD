from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import socket
import subprocess

import pytest

from benchmark.camera_cal_scene_level import uniform, composition, orchestrator
from benchmark.evaluator.judgement_coverage import aggregate, metric_projection, WEIGHTS
from benchmark.visual_judge.group_context import project_group
from benchmark.visual_judge.evidence_resolution import FALLBACK_POLICY, evidence_policy_scope
from test_evidence_adaptive_judgement import Model, picture, complete_required_rows, request


def coverage(accepted, total):
    return {'planned_ids': [str(i) for i in range(total)], 'units': [
        {'unit_id': str(i), 'accepted': i < accepted, 'status': 'evaluated' if i < accepted else 'not_evaluable',
         'score': 1 if i < accepted else None, 'execution_complete': True,
         'visual_observation_complete': False, 'model_judgement_completed': i < accepted,
         'decision_source': 'model' if i < accepted else None,
         'failure_category': None if i < accepted else 'evidence_gap'} for i in range(total)]}


def test_group_projection_only_removes_routing_audit_and_is_idempotent():
    group = {'group_id': 'g', 'object_ids': ['a', 'b'], 'formation_edges': ['x' * 150000],
             'edge_reasons': ['distance'], 'region_id': 'r', 'unknown_future_fact': {'keep': True}}
    before = deepcopy(group)
    projected = project_group(group)
    assert group == before
    assert projected['object_ids'] == group['object_ids']
    assert projected['unknown_future_fact'] == group['unknown_future_fact']
    assert set(group) - set(projected) == {'formation_edges', 'edge_reasons'}
    assert len(projected['_routing_audit_projection']['source_group_sha256']) == 64
    assert project_group(projected) == projected


@pytest.mark.parametrize('metric', ['style_consistency', 'semantic_placement_consistency'])
@pytest.mark.parametrize('verdict', ['valid', 'invalid'])
def test_oversized_routing_audit_reaches_real_public_metric_report(picture, metric, verdict):
    from benchmark.evaluator.scene_quality import evaluate_scene_quality_interfaces, SCENE_QUALITY_INTERFACE_METRICS
    from benchmark.visual_judge import OpenAICompatibleVLMJudge
    from benchmark.visual_judge.adapters.legacy_judge import ControlledVLMJudge
    from test_postrun_policy_boundary import v2_control
    scene = request(metric)['scene_summary']
    scene['objects'].append({**deepcopy(scene['objects'][0]), 'id': 'desk', 'category': 'desk', 'center': [3, 3, 1]})
    seen = []
    def answer(messages):
        value = complete_required_rows(messages)
        context = json.loads(messages[1]['content'][0]['text'].split('\n', 1)[1])
        seen.append(context)
        if verdict == 'invalid':
            if metric == 'style_consistency':
                value.update(verdict='invalid', defects=[{'scope': 'significant_visible_style_incompatibility',
                    'target_ids': ['chair'], 'relation': 'visible design-language conflict',
                    'category': 'style_outlier', 'severity': 'gross', 'reason': 'Strongly conflicting style.'}])
            else:
                rows = value.get('placement_check_results') or []
                required = context.get('required_placement_checks') or []
                if rows:
                    row, check = rows[0], required[0]
                    row.update(conclusion='invalid', observation_status='inferred_under_budget')
                    value.update(verdict='invalid', defects=[{
                        'scope': 'semantically_inappropriate_scene_zone', 'target_ids': [check['subject_id']],
                        'relation': 'scene_zone', 'reason': 'Wrong semantic zone.',
                        'severity': 'material_contextual_mismatch', 'check_id': check['check_id'],
                        'check_type': check['check_type']}])
                elif context.get('evidence_phase') == 'global_discovery':
                    value.update(verdict='invalid', defects=[{
                        'scope': 'semantically_inappropriate_scene_zone', 'target_ids': ['chair'],
                        'relation': 'scene_zone', 'reason': 'Wrong room zone.',
                        'severity': 'material_contextual_mismatch', 'check_id': 'proposal-zone'}],
                        judge_originated_placement_results=[{'proposal_id': 'proposal-zone',
                            'subject_id': 'chair', 'context_ids': [], 'check_type': 'scene_zone',
                            'observation_goal': 'Inspect room zone.', 'observation_status': 'observed',
                            'conclusion': 'invalid', 'reason': 'Wrong room zone.',
                            'severity': 'material_contextual_mismatch'}])
        for defect in value.get('defects', []):
            defect.setdefault('attribution_mode', 'unary')
            if metric == 'semantic_placement_consistency':
                defect.setdefault('category', 'zone_placement_mismatch')
                defect['severity'] = 'atypical'
        return value
    model = Model(answer)
    judge = OpenAICompatibleVLMJudge(model, max_context_chars=120000,
             evidence_resolution_policy=FALLBACK_POLICY, terminal_evidence_policy='best_available_final_v1')
    wrapper = ControlledVLMJudge(judge, control=v2_control(budgets={'max_evidence_rounds': 0}))
    groups = [{'group_id': 'work', 'object_ids': ['chair', 'desk'],
               'formation_edges': ['routing-only' * 15000], 'edge_reasons': ['nearby']}]
    result = evaluate_scene_quality_interfaces(scene,
        config={'enabled': True, 'metrics': {m: {'enabled': m == metric} for m in SCENE_QUALITY_INTERFACE_METRICS}},
        object_grouping_report={'object_groups': groups}, render_evidence=[picture], vlm_judge=wrapper,
        metric_applicability={metric: {'applicability': 'relevant'}})['metrics'][metric]
    assert result['status'] == 'evaluated', (result.get('reason'), result.get('judgement', {}).get('error'))
    assert result['judgement']['verdict'] == verdict
    assert model.calls
    assert all('formation_edges' not in g for ctx in seen for g in ctx.get('object_groups', []))
    assert 'formation_edges' in groups[0]
    assert all(len(json.dumps(ctx, separators=(',', ':'))) <= 120000 for ctx in seen)


@pytest.mark.parametrize('accepted,total,eligible', [(88, 99, True), (4, 5, True), (79, 100, False), (0, 99, False)])
def test_whole_scene_coverage_not_visual_or_eight_of_eight(accepted, total, eligible):
    projections = {m: {'judgement_fraction': accepted / total,
                      'observed_score': 0.7 if accepted else None} for m in WEIGHTS}
    result = aggregate(projections)
    assert result['eligible'] is eligible
    assert result['judgement_coverage_fraction'] == pytest.approx(accepted / total)
    assert result['score'] == (pytest.approx(0.7) if eligible else None)


def test_missing_metric_excluded_from_score_but_not_planned_coverage():
    projections = {m: {'judgement_fraction': 1, 'observed_score': 0.7} for m in WEIGHTS}
    projections['style_consistency'] = {'judgement_fraction': 0, 'observed_score': None}
    result = aggregate(projections)
    assert result['score'] == pytest.approx(0.7)
    assert result['judgement_coverage_fraction'] == pytest.approx(1 - WEIGHTS['style_consistency'])
    assert result['effective_metric_weights']['style_consistency'] == 0


def test_partial_metric_uses_original_N_and_retains_invalid_burden():
    report = {'status': 'not_evaluable', 'score': None, 'resolution_coverage': coverage(1, 2),
              'observed_burden_input': {'schema_version': 'metric_owned_observed_defects_v1',
                  'defects': [{'scope': 'object', 'target_ids': ['a'], 'category': 'oversized',
                               'severity': 'gross', 'reason': 'Oversized.'}]}}
    before = deepcopy(report)
    p = metric_projection('scale_consistency', report, ['a', 'b', 'c', 'd'])
    assert p['judgement_fraction'] == 0.5 and p['visual_fraction'] == 0
    assert p['observed_scoring']['n_scene'] == 4
    assert p['observed_scoring']['event_count'] == 1
    assert p['observed_score'] < 1
    assert report == before


def test_no_partial_ledger_and_program_default_fail_closed():
    report = {'status': 'not_evaluable', 'resolution_coverage': coverage(1, 2)}
    with pytest.raises(ValueError, match='metric-owned ledger'):
        metric_projection('scale_consistency', report, ['a', 'b'])
    report['resolution_coverage']['units'][0]['defaulted'] = True
    with pytest.raises(ValueError, match='program default'):
        metric_projection('scale_consistency', report, ['a', 'b'])


def test_partial_l1_only_accepted_verdicts_contribute_burden():
    audit = coverage(1, 2)
    audit['planned_ids'] = ['a', 'b']
    for unit, name in zip(audit['units'], ['a', 'b']):
        unit['unit_id'] = name
    report = {'status': 'not_evaluable', 'resolution_coverage': audit,
              'objects': [{'object_id': 'a', 'final_verdict': 'valid'},
                          {'object_id': 'b', 'final_verdict': 'invalid'}]}
    p = metric_projection('oob', report, ['a', 'b'])
    assert p['judgement_fraction'] == 0.5
    assert p['observed_score'] == 1 and not p['observed_scoring']['events']
    assert p['observed_scoring']['n_scene'] == 2


def test_hard_failure_is_not_evidence_gap_or_zero_score():
    projections = {m: {'judgement_fraction': 1, 'observed_score': 0.7} for m in WEIGHTS}
    projections['support'] = {'judgement_fraction': 0.5, 'observed_score': None, 'infrastructure_failure': True}
    result = aggregate(projections)
    assert result['score'] is None and not result['eligible']
    assert result['status'] == 'infrastructure_failure'


def test_uniform_projection_flows_through_public_report_and_persisted_summary(tmp_path):
    from benchmark.api.evaluation import run_evaluate
    from benchmark.evaluator.judgement_coverage import apply_report
    from benchmark.visual_judge import OpenAICompatibleVLMJudge
    from benchmark.camera_cal_scene_level.persisted_scoring import case_scoring_summary
    from test_current_evaluation_profile import _scene
    report = run_evaluate(scene=_scene(), out=tmp_path / 'raw.json',
        vlm_judge=OpenAICompatibleVLMJudge(Model(complete_required_rows),
            evidence_resolution_policy=FALLBACK_POLICY, terminal_evidence_policy='best_available_final_v1'),
        asset_policy={'mode': 'generated_or_open_assets', 'arrangement_owner': 'generator',
                      'category_selection_owner': 'generator', 'scale_owner': 'generator', 'appearance_owner': 'generator'},
        render_evidence=[], vlm_evaluation_control={'evidence_resolution_policy': FALLBACK_POLICY,
                                                   'budgets': {'max_evidence_rounds': 0}})
    raw = deepcopy(report)
    result = apply_report(report)
    l1 = result['layer_reports']['l1_physical_plausibility']
    l3 = result['reports']['scene_quality']
    summary = case_scoring_summary(case_id='example', case_manifest=result, l1_report=l1, l3_report=l3)
    assert summary['combined_score_100'] == result['benchmark_score_100']
    assert set(summary['judgement_coverage_summary']['metric_projections']) == set(WEIGHTS)
    assert result['pre_uniform_score']['benchmark_score'] == raw['benchmark_score']
    for m in l3['metrics']:
        assert l3['metrics'][m]['score'] == raw['reports']['scene_quality']['metrics'][m]['score']
        assert 'judgement_coverage_projection' in l3['metrics'][m]


def test_same_native_options_for_both_modes(tmp_path):
    common = ['--dataset-root', str(tmp_path), '--output-root', str(tmp_path / 'new')]
    one = uniform.parser().parse_args(['--mode', 'open-space', *common])
    many = uniform.parser().parse_args(['--mode', 'multi-room', *common])
    assert uniform.native_argv(one) == uniform.native_argv(many)
    from benchmark.camera_cal_scene_level.planning import resolved_control
    control = resolved_control()
    resolved = orchestrator.resolve_run_control(control, FALLBACK_POLICY)
    assert resolved.evidence_resolution_policy == FALLBACK_POLICY
    with pytest.raises(ValueError):
        orchestrator.resolve_run_control(control, 'unknown')


def test_release_rejects_missing_pin_and_config_drift(monkeypatch, tmp_path):
    source = tmp_path / 'source'
    for rel in ('src/benchmark/a.py', 'configs/runners/model_floorplan_unified_v1.json',
                'scripts/run_uniform_model_evaluation.py', 'pyproject.toml'):
        path = source/rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
    monkeypatch.setattr(uniform, 'ROOT', source)
    monkeypatch.setattr(uniform, 'PROTOCOL_PATH', source/'configs/runners/model_floorplan_unified_v1.json')
    monkeypatch.setattr(uniform, 'source_files', lambda: ['src/benchmark/a.py', 'configs/runners/model_floorplan_unified_v1.json',
                'scripts/run_uniform_model_evaluation.py', 'pyproject.toml'])
    files = {rel: uniform.digest(source/rel) for rel in uniform.source_files()}
    manifest = tmp_path/'release.json'
    release = {'schema_version': 'uniform_model_evaluator_release_v1', 'source_root': str(source),
               'protocol_sha256': uniform.digest(uniform.PROTOCOL_PATH), 'files': files,
               'source_tree_sha256': uniform.tree_digest(files)}
    release['files'].pop('src/benchmark/a.py')
    manifest.write_text(json.dumps(release))
    with pytest.raises(ValueError, match='inventory changed'):
        uniform.verify_source(manifest, required=True)
    (uniform.PROTOCOL_PATH).write_text('{"drift":true}')
    with pytest.raises(ValueError, match='protocol changed'):
        uniform.verify_source(manifest, required=True)
    with pytest.raises(ValueError, match='sealed release'):
        uniform.verify_source(None, required=True)


def test_live_input_drift_fails_before_output_or_execution(monkeypatch, tmp_path):
    monkeypatch.setattr(uniform, 'verify_source', lambda *a, **kw: {'sealed': True})
    monkeypatch.setattr(uniform, 'inspect_inputs', lambda _: [{'case_id': 'now-changed'}])
    receipt = tmp_path/'approved.json'
    receipt.write_text(json.dumps({'schema_version': 'uniform_model_evaluator_run_v1',
        'live_run': False, 'mode': 'open-space', 'protocol_sha256': uniform.digest(uniform.PROTOCOL_PATH),
        'identity': {'sealed': True}, 'cases': [{'case_id': 'original'}]}))
    output = tmp_path/'output'
    with pytest.raises(ValueError, match='differs from reviewed manifest: cases'):
        uniform.main(['--mode', 'open-space', '--dataset-root', str(tmp_path),
            '--output-root', str(output), '--input-manifest', str(receipt), '--run'])
    assert not output.exists()


@pytest.mark.requires_local_data
@pytest.mark.parametrize('metric', ['style_consistency', 'semantic_placement_consistency'])
def test_actual_atelier_all_173_objects_survive_context_projection(metric):
    from benchmark.visual_judge import OpenAICompatibleVLMJudge
    from benchmark.evaluator.context_projection import project_scene_for_evaluator_context
    from benchmark.evaluator.scene_quality.interfaces import _judge_request
    from benchmark.visual_judge.adaptive_context import budget_adaptive_context
    from benchmark.visual_judge.evidence_resolution import ContextCapacityError
    root = Path('/Users/han_mohan/Desktop/Layout_DDD/Support/artifacts/outputs/atelier_v2_evaluation_r15_v2')
    case = root/'input/dataset/ATELIER'
    if not case.is_dir():
        pytest.skip('retained Atelier artifacts unavailable')
    scene = project_scene_for_evaluator_context(json.loads((case/'scene/canonical_scene.json').read_text()))
    groups = json.loads((root/'evaluation/cases/ATELIER/grouping.json').read_text())['object_groups']
    model = Model(complete_required_rows)
    judge = OpenAICompatibleVLMJudge(model, max_context_chars=120000,
        evidence_resolution_policy=FALLBACK_POLICY, terminal_evidence_policy='best_available_final_v1')
    with evidence_policy_scope({'evidence_resolution_policy': FALLBACK_POLICY}):
        req = _judge_request(metric_name=metric, scene=scene, prompt=None,
            render_evidence=[str(case/'evidence/standardized_perspective.png')],
            selected_object_ids=[o['id'] for o in scene['objects']], selected_group_ids=[],
            groups=groups, authorized_deviations=[], visual_style_spec=None,
            evidence_phase='global_screen' if metric == 'style_consistency' else 'global_discovery',
            decision_mode='screen')
        req['adaptive_terminal'] = True
        result = judge._adjudicate_scene_quality_raw(req)
    assert result['verdict'] == 'valid'
    context = json.loads(model.calls[0][1]['content'][0]['text'].split('\n', 1)[1])
    actual_ids = [o['id'] for o in context['adaptive_evidence']['structured_evidence']['objects']]
    assert actual_ids == [o['id'] for o in scene['objects']]
    assert len(actual_ids) == 173
    assert [g['object_ids'] for g in context['object_groups']] == [g['object_ids'] for g in groups]
    assert len(json.dumps(context, separators=(',', ':'))) <= 120000
    old_context = {**context, 'object_groups': groups}
    with pytest.raises(ContextCapacityError):
        budget_adaptive_context(old_context, 120000)


@pytest.mark.requires_local_data
@pytest.mark.parametrize('workers', [1, 2])
@pytest.mark.parametrize('mode,dataset,case_id', [
    ('open-space', 'Support/artifacts/outputs/atelier_v2_evaluation_r15_v2/input/dataset', 'ATELIER'),
    ('multi-room', 'Support/datasets/multi_room_floorplans_v2_hy4_preview_tokenhub_wall_eval_v1/hy4-preview-tokenhub',
     'mr.hy4-preview-tokenhub.layout_01.room_000')])
def test_native_cli_to_case_to_real_judge_factory_no_live_calls(monkeypatch, tmp_path, mode, dataset, case_id, workers):
    root = Path('/Users/han_mohan/Desktop/Layout_DDD') / dataset
    if not root.is_dir():
        pytest.skip('prepared local input unavailable')
    captured = {}
    class Reached(BaseException):
        pass
    def boundary(**kwargs):
        from benchmark.visual_judge.control_config import resolve_vlm_evaluation_control
        control = resolve_vlm_evaluation_control(kwargs['vlm_evaluation_control'])
        judge = kwargs['vlm_judge']
        assert control.evidence_resolution_policy == FALLBACK_POLICY
        assert judge.evidence_resolution_policy == FALLBACK_POLICY
        assert judge.terminal_evidence_policy == 'best_available_final_v1'
        assert kwargs['collision_geometry']['objects']
        captured.update(objects=len(kwargs['scene']['objects']))
        raise Reached()
    def forbidden(*a, **kw):
        pytest.fail('Offline check attempted network or process creation')
    monkeypatch.setattr(composition, 'run_evaluate', boundary)
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(socket.socket, 'connect_ex', forbidden)
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    monkeypatch.setenv('JUDGE_ENDPOINT', 'http://127.0.0.1:9/v1')
    monkeypatch.setenv('JUDGE_MODEL', 'gpt-5.6-sol')
    monkeypatch.setenv('JUDGE_API_KEY_ENV', 'UNIFORM_OFFLINE_KEY')
    monkeypatch.setenv('UNIFORM_OFFLINE_KEY', 'offline-test-only')
    args = uniform.parser().parse_args(['--mode', mode, '--dataset-root', str(root),
            '--case-id', case_id, '--output-root', str(tmp_path / 'out'), '--max-workers', str(workers)])
    deps = uniform.dependencies(uniform.native_argv(args))
    deps = replace(deps, execution=replace(deps.execution,
        endpoint_preflight=lambda **_: {'attempts_required': 1, 'api_invocations': 0}))
    with pytest.raises(Reached):
        orchestrator.run_main(deps=deps)
    assert captured['objects'] > 0
