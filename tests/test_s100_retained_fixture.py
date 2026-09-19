"""Small actual-report subset retained before campaign artifact cleanup.

No source campaign path, Blender, image bytes, network or model needed.
Full historical replay is evidenced separately, not claimed from this subset.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from benchmark.evaluator.scene_quality.consistency_acceptance_v2 import summarize_metric, terminalize_scope
from benchmark.evaluator.scene_quality.group_scoped import _combine_functional_check_episodes
from benchmark.scene_generation.feedback_loop.frozen_report import normalized_feedback_scores

FIXTURE = Path(__file__).parent / 'fixtures/s100_scope_failure_minimal.json'


@pytest.fixture
def retained():
    return json.loads(FIXTURE.read_text())


def test_actual_residual_requires_receipt_not_bare_verdict(retained):
    row = retained['metrics']['semantic_placement_consistency']['judgement']['residual_global_placement_judgement']
    report = {'status': 'evaluated', 'score': 1., 'judgement': {'residual_global_placement_judgement': row}}
    result = summarize_metric(report, ['residual:0'])
    assert result['complete'] and result['accepted_count'] == 1
    assert result['units'][0]['score'] == 1.
    row.pop('evidence_resolution')
    assert summarize_metric(report, ['residual:0'])['accepted_count'] == 0


def test_actual_atomic_episode_aggregation_subset(retained):
    groups = retained['metrics']['functional_consistency']['group_results']
    assert len(groups) == 4 and sum(len(g['check_episodes']) for g in groups) == 13
    for group in groups:
        episodes = group['check_episodes']
        assert group['status'] == 'failed' and all(e['status'] == 'evaluated' for e in episodes)
        checks = [c for e in episodes for c in e['functional_probe_evidence']['required_checks']]
        packet = {'group': {'group_id': group['group_id']}, 'functional_probe_evidence': {'required_checks': checks}}
        # The fixture deliberately omits detailed causal defect/measurement
        # payloads. Exercise receipt propagation at the post-aggregation
        # boundary, not full semantic defect revalidation or metric scoring.
        result = {'status': 'evaluated', 'score': min(e['score'] for e in episodes),
                  'judgement': {'aggregation': 'atomic_group_local_checks'}, 'check_episodes': episodes}
        terminalize_scope(result, phase='offline_retained_group')
        assert result['status'] == 'evaluated'
        assert result['score'] == min(e['score'] for e in episodes)
        if len(episodes) > 1:
            with pytest.raises(ValueError, match='do not match the required ledger'):
                _combine_functional_check_episodes(packets=[packet], episode_results=episodes[:-1])


def test_actual_failed_scores_serialize_with_positive_coverage(retained):
    # Synthetic complete L1 rows only satisfy the feedback subset envelope;
    # the two failing L3 rows below are taken from retained real projections.
    rows = [{'metric': m, 'layer': 'L1', 'score': .5, 'status': 'evaluated',
             'overall_weight': .1, 'coverage_fraction': 1.} for m in ('collision', 'support', 'oob')]
    for metric, weight in [('functional_consistency', .364), ('semantic_placement_consistency', .196)]:
        source = retained['metrics'][metric]
        projection = source['judgement_coverage_projection']
        assert projection['observed_score'] is None and projection['judgement_fraction'] > 0
        rows.append({'metric': metric, 'layer': 'L3', 'score': projection['observed_score'],
                     'status': source['status'], 'overall_weight': weight,
                     'coverage_fraction': projection['judgement_fraction']})
    value, audit = normalized_feedback_scores({'metrics': rows, 'combined_score_100': None})
    assert value['feedback_score'] is None
    assert len(audit['unavailable_metrics']) == 2
    json.dumps((value, audit), allow_nan=False)


def test_fixture_is_explicitly_partial_and_redacted(retained):
    assert retained['provenance']['extraction_complete'] is False
    text = FIXTURE.read_text()
    assert '/Users/' not in text
    assert 'raw_response' not in text and 'api_key' not in text
    assert retained['redaction']['images'] is False
