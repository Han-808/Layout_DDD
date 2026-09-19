"""Uniform Model reporting from metric-owned judgements, not visual coverage.

This projection never calls a model or supplies a verdict for an unknown unit.
Canonical N and the existing burden/Placement component formulas are unchanged.
Only metric-owned, post-exemption observations may enter a partial ledger.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any

from benchmark.evaluator import scoring
from benchmark.visual_judge.evidence_gap_v2 import coverage_from_plan
from benchmark.visual_judge.evidence_resolution import finite_score

POLICY = 'model_judgement_coverage_v1'
MIN_COVERAGE = 0.80
WEIGHTS = {**{m: 0.3 * w for m, w in scoring.L1_METRIC_WEIGHTS.items()},
           **{m: 0.7 * w for m, w in scoring.L3_METRIC_WEIGHTS.items()}}


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def metric_projection(metric: str, report: dict, object_ids: list[str], *,
                      deduction_multiplier: float = 2.0) -> dict:
    """Recheck the fixed obligation inventory and project observed burden only."""
    original = report.get('resolution_coverage') or {}
    planned = original.get('planned_ids')
    if not isinstance(planned, list) or not planned:
        raise ValueError(metric + ': missing fixed judgement inventory')
    units = deepcopy(original.get('units') or [])
    if metric == 'functional_consistency':
        from benchmark.visual_judge.functional_relation_recovery import unresolved_inventory_units
        required = unresolved_inventory_units(report)
        by_id = {u['unit_id']:u for u in units}
        if any(u['unit_id'] not in planned or by_id.get(u['unit_id']) != u for u in required):
            raise ValueError(metric + ': unresolved relation discovery missing from fixed inventory')
    for unit in units:
        if unit.get('accepted') and (unit.get('defaulted') or unit.get('decision_source') in {
            'program_default', 'default_valid', 'style_policy_no_deduction',
        }):
            raise ValueError(metric + ': program default is not an accepted judgement')
    coverage = coverage_from_plan(planned, units)
    accepted = {u['unit_id'] for u in units if u.get('accepted') is True}
    hard = bool(coverage['failure_count'] or report.get('infrastructure_failures')
                or report.get('status') == 'failed')
    projection = None
    if accepted and not hard:
        if metric in scoring.L1_METRIC_WEIGHTS:
            key = 'pairs' if metric == 'collision' else 'objects'
            def identity(row):
                return '|'.join(sorted((str(row['object_a']), str(row['object_b'])))) \
                    if metric == 'collision' else str(row['object_id'])
            selected = [deepcopy(row) for row in report.get(key) or [] if identity(row) in accepted]
            for row in selected:
                if row.get('final_verdict') not in {'valid', 'invalid'}:
                    raise ValueError(metric + ': accepted row has no verdict')
            # Keep Support's causal graph and all metric-owned magnitude facts.
            observed = {**report, key: selected}
            projection = getattr(scoring, 'score_' + metric + '_report')(
                observed, ordered_object_ids=object_ids, deduction_multiplier=deduction_multiplier)
        else:
            if coverage['evaluation_complete']:
                observed = report
            else:
                # Produced by the metric aggregator AFTER ownership, exclusions
                # and deduplication, never by scanning raw model responses.
                partial = report.get('observed_burden_input') or {}
                if partial.get('schema_version') != 'metric_owned_observed_defects_v1':
                    raise ValueError(metric + ': partial observations lack a metric-owned ledger')
                observed = {'judgement': {'verdict': 'invalid' if partial['defects'] else 'valid',
                                          'defects': deepcopy(partial['defects'])}}
                # This derived polarity is only an input to the pure burden
                # function, NOT a new scene/scope verdict or validity certificate.
            subscore = report.get('placement_subscore_policy') or {}
            kwargs = dict(ordered_object_ids=object_ids, deduction_multiplier=deduction_multiplier)
            if metric == 'semantic_placement_consistency' and subscore.get('enabled'):
                projection = scoring.score_placement_metric_report(observed, **kwargs,
                    residual_weight=float(subscore['residual_global_review_weight']))
            else:
                projection = scoring.score_l3_metric_report(metric, observed, **kwargs)
    score = projection.get('score') if projection is not None else None
    if score is not None and not finite_score(score):
        raise ValueError(metric + ': nonfinite observed burden score')
    return {'schema_version': POLICY, 'metric': metric,
            'source_report_sha256': fingerprint(report),
            'planned_count': len(planned), 'accepted_count': len(accepted),
            'judgement_fraction': len(accepted) / len(planned),
            'visual_fraction': coverage['visual_evidence_fraction'],
            'evaluation_complete': coverage['evaluation_complete'],
            'infrastructure_failure': hard, 'observed_score': score,
            'coverage': coverage, 'observed_scoring': projection,
            'canonical_object_ids': list(object_ids),
            'unknown_is_neither_valid_nor_invalid': True}


def aggregate(projections: dict[str, dict], weights: dict[str, float] = WEIGHTS) -> dict:
    if set(projections) != set(weights):
        raise ValueError('The complete planned metric inventory is required')
    total = sum(weights.values())
    if not total or any(not math.isfinite(w) or w < 0 for w in weights.values()):
        raise ValueError('Invalid metric weights')
    mass, points, failures = {}, {}, []
    for metric, item in projections.items():
        c = item['judgement_fraction']
        if not math.isfinite(c) or not 0 <= c <= 1:
            raise ValueError('Invalid judgement fraction')
        score = item.get('observed_score')
        mass[metric] = weights[metric] * c
        if item.get('infrastructure_failure'):
            failures.append(metric)
        elif c > 0 and not finite_score(score):
            raise ValueError(metric + ': accepted observations have no verified score')
        points[metric] = mass[metric] * score if finite_score(score) else 0.0
    covered = sum(mass.values())
    coverage = covered / total
    observed = sum(points.values()) / covered if covered and not failures else None
    eligible = observed is not None and coverage >= MIN_COVERAGE - 1e-12
    return {'schema_version': POLICY, 'judgement_coverage_fraction': coverage,
            'minimum_judgement_coverage': MIN_COVERAGE,
            'score': observed if eligible else None, 'observed_score': observed,
            'eligible': eligible, 'infrastructure_failure_metrics': failures,
            'status': 'infrastructure_failure' if failures else 'complete' if eligible and coverage == 1
                      else 'partial_coverage' if eligible else 'insufficient_judgement_coverage',
            'effective_metric_weights': {m: w / covered if covered else 0 for m, w in mass.items()},
            'metric_projections': projections,
            'formula': 'sum(w*c*s)/sum(w*c); coverage=sum(w*c)/sum(w)',
            'canonical_N_unchanged': True, 'score_multiplied_by_coverage_once': True}


def apply_report(report: dict) -> dict:
    """Called exclusively by the uniform runner; historical callers are untouched."""
    layers = report['layer_reports']
    sources = {**layers['l1_physical_plausibility']['metrics'],
               **report['reports']['scene_quality']['metrics']}
    ids = report['canonical_object_denominator']['ordered_object_ids']
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Uniform scoring requires a nonempty canonical object inventory')
    projections = {m: metric_projection(m, sources[m], ids) for m in WEIGHTS}
    summary = aggregate(projections)
    for metric, projection in projections.items():
        # The API persists raw backends and canonical envelope copies. Annotate
        # every explicit metric view so neither case files nor viewers revert
        # to the predecessor's visual/all-eight acceptance gate.
        views = [sources, layers['l1_physical_plausibility']['metrics'],
                 layers['l3_scene_quality']['metrics'],
                 report['reports'].get('generic_validity', {}).get('metrics', {}),
                 layers['l1_physical_plausibility'].get('backend_report', {}).get('metrics', {})]
        for view in views:
            if metric in view:
                view[metric]['judgement_coverage_projection'] = deepcopy(projection)
    report['pre_uniform_score'] = {k: deepcopy(report.get(k)) for k in
                                  ('benchmark_score', 'benchmark_score_100', 'benchmark_score_status')}
    report['judgement_coverage_summary'] = summary
    report.update(benchmark_score=summary['score'],
                  benchmark_score_100=summary['score'] * 100 if summary['score'] is not None else None,
                  benchmark_score_status=summary['status'])
    report['scoring_reliability']['uniform_judgement_coverage'] = {
        k: deepcopy(summary[k]) for k in ('schema_version', 'eligible', 'status',
                                         'judgement_coverage_fraction', 'infrastructure_failure_metrics')}
    return report


def finish_persisted(result: dict, sources: dict) -> dict:
    projections = {m: sources[m]['judgement_coverage_projection'] for m in WEIGHTS}
    summary = aggregate(projections)
    for item in result['metrics']:
        p = projections[item['metric']]
        score = p['observed_score']
        item.update(score=score, observed_score=score, accepted=score is not None,
                    coverage_fraction=p['judgement_fraction'],
                    visual_coverage_fraction=p['visual_fraction'],
                    coverage_complete=p['evaluation_complete'],
                    score_status='observed' if score is not None else 'no_judgement',
                    effective_overall_weight=summary['effective_metric_weights'][item['metric']],
                    coverage_threshold_passed=None,
                    grounded_overall_weight=item['overall_weight'] * p['judgement_fraction'],
                    weighted_points=score * item['overall_weight'] * p['judgement_fraction'] * 100 if score is not None else None,
                    judgement_coverage_projection=deepcopy(p))
    result.update(judgement_coverage_summary=summary,
                  combined_score_100=summary['score'] * 100 if summary['score'] is not None else None,
                  combined_observed_score_100=summary['observed_score'] * 100 if summary['observed_score'] is not None else None,
                  combined_status=summary['status'], combined_coverage_fraction=summary['judgement_coverage_fraction'])
    for layer in result.get('layers') or []:
        rows = [item for item in result['metrics'] if item['layer'] == layer['layer']]
        planned = sum(item['overall_weight'] for item in rows)
        observed_mass = sum(item['overall_weight'] * item['coverage_fraction'] for item in rows)
        points = sum(item['weighted_points'] or 0 for item in rows)
        failed = any(projections[item['metric']]['infrastructure_failure'] for item in rows)
        score = points / (100 * observed_mass) if observed_mass and not failed else None
        layer.update(score=score, observed_score=score,
                     score_status='observed' if score is not None else 'not_evaluable')
        layer['coverage'] = {'fraction': observed_mass / planned if planned else 0,
                             'basis': POLICY, 'threshold_applied_at': 'whole_scene_only'}
    return result
