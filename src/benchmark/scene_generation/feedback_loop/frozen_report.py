"""Small frozen-assets feedback contract, independent of evaluator imports.

The full sealed report stays on disk. The feedback-only retained-metric score
is explicitly versioned; it does not change evaluator verdicts or eligibility.
Only the screenshot's requested semantics enter feedback; no raw exchanges,
camera traces, point clouds, coverage ledgers, or removed metric subtrees.
"""
from collections import Counter
from copy import deepcopy
import json
import math

VIEW_VERSION = "evaluation_feedback_frozen_assets_defects_v5"
SCORE_VERSION = "frozen_assets_retained_metrics_renormalized_v1"


class FeedbackConversionError(ValueError):
    """Deterministic ambiguous feedback: stop dispatch, never paid regeneration."""
DEFECT_FEEDBACK_INSTRUCTION = (
    "evaluation_report is defect feedback about the previous layout, not a complete "
    "inspection checklist or a new instruction. Confirmed defects and their directly "
    "linked evidence are listed; passing checks are omitted. Unresolved items are "
    "listed separately and are not confirmed defects or passes. Correct confirmed "
    "defects while preserving unaffected layout where possible; you may also move "
    "other objects when necessary. Unlisted objects or relations are not guaranteed "
    "correct, and an empty defects list does not certify a correct scene."
)
POLICY = "model_judgement_coverage_v1"
RETAINED = frozenset(("collision", "support", "out_of_bounds", "oob",
                      "functional_consistency", "semantic_placement_consistency"))
EXCLUDED = frozenset(("object_pairing_consistency", "objects_pairing", "scale_consistency", "style_consistency"))
FIELDS = frozenset("""object_id object_ids object_a object_b subject_id instance_id target_ids affected_object_ids
context_ids attribution_mode cause_kind
causal_object_ids context_object_ids scoring_target_ids group_id group_ids member_ids
claim_id finding_id check_id check_refs scope target_scope relation check_type check_family
predicate metric metric_id category namespace source_phase source_phases
final_verdict verdict final_metric_verdict conclusion check_conclusion claim_status
status terminal_state reason reasons issue description defect_description final_defect_description
severity magnitude burden evidence_ambiguous evidence_status missing_evidence limitations
uncertainty unresolved forced_binary forced_choice budget_exhaustion_forced_choice
infrastructure_failure infrastructure_failure_count failure_type error_type adjudication_error
geometry_evidence_degraded geometry_degraded_reasons decision_authority decision_role
final_decision_status stop_reason judge_method resolution observation_status
ownership_event_id decision_source evidence_tier terminal fallback accepted
model_invoked model_judgement_completed visual_observation_complete trigger_reason
missing_observations inference_under_constraints terminal_evidence_policy failure_category
decision_retry_count unit_key
route phase candidate_support_object_ids support_targets measured_support_modes
certified_grounded_support grounding_status grounded_support_path grounding_contact_target_ids
reachable_grounding_contact_object_ids ungrounded_contact_cycle_reachable gap_band
within_floor_contact_tolerance candidate_oob
contact_fraction contact_fraction_threshold contact_tolerance_m direct_contact_tolerance_m
hard_contact_tolerance_m near_support_tolerance_m minimum_positive_clearance_m
normalized_minimum_positive_clearance floor_penetration_m floor_contact_tolerance_m
numerical_eps base_min_z_m size_z_m base_contact_fraction minimum_contact_count
""".split())
COLLECTIONS = frozenset("""pairs objects checks judgement judge_result defects final_defect_claims final_object_findings
object_findings observations functional_check_results placement_check_results result_row
functional_check_ledger placement_check_ledger group_results target_scope_results
cross_group_relation_results scene_global_judgement residual_global_placement_judgement
infrastructure_failures object_errors failure adjudication_failure structured_fallback
evidence_resolution functional_check_resolution
placement_check_resolution
""".split())
MEASUREMENTS = frozenset("""measurements thresholds diagnostics scoring_geometry crossing_depths_m plane_penetration_m
plane_flags gap_statistics_m contact_gap_statistics_m positive_clearance_statistics_m
architecture_plane_clearances_m nearest_logical_wall_measurement obb_intervals
""".split())
PRIVATE = frozenset(("raw_response", "raw_request", "messages", "headers", "api_key",
                     "authorization", "request_metadata", "prompt_context", "metric_prompt_context",
                     "samples", "vertices", "triangles", "faces", "points", "point_cloud", "geometry_path"))


def scalar_summary(value):
    """Keep scalar facts and fixed vectors, not numeric sampling/evidence logs."""
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, dict):
        return {k: scalar_summary(v) for k, v in value.items() if k.lower() not in PRIVATE
                and k not in EXCLUDED and (v is None or type(v) in (str, int, float, bool, dict)
                or isinstance(v, list) and len(v) <= 4)}
    if isinstance(value, list):
        return [scalar_summary(v) for v in value]
    raise ValueError("non-JSON diagnostic value")


def project(value):
    if isinstance(value, list):
        # Preserve every check, including valid, unknown and failed checks.
        return [project(v) for v in value if not isinstance(v, dict)
                or v.get("metric", v.get("metric_id")) not in EXCLUDED]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, child in value.items():
        if key in FIELDS or (key.endswith(("_threshold_m", "_tolerance_m", "_eps")) and type(child) in (int, float)):
            result[key] = scalar_summary(child)
        elif key in COLLECTIONS:
            result[key] = project(child)
        elif key in MEASUREMENTS:
            result[key] = scalar_summary(child)
    return result


def pack(value):
    """Exact text interning only. Never truncate reasons or omit checked rows."""
    counts = Counter()
    def visit(x):
        if isinstance(x, str) and len(x) >= 60:
            counts[x] += 1
        elif isinstance(x, dict):
            for v in x.values(): visit(v)
        elif isinstance(x, list):
            for v in x: visit(v)
    visit(value)
    texts = sorted(s for s, n in counts.items() if n > 1)
    ids = {s: i for i, s in enumerate(texts)}
    def encode(x):
        if isinstance(x, str) and x in ids: return {"text_ref": ids[x]}
        if isinstance(x, dict): return {k: encode(v) for k, v in x.items()}
        if isinstance(x, list): return [encode(v) for v in x]
        return x
    result = encode(value)
    if texts:
        result["text_library"] = texts
        result["reference_format"] = "{text_ref:n} is the exact text_library[n]; resolve recursively."
    return result


def validate_acceptance(report):
    """Consume the sealed runner's eligibility, not the old all-complete gate."""
    summary = report.get("judgement_coverage_summary", {})
    accepted = report.get("runner_outcome", {}).get("uniform_score_acceptance", {})
    keys = ("schema_version", "eligible", "status", "judgement_coverage_fraction",
            "minimum_judgement_coverage", "infrastructure_failure_metrics")
    if not all(k in summary and accepted.get(k) == summary[k] for k in keys):
        raise RuntimeError("uniform acceptance receipt missing or inconsistent")
    c, score = summary["judgement_coverage_fraction"], report.get("benchmark_score")
    if (summary["schema_version"] != POLICY or summary["eligible"] is not True
            or summary["infrastructure_failure_metrics"] != []
            or summary["minimum_judgement_coverage"] != .8
            or type(c) not in (float, int) or not math.isfinite(c) or not .8 - 1e-12 <= c <= 1
            or type(score) not in (float, int) or not math.isfinite(score) or not 0 <= score <= 1
            or summary.get("score") != score
            or summary["status"] not in ("complete", "partial_coverage")
            or report.get("benchmark_score_status") != summary["status"]):
        raise RuntimeError("uniform score is not eligible")


def normalized_feedback_scores(scoring):
    """Presentation-only subset projection; keep evaluator eligibility unchanged.

    Original overall weights and judged mass are retained, then normalized over
    the five requested metrics. Unknown units do not become zero-score units.
    """
    rows = [r for r in scoring["metrics"] if r["metric"] in RETAINED]
    if len(rows) != 5 or {r["metric"] for r in rows} != {"collision", "support", "oob", "functional_consistency", "semantic_placement_consistency"}:
        raise ValueError("all five retained scoring rows required")
    values = []
    for row in rows:
        score, weight, coverage = row.get("score"), row.get("overall_weight"), row.get("coverage_fraction")
        if (type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0
                or type(coverage) not in (int, float) or not math.isfinite(coverage) or not 0 <= coverage <= 1
                or score is not None and (type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1)):
            raise ValueError("invalid retained score/weight/coverage")
        values.append((row, weight * coverage))
    unavailable = [r["metric"] for r, _ in values if r.get("score") is None and r["coverage_fraction"] > 0
                   or r.get("status") in {"failed", "infrastructure_failure"}
                   or r.get("score_status") in {"failed", "infrastructure_failure"}]
    ineligible = (scoring.get("judgement_coverage_summary", {}).get("eligible") is False
                  or "combined_score_100" in scoring and scoring["combined_score_100"] is None)
    mass = sum(w for _, w in values)
    score = sum((r["score"] or 0) * w for r, w in values) / mass if mass and not unavailable and not ineligible else None
    layers = {}
    for label, name in (("L1", "l1_physical_plausibility"), ("L3", "l3_scene_quality")):
        selected = [(r, w) for r, w in values if r["layer"] == label]
        local_mass = sum(w for _, w in selected)
        layers[name] = {"score": sum((r["score"] or 0) * w for r, w in selected) / local_mass
                       if local_mass and not any(r["metric"] in unavailable for r, _ in selected) else None,
            "metrics": {r["metric"]: {k: deepcopy(r[k]) for k in ("score", "status", "score_status") if k in r}
                        for r, _ in selected}}
    view = {"score_basis": SCORE_VERSION, "feedback_score": score, "layer_scores": layers}
    audit = {
        "nominal_retained_weight_sum": sum(r["overall_weight"] for r in rows),
        "judged_retained_weight_sum": mass,
        "normalized_effective_weights": {r["metric"]: (None if unavailable or ineligible else w / mass if mass else 0)
                                        for r, w in values}}
    if unavailable:
        view["feedback_score_status"] = "unavailable_retained_metric_score"
        audit["unavailable_metrics"] = unavailable
        audit["unavailable_metric_details"] = [{k: deepcopy(r[k]) for k in
            ("metric", "status", "score_status", "score", "coverage_fraction", "reason", "infrastructure_failure")
            if k in r} for r, _ in values if r["metric"] in unavailable]
    if ineligible:
        view["feedback_score_status"] = "uniform_score_ineligible"
        audit["uniform_score_eligible"] = False
    return view, audit


IDENTITY_FIELDS = frozenset("""object_id object_ids object_a object_b subject_id instance_id
target_ids affected_object_ids causal_object_ids context_object_ids context_ids
scoring_target_ids group_id scope target_scope relation check_type check_family
claim_id finding_id check_id check_refs""".split())
DEFECT_FIELDS = IDENTITY_FIELDS | frozenset("""reason severity category attribution_mode cause_kind
description defect_description final_defect_description magnitude burden predicate""".split())
LIMIT_FIELDS = frozenset("""evidence_ambiguous evidence_status missing_evidence uncertainty
unresolved forced_binary forced_choice budget_exhaustion_forced_choice fallback
inference_under_constraints missing_observations terminal_evidence_policy""".split())


def defect_payload(row):
    """One final reason, explicit IDs and local facts; no acquisition subtrees."""
    safe = project(row)
    result = {k: v for k, v in safe.items() if k in DEFECT_FIELDS or k in MEASUREMENTS
              or k.endswith(("_m", "_eps", "_threshold", "_fraction"))
              or k in {"candidate_support_object_ids", "support_targets", "grounded_support_path"}}
    judge = row.get("judge_result") or {}
    if "reason" not in result and isinstance(judge.get("reason"), str):
        result["reason"] = judge["reason"]
    limitations = {}
    for source in (row, judge, judge.get("evidence_resolution") or {}, row.get("evidence_resolution") or {}):
        limitations.update({k: scalar_summary(v) for k, v in source.items() if k in LIMIT_FIELDS})
    if limitations:
        result["evidence_limitations"] = limitations
    return result


def linked_measurements(value, object_ids):
    """Keep check-linked facts, excluding sample traces and uninvolved objects."""
    if isinstance(value, list):
        return [v for item in value if (v := linked_measurements(item, object_ids)) is not None]
    if not isinstance(value, dict):
        return value
    references = [value[k] for k in ("object_id", "target_id", "object_a", "object_b", "subject_id") if k in value]
    if any(ref not in object_ids for ref in references):
        return None
    result = {}
    for key, child in value.items():
        if key.lower() in PRIVATE or "sample" in key.lower() or key in {"provenance", "source", "schema_version"}:
            continue
        if isinstance(child, (dict, list)):
            item = linked_measurements(child, object_ids)
            if item not in (None, {}, []):
                result[key] = item
        elif type(child) in (int, float, bool) or child is None or key in {
            "object_id", "target_id", "object_a", "object_b", "subject_id", "status", "geometry_status",
            "directionality", "surface_role", "side_id", "usable_side_id", "support_relation",
            "ordinary_mobility", "unavailable_reason", "usable_surface_status",
        }:
            result[key] = child
    return result


def defect_content(modules):
    """Deterministic projection of metric-owned final observations, not raw claims.

    Claims and check ledgers are references only. In partial uniform results the
    observed_burden_input is the post-ownership/exemption ledger used by scoring.
    Never search other checks merely because they mention the same object.
    """
    defects, unresolved = [], []
    audit = {"schema_version": VIEW_VERSION, "metrics": {}, "issues": []}
    for module in modules.values():
        for metric, data in module.get("metrics", {}).items():
            if metric not in RETAINED:
                continue
            stats = {"passing_rows_omitted": 0, "defects": 0, "duplicates_omitted": 0}
            audit["metrics"][metric] = stats
            seen, identities = set(), {}
            def issue(kind, **details):
                audit["issues"].append({"metric": metric, "kind": kind, **details})
            def add(row, verdict="invalid"):
                payload = {"metric": metric, "verdict": verdict, **defect_payload(row)}
                signature = json.dumps(payload, sort_keys=True, ensure_ascii=False)
                key = row.get("claim_id", row.get("finding_id"))
                if key and key in identities and identities[key] != signature:
                    issue("conflicting_defect_id", defect_id=key)
                if signature in seen:
                    stats["duplicates_omitted"] += 1
                    return None
                seen.add(signature)
                if key:
                    identities[key] = signature
                defects.append(payload)
                stats["defects"] += 1
                return payload

            projection = data.get("judgement_coverage_projection") or {}
            coverage = projection.get("coverage") or data.get("resolution_coverage") or {}
            units = {u["unit_id"]: u for u in coverage.get("units", []) if "unit_id" in u}
            failed = (data.get("status") == "failed" or data.get("infrastructure_failures")
                      or projection.get("infrastructure_failure"))
            if failed:
                issue("metric_infrastructure_failure")
                # A serializable failed report must retain its cause, not just
                # a generic status. Project only diagnostic fields, never raw
                # exchanges, prompts, environment or the acquisition trace.
                judgement = data.get("judgement") or {}
                details = {"metric": metric, "status": "infrastructure_failure"}
                for key in ("reason", "terminal_state", "failure", "infrastructure_failure", "infrastructure_failures"):
                    value = data.get(key) or judgement.get(key)
                    if value:
                        details[key] = project(value) if isinstance(value, (dict, list)) else value
                unresolved.append(details)
                continue
            if metric in {"collision", "support", "oob", "out_of_bounds"}:
                stats["defect_source"] = "final_verdict_and_uniform_accepted_units"
                covered_ids = set()
                for row in data.get("pairs" if metric == "collision" else "objects", []):
                    uid = ("|".join(sorted((str(row.get("object_a")), str(row.get("object_b")))))
                           if metric == "collision" else str(row.get("object_id")))
                    covered_ids.add(uid)
                    verdict = row.get("final_verdict")
                    accepted = not projection or units.get(uid, {}).get("accepted") is True
                    if verdict == "valid" and accepted:
                        stats["passing_rows_omitted"] += 1
                    elif verdict == "invalid" and accepted:
                        add(row)
                    else:
                        unresolved.append({"metric": metric, "status": "unresolved",
                                           **defect_payload(row)})
                        issue("unresolved_final_judgement", unit_id=uid)
                for uid in coverage.get("planned_ids", []):
                    if uid not in covered_ids:
                        unresolved.append({"metric": metric, "unit_id": uid, "status": "missing_result"})
                        issue("missing_l1_result", unit_id=uid)
                continue

            judgement = data.get("judgement") or {}
            observed = data.get("observed_burden_input") or {}
            if projection and projection.get("evaluation_complete") is not True:
                if observed.get("schema_version") == "metric_owned_observed_defects_v1":
                    final_rows = observed.get("defects", [])
                    stats["defect_source"] = "observed_burden_input.defects"
                else:
                    final_rows = []
                    issue("missing_metric_owned_observations")
                    unresolved.append({"metric": metric, "status": "unverified_final_defects"})
            elif judgement.get("verdict") in {"valid", "invalid"}:
                final_rows = judgement.get("defects", []) if judgement["verdict"] == "invalid" else []
                stats["defect_source"] = "judgement.defects"
                if judgement["verdict"] == "valid" and judgement.get("defects"):
                    issue("valid_judgement_with_defects")
            else:
                final_rows = []
                issue("no_final_metric_judgement")
                unresolved.append({"metric": metric, "status": "unresolved",
                    **{k: deepcopy(judgement[k]) for k in ("reason", "missing_evidence") if k in judgement}})

            checks = {}
            for name in ("functional_check_ledger", "placement_check_ledger"):
                for check in (data.get(name) or {}).get("checks", []):
                    cid = check.get("check_id")
                    if cid:
                        checks.setdefault(cid, []).append(check)
            measurements = {}
            for measurement in (data.get("functional_measurement_bank") or {}).get("check_measurements", []):
                if measurement.get("check_id"):
                    measurements.setdefault(measurement["check_id"], []).append(measurement)
            for row in final_rows:
                if (row.get("claim_status", "final") != "final" or row.get("exempted") is True
                        or row.get("excluded") is True or row.get("withdrawn") is True):
                    issue("nonfinal_record_in_final_ledger", defect_id=row.get("claim_id"))
                    continue
                payload = add(row)
                if payload is None:
                    continue
                refs = row.get("check_refs") or []
                if not isinstance(refs, list):
                    issue("malformed_check_refs")
                    refs = []
                refs = list(dict.fromkeys([*refs, *([row["check_id"]] if row.get("check_id") else [])]))
                linked = []
                for ref in refs:
                    matches = checks.get(ref, [])
                    if len(matches) != 1:
                        issue("missing_check_reference" if not matches else "ambiguous_check_reference", check_id=ref)
                        continue
                    # Attach identity/measurement facts only, never another
                    # check's verdict/reason or its full acquisition history.
                    check = matches[0]
                    facts = defect_payload(check)
                    facts.update({k: v for k, v in defect_payload(check.get("result_row") or {}).items()
                                  if k in MEASUREMENTS})
                    facts = {k: v for k, v in facts.items() if k in IDENTITY_FIELDS or k in MEASUREMENTS
                             or k.endswith(("_m", "_eps", "_threshold", "_fraction"))}
                    bank = measurements.get(ref, [])
                    if len(bank) > 1:
                        issue("ambiguous_measurement_reference", check_id=ref)
                    elif bank:
                        involved = set()
                        for source in (row, check):
                            for key in IDENTITY_FIELDS:
                                if key.endswith("_ids"):
                                    involved.update(v for v in source.get(key, []) if isinstance(v, str))
                                elif key in {"object_id", "subject_id", "object_a", "object_b"} and source.get(key):
                                    involved.add(source[key])
                        facts["measurements"] = linked_measurements({k: bank[0][k] for k in (
                            "target_measurements", "pair_measurements", "measurement_extensions") if k in bank[0]}, involved)
                    linked.append(facts)
                if linked:
                    payload["related_checks"] = linked
            for uid in coverage.get("planned_ids", []):
                unit = units.get(uid, {})
                if unit.get("accepted") is not True:
                    unresolved.append({"metric": metric, "unit_id": uid, "status": "unresolved",
                        **{k: deepcopy(unit[k]) for k in ("target_ids", "subject_id", "missing_observations") if k in unit}})
            if not coverage and judgement.get("missing_evidence") and final_rows:
                unresolved.append({"metric": metric, "status": "unresolved",
                                   "missing_evidence": deepcopy(judgement["missing_evidence"])})
    return {"defects": defects, "unresolved": unresolved}, audit


def compact_report(report, *, scoring_summary=None):
    """Worker envelope: internal acceptance stays separate from model feedback."""
    if report.get("report_schema_version") != "scene_evaluation_report_v2":
        raise ValueError("unknown evaluator report contract")
    layer_scores = {}
    modules = {}
    for name, layer in report["layer_reports"].items():
        if name not in ("l1_physical_plausibility", "l3_scene_quality"):
            continue
        layer_scores[name] = {k: deepcopy(layer[k]) for k in ("score", "status", "terminal_state") if k in layer}
        layer_scores[name]["metrics"] = {
            m: {k: deepcopy(v[k]) for k in ("score", "partial_score", "status", "terminal_state") if k in v}
            for m, v in layer.get("metrics", {}).items() if m in RETAINED}
    if scoring_summary is not None:
        # Consume the sealed evaluator's persisted projection. Raw layer scores
        # predate its judgement-coverage projection and may not be scoreable.
        actual = scoring_summary.get("combined_score_100")
        expected = report.get("benchmark_score")
        if ((actual is None) != (expected is None)
                or actual is not None and not math.isclose(actual, expected * 100, abs_tol=1e-9)):
            raise ValueError("sealed persisted score disagrees with evaluation report")
        for layer in scoring_summary["layers"]:
            name = {"L1": "l1_physical_plausibility", "L3": "l3_scene_quality"}[layer["layer"]]
            layer_scores[name] = {k: deepcopy(layer[k]) for k in ("score", "status", "score_status") if k in layer}
            layer_scores[name]["metrics"] = {
                row["metric"]: {k: deepcopy(row[k]) for k in ("score", "status", "score_status") if k in row}
                for row in scoring_summary["metrics"]
                if row["layer"] == layer["layer"] and row["metric"] in RETAINED}
    for name in ("generic_validity", "scene_quality"):
        source = report.get("reports", {}).get(name, {})
        # Some envelopes place the final L1 metric rows only under layer_reports.
        if name == "generic_validity" and not source.get("metrics"):
            source = report["layer_reports"].get("l1_physical_plausibility", {})
        modules[name] = {"metrics": {m: v for m, v in source.get("metrics", {}).items() if m in RETAINED}}
    content, conversion_audit = defect_content(modules)
    score_view, score_audit = normalized_feedback_scores(scoring_summary) if scoring_summary is not None else (None, None)
    if score_view is not None:
        try:
            validate_acceptance(report)
        except RuntimeError:
            # Persist the failed/under-covered report and diagnostics so the
            # caller can stop/recover explicitly. Never normalize the remaining
            # good metrics into an apparently valid feedback score.
            score_view["feedback_score"] = None
            score_view["feedback_score_status"] = "uniform_score_ineligible"
            score_audit["uniform_score_eligible"] = False
        else:
            if score_view["feedback_score"] is None:
                raise ValueError("eligible uniform report has no retained feedback score")
    summary = report.get("judgement_coverage_summary", {})
    internal = {k: deepcopy(summary[k]) for k in (
        "schema_version", "eligible", "status", "judgement_coverage_fraction",
        "minimum_judgement_coverage", "infrastructure_failure_metrics", "score") if k in summary}
    result = {
        "report_schema_version": "scene_evaluation_report_v2", "feedback_view_version": VIEW_VERSION,
        **{k: deepcopy(report[k]) for k in ("evaluation_status", "benchmark_score", "benchmark_score_status")},
        "layer_reports": layer_scores, "defect_content": content, "conversion_audit": conversion_audit,
        "feedback_scores": score_view, "feedback_score_audit": score_audit,
        "judgement_coverage_summary": internal,
        "runner_outcome": {"uniform_score_acceptance": deepcopy(
            report.get("runner_outcome", {}).get("uniform_score_acceptance", {}))},
    }
    return result


def feedback_view(report, mode):
    if mode not in ("none", "scores", "diagnostics"):
        raise ValueError("unknown feedback mode")
    if report.get("feedback_view_version") != VIEW_VERSION:
        raise ValueError("expected defect-feedback v5 report; no implicit old-report migration")
    audit = {"schema_version": VIEW_VERSION, "mode": mode, "truncated": False,
             "removed_metrics": sorted(EXCLUDED), "passing_judgements_omitted": True,
             "coverage_internal_only": True, "full_report_preserved_separately": True}
    audit["conversion"] = deepcopy(report.get("conversion_audit", {}))
    if mode == "none": return {}, audit
    if report.get("feedback_scores") is None:
        raise ValueError("sealed scoring projection required for normalized feedback")
    view = {"schema_version": VIEW_VERSION, "score_range": [0, 1],
            "evaluation_status": report["evaluation_status"],
            "score_status": report["benchmark_score_status"], **deepcopy(report["feedback_scores"])}
    if mode == "diagnostics":
        if any(row["kind"] in {"conflicting_defect_id", "ambiguous_check_reference", "ambiguous_measurement_reference",
                               "missing_metric_owned_observations", "valid_judgement_with_defects", "nonfinal_record_in_final_ledger"}
               for row in report.get("conversion_audit", {}).get("issues", [])):
            raise FeedbackConversionError("ambiguous final defect conversion; review audit before continuation")
        view.update(deepcopy(report["defect_content"]))
    return pack(view), audit
