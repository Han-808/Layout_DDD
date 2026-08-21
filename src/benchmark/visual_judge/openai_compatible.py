from __future__ import annotations

import base64
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
from typing import Any

from benchmark.models import OpenAICompatibleModel, parse_json_object
from benchmark.visual_judge.adapters.legacy_openai_camera import (
    CAMERA_SELECTION_SYSTEM_PROMPT,
    LegacyOpenAICameraSelectionAdapter,
    _selector_candidate_order_key,
)
from benchmark.visual_judge.contracts import (
    validate_binary_judge_response,
    validate_canonical_metric_response,
    validate_generic_visual_response,
)
from benchmark.visual_judge.response_repair import (
    repair_binary_response_schema_once,
    repair_canonical_response_schema_once,
)
from benchmark.visual_judge.roles import (
    DecisionContract,
    VLMRole,
    vlm_audit_metadata,
)
from benchmark.visual_judge.interfaces import JudgeResult
from benchmark.visual_judge.l3_prompts import (
    L3_METRIC_PHASE_PROMPTS,
    L3_METRIC_RUBRICS,
)
from benchmark.visual_judge.camera_dsl import (
    CAMERA_OBSERVATIONS,
    METRIC_CAMERA_REQUIREMENTS,
    canonical_camera_metric,
)


SYSTEM_PROMPT = """You are the visual judge for a 3D scene-generation benchmark.
Use the rendered views as primary visual evidence. The canonical scene summary and deterministic
findings are navigation aids, not a replacement for inspecting the images. Do not invent hidden
objects or relations. Return exactly one JSON object with:
{"applicable":true,"score":0.0,"confidence":0.0,"summary":"...","issues":["..."],"evidence":["..."]}.
score and confidence must be between 0 and 1. If the supplied views cannot support this category,
return applicable=false, score=null, and explain why in summary."""

CANONICAL_METRIC_SYSTEM_PROMPT = """You judge exactly one narrowly scoped
metric for a frozen 3D scene.

Use the original request, authorized deviations, structured context, metric
rubric, metric-boundary rules, and supplied images. Do not judge neighboring
metrics.

Follow this decision order:

1. Establish the observable facts supported by the current evidence.
2. Decide whether a missing observable fact prevents application of the metric
   rubric.
3. If another visual observation could materially resolve that missing fact,
   return insufficient evidence with a structured evidence_request.
4. Otherwise apply the metric rubric and ownership rules to the established
   facts.
5. Return invalid only when an explicit observed fact meets the rubric's
   significant in-scope defect threshold.
6. Do not excuse an observed defect by imagining that scene objects are moved,
   rotated, resized, replaced, or removed, except when the metric rubric
   explicitly requires a counterfactual.
7. Ordinary articulation or handling intrinsic to using an object may be
   considered; rearranging the authored layout to repair the scene may not.
8. Semantic uncertainty is not evidence insufficiency. If the observable facts
   are adequate but the case remains a normative borderline below the stated
   invalid threshold, return valid and express uncertainty through confidence
   and reason.

Additional visual evidence can be acquired only through evidence_request.
Request evidence only for a missing observable fact, not to resolve subjective
preference or an undefined semantic threshold. Apply any supplied
metric_boundary_rules as authoritative metric-ownership rules.

Return exactly one JSON object:
{"evidence_status":"sufficient","verdict":"valid","confidence":0.0,"reason":"...",
"missing_evidence":[],"defects":[],"evidence_request":null}.
evidence_status must be sufficient or insufficient. verdict must be valid, invalid, or ambiguous.
If evidence is insufficient, verdict must be ambiguous, defects must be empty, and
evidence_request must be:
{"target_ids":["object_id"],"missing_observations":["exact_token"],
"view_goal":"concrete visual goal","metadata":{}}.
Use only target IDs in allowed_evidence_request_target_ids and exact observation tokens in
allowed_missing_observations from the user context. evidence_request is authoritative;
missing_evidence may be [] or an exact duplicate of its missing_observations. Do not place prose in
missing_evidence or missing_observations. Do not define camera constraints, poses, repair plans,
metric scope, or rubric in evidence_request. Invalid requires one or more explicit significant
defects in defects; otherwise return valid when evidence is sufficient. Each defect must contain
scope, target_ids, relation, and reason. Each defect must identify only the
objects whose own state fails this metric. If response_contract.defects lists
additional required fields or exact allowed values, include them without
renaming or free-form alternatives. confidence must be between 0 and 1.

When response_contract requires functional_check_results, resolve every listed
required functional check exactly once. Include check_id, exact target_ids,
observation_status, conclusion, and a brief evidence-grounded reason. A valid
verdict requires every listed check to resolve valid. Missing observation must
be marked missing/unresolved and must not be silently treated as normal. For
an evidence-complete response, every invalid required check must include its
exact check_id in exactly one defect.check_refs list. If any required check is
still missing/unresolved, the outer response remains evidence_status=insufficient
and verdict=ambiguous, defects must be empty, and already-observed invalid rows
remain only in functional_check_results until the requested coverage completes.
One physical defect may reference multiple invalid
checks only when defect.target_ids is a non-empty subset of every referenced
check's target_ids. If invalid checks have different target scopes, emit one
atomic defect per check instead of combining their target unions.
Baseline defects that did not originate from a required check may omit
check_refs. For an invalid clearance check, also return affected_object_ids,
cause_kind (external_object or self_layout), and causal_object_ids. Identify
the actual external blocker in causal_object_ids; for self_layout,
causal_object_ids must equal the affected clearance target IDs and must never
be empty. scoring_target_ids is deterministic bookkeeping derived from the
validated causal_object_ids, so do not make a second ownership decision with
that field.
Nearby causal candidates are routing priors, not a whitelist; use any legal
scene object supported by the evidence. The defect target_ids must equal the
deterministically derived scoring_target_ids in the validated result.

For semantic placement, resolve every required placement check exactly once.
Use only support_and_height, scene_zone, or contextual_anchor. A placement
defect must reference its check_id and may target only that check's subject_id;
context_ids are never defect owners. First resolve the static-location claim
independently. Only after finding that same Placement claim invalid may you
deduplicate its burden against one supplied final Functional event by returning
conclusion=excluded_function_owned, that exact function_event_ref, and
same_physical_event=true. Shared object identity, a similar reason, or mere
role overlap is never sufficient. If baseline review finds a missed placement
issue, emit a strictly
typed judge_originated_placement_results item with exactly proposal_id,
subject_id, context_ids, check_type, observation_goal, observation_status,
conclusion, reason, and severity. Its defect.check_id must equal proposal_id;
the Controller derives the stable check ID. Resolve it in the same call only
when current evidence is sufficient; otherwise request evidence and place a
proposal containing exactly proposal_id, subject_id, context_ids, check_type,
and observation_goal in evidence_request.metadata.placement_check_proposal.
Never emit observation_kind or placement_check_type."""

# Appended separately so the generic canonical response envelope remains
# stable while Residual Placement can require an auditable two-stage synthesis.
CANONICAL_METRIC_SYSTEM_PROMPT += """

When response_contract requires group_global_observations, inspect every exact
group once before the final verdict. Return one row per exact group_id with its
unchanged object_ids, a concise global_position_observation, related_group_ids,
an inter_group_observation, evidence_sufficiency, and residual_issue_candidate.
These rows organize visual reasoning and have no independent scoring authority.
Use only group membership; never infer truth from grouping labels, reasons, or
formation edges. After all rows are complete, synthesize the final verdict and
emit only novel scene_zone or contextual_anchor Placement results.
"""

_BUDGET_EXHAUSTION_FORCED_CHOICE_INSTRUCTION = """This is the terminal
budget-exhaustion adjudication. No additional visual evidence can be acquired.
The supplied images are all available views that fit the configured visual
context limit. You must choose the more defensible binary conclusion now.
Return evidence_status="sufficient" and verdict exactly "valid" or "invalid".
"ambiguous", "insufficient", and evidence_request are forbidden. If no clear,
significant in-scope defect is established, choose valid. Choose invalid only
with one or more explicit in-scope defects. Express residual uncertainty only
through confidence and reason. For this terminal call, "sufficient" means
sufficient to make the required binary choice under the remaining uncertainty;
it does not claim that every possible camera view was observed. When required
functional or placement checks are present, return every exact check row and use
observation_status="inferred_under_budget" for a check that must be resolved
from the bounded available context; missing/unresolved is not allowed in this
terminal call."""

P0B_SYSTEM_PROMPT = """You adjudicate one ambiguous geometry event in a 3D scene benchmark.
Use the natural-language prompt and extracted relationships only to understand intended semantics.
Use detector evidence for measured geometry and inspect supplied images when present. Generator
relationships are claims, not automatic exemptions. Highlighted diagnostic views use the supplied
color legend; gray geometry is non-target context, not missing scene content. Decide whether this event constitutes a
structural error. Return exactly one JSON object:
{"verdict":"valid","confidence":0.0,"reason":"..."}.
verdict must be exactly valid or invalid. No abstention, not-applicable, insufficient-evidence,
continuous score, or third verdict is allowed. confidence must be between 0 and 1."""

RELATION_SYSTEM_PROMPT = """You adjudicate one explicit spatial relationship in a 3D scene benchmark.
The relation was routed because it is outside the frozen deterministic registry or because deterministic
proxy evidence could not safely resolve its semantic realization. Routing carries no valid or invalid
prior. Judge only whether that exact relationship is satisfied by the generated scene. Use the original
prompt, structured relation, detector evidence, and supplied rendered views. Canonical object geometry
is supporting context, not permission to invent hidden evidence. Return exactly one JSON object:
{"verdict":"valid","confidence":0.0,"reason":"..."}.
verdict must be exactly valid or invalid. No abstention, not-applicable, insufficient-evidence,
continuous score, or third verdict is allowed. confidence must be between 0 and 1."""

_EVIDENCE_AWARE_BINARY_OUTPUT_CONTRACT = """Return exactly one JSON object:
{"status":"valid","confidence":0.0,"reason":"...","defects":[],
"evidence_request":null}.
status must be valid, invalid, or need_more_evidence. valid and invalid must use
evidence_request=null. defects must always be exactly the empty JSON list []; put every semantic
explanation in reason and never return strings or objects inside defects. If the supplied views
cannot support a safe binary conclusion, do not guess: return need_more_evidence with defects=[]
and a structured evidence_request:
{"target_ids":["object_id"],"missing_observations":["..."],"view_goal":"...","metadata":{}}.
The request must identify the targets, missing technical observation, and a concrete evidence view
goal. missing_observations must contain only exact Camera DSL tokens from the
allowed_missing_observations supplied in the user context. The finite vocabulary is:
target_visible, joint_visibility, contact_surface_visible, support_chain_visible,
architecture_plane_visible, front_back_disambiguated, depth_baseline_available,
group_context_visible, interaction_side_visible, approach_zone_visible,
limited_local_context, global_context_preserved, occluder_avoided.
Do not put prose in missing_observations. confidence must be between 0 and 1."""

P0B_CONTROL_SYSTEM_PROMPT = (
    P0B_SYSTEM_PROMPT.split("Return exactly one JSON object:", 1)[0]
    + _EVIDENCE_AWARE_BINARY_OUTPUT_CONTRACT
)
RELATION_CONTROL_SYSTEM_PROMPT = (
    RELATION_SYSTEM_PROMPT.split("Return exactly one JSON object:", 1)[0]
    + _EVIDENCE_AWARE_BINARY_OUTPUT_CONTRACT
)

SPATIAL_FIDELITY_SYSTEM_PROMPT = """You adjudicate one candidate Spatial Fidelity issue in a
coarse-grained 3D scene benchmark. A statistical detector routed this event; routing is evidence to
inspect, not a valid/invalid prior. For scale, decide whether the object's visible real-world size is
semantically implausible, allowing legitimate variants and unusual but coherent designs. For
co-occurrence, decide whether the object-category combination is semantically incoherent in this
particular requested scene; dataset rarity or absence alone is never an error. Use the canonical
measurements and statistical packet together with the rendered views. Return exactly one JSON object:
{"verdict":"valid","confidence":0.0,"reason":"..."}.
verdict must be exactly valid or invalid. No abstention, not-applicable, insufficient-evidence,
continuous score, or third verdict is allowed. confidence must be between 0 and 1."""

CATEGORY_RUBRICS = {
    "visual_quality": (
        "Judge holistic visual coherence, commonsense plausibility, style/appearance consistency, "
        "proportions, and whether the scene looks intentionally composed. Do not deduct twice for "
        "a deterministic geometry issue unless its visual consequence is visible."
    ),
    "prompt_fidelity": (
        "Judge only whether the visible scene follows the supplied natural-language request. "
        "Check requested objects, attributes, rough counts, and explicitly requested spatial relations. "
        "Treat ordinary synonyms and subtypes as the same broad object identity. Internal object-mapping "
        "confidence, unmatched candidates, resolver requests, and retrieval metadata are non-scoring alignment "
        "diagnostics: do not reward or penalize the generator for them. Judge visible prompt adherence directly, "
        "including color, material, style, count, placement, and relations."
    ),
    "structural_validity": (
        "Use the images to resolve only ambiguous structural findings called out by deterministic evidence. "
        "Do not override exact schema, boundary, or collision calculations without visible contradictory evidence."
    ),
    "functional_semantic_fidelity": (
        "Judge prompt-conditioned functional semantics only. Global evidence checks the requested "
        "room/scene type, broad functional intent, and required functional areas. Check local "
        "functionality only when the frozen prompt contract explicitly requests that function. "
        "Do not judge generic object pairing, scale, style, physical plausibility, or unspecified "
        "local functionality. Invalid requires at least one explicit significant unmet requirement; "
        "otherwise return valid when evidence is sufficient."
    ),
    **L3_METRIC_RUBRICS,
}


DEFAULT_JUDGE_COMPLETION_MAX_TOKENS = 8192


class OpenAICompatibleVLMJudge:
    """Multimodal judge shared by MNET localhost and remote OpenAI-style APIs."""

    def __init__(
        self,
        model: OpenAICompatibleModel,
        *,
        max_images: int = 8,
        max_context_chars: int = 30000,
        response_format_json: bool | None = None,
    ) -> None:
        self.model = model
        self.max_images = max(1, int(max_images))
        self.max_context_chars = max(1000, int(max_context_chars))
        self.response_format_json = (
            bool(getattr(model, "response_format_json", True))
            if response_format_json is None
            else bool(response_format_json)
        )

    def _audit_result(
        self,
        result: dict[str, Any],
        *,
        role: VLMRole,
        decision_contract: DecisionContract,
        judge_method: str,
        images_used: list[str],
        request_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result.update(
            vlm_audit_metadata(
                role,
                decision_contract=decision_contract,
                judge_method=judge_method,
            )
        )
        result["model"] = self.model.model_id
        result["endpoint"] = self.model.endpoint
        result["images_used"] = list(images_used)
        result["request_metadata"] = (
            dict(self.model.last_request_metadata)
            if request_metadata is None
            else dict(request_metadata)
        )
        return result

    def evaluate(self, request: dict) -> dict:
        """Compatibility score wrapper with a deterministic evidence preflight."""

        if not isinstance(request, dict):
            raise TypeError("VLM judge request must be a JSON object")
        gate = _check_standalone_visual_evidence(request)
        if gate is not None:
            return self._audit_result(
                {
                    "applicable": False,
                    "score": None,
                    "confidence": 0.0,
                    "summary": gate,
                    "issues": [],
                    "evidence": [],
                },
                role=VLMRole.JUDGE,
                decision_contract=DecisionContract.GENERIC_VISUAL_SCORE,
                judge_method="evaluate",
                images_used=[],
                request_metadata={},
            )
        return self._evaluate_raw(request)

    def _evaluate_raw(self, request: dict) -> dict:
        if not isinstance(request, dict):
            raise TypeError("VLM judge request must be a JSON object")
        category = str(request.get("category") or "visual_quality")
        paths = [Path(str(value)).expanduser() for value in request.get("render_evidence", [])]
        selected = paths[: self.max_images]
        missing = [str(path) for path in selected if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"VLM render evidence does not exist: {missing}")

        audit = vlm_audit_metadata(
            VLMRole.JUDGE,
            decision_contract=DecisionContract.GENERIC_VISUAL_SCORE,
            judge_method="evaluate",
        )
        context = {
            **audit,
            "category": category,
            "rubric": CATEGORY_RUBRICS.get(category, "Judge this category from the supplied evidence."),
            "natural_language_request": request.get("prompt"),
            "canonical_scene": request.get("scene_summary"),
            "deterministic_evidence": request.get("deterministic_evidence"),
            "view_names": _generic_view_names(selected),
        }
        context_text = _budgeted_context_json(
            context,
            self.max_context_chars,
            priority_keys=(
                "vlm_role",
                "decision_contract",
                "judge_method",
                "category",
                "deterministic_evidence",
                "rubric",
                "natural_language_request",
                "view_names",
            ),
        )
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": "Evaluate this scene under the specified category.\n" + context_text,
            }
        ]
        content.extend(
            {"type": "image_url", "image_url": {"url": _image_data_url(path)}}
            for path in selected
        )
        raw = self.model.chat_messages(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            response_format_json=self.response_format_json,
            call_type=f"vlm_judge.{category}",
        )
        result = parse_json_object(raw)
        validate_generic_visual_response(result)
        return self._audit_result(
            result,
            role=VLMRole.JUDGE,
            decision_contract=DecisionContract.GENERIC_VISUAL_SCORE,
            judge_method="evaluate",
            images_used=[str(path.resolve()) for path in selected],
        )

    def adjudicate_scene_quality(self, request: dict) -> dict:
        """Return the strict canonical L3 verdict contract."""

        return self._controlled_adjudicate(
            "adjudicate_scene_quality",
            request,
        )

    def screen_scene_quality(self, request: dict) -> dict:
        """Run an explicit non-rendering L3 routing screen.

        Screening is owned by the metric evaluator.  It deliberately bypasses
        the camera Controller because an invalid/insufficient screen is a route
        to visual confirmation rather than a final metric verdict.
        """

        if (
            str(request.get("decision_mode") or "").strip().lower()
            != "screen"
        ):
            raise ValueError(
                "screen_scene_quality requires decision_mode='screen'"
            )
        return self._adjudicate_scene_quality_raw(request)

    def _adjudicate_scene_quality_raw(self, request: dict) -> dict:
        return self._adjudicate_canonical_metric(
            request,
            family="scene_quality",
            judge_method="adjudicate_scene_quality",
        )

    def adjudicate_functional_semantic(self, request: dict) -> dict:
        """Return the strict canonical L2 functional-semantic verdict contract."""

        return self._controlled_adjudicate(
            "adjudicate_functional_semantic",
            request,
        )

    def _adjudicate_functional_semantic_raw(
        self,
        request: dict,
    ) -> dict:
        result = self._adjudicate_canonical_metric(
            request,
            family="specification_fidelity",
            judge_method="adjudicate_functional_semantic",
        )
        if result.get("verdict") == "ambiguous":
            result["canonical_verdict"] = "ambiguous"
            result["response_adapter"] = (
                "functional_semantic_insufficient_evidence_compat_v1"
            )
            result["verdict"] = "insufficient_evidence"
        result.setdefault(
            "router_state",
            "not_suspicious"
            if result.get("verdict") == "valid"
            else "suspicious"
            if result.get("verdict") == "invalid"
            else "insufficient_evidence",
        )
        return result

    def _adjudicate_canonical_metric(
        self,
        request: dict,
        *,
        family: str,
        judge_method: str,
    ) -> dict:
        if not isinstance(request, dict):
            raise TypeError("canonical metric judge request must be a JSON object")
        metric = str(request.get("metric") or request.get("category") or "")
        if metric not in {
            "functional_semantic_fidelity",
            "scale_consistency",
            "object_pairing_consistency",
            "style_consistency",
            "functional_consistency",
            "semantic_placement_consistency",
        }:
            raise ValueError(f"unsupported canonical metric {metric!r}")
        allowed_missing_observations = (
            _allowed_canonical_camera_observations(metric)
        )
        allowed_evidence_request_target_ids = (
            _canonical_evidence_request_target_ids(request)
        )
        allowed_defect_target_ids = _canonical_defect_target_ids(request)
        available_paths = [
            Path(str(value)).expanduser()
            for value in request.get("render_evidence", [])
        ]
        forced_choice = _budget_exhaustion_finalization(request)
        selected, forced_choice_evidence = _select_judge_visual_paths(
            available_paths,
            max_images=self.max_images,
            forced_choice=forced_choice is not None,
        )
        missing_paths = [str(path) for path in selected if not path.is_file()]
        if missing_paths:
            raise FileNotFoundError(
                f"canonical metric render evidence does not exist: {missing_paths}"
            )
        text_only_screen = bool(
            metric
            in {"scale_consistency", "object_pairing_consistency"}
            and request.get("evidence_phase") == "json_screen"
            and request.get("decision_mode") == "screen"
        )
        if not selected and not text_only_screen:
            if forced_choice is not None:
                raise ValueError(
                    "terminal budget-exhaustion adjudication requires at "
                    "least one rendered visual"
                )
            return self._audit_result(
                {
                    "evidence_status": "insufficient",
                    "verdict": "ambiguous",
                    "confidence": 0.0,
                    "reason": "no rendered evidence was supplied",
                    "missing_evidence": ["target_visible"],
                    "defects": [],
                    "evidence_request": {
                        "target_ids": list(
                            allowed_evidence_request_target_ids[:1]
                            or ("scene",)
                        ),
                        "missing_observations": ["target_visible"],
                        "view_goal": (
                            "show the requested metric scope in a "
                            "decodable visual view"
                        ),
                        "metadata": {},
                    },
                },
                role=VLMRole.JUDGE,
                decision_contract=DecisionContract.CANONICAL_METRIC,
                judge_method=judge_method,
                images_used=[],
                request_metadata={},
            )
        audit = vlm_audit_metadata(
            VLMRole.JUDGE,
            decision_contract=DecisionContract.CANONICAL_METRIC,
            judge_method=judge_method,
        )
        required_functional_checks = (
            _required_functional_checks_from_request(request)
        )
        required_functional_checks_context = (
            _judge_facing_required_functional_checks(
                required_functional_checks
            )
        )
        functional_preflight = _functional_preflight_response(
            request,
            selected_paths=selected,
            required_checks=required_functional_checks,
            allowed_missing_observations=allowed_missing_observations,
            allowed_target_ids=allowed_evidence_request_target_ids,
            forced_choice=forced_choice is not None,
        )
        if functional_preflight is not None:
            return self._audit_result(
                functional_preflight,
                role=VLMRole.JUDGE,
                decision_contract=DecisionContract.CANONICAL_METRIC,
                judge_method=judge_method,
                images_used=[
                    str(path.resolve()) for path in selected
                ],
                request_metadata={
                    "functional_evidence_preflight": {
                        "applied": True,
                        "model_call_skipped": True,
                    }
                },
            )
        required_placement_checks = (
            _required_placement_checks_from_request(request)
        )
        required_placement_checks_context = (
            _judge_facing_required_placement_checks(
                required_placement_checks
            )
        )
        deferred_placement_checks_context = (
            _judge_facing_required_placement_checks(
                request.get("deferred_placement_checks")
            )
        )
        functional_ownership_ledger = request.get(
            "functional_ownership_ledger"
        )
        if (
            metric == "semantic_placement_consistency"
            and isinstance(functional_ownership_ledger, dict)
        ):
            from benchmark.evaluator.scene_quality.functional_ownership import (
                validate_functional_ownership_ledger,
            )

            functional_ownership_ledger = (
                validate_functional_ownership_ledger(
                    functional_ownership_ledger,
                    known_object_ids=_scene_known_ids_for_request(request),
                )
            )
        raw_functional_probe_context = request.get(
            "functional_probe_evidence"
        )
        raw_functional_measurements_context = (
            deepcopy(
                raw_functional_probe_context.get(
                    "functional_measurements"
                )
            )
            if isinstance(raw_functional_probe_context, dict)
            and isinstance(
                raw_functional_probe_context.get(
                    "functional_measurements"
                ),
                dict,
            )
            else None
        )
        (
            functional_measurements_context,
            functional_measurement_scope_audit,
        ) = _scope_soft_functional_measurements(
            raw_functional_measurements_context,
            required_checks=required_functional_checks_context,
        )
        functional_probe_context = _judge_facing_functional_probe_context(
            raw_functional_probe_context,
            required_checks_supplied=bool(
                required_functional_checks_context
            ),
        )
        residual_placement_phase = bool(
            metric == "semantic_placement_consistency"
            and str(request.get("evidence_phase") or "")
            == "residual_global_placement_review"
        )
        canonical_scene_context = request.get("scene_summary")
        natural_language_request = request.get("prompt")
        if residual_placement_phase:
            canonical_scene_context = (
                _residual_placement_canonical_scene(
                    request.get("scene_summary")
                )
            )
            natural_language_request = None
        context = {
            **audit,
            "family": family,
            "metric": metric,
            "metric_prompt_version": request.get("metric_prompt_version"),
            "metric_boundary_rules": request.get("metric_boundary_rules"),
            "rubric": request.get("metric_rubric") or CATEGORY_RUBRICS[metric],
            "metric_rubric": request.get("metric_rubric"),
            "natural_language_request": natural_language_request,
            "authorized_deviations": request.get("authorized_deviations"),
            "metric_scope": request.get("judgment_scope"),
            "response_contract": request.get("response_contract"),
            "claims": request.get("claims"),
            "components": request.get("components"),
            "object_groups": request.get("object_groups"),
            "target_scope": request.get("target_scope"),
            "context_object_ids": request.get("context_object_ids"),
            "defect_attribution": request.get("defect_attribution"),
            "structured_context_policy": request.get(
                "structured_context_policy"
            ),
            "functional_probe_evidence": functional_probe_context,
            "functional_measurements": functional_measurements_context,
            "required_functional_checks": (
                required_functional_checks_context
            ),
            "required_placement_checks": (
                required_placement_checks_context
            ),
            "deferred_placement_checks": (
                deferred_placement_checks_context
            ),
            "functional_relation_scope": request.get(
                "functional_relation_scope"
            ),
            "placement_discovery": request.get(
                "placement_discovery"
            ),
            "placement_residual_context": request.get(
                "placement_residual_context"
            ),
            "functional_ownership_ledger": (
                _judge_facing_functional_ownership_ledger(
                    functional_ownership_ledger
                )
            ),
            "placement_check_policy": request.get(
                "placement_check_policy"
            ),
            "causal_object_catalog": request.get(
                "causal_object_catalog"
            ),
            "placement_severity_policy": request.get(
                "placement_severity_policy"
            ),
            "routed_screen_claims": request.get(
                "routed_screen_claims"
            ),
            "visual_style_spec": request.get("visual_style_spec"),
            "canonical_scene": canonical_scene_context,
            "deterministic_evidence": request.get("deterministic_evidence"),
            "allowed_missing_observations": list(
                allowed_missing_observations
            ),
            "allowed_evidence_request_target_ids": list(
                allowed_evidence_request_target_ids
            ),
            "allowed_defect_target_ids": list(
                allowed_defect_target_ids
            ),
            "evidence_phase": request.get("evidence_phase"),
            "decision_mode": request.get("decision_mode"),
            "phase_instruction": _canonical_phase_instruction(
                metric=metric,
                evidence_phase=request.get("evidence_phase"),
                decision_mode=request.get("decision_mode"),
            ),
            "view_names": _generic_view_names(selected),
        }
        if required_functional_checks_context:
            context["functional_measurement_policy"] = {
                "evidence_role": "soft_supporting_evidence",
                "decision_authority": "none",
                "missing_or_unavailable_policy": (
                    "Continue the visual judgement and make the most "
                    "educated binary choice permitted by the evidence. "
                    "Do not treat a missing deterministic measurement "
                    "alone as a reason to refuse judgement."
                ),
            }
        if deferred_placement_checks_context:
            context["phase_instruction"] = (
                str(context["phase_instruction"]).rstrip()
                + " Checks in deferred_placement_checks belong to a later "
                "group-local stage. Do not adjudicate them and do not emit "
                "their defects or result rows in this global call."
            )
        if forced_choice is not None:
            context["budget_exhaustion_finalization"] = {
                **forced_choice,
                **forced_choice_evidence,
            }
            context["phase_instruction"] = (
                str(context["phase_instruction"]).rstrip()
                + " "
                + _BUDGET_EXHAUSTION_FORCED_CHOICE_INSTRUCTION
            )
        context_text = _budgeted_context_json(
            context,
            self.max_context_chars,
            priority_keys=(
                "vlm_role",
                "decision_contract",
                "judge_method",
                "metric",
                "placement_residual_context",
                "required_functional_checks",
                "required_placement_checks",
                "deferred_placement_checks",
                "response_contract",
                "functional_measurement_policy",
                "functional_measurements",
                "functional_probe_evidence",
                "metric_prompt_version",
                "metric_boundary_rules",
                "rubric",
                "metric_rubric",
                "natural_language_request",
                "authorized_deviations",
                "metric_scope",
                "claims",
                "components",
                "object_groups",
                "target_scope",
                "context_object_ids",
                "defect_attribution",
                "structured_context_policy",
                "functional_relation_scope",
                "placement_discovery",
                "functional_ownership_ledger",
                "placement_check_policy",
                "causal_object_catalog",
                "placement_severity_policy",
                "routed_screen_claims",
                "visual_style_spec",
                "deterministic_evidence",
                "allowed_missing_observations",
                "allowed_evidence_request_target_ids",
                "allowed_defect_target_ids",
                "evidence_phase",
                "decision_mode",
                "phase_instruction",
                "budget_exhaustion_finalization",
                "view_names",
            ),
        )
        _require_functional_checks_in_model_context(
            context_text,
            required_checks=required_functional_checks_context,
        )
        functional_measurement_delivery_audit = (
            _functional_measurement_delivery_audit(
                context_text,
                required_checks=required_functional_checks_context,
                measurements=functional_measurements_context,
                source_scope_audit=functional_measurement_scope_audit,
            )
        )
        _require_placement_checks_in_model_context(
            context_text,
            required_checks=required_placement_checks_context,
        )
        placement_residual_context_delivery = (
            _require_placement_residual_context_in_model_context(
                context_text,
                source_context=request.get(
                    "placement_residual_context"
                ),
            )
        )
        _require_functional_ownership_in_model_context(
            context_text,
            ledger=context.get("functional_ownership_ledger"),
        )
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": "Adjudicate this canonical metric only.\n" + context_text,
            }
        ]
        content.extend(
            {"type": "image_url", "image_url": {"url": _image_data_url(path)}}
            for path in selected
        )
        call_type = (
            f"vlm_judge.screen.{metric}"
            if request.get("decision_mode") == "screen"
            else f"vlm_judge.canonical.{metric}"
        )
        if forced_choice is not None:
            call_type += ".forced_choice"
        messages = [
            {
                "role": "system",
                "content": (
                    CANONICAL_METRIC_SYSTEM_PROMPT
                    + (
                        "\n\n"
                        + _BUDGET_EXHAUSTION_FORCED_CHOICE_INSTRUCTION
                        if forced_choice is not None
                        else ""
                    )
                ),
            },
            {"role": "user", "content": content},
        ]
        allowed_scopes = (
            (
                request.get("judgment_scope")
                if isinstance(request.get("judgment_scope"), dict)
                else {}
            ).get("included")
            or []
        )
        response_contract = (
            request.get("response_contract")
            if isinstance(request.get("response_contract"), dict)
            else {}
        )
        defect_contract = (
            response_contract.get("defects")
            if isinstance(response_contract.get("defects"), dict)
            else {}
        )
        required_defect_fields = tuple(
            str(item)
            for item in defect_contract.get("fields") or ()
            if isinstance(item, str) and item.strip()
        )
        allowed_defect_field_values = (
            defect_contract.get("allowed_field_values")
            if isinstance(
                defect_contract.get("allowed_field_values"),
                dict,
            )
            else {}
        )

        def validate_response(result: dict[str, Any]) -> dict[str, Any]:
            judge_originated_placement_checks: list[
                dict[str, Any]
            ] = []
            if metric == "semantic_placement_consistency":
                from benchmark.evaluator.scene_quality.placement_checks import (
                    canonicalize_placement_defect_linkage,
                    normalize_judge_originated_placement_results,
                )

                (
                    result,
                    judge_originated_placement_checks,
                ) = normalize_judge_originated_placement_results(
                    result,
                    known_ids=_placement_known_ids_for_request(
                        request
                    ),
                    groups=_placement_groups_for_request(request),
                    existing_checks=deepcopy(
                        required_placement_checks
                    ),
                    expected_owner_stage=(
                        _expected_placement_owner_stage(request)
                    ),
                )
                result = canonicalize_placement_defect_linkage(
                    result,
                    required_checks=[
                        *deepcopy(required_placement_checks),
                        *judge_originated_placement_checks,
                    ],
                )
            if (
                metric
                in {
                    "functional_consistency",
                    "semantic_placement_consistency",
                }
                and (
                    required_functional_checks
                    or required_placement_checks
                    or judge_originated_placement_checks
                )
            ):
                from benchmark.evaluator.scene_quality.functional_checks import (
                    canonicalize_typed_invalid_envelope,
                )

                result = canonicalize_typed_invalid_envelope(result)
            if (
                metric == "functional_consistency"
                and required_functional_checks
            ):
                from benchmark.evaluator.scene_quality.functional_checks import (
                    canonicalize_clearance_causal_attribution,
                    canonicalize_functional_defect_check_linkage,
                )

                result = canonicalize_clearance_causal_attribution(
                    result,
                    required_checks=deepcopy(required_functional_checks),
                )
                result = canonicalize_functional_defect_check_linkage(
                    result,
                    required_checks=deepcopy(required_functional_checks),
                )
            result = _project_group_local_defects_to_scope(
                result,
                request=request,
                required_functional_checks=required_functional_checks,
            )
            if metric == "semantic_placement_consistency":
                _validate_pending_placement_proposal(
                    result,
                    request=request,
                )
            normalized = validate_canonical_metric_response(
                result,
                allowed_scopes=allowed_scopes,
                allowed_missing_observations=(
                    allowed_missing_observations
                ),
                allowed_defect_target_ids=allowed_defect_target_ids,
                allowed_evidence_request_target_ids=(
                    allowed_evidence_request_target_ids
                ),
                required_defect_fields=required_defect_fields,
                allowed_defect_field_values=(
                    allowed_defect_field_values
                ),
            )
            if (
                metric == "functional_consistency"
                and required_functional_checks
            ):
                # Keep the low-level OpenAI-compatible transport importable
                # without eagerly initializing the evaluator package.
                from benchmark.evaluator.scene_quality.functional_checks import (
                    validate_functional_check_results,
                )
                validate_functional_check_results(
                    normalized,
                    required_checks=deepcopy(
                        required_functional_checks
                    ),
                    invalid_verdict_requires_invalid_check=(
                        str(request.get("evidence_phase") or "")
                        == "cross_group_relation_review"
                    ),
                )
            if metric == "semantic_placement_consistency":
                from benchmark.evaluator.scene_quality.placement_checks import (
                    validate_placement_check_results,
                    validate_residual_group_global_observations,
                )

                ownership = request.get(
                    "functional_ownership_ledger"
                )
                validate_placement_check_results(
                    normalized,
                    required_checks=[
                        *deepcopy(required_placement_checks),
                        *judge_originated_placement_checks,
                    ],
                    function_events=list(
                        (
                            ownership
                            if isinstance(ownership, dict)
                            else {}
                        ).get("events")
                        or []
                    ),
                )
                if residual_placement_phase:
                    validated_group_coverage = (
                        validate_residual_group_global_observations(
                        normalized,
                        groups=_placement_groups_for_request(request),
                        )
                    )
                    supplied_group_coverage = normalized.get(
                        "group_global_observation_coverage"
                    )
                    if isinstance(supplied_group_coverage, dict) and isinstance(
                        supplied_group_coverage.get("source_by_group"),
                        dict,
                    ):
                        ungrounded = set(
                            validated_group_coverage.get(
                                "ungrounded_group_ids"
                            )
                            or []
                        )
                        source_by_group = deepcopy(
                            supplied_group_coverage["source_by_group"]
                        )
                        validated_group_coverage[
                            "source_by_group"
                        ] = source_by_group
                        validated_group_coverage[
                            "defaulted_group_ids"
                        ] = [
                            group_id
                            for group_id, source in source_by_group.items()
                            if source == "defaulted"
                            and group_id in ungrounded
                        ]
                    normalized[
                        "group_global_observation_coverage"
                    ] = validated_group_coverage
                    if normalized[
                        "group_global_observation_coverage"
                    ].get("complete") is not True:
                        normalized["evidence_ambiguous"] = True
            if forced_choice is not None:
                if normalized.get("evidence_status") != "sufficient":
                    raise ValueError(
                        "terminal budget-exhaustion adjudication requires "
                        "evidence_status=sufficient"
                    )
                if normalized.get("verdict") not in {"valid", "invalid"}:
                    raise ValueError(
                        "terminal budget-exhaustion adjudication forbids "
                        "ambiguous verdicts"
                    )
                if normalized.get("evidence_request") is not None:
                    raise ValueError(
                        "terminal budget-exhaustion adjudication cannot "
                        "request more evidence"
                    )
            return normalized

        fail_soft_fallback = None
        if metric == "functional_consistency" and required_functional_checks:
            from benchmark.evaluator.scene_quality.functional_checks import (
                salvage_functional_judge_response,
            )

            def fail_soft_fallback(
                value: dict[str, Any],
                initial_value: dict[str, Any],
            ) -> dict[str, Any]:
                candidate = salvage_functional_judge_response(
                    value,
                    required_checks=deepcopy(required_functional_checks),
                    fallback_value=initial_value,
                    retain_invalid=True,
                )
                try:
                    validate_response(candidate)
                except (TypeError, ValueError, KeyError):
                    return salvage_functional_judge_response(
                        value,
                        required_checks=deepcopy(
                            required_functional_checks
                        ),
                        fallback_value=initial_value,
                        retain_invalid=False,
                    )
                return candidate
        elif metric == "semantic_placement_consistency":
            from benchmark.evaluator.scene_quality.placement_checks import (
                salvage_residual_group_global_observations,
                salvage_placement_judge_response,
            )

            ownership = request.get("functional_ownership_ledger")
            function_events = list(
                (
                    ownership if isinstance(ownership, dict) else {}
                ).get("events")
                or []
            )

            def fail_soft_fallback(
                value: dict[str, Any],
                initial_value: dict[str, Any],
            ) -> dict[str, Any]:
                candidate = salvage_placement_judge_response(
                    value,
                    required_checks=deepcopy(required_placement_checks),
                    function_events=function_events,
                    retain_invalid=True,
                    fallback_value=initial_value,
                    known_ids=_placement_known_ids_for_request(request),
                    groups=_placement_groups_for_request(request),
                    expected_owner_stage=(
                        _expected_placement_owner_stage(request)
                    ),
                )
                if residual_placement_phase:
                    candidate = salvage_residual_group_global_observations(
                        candidate,
                        groups=_placement_groups_for_request(request),
                        fallback_value=initial_value,
                    )
                try:
                    validate_response(candidate)
                except (TypeError, ValueError, KeyError):
                    candidate = salvage_placement_judge_response(
                        value,
                        required_checks=deepcopy(
                            required_placement_checks
                        ),
                        function_events=function_events,
                        retain_invalid=False,
                        fallback_value=initial_value,
                        known_ids=_placement_known_ids_for_request(request),
                        groups=_placement_groups_for_request(request),
                        expected_owner_stage=(
                            _expected_placement_owner_stage(request)
                        ),
                    )
                    if residual_placement_phase:
                        candidate = (
                            salvage_residual_group_global_observations(
                                candidate,
                                groups=_placement_groups_for_request(
                                    request
                                ),
                                fallback_value=initial_value,
                            )
                        )
                    return candidate
                # Return the raw atomically salvaged shape.  The generic
                # repair wrapper validates it exactly once; returning the
                # already-normalized Placement value here would normalize
                # judge-originated checks a second time and lose their
                # derived registration ledger.
                return candidate

        result, schema_audit = repair_canonical_response_schema_once(
            model=self.model,
            messages=messages,
            response_format_json=self.response_format_json,
            call_type=call_type,
            judge_label=f"canonical {metric} Judge",
            validator=validate_response,
            force_binary_choice=forced_choice is not None,
            allowed_scopes=tuple(str(item) for item in allowed_scopes),
            allowed_target_ids=tuple(
                str(item)
                for item in allowed_defect_target_ids
            ),
            allowed_missing_observations=tuple(
                str(item) for item in allowed_missing_observations
            ),
            fail_soft_fallback=fail_soft_fallback,
        )
        request_metadata = dict(self.model.last_request_metadata)
        request_metadata["response_schema_validation"] = schema_audit
        if required_functional_checks_context:
            request_metadata["functional_measurement_delivery"] = deepcopy(
                functional_measurement_delivery_audit
            )
            if functional_measurement_delivery_audit.get(
                "evidence_ambiguous"
            ):
                result["evidence_ambiguous"] = True
                result["functional_measurement_delivery_audit"] = deepcopy(
                    functional_measurement_delivery_audit
                )
        if placement_residual_context_delivery is not None:
            request_metadata[
                "placement_residual_context_delivery"
            ] = deepcopy(placement_residual_context_delivery)
        if forced_choice is not None:
            request_metadata["budget_exhaustion_finalization"] = {
                **forced_choice,
                **forced_choice_evidence,
            }
        if request.get("metric_prompt_version") is not None:
            request_metadata["metric_prompt_version"] = str(
                request["metric_prompt_version"]
            )
        return self._audit_result(
            result,
            role=VLMRole.JUDGE,
            decision_contract=DecisionContract.CANONICAL_METRIC,
            judge_method=judge_method,
            images_used=[str(path.resolve()) for path in selected],
            request_metadata=request_metadata,
        )

    def adjudicate_p0b(self, request: dict) -> dict:
        return self._controlled_adjudicate("adjudicate_p0b", request)

    def _adjudicate_p0b_control(self, request: dict) -> dict:
        return self._adjudicate_p0b_raw(
            request,
            _allow_need_more_evidence=True,
        )

    def _adjudicate_p0b_raw(
        self,
        request: dict,
        *,
        _allow_need_more_evidence: bool = False,
    ) -> dict:
        if not isinstance(request, dict):
            raise TypeError("P0b judge request must be a JSON object")
        paths = [
            Path(str(value)).expanduser()
            for value in request.get("render_evidence", [])
        ]
        forced_choice = _budget_exhaustion_finalization(request)
        selected, forced_choice_evidence = _select_judge_visual_paths(
            paths,
            max_images=self.max_images,
            forced_choice=forced_choice is not None,
        )
        allow_need_more_evidence = (
            _allow_need_more_evidence and forced_choice is None
        )
        missing = [str(path) for path in selected if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"P0b render evidence does not exist: {missing}")
        audit = vlm_audit_metadata(
            VLMRole.JUDGE,
            decision_contract=DecisionContract.P0B_BINARY,
            judge_method="adjudicate_p0b",
        )
        context = {
            **audit,
            "metric": request.get("metric"),
            "metric_rubric": request.get("metric_rubric"),
            "candidate_selection_policy": request.get("candidate_selection_policy"),
            "collision_evidence_style_guide": request.get("collision_evidence_style_guide"),
            "visual_evidence_policy": request.get("visual_evidence_policy"),
            "event": request.get("event"),
            "detector_evidence": request.get("detector_evidence"),
            "objects": request.get("objects"),
            "architecture": request.get("architecture"),
            "natural_language_prompt": request.get("natural_language_prompt"),
            "extracted_relationships": request.get("extracted_relationships"),
            "view_names": _generic_view_names(selected),
            "view_evidence": _sanitize_outbound_view_evidence(
                request.get("local_render_evidence_metadata")
            ),
            "allowed_missing_observations": (
                _allowed_binary_camera_observations(
                    request.get("metric")
                )
            ),
        }
        if forced_choice is not None:
            context["budget_exhaustion_finalization"] = {
                **forced_choice,
                **forced_choice_evidence,
            }
        context_text = _budgeted_context_json(
            context,
            self.max_context_chars,
            priority_keys=(
                "vlm_role",
                "decision_contract",
                "judge_method",
                "metric",
                "detector_evidence",
                "event",
                "natural_language_prompt",
                "metric_rubric",
                "candidate_selection_policy",
                "collision_evidence_style_guide",
                "visual_evidence_policy",
                "view_names",
                "view_evidence",
                "allowed_missing_observations",
                "budget_exhaustion_finalization",
            ),
        )
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": "Adjudicate this P0b event.\n" + context_text,
            }
        ]
        content.extend(
            {"type": "image_url", "image_url": {"url": _image_data_url(path)}}
            for path in selected
        )
        messages = [
            {
                "role": "system",
                "content": (
                    P0B_CONTROL_SYSTEM_PROMPT
                    if allow_need_more_evidence
                    else P0B_SYSTEM_PROMPT
                ),
            },
            {"role": "user", "content": content},
        ]
        call_type = f"vlm_judge.p0b.{request.get('metric') or 'event'}"
        if forced_choice is not None:
            call_type += ".forced_choice"
        schema_audit = None
        if allow_need_more_evidence:
            result, schema_audit = (
                repair_binary_response_schema_once(
                    model=self.model,
                    messages=messages,
                    response_format_json=self.response_format_json,
                    call_type=call_type,
                    judge_label="P0b judge",
                    validator=lambda value: (
                        _normalize_evidence_aware_binary_response(
                            value,
                            judge_label="P0b judge",
                        )
                    ),
                )
            )
        else:
            raw = self.model.chat_messages(
                messages,
                response_format_json=self.response_format_json,
                call_type=call_type,
            )
            result = parse_json_object(raw)
            validate_binary_judge_response(
                result,
                judge_label="P0b judge",
                confidence_label="VLM judge",
            )
        request_metadata = dict(self.model.last_request_metadata)
        if schema_audit is not None:
            request_metadata["response_schema_validation"] = schema_audit
        if forced_choice is not None:
            request_metadata["budget_exhaustion_finalization"] = {
                **forced_choice,
                **forced_choice_evidence,
            }
        return self._audit_result(
            result,
            role=VLMRole.JUDGE,
            decision_contract=DecisionContract.P0B_BINARY,
            judge_method="adjudicate_p0b",
            images_used=[str(path.resolve()) for path in selected],
            request_metadata=request_metadata,
        )

    def adjudicate_relation(self, request: dict) -> dict:
        return self._controlled_adjudicate(
            "adjudicate_relation",
            request,
        )

    def _adjudicate_relation_control(self, request: dict) -> dict:
        return self._adjudicate_relation_raw(
            request,
            _allow_need_more_evidence=True,
        )

    def _adjudicate_relation_raw(
        self,
        request: dict,
        *,
        _allow_need_more_evidence: bool = False,
    ) -> dict:
        if not isinstance(request, dict):
            raise TypeError("relationship judge request must be a JSON object")
        paths = [
            Path(str(value)).expanduser()
            for value in request.get("render_evidence", [])
        ]
        forced_choice = _budget_exhaustion_finalization(request)
        selected, forced_choice_evidence = _select_judge_visual_paths(
            paths,
            max_images=self.max_images,
            forced_choice=forced_choice is not None,
        )
        allow_need_more_evidence = (
            _allow_need_more_evidence and forced_choice is None
        )
        missing = [str(path) for path in selected if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"relationship render evidence does not exist: {missing}")
        if not selected:
            raise ValueError("relationship adjudication requires at least one rendered view")
        audit = vlm_audit_metadata(
            VLMRole.JUDGE,
            decision_contract=DecisionContract.RELATION_BINARY,
            judge_method="adjudicate_relation",
        )
        context = {
            **audit,
            "family": request.get("family"),
            "explicit_relation_claim": request.get("relation"),
            "natural_language_prompt": request.get("natural_language_prompt"),
            "detector_evidence": request.get("detector_evidence"),
            "involved_objects": request.get("involved_objects"),
            "canonical_scene": request.get("scene_summary"),
            "view_names": _generic_view_names(selected),
            "allowed_missing_observations": (
                _allowed_binary_camera_observations(
                    "relation",
                    relation_type=_relation_type_from_request(request),
                )
            ),
        }
        if forced_choice is not None:
            context["budget_exhaustion_finalization"] = {
                **forced_choice,
                **forced_choice_evidence,
            }
        context_text = _budgeted_context_json(
            context,
            self.max_context_chars,
            priority_keys=(
                "vlm_role",
                "decision_contract",
                "judge_method",
                "family",
                "detector_evidence",
                "explicit_relation_claim",
                "natural_language_prompt",
                "involved_objects",
                "view_names",
                "allowed_missing_observations",
                "budget_exhaustion_finalization",
            ),
        )
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": "Adjudicate this explicit spatial relationship.\n" + context_text,
            }
        ]
        content.extend(
            {"type": "image_url", "image_url": {"url": _image_data_url(path)}}
            for path in selected
        )
        messages = [
            {
                "role": "system",
                "content": (
                    RELATION_CONTROL_SYSTEM_PROMPT
                    if allow_need_more_evidence
                    else RELATION_SYSTEM_PROMPT
                ),
            },
            {"role": "user", "content": content},
        ]
        call_type = (
            "vlm_judge.relationship."
            f"{request.get('family') or 'unknown'}"
        )
        if forced_choice is not None:
            call_type += ".forced_choice"
        schema_audit = None
        if allow_need_more_evidence:
            result, schema_audit = (
                repair_binary_response_schema_once(
                    model=self.model,
                    messages=messages,
                    response_format_json=self.response_format_json,
                    call_type=call_type,
                    judge_label="relationship judge",
                    validator=lambda value: (
                        _normalize_evidence_aware_binary_response(
                            value,
                            judge_label="relationship judge",
                        )
                    ),
                )
            )
        else:
            raw = self.model.chat_messages(
                messages,
                response_format_json=self.response_format_json,
                call_type=call_type,
            )
            result = parse_json_object(raw)
            validate_binary_judge_response(
                result,
                judge_label="relationship judge",
                confidence_label="VLM judge",
            )
        request_metadata = dict(self.model.last_request_metadata)
        if schema_audit is not None:
            request_metadata["response_schema_validation"] = schema_audit
        if forced_choice is not None:
            request_metadata["budget_exhaustion_finalization"] = {
                **forced_choice,
                **forced_choice_evidence,
            }
        return self._audit_result(
            result,
            role=VLMRole.JUDGE,
            decision_contract=DecisionContract.RELATION_BINARY,
            judge_method="adjudicate_relation",
            images_used=[str(path.resolve()) for path in selected],
            request_metadata=request_metadata,
        )

    def adjudicate_spatial_fidelity(self, request: dict) -> dict:
        return self._controlled_adjudicate(
            "adjudicate_spatial_fidelity",
            request,
        )

    def _adjudicate_spatial_fidelity_raw(
        self,
        request: dict,
    ) -> dict:
        if not isinstance(request, dict):
            raise TypeError("Spatial Fidelity judge request must be a JSON object")
        paths = [
            Path(str(value)).expanduser()
            for value in request.get("render_evidence", [])
        ]
        forced_choice = _budget_exhaustion_finalization(request)
        selected, forced_choice_evidence = _select_judge_visual_paths(
            paths,
            max_images=self.max_images,
            forced_choice=forced_choice is not None,
        )
        missing = [str(path) for path in selected if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Spatial Fidelity render evidence does not exist: {missing}")
        if not selected:
            raise ValueError("Spatial Fidelity adjudication requires at least one rendered view")
        audit = vlm_audit_metadata(
            VLMRole.JUDGE,
            decision_contract=DecisionContract.SPATIAL_FIDELITY_BINARY,
            judge_method="adjudicate_spatial_fidelity",
        )
        context = {
            **audit,
            "metric": request.get("metric"),
            "event": request.get("event"),
            "detector_evidence": request.get("detector_evidence"),
            "involved_objects": request.get("involved_objects"),
            "canonical_scene": request.get("scene_summary"),
            "natural_language_prompt": request.get("natural_language_prompt"),
            "view_names": _generic_view_names(selected),
        }
        if forced_choice is not None:
            context["budget_exhaustion_finalization"] = {
                **forced_choice,
                **forced_choice_evidence,
            }
        context_text = _budgeted_context_json(
            context,
            self.max_context_chars,
            priority_keys=(
                "vlm_role",
                "decision_contract",
                "judge_method",
                "metric",
                "event",
                "detector_evidence",
                "involved_objects",
                "natural_language_prompt",
                "view_names",
                "budget_exhaustion_finalization",
            ),
        )
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": "Adjudicate this routed Spatial Fidelity event.\n" + context_text,
            }
        ]
        content.extend(
            {"type": "image_url", "image_url": {"url": _image_data_url(path)}}
            for path in selected
        )
        call_type = (
            "vlm_judge.spatial_fidelity."
            f"{request.get('metric') or 'event'}"
        )
        if forced_choice is not None:
            call_type += ".forced_choice"
        raw = self.model.chat_messages(
            [
                {"role": "system", "content": SPATIAL_FIDELITY_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            response_format_json=self.response_format_json,
            call_type=call_type,
        )
        result = parse_json_object(raw)
        validate_binary_judge_response(
            result,
            judge_label="Spatial Fidelity judge",
            confidence_label="VLM judge",
        )
        request_metadata = dict(self.model.last_request_metadata)
        if forced_choice is not None:
            request_metadata["budget_exhaustion_finalization"] = {
                **forced_choice,
                **forced_choice_evidence,
            }
        return self._audit_result(
            result,
            role=VLMRole.JUDGE,
            decision_contract=DecisionContract.SPATIAL_FIDELITY_BINARY,
            judge_method="adjudicate_spatial_fidelity",
            images_used=[str(path.resolve()) for path in selected],
            request_metadata=request_metadata,
        )

    def _controlled_adjudicate(
        self,
        method_name: str,
        request: dict,
    ) -> dict:
        """Route direct public calls through the same strict compatibility loop."""

        # Lazy imports keep the low-level OpenAI transport independent from the
        # orchestration module at import time.  ``ControlledVLMJudge`` resolves
        # the private raw method, so this wrapper cannot recurse.
        from benchmark.visual_judge.control_config import (
            resolve_vlm_evaluation_control,
        )
        from benchmark.visual_judge.runtime import ControlledVLMJudge

        control = resolve_vlm_evaluation_control(
            existing_selector_available=False,
            judge_max_images=self.max_images,
        )
        wrapper = ControlledVLMJudge(
            self,
            control=control,
            strict=True,
        )
        call = getattr(wrapper, method_name)
        return call(request)

    def select_camera_views(self, request: dict) -> dict:
        return select_openai_compatible_camera_views(
            model=self.model,
            request=request,
            max_images=self.max_images,
            max_context_chars=self.max_context_chars,
            response_format_json=self.response_format_json,
        )


def select_openai_compatible_camera_views(
    *,
    model: OpenAICompatibleModel,
    request: dict[str, Any],
    max_images: int = 6,
    max_context_chars: int = 30000,
    response_format_json: bool | None = None,
) -> dict[str, Any]:
    """Execute the frozen query-cov/active camera contract without a Judge."""

    return LegacyOpenAICameraSelectionAdapter(
        model,
        context_serializer=_budgeted_context_json,
        image_encoder=_normalized_rgb_png_data_url,
        max_images=max_images,
        max_context_chars=max_context_chars,
        response_format_json=response_format_json,
    ).select(request)


def _normalize_evidence_aware_binary_response(
    value: dict[str, Any],
    *,
    judge_label: str,
) -> dict[str, Any]:
    if value.get("status") is not None:
        result = JudgeResult.from_value(value)
        if result.defects:
            raise ValueError(
                f"{judge_label} defects must be exactly an empty JSON list"
            )
        return result.to_dict()
    validate_binary_judge_response(
        value,
        judge_label=judge_label,
        confidence_label="VLM judge",
    )
    return JudgeResult.from_value(
        {
            "status": value["verdict"],
            "confidence": value["confidence"],
            "reason": value["reason"],
            "defects": [],
        }
    ).to_dict()


def _budget_exhaustion_finalization(
    request: dict[str, Any],
) -> dict[str, Any] | None:
    value = request.get("budget_exhaustion_finalization")
    if not isinstance(value, dict) or value.get("required") is not True:
        return None
    stop_reason = str(value.get("trigger_stop_reason") or "").strip()
    if not stop_reason:
        raise ValueError(
            "budget_exhaustion_finalization requires trigger_stop_reason"
        )
    return {
        "required": True,
        "trigger": "camera_or_evidence_budget_exhausted",
        "trigger_stop_reason": stop_reason,
        "termination_kind": (
            "budget_exhausted"
            if "exhausted" in stop_reason
            else "acquisition_unavailable"
        ),
        "ambiguity_before_forcing": bool(
            value.get("ambiguity_before_forcing")
        ),
        "allowed_verdicts": ["valid", "invalid"],
        "previous_missing_observations": [
            str(item)
            for item in value.get("previous_missing_observations") or []
            if str(item).strip()
        ],
        "previous_evidence_request": deepcopy(
            value.get("previous_evidence_request")
        ),
    }


def _select_judge_visual_paths(
    paths: list[Path],
    *,
    max_images: int,
    forced_choice: bool,
) -> tuple[list[Path], dict[str, Any]]:
    limit = max(1, int(max_images))
    if len(paths) <= limit:
        selected = list(paths)
        policy = "all_available_within_visual_context"
    elif limit == 1:
        # Once the global context cannot coexist with a repair view, the most
        # recent targeted observation is the strongest available evidence.
        selected = [paths[-1]]
        policy = "most_recent_targeted_only"
    else:
        # Packet order is global/context first and newly acquired evidence
        # last. Preserve the anchor, then spend the remaining visual context
        # on the most recent repair observations. This applies before terminal
        # forcing too: otherwise a successful late repair could be acquired
        # but omitted from the next ordinary Judge call.
        selected = [paths[0], *paths[-(limit - 1) :]]
        policy = "global_anchor_plus_most_recent"
    return selected, {
        "available_visual_count": len(paths),
        "selected_visual_count": len(selected),
        "dropped_visual_count": max(0, len(paths) - len(selected)),
        "configured_visual_context_limit": limit,
        "visual_selection_policy": policy,
    }


def _generic_view_names(paths: list[Path]) -> list[str]:
    """Return request-local aliases instead of exposing local filenames."""

    return [f"image_{index:02d}" for index, _ in enumerate(paths)]


def _canonical_phase_instruction(
    *,
    metric: str,
    evidence_phase: Any,
    decision_mode: Any,
) -> str:
    phase = str(evidence_phase or "final").strip().lower()
    mode = str(decision_mode or "final").strip().lower()
    acquire_more = "Additional visual evidence can be acquired."
    if (
        phase == "json_screen"
        and mode == "screen"
        and metric == "scale_consistency"
    ):
        return (
            "This is a structured-data routing screen, not a final scale "
            "decision. Use category, physical dimensions, authorized "
            "deviations, and nearby structured size references. Route a "
            "target to visual confirmation only when its physical dimensions "
            "remain materially suspicious after allowing plausible subtype "
            "and product variation. Do not use image-space appearance, "
            "placement, orientation, style, or functionality. In this "
            "screening contract, invalid means visual confirmation required, "
            "not a final scale defect. A weak, uncertain, or metadata-dependent "
            "discrepancy is review_required rather than a defect. Return "
            "insufficient/ambiguous when structured identity or dimensions "
            "cannot safely complete the screen. "
            + acquire_more
        )
    if (
        phase == "json_screen"
        and mode == "screen"
        and metric == "object_pairing_consistency"
    ):
        return (
            "This is a structured-data routing screen, not a final "
            "object-pairing decision. Apply the relocation test using object "
            "identity, semantic role, scene type, and local ensemble "
            "membership. Route a target only when its identity or role may "
            "remain inappropriate even after reasonable relocation within "
            "the same scene. Do not route merely because of current support "
            "surface, zone, height, orientation, distance, access, clearance, "
            "scale, style, or collision. In this screening contract, invalid "
            "means visual confirmation required, not a final pairing defect. "
            "Return insufficient/ambiguous only when the structured data "
            "cannot safely complete the screen. "
            + acquire_more
        )
    if metric == "style_consistency" and phase == "global_screen":
        return (
            "Compatibility phase name: treat this as the required style "
            "scene-global pass. Return a final verdict for what the global "
            "view establishes, localized to exact object IDs. This verdict "
            "does not short-circuit the mandatory eligible group-local pass. "
            + acquire_more
        )
    if metric == "style_consistency" and phase == "local_confirmation":
        return (
            "Compatibility phase name: treat this as a mandatory independent "
            "style group-local pass. Use the global view as scene context and "
            "the local view for the supplied group; its presence carries no "
            "invalidity prior. "
            + acquire_more
        )
    if metric == "scale_consistency" and phase == "visual_confirmation":
        return (
            "This is visual confirmation of a structured-data scale "
            "candidate. Use the global view for scene context and the "
            "group-local view to verify target identity, relative-scale "
            "context, and perspective. The prior screen carries no invalidity "
            "prior. Confirm a final defect only when the physical-size "
            "mismatch remains material after the rubric's permitted "
            "variation. Do not count the routed candidate and its "
            "confirmation as two defects. "
            + acquire_more
        )
    if (
        metric == "object_pairing_consistency"
        and phase == "visual_confirmation"
    ):
        return (
            "This is visual confirmation of a structured-data object-pairing "
            "candidate. Use the global view for scene context and the "
            "group-local view to verify visible object identity and semantic "
            "role. The prior screen carries no invalidity prior. Apply the "
            "relocation test again; current support surface, zone, "
            "orientation, access, or clearance cannot confirm an "
            "object-pairing defect. Do not count the routed candidate and its "
            "confirmation as two defects. "
            + acquire_more
        )
    if metric == "functional_consistency" and phase == "global_discovery":
        stage_emphasis = L3_METRIC_PHASE_PROMPTS[metric][phase]
        return (
            "This is the required overall scene-level functional pass. "
            "Judge only holistic scene-level functional organization. "
            "Functional Discovery relations are owned by later isolated "
            "cross-group episodes, while usable frontage, local clearance, "
            "and within-group correspondence are owned by group-local "
            "episodes. Do not report those claims here. "
            + stage_emphasis
            + " Localize any genuinely scene-level defect to exact target "
            "IDs. Later phases still run and carry no validity prior. "
            + acquire_more
        )
    if (
        metric == "functional_consistency"
        and phase == "cross_group_relation_review"
    ):
        stage_emphasis = L3_METRIC_PHASE_PROMPTS[metric][phase]
        return (
            "This is one isolated cross-group functional-relation episode. "
            "Use the global image only as context and the pair-specific image "
            "for the supplied relation. "
            + stage_emphasis
            + " Return valid or invalid only for this relation. If a named "
            "observation is still unavailable, request that exact evidence "
            "through the normal camera loop. "
            + acquire_more
        )
    if (
        metric == "semantic_placement_consistency"
        and phase == "global_discovery"
    ):
        stage_emphasis = L3_METRIC_PHASE_PROMPTS[metric][phase]
        return (
            "This is the required scene-global placement pass. Groups are "
            "evidence partitions, not reasoning boundaries; scene-zone and "
            "cross-group contextual anchors remain in scope. "
            + stage_emphasis
            + " A newly noticed support-and-height or within-group anchor "
            "check belongs to the later group-local stage: request evidence "
            "with a typed placement_check_proposal so it can be handed off; "
            "do not resolve or score that local check here. Localize every "
            "scene-global defect to exact target IDs. "
            + acquire_more
        )
    if (
        metric == "semantic_placement_consistency"
        and phase == "residual_global_placement_review"
    ):
        stage_emphasis = L3_METRIC_PHASE_PROMPTS[metric][phase]
        return (
            "This is the final residual scene-global Placement pass. The "
            "typed global, group-local, and target-local passes have already "
            "run; their results are neutral suppression context, not a "
            "validity prior. Use the angled overview for appearance and the "
            "top/identity views for scene-scale organization and object "
            "grounding. scene_program gives only the broad room program; "
            "do not turn it into a style or inventory stereotype. Treat the "
            "complete object_inventory as an identity roster, not as proof "
            "that any two objects form a relation. Visual layout remains the "
            "primary decision evidence. placement_residual_context."
            "scene_distribution_descriptors are "
            "neutral measurements only: use them to focus inspection and "
            "confirm any claimed distribution failure in the images. Complete "
            "one concise "
            "group_global_observations row for every exact group before "
            "synthesizing the final verdict; those rows are audit-only, not "
            "independent votes or defects. "
            + stage_emphasis
            + " Return a judge-originated typed result only for a concrete "
            "new scene_zone or contextual_anchor failure. At most one "
            "collective scene-zone distribution finding may be emitted, with "
            "one accounting subject and other materially involved objects as "
            "non-owning context. If the current "
            "global packet is missing a necessary observable fact, use the "
            "normal evidence-request loop; otherwise make a binary choice. "
            + acquire_more
        )
    if metric == "style_consistency" and phase == "global_discovery":
        stage_emphasis = L3_METRIC_PHASE_PROMPTS[metric][phase]
        return (
            "This is the required scene-global pass. Groups are evidence "
            "partitions, not reasoning boundaries; cross-group facts remain "
            "in scope. Follow any supplied image_order, and treat every "
            "discovery or probe record as a neutral observation aid. "
            + stage_emphasis
            + " Decide only facts established at scene scope and localize "
            "every defect to exact target IDs. A later local pass still runs; "
            "do not summarize unseen local detail or short-circuit it. "
            + acquire_more
        )
    if (
        metric
        in {
            "style_consistency",
            "functional_consistency",
            "semantic_placement_consistency",
        }
        and phase == "group_local_review"
    ):
        stage_emphasis = L3_METRIC_PHASE_PROMPTS[metric][phase]
        return (
            "This is the group-local pass. Use global images only as anchors "
            "and local images for the supplied group. Review carries no "
            "validity prior. Defect target IDs must belong to this group; do "
            "not restate cross-group claims with missing participants. "
            + stage_emphasis
            + " "
            + acquire_more
        )
    if phase == "target_local_confirmation":
        return (
            "This is an independent target-centred confirmation episode, "
            "not a group judgement. The target object is the only default "
            "defect owner; nearby context objects are included only to make "
            "the local scene relationship legible. Do not infer group "
            "membership from the framing. "
            + acquire_more
        )
    if (
        metric
        in {
            "functional_consistency",
            "semantic_placement_consistency",
        }
        and phase == "initial_visual"
    ):
        return (
            "This is the initial global-plus-local visual pass. Judge only "
            "what the current scene context and target-local view establish. "
            + acquire_more
        )
    return acquire_more


def _sanitize_outbound_view_evidence(value: Any) -> Any:
    """Remove filesystem, content-hash, provenance, and dataset linkage fields.

    Rich evidence metadata is retained in local artifacts, while the remote
    judge receives only the role, pose, target, and diagnostic legend needed
    to interpret the images.
    """

    if isinstance(value, list):
        sanitized_items = [
            _sanitize_outbound_view_evidence(item)
            for item in value
        ]
        # Judge-visible image identity is request-local and fixed-shape.  It
        # must not reveal whether a slot came from deterministic or active
        # camera search through an action-suffixed internal view ID.
        for index, item in enumerate(sanitized_items):
            if not isinstance(item, dict):
                continue
            if "view_id" in item:
                item["view_id"] = f"slot_{index:02d}"
            if "pair_id" in item:
                item["pair_id"] = f"slot_{index:02d}"
        return sanitized_items
    if not isinstance(value, dict):
        return value
    sanitized: dict[str, Any] = {}
    for raw_key, item in value.items():
        key = str(raw_key)
        normalized = key.lower()
        if (
            "path" in normalized
            or "sha256" in normalized
            or "degradation" in normalized
            or "exception" in normalized
            or normalized in {
                "hash",
                "content_hash",
                "case_id",
                "dataset_id",
                "asset_jid",
                "jid",
                "provenance",
                "geometry_provenance",
                "active_camera_fallback",
                "camera_action",
                "camera_action_protocol",
                "camera_action_parameters",
                "parent_view_id",
                "policy_source",
                "proposal_id",
                "proposal_fingerprint",
                "trajectory",
                "selection_phase",
                "active_repair",
                "active_corrective_proposal_version",
                "trigger_recommended",
                "camera_repairable",
                "repairability",
            }
        ):
            continue
        if normalized == "pose":
            sanitized[key] = _minimal_judge_pose(item)
            continue
        sanitized[key] = _sanitize_outbound_view_evidence(item)
    return sanitized


def _minimal_judge_pose(value: Any) -> dict[str, Any]:
    """Keep framing semantics while stripping camera-policy lineage."""

    pose = value if isinstance(value, dict) else {}
    result: dict[str, Any] = {}
    for key in ("location", "target"):
        vector = pose.get(key)
        if isinstance(vector, (list, tuple)) and len(vector) == 3:
            result[key] = [
                round(float(component), 5)
                for component in vector
            ]
    if isinstance(pose.get("lens_mm"), (int, float)):
        result["lens_mm"] = round(float(pose["lens_mm"]), 3)
    camera_type = str(pose.get("camera_type") or "").strip().upper()
    if camera_type in {"PERSP", "ORTHO"}:
        result["camera_type"] = camera_type
    return result


def build_openai_compatible_vlm_judge(config: dict[str, Any]) -> OpenAICompatibleVLMJudge:
    if not isinstance(config, dict):
        raise TypeError("VLM judge config must be a JSON object")
    if "api_key" in config:
        raise ValueError(
            "VLM judge config must not contain literal api_key; use api_key_env instead"
        )
    endpoint = config.get("endpoint") or config.get("base_url")
    model_id = config.get("model") or config.get("model_id")
    if not endpoint or not model_id:
        raise ValueError("VLM judge config requires endpoint and model")
    model = OpenAICompatibleModel(
        name=str(config.get("name") or "openai-compatible-vlm-judge"),
        endpoint=str(endpoint),
        model_id=str(model_id),
        api_key_env=str(config["api_key_env"]) if config.get("api_key_env") else None,
        temperature=float(config.get("temperature", 0.0)),
        max_tokens=int(
            config.get(
                "max_tokens",
                DEFAULT_JUDGE_COMPLETION_MAX_TOKENS,
            )
        ),
        context_length=int(config["context_length"]) if config.get("context_length") is not None else None,
        timeout_seconds=int(config.get("timeout_seconds", 300)),
        response_format_json=bool(config.get("response_format_json", True)),
        max_retries=int(config.get("max_retries", 1)),
        retry_backoff_seconds=float(config.get("retry_backoff_seconds", 1.0)),
        min_request_interval_seconds=float(
            config.get("min_request_interval_seconds", 0.0)
        ),
        max_tokens_field=str(config.get("max_tokens_field", "max_tokens")),
        send_temperature=bool(config.get("send_temperature", True)),
        require_api_key=(
            bool(config["require_api_key"])
            if config.get("require_api_key") is not None
            else None
        ),
    )
    return OpenAICompatibleVLMJudge(
        model,
        max_images=int(config.get("max_images", 8)),
        max_context_chars=int(config.get("max_context_chars", 30000)),
        response_format_json=bool(config.get("response_format_json", True)),
    )


def _check_standalone_visual_evidence(
    request: dict[str, Any],
) -> str | None:
    """Return a compatibility summary when deterministic evidence is not ready."""

    from benchmark.visual_judge.evidence_gate import (
        DeterministicEvidenceGate,
    )
    from benchmark.visual_judge.interfaces import EvidenceGateRequest
    from benchmark.visual_judge.runtime import _judge_request

    core = _judge_request(request)
    gate = DeterministicEvidenceGate().check(
        EvidenceGateRequest(
            task=core.task,
            metric=core.metric,
            target_ids=("scene",),
            scene=deepcopy(core.scene_context),
            visual_evidence=tuple(deepcopy(core.visual_evidence)),
            evidence_goal={},
            context=deepcopy(request),
        )
    )
    if gate.ready:
        return None
    reasons = list(gate.reason_codes) or [
        "technical_visual_evidence_not_ready"
    ]
    return "visual evidence is not technically ready: " + ", ".join(
        reasons
    )


def _image_data_url(path: Path) -> str:
    return _normalized_rgb_png_data_url(path, label="judge_evidence")


def _normalized_rgb_png_data_url(path: Path, *, label: str) -> str:
    """Decode and re-encode one outbound image without file metadata or alpha."""

    try:
        from PIL import Image, ImageOps, UnidentifiedImageError
    except ImportError as exc:  # pragma: no cover - dependency contract guard
        raise RuntimeError(
            "Pillow is required to sanitize outbound VLM images"
        ) from exc

    try:
        with Image.open(path) as source:
            source.load()
            normalized = ImageOps.exif_transpose(source).convert("RGBA")
            flattened = Image.new("RGB", normalized.size, (255, 255, 255))
            flattened.paste(normalized, mask=normalized.getchannel("A"))
            output = BytesIO()
            flattened.save(output, format="PNG")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(
            f"outbound image {label} is not a valid decodable image"
        ) from exc
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _allowed_canonical_camera_observations(metric: Any) -> list[str]:
    return _allowed_binary_camera_observations(metric)


def _canonical_evidence_request_target_ids(
    request: dict[str, Any],
) -> tuple[str, ...]:
    raw_external = request.get(
        "allowed_external_evidence_target_ids"
    )
    external_ids = (
        [
            str(value)
            for value in raw_external
            if isinstance(value, (str, int)) and str(value).strip()
        ]
        if isinstance(raw_external, list)
        else []
    )
    group_scope = request.get("group_scope")
    if isinstance(group_scope, dict):
        members = group_scope.get("member_ids")
        if isinstance(members, list):
            values = tuple(
                dict.fromkeys(
                    str(value)
                    for value in members
                    if isinstance(value, (str, int))
                    and str(value).strip()
                )
            )
            if values:
                return tuple(
                    dict.fromkeys((*values, *external_ids))
                )

    values: list[str] = []
    for key in ("target_object_ids", "object_ids"):
        raw = request.get(key)
        if isinstance(raw, list):
            values.extend(
                str(value)
                for value in raw
                if isinstance(value, (str, int))
                and str(value).strip()
            )
    scene = request.get("scene_summary")
    if isinstance(scene, dict):
        for item in scene.get("objects") or []:
            if not isinstance(item, dict):
                continue
            object_id = item.get("id") or item.get("object_id")
            if object_id is not None and str(object_id).strip():
                values.append(str(object_id))
    return tuple(
        dict.fromkeys((*values, *external_ids, "scene"))
    )


def _canonical_defect_target_ids(
    request: dict[str, Any],
) -> tuple[str, ...]:
    contract = request.get("response_contract")
    contract = contract if isinstance(contract, dict) else {}
    defects = contract.get("defects")
    defects = defects if isinstance(defects, dict) else {}
    raw = defects.get("allowed_target_ids")
    if not isinstance(raw, list):
        group_scope = request.get("group_scope")
        raw = (
            group_scope.get("member_ids")
            if isinstance(group_scope, dict)
            and isinstance(group_scope.get("member_ids"), list)
            else request.get("target_object_ids")
        )
    if not isinstance(raw, list) or not raw:
        scene = request.get("scene_summary")
        raw = [
            item.get("id") or item.get("object_id")
            for item in (
                scene.get("objects")
                if isinstance(scene, dict)
                else []
            )
            or []
            if isinstance(item, dict)
        ]
    values = tuple(
        dict.fromkeys(
            str(item)
            for item in raw
            if isinstance(item, (str, int)) and str(item).strip()
        )
    )
    return tuple(dict.fromkeys((*values, "scene")))


def _project_group_local_defects_to_scope(
    result: dict[str, Any],
    *,
    request: dict[str, Any],
    required_functional_checks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Discard visual findings that do not belong to this group episode.

    Global context is evidence, not decision scope.  The sole exception is an
    external Functional clearance owner explicitly linked to its typed check
    and exact ``scoring_target_ids``.  This projection is deterministic scope
    enforcement, not response-schema compatibility repair.
    """

    if request.get("evidence_phase") != "group_local_review":
        return result
    group_scope = request.get("group_scope")
    members = {
        str(item)
        for item in (
            group_scope.get("member_ids")
            if isinstance(group_scope, dict)
            else []
        )
        or []
        if str(item).strip()
    }
    defects = result.get("defects")
    if not members or not isinstance(defects, list):
        return result
    checks = {
        str(item.get("check_id") or ""): item
        for item in required_functional_checks
        if isinstance(item, dict) and item.get("check_id")
    }
    rows = {
        str(item.get("check_id") or ""): item
        for item in result.get("functional_check_results") or []
        if isinstance(item, dict) and item.get("check_id")
    }

    def in_scope(defect: Any) -> bool:
        if not isinstance(defect, dict):
            return True
        targets = {
            str(item)
            for item in defect.get("target_ids") or []
            if str(item).strip()
        }
        if targets and targets <= members:
            return True
        refs = defect.get("check_refs")
        if not isinstance(refs, list) or len(refs) != 1:
            return False
        check_id = str(refs[0])
        check = checks.get(check_id)
        row = rows.get(check_id)
        return bool(
            isinstance(check, dict)
            and check.get("check_type") == "clearance"
            and isinstance(row, dict)
            and row.get("conclusion") == "invalid"
            and targets
            == {
                str(item)
                for item in row.get("scoring_target_ids") or []
                if str(item).strip()
            }
        )

    retained = [deepcopy(item) for item in defects if in_scope(item)]
    if len(retained) == len(defects):
        return result
    projected = deepcopy(result)
    projected["defects"] = retained
    projected["group_scope_projection"] = {
        "policy": "group_members_or_linked_clearance_owner_v1",
        "discarded_defect_count": len(defects) - len(retained),
    }
    typed_invalid = any(
        isinstance(row, dict) and row.get("conclusion") == "invalid"
        for row in [
            *(projected.get("functional_check_results") or []),
            *(projected.get("placement_check_results") or []),
        ]
    )
    if projected.get("verdict") == "invalid" and not retained:
        if typed_invalid:
            # A typed invalid without its exact authorized defect is a real
            # contract failure and must proceed to schema repair or the
            # metric-specific atomic-row fallback.
            return projected
        projected["verdict"] = "valid"
        projected["reason"] = (
            "The model described only defects outside the immutable group-local "
            "decision scope; no in-scope defect remains."
        )
    return projected


def _validate_pending_placement_proposal(
    result: dict[str, Any],
    *,
    request: dict[str, Any],
) -> None:
    evidence_request = result.get("evidence_request")
    if not isinstance(evidence_request, dict):
        return
    metadata = evidence_request.get("metadata")
    if not isinstance(metadata, dict):
        return
    proposal = metadata.get("placement_check_proposal")
    if proposal is None:
        return
    from benchmark.evaluator.scene_quality.placement_checks import (
        build_pending_placement_check,
        canonicalize_placement_proposal_transport,
    )

    known_ids = _placement_known_ids_for_request(request)
    proposal, warnings = canonicalize_placement_proposal_transport(
        proposal,
        known_ids=known_ids,
    )
    metadata["placement_check_proposal"] = proposal
    if warnings:
        metadata["placement_transport_normalization"] = {
            "policy": "narrow_transport_only_v1",
            "warnings": warnings,
        }

    pending = build_pending_placement_check(
        proposal,
        known_ids=known_ids,
        groups=_placement_groups_for_request(request),
        source_ref=str(proposal["proposal_id"]),
    )
    expected_stage = (
        "group_local"
        if request.get("evidence_phase") == "group_local_review"
        else "scene_global"
    )
    if pending.get("owner_stage") != expected_stage:
        raise ValueError(
            "placement_check_proposal belongs to "
            f"{pending.get('owner_stage')!r}, not active stage "
            f"{expected_stage!r}"
        )


def _allowed_binary_camera_observations(
    metric: Any,
    *,
    relation_type: str | None = None,
) -> list[str]:
    """Expose only controller-validated DSL tokens to the internal Judge."""

    try:
        metric_name = canonical_camera_metric(
            metric,
            relation_type=relation_type,
        )
    except ValueError:
        # target_visible belongs to every metric registry entry and remains a
        # safe technical fallback when a legacy request omits its subtype.
        return ["target_visible"]
    allowed = METRIC_CAMERA_REQUIREMENTS[
        metric_name
    ].allowed_observations
    return sorted(set(allowed) & CAMERA_OBSERVATIONS)


def _relation_type_from_request(request: dict[str, Any]) -> str | None:
    for source in (
        request.get("relation"),
        request.get("event"),
        request.get("detector_evidence"),
        request,
    ):
        if isinstance(source, str) and source.strip():
            return source.strip()
        if not isinstance(source, dict):
            continue
        for key in ("relation_type", "event_type", "predicate", "type"):
            value = source.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        nested = source.get("relation")
        if isinstance(nested, str) and nested.strip():
            return nested.strip()
    return None


def _required_functional_checks_from_request(
    request: dict[str, Any],
) -> list[Any]:
    value = request.get("required_functional_checks")
    if not isinstance(value, list):
        packet = request.get("functional_probe_evidence")
        value = (
            packet.get("required_checks")
            if isinstance(packet, dict)
            else None
        )
    return deepcopy(value) if isinstance(value, list) else []


def _judge_facing_required_functional_checks(
    value: Any,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for check in value:
        if not isinstance(check, dict):
            raise TypeError(
                "required_functional_checks must contain JSON objects"
            )
        check_id = str(check.get("check_id") or "").strip()
        target_ids = check.get("target_ids")
        if (
            not check_id
            or check_id in seen_ids
            or not isinstance(target_ids, list)
            or not target_ids
        ):
            raise ValueError(
                "required_functional_checks require unique IDs and "
                "non-empty target lists"
            )
        seen_ids.add(check_id)
        compact = {
            key: deepcopy(check[key])
            for key in (
                "check_id",
                "check_type",
                "predicate",
                "owner_stage",
                "target_ids",
                "group_ids",
                "owning_group_id",
                "relation",
                "required_observations",
                "observation_goals",
                "observation_kinds",
                "surface_roles",
                "need_clearance",
                "causal_candidate_ids",
                "causal_candidates",
                "causal_candidates_are_routing_prior",
                "allowed_causal_object_ids",
                "acquisition_status",
                "artifact_rendered",
                "machine_view_coverage_status",
                "machine_view_coverage_usable",
            )
            if key in check
        }
        affordances = []
        for affordance in check.get("target_affordances") or []:
            if not isinstance(affordance, dict):
                continue
            affordances.append(
                {
                    key: deepcopy(affordance[key])
                    for key in (
                        "target_id",
                        "directionality",
                        "surface_roles",
                        "need_clearance",
                    )
                    if key in affordance
                }
            )
        if affordances:
            compact["target_affordances"] = affordances
        result.append(compact)
    return result


def _required_placement_checks_from_request(
    request: dict[str, Any],
) -> list[Any]:
    value = request.get("required_placement_checks")
    return deepcopy(value) if isinstance(value, list) else []


def _judge_facing_required_placement_checks(
    value: Any,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for check in value:
        if not isinstance(check, dict):
            raise TypeError(
                "required_placement_checks must contain JSON objects"
            )
        check_id = str(check.get("check_id") or "").strip()
        subject_id = str(check.get("subject_id") or "").strip()
        context_ids = check.get("context_ids")
        if (
            not check_id
            or check_id in seen_ids
            or not subject_id
            or not isinstance(context_ids, list)
        ):
            raise ValueError(
                "required_placement_checks require unique IDs, a subject, "
                "and a context list"
            )
        seen_ids.add(check_id)
        result.append(
            {
                key: deepcopy(check[key])
                for key in (
                    "check_id",
                    "check_type",
                    "subject_id",
                    "context_ids",
                    "owner_stage",
                    "owning_group_id",
                    "group_ids",
                    "required_observations",
                    "observation_goals",
                    "origin",
                    "acquisition_status",
                )
                if key in check
            }
        )
    return result


def _judge_facing_functional_ownership_ledger(
    value: Any,
) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    events = []
    for event in value.get("events") or []:
        if not isinstance(event, dict):
            raise TypeError(
                "functional ownership events must contain JSON objects"
            )
        events.append(
            {
                key: deepcopy(event[key])
                for key in (
                    "event_id",
                    "owning_metric",
                    "affected_object_ids",
                    "cause_kind",
                    "causal_object_ids",
                    "scoring_target_ids",
                    "counterpart_object_ids",
                    "scope",
                    "relation",
                    "reason",
                    "check_refs",
                    "decision_ref",
                )
                if key in event
            }
        )
    return {
        "schema_version": value.get("schema_version"),
        "events": events,
        "event_count": len(events),
        "decision_authority": "none",
    }


def _expected_placement_owner_stage(
    request: dict[str, Any],
) -> str | None:
    if isinstance(request.get("target_scope"), dict):
        return "target_local"
    if isinstance(request.get("group_scope"), dict):
        return "group_local"
    phase = str(request.get("evidence_phase") or "")
    if phase in {
        "global_discovery",
        "scene_global",
        "residual_global_placement_review",
    }:
        return "scene_global"
    # Direct adapter callers may use the group-local phase without supplying
    # a trusted group.  In that case owner assignment is derived from the
    # available group map and may legitimately become target-local.
    return None


def _placement_known_ids_for_request(
    request: dict[str, Any],
) -> set[str]:
    group_scope = request.get("group_scope")
    if isinstance(group_scope, dict) and isinstance(
        group_scope.get("member_ids"),
        list,
    ):
        members = {
            str(item)
            for item in group_scope["member_ids"]
            if isinstance(item, (str, int)) and str(item).strip()
        }
        if members:
            return members
    for key in ("camera_scene_context", "scene_summary"):
        scene = request.get(key)
        if not isinstance(scene, dict):
            continue
        result = {
            str(item.get("id") or item.get("object_id"))
            for item in scene.get("objects") or []
            if isinstance(item, dict)
            and (item.get("id") or item.get("object_id"))
        }
        if result:
            return result
    raise ValueError(
        "semantic placement request has no trusted object identities"
    )


def _scene_known_ids_for_request(
    request: dict[str, Any],
) -> set[str]:
    for key in ("camera_scene_context", "scene_summary"):
        scene = request.get(key)
        if not isinstance(scene, dict):
            continue
        result = {
            str(item.get("id") or item.get("object_id"))
            for item in scene.get("objects") or []
            if isinstance(item, dict)
            and (item.get("id") or item.get("object_id"))
        }
        if result:
            return result
    raise ValueError(
        "semantic placement ownership has no trusted scene identities"
    )


def _placement_groups_for_request(
    request: dict[str, Any],
) -> list[dict[str, Any]]:
    groups = request.get("object_groups")
    return [
        {
            "group_id": str(item.get("group_id") or ""),
            "object_ids": [
                str(object_id)
                for object_id in item.get("object_ids") or []
            ],
        }
        for item in groups or []
        if isinstance(item, dict)
    ]


def _judge_facing_functional_probe_context(
    value: Any,
    *,
    required_checks_supplied: bool,
) -> Any:
    if not isinstance(value, dict):
        return deepcopy(value)
    result = deepcopy(value)
    if isinstance(result.pop("functional_measurements", None), dict):
        # The compact bank is a separately budget-protected top-level field.
        # Keeping a second copy inside this potentially large packet can force
        # both copies through lossy JSON-prefix compaction.
        result["functional_measurements_reference"] = (
            "functional_measurements"
        )
    if required_checks_supplied:
        result.pop("required_checks", None)
        result["required_checks_reference"] = (
            "required_functional_checks"
        )
    return result


def _functional_preflight_response(
    request: dict[str, Any],
    *,
    selected_paths: list[Path],
    required_checks: list[dict[str, Any]],
    allowed_missing_observations: tuple[str, ...],
    allowed_target_ids: tuple[str, ...],
    forced_choice: bool,
) -> dict[str, Any] | None:
    """Emit one controller-visible evidence request before binary judging.

    A failed usable-side decode is machine-observable.  Letting the semantic
    Judge immediately return a confident binary answer in that state defeats
    the evidence loop.  The preflight therefore requests one replacement view
    without consuming a model call.  Once the active packet contains an
    artifact not present in the initial packet, ordinary Judge authority
    resumes.  Terminal forced choice also bypasses this guard by design.
    """

    value = request.get("functional_evidence_preflight")
    if (
        forced_choice
        or not isinstance(value, dict)
        or value.get("active") is not True
        or not required_checks
    ):
        return None

    def artifact_key(raw: Any) -> str:
        try:
            return str(Path(str(raw)).expanduser().resolve())
        except (OSError, RuntimeError, ValueError):
            return str(raw)

    initial = {
        artifact_key(item)
        for item in value.get("initial_evidence_refs") or []
        if str(item).strip()
    }
    current = {artifact_key(path) for path in selected_paths}
    if current - initial:
        return None

    allowed_missing = set(allowed_missing_observations)
    missing = list(
        dict.fromkeys(
            str(item)
            for item in value.get("missing_observations") or []
            if str(item) in allowed_missing
        )
    )
    if not missing:
        missing = [
            item
            for item in (
                "interaction_side_visible",
                "front_back_disambiguated",
                "target_visible",
            )
            if item in allowed_missing
        ][:1]
    allowed_targets = set(allowed_target_ids)
    targets = list(
        dict.fromkeys(
            str(item)
            for item in value.get("target_ids") or []
            if str(item) in allowed_targets
        )
    )
    if not targets:
        targets = list(allowed_target_ids[:1]) or ["scene"]
    rows = [
        {
            "check_id": str(check.get("check_id") or ""),
            "target_ids": [
                str(item) for item in check.get("target_ids") or []
            ],
            "observation_status": "missing",
            "conclusion": "unresolved",
            "reason": (
                "The deterministic visual preflight could not bind the "
                "required directed-side observation; acquire a replacement "
                "view before binary judgement."
            ),
        }
        for check in required_checks
    ]
    return {
        "evidence_status": "insufficient",
        "verdict": "ambiguous",
        "confidence": 0.0,
        "reason": (
            "Required directed-side evidence was not machine-resolved in "
            "the initial packet."
        ),
        "missing_evidence": missing,
        "defects": [],
        "evidence_request": {
            "target_ids": targets,
            "missing_observations": missing,
            "view_goal": (
                "show the directed usable or interaction side with enough "
                "front/back context to resolve the requested check"
            ),
            "metadata": {
                "source": "functional_evidence_preflight_v1",
                "check_id": value.get("check_id"),
                "reason_codes": deepcopy(value.get("reason_codes") or []),
            },
        },
        "functional_check_results": rows,
        "functional_evidence_preflight": {
            "applied": True,
            "reason_codes": deepcopy(value.get("reason_codes") or []),
        },
    }


def _require_functional_checks_in_model_context(
    context_text: str,
    *,
    required_checks: list[dict[str, Any]],
) -> None:
    if not required_checks:
        return
    parsed = json.loads(context_text)
    delivered = parsed.get("required_functional_checks")
    expected_signature = [
        (
            str(check.get("check_id") or ""),
            tuple(
                sorted(
                    str(target_id)
                    for target_id in check.get("target_ids") or []
                )
            ),
        )
        for check in required_checks
    ]
    delivered_signature = (
        [
            (
                str(check.get("check_id") or ""),
                tuple(
                    sorted(
                        str(target_id)
                        for target_id in check.get("target_ids") or []
                    )
                ),
            )
            for check in delivered
            if isinstance(check, dict)
        ]
        if isinstance(delivered, list)
        else []
    )
    if delivered_signature != expected_signature:
        raise ValueError(
            "required functional checks exceed the model context budget; "
            "partial check delivery is forbidden"
        )


def _scope_soft_functional_measurements(
    measurements: dict[str, Any] | None,
    *,
    required_checks: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Project non-authoritative measurements onto the exact Judge scope."""

    expected_ids = list(
        dict.fromkeys(
            str(check.get("check_id") or "")
            for check in required_checks
            if str(check.get("check_id") or "").strip()
        )
    )
    if not expected_ids:
        return None, {
            "schema_version": "functional_measurement_scope_audit_v1",
            "status": "not_applicable",
            "expected_check_ids": [],
            "source_check_ids": [],
            "source_unexpected_check_ids": [],
            "source_duplicate_check_ids": [],
            "missing_check_ids": [],
            "evidence_role": "soft_supporting_evidence",
            "decision_authority": "none",
        }

    source = deepcopy(measurements) if isinstance(measurements, dict) else {}
    rows = [
        deepcopy(item)
        for item in source.get("check_measurements") or []
        if isinstance(item, dict)
        and str(item.get("check_id") or "").strip()
    ]
    source_ids = [str(item.get("check_id")) for item in rows]
    expected = set(expected_ids)
    source_unexpected = list(
        dict.fromkeys(item for item in source_ids if item not in expected)
    )
    source_duplicates = sorted(
        {
            item
            for item in source_ids
            if source_ids.count(item) > 1
        }
    )
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        check_id = str(row["check_id"])
        if check_id in expected and check_id not in by_id:
            by_id[check_id] = row
    scoped_rows = [
        by_id[check_id]
        for check_id in expected_ids
        if check_id in by_id
    ]
    missing = [
        check_id for check_id in expected_ids if check_id not in by_id
    ]
    scoped = source
    scoped.pop("_truncated", None)
    scoped["status"] = (
        "complete"
        if not missing
        else "partial"
        if scoped_rows
        else "unavailable"
    )
    scoped["measurement_role"] = (
        "deterministic_spatial_evidence_not_verdict"
    )
    scoped["decision_authority"] = "none"
    scoped["requested_check_ids"] = list(expected_ids)
    scoped["check_measurements"] = scoped_rows
    audit = {
        "schema_version": "functional_measurement_scope_audit_v1",
        "status": scoped["status"],
        "expected_check_ids": list(expected_ids),
        "source_check_ids": source_ids,
        "source_unexpected_check_ids": source_unexpected,
        "source_duplicate_check_ids": source_duplicates,
        "missing_check_ids": missing,
        "evidence_role": "soft_supporting_evidence",
        "decision_authority": "none",
    }
    return scoped, audit


def _functional_measurement_delivery_audit(
    context_text: str,
    *,
    required_checks: list[dict[str, Any]],
    measurements: dict[str, Any] | None,
    source_scope_audit: dict[str, Any],
) -> dict[str, Any]:
    """Audit soft-evidence delivery without blocking a visual verdict."""

    expected_ids = list(
        dict.fromkeys(
            str(check.get("check_id") or "")
            for check in required_checks
            if str(check.get("check_id") or "").strip()
        )
    )
    if not expected_ids:
        return {
            "schema_version": "functional_measurement_delivery_audit_v1",
            "status": "not_applicable",
            "expected_check_ids": [],
            "delivered_check_ids": [],
            "missing_check_ids": [],
            "unexpected_check_ids": [],
            "truncated": False,
            "evidence_ambiguous": False,
            "blocks_judge": False,
            "evidence_role": "soft_supporting_evidence",
            "decision_authority": "none",
        }
    parsed = json.loads(context_text)
    delivered = parsed.get("functional_measurements")
    truncated = bool(
        isinstance(delivered, dict)
        and delivered.get("_truncated") is True
    )
    delivered_ids = (
        [
            str(item.get("check_id") or "")
            for item in delivered.get("check_measurements") or []
            if isinstance(item, dict)
            and str(item.get("check_id") or "").strip()
        ]
        if isinstance(delivered, dict) and not truncated
        else []
    )
    expected = set(expected_ids)
    missing = [
        check_id for check_id in expected_ids if check_id not in delivered_ids
    ]
    unexpected = list(
        dict.fromkeys(
            check_id
            for check_id in delivered_ids
            if check_id not in expected
        )
    )
    source_duplicates = list(
        source_scope_audit.get("source_duplicate_check_ids") or []
    )
    ambiguous = bool(
        missing or unexpected or truncated or source_duplicates
    )
    status = (
        "complete"
        if not ambiguous
        else "partial"
        if delivered_ids
        else "unavailable"
    )
    return {
        "schema_version": "functional_measurement_delivery_audit_v1",
        "status": status,
        "expected_check_ids": expected_ids,
        "delivered_check_ids": delivered_ids,
        "missing_check_ids": missing,
        "unexpected_check_ids": unexpected,
        "truncated": truncated,
        "source_measurements_present": isinstance(measurements, dict),
        "source_scope_audit": deepcopy(source_scope_audit),
        "evidence_ambiguous": ambiguous,
        "blocks_judge": False,
        "evidence_role": "soft_supporting_evidence",
        "decision_authority": "none",
    }


def _require_placement_checks_in_model_context(
    context_text: str,
    *,
    required_checks: list[dict[str, Any]],
) -> None:
    if not required_checks:
        return
    parsed = json.loads(context_text)
    delivered = parsed.get("required_placement_checks")
    expected_signature = [
        (
            str(check.get("check_id") or ""),
            str(check.get("subject_id") or ""),
            tuple(
                sorted(
                    str(item)
                    for item in check.get("context_ids") or []
                )
            ),
        )
        for check in required_checks
    ]
    delivered_signature = (
        [
            (
                str(check.get("check_id") or ""),
                str(check.get("subject_id") or ""),
                tuple(
                    sorted(
                        str(item)
                        for item in check.get("context_ids") or []
                    )
                ),
            )
            for check in delivered
            if isinstance(check, dict)
        ]
        if isinstance(delivered, list)
        else []
    )
    if delivered_signature != expected_signature:
        raise ValueError(
            "required placement checks exceed the model context budget; "
            "partial check delivery is forbidden"
        )


def _require_placement_residual_context_in_model_context(
    context_text: str,
    *,
    source_context: Any,
) -> dict[str, Any] | None:
    if not isinstance(source_context, dict):
        return None
    parsed = json.loads(context_text)
    delivered = parsed.get("placement_residual_context")
    if not isinstance(delivered, dict):
        raise ValueError(
            "residual Placement context exceeds the model context budget"
        )
    source_program = source_context.get("scene_program")
    delivered_program = delivered.get("scene_program")
    source_program = (
        source_program if isinstance(source_program, dict) else {}
    )
    delivered_program = (
        delivered_program if isinstance(delivered_program, dict) else {}
    )
    source_inventory = source_context.get("object_inventory")
    delivered_inventory = delivered.get("object_inventory")
    source_inventory = (
        source_inventory if isinstance(source_inventory, list) else []
    )
    delivered_inventory = (
        delivered_inventory
        if isinstance(delivered_inventory, list)
        else []
    )
    expected_ids = [
        str(item.get("object_id") or "")
        for item in source_inventory
        if isinstance(item, dict)
    ]
    delivered_ids = [
        str(item.get("object_id") or "")
        for item in delivered_inventory
        if isinstance(item, dict)
    ]
    expected_scene_type = str(source_program.get("scene_type") or "")
    delivered_scene_type = str(
        delivered_program.get("scene_type") or ""
    )
    source_groups = source_context.get("groups")
    delivered_groups = delivered.get("groups")
    top_level_groups = parsed.get("object_groups")
    source_groups = source_groups if isinstance(source_groups, list) else []
    delivered_groups = (
        delivered_groups if isinstance(delivered_groups, list) else []
    )
    top_level_groups = (
        top_level_groups if isinstance(top_level_groups, list) else []
    )

    def group_roster(value: list[Any]) -> list[dict[str, Any]]:
        return [
            {
                "group_id": str(item.get("group_id") or ""),
                "object_ids": list(item.get("object_ids") or []),
            }
            for item in value
            if isinstance(item, dict)
        ]

    expected_groups = group_roster(source_groups)
    delivered_group_roster = group_roster(delivered_groups)
    top_level_group_roster = group_roster(top_level_groups)
    leaked_group_semantics = any(
        isinstance(item, dict)
        and set(item) - {"group_id", "object_ids"}
        for item in top_level_groups
    )
    canonical_scene = parsed.get("canonical_scene")
    canonical_objects = (
        canonical_scene.get("objects")
        if isinstance(canonical_scene, dict)
        and isinstance(canonical_scene.get("objects"), list)
        else []
    )
    leaked_description = any(
        isinstance(item, dict)
        and any(key in item for key in ("description", "desc"))
        for item in canonical_objects
    )
    if (
        delivered_scene_type != expected_scene_type
        or delivered_ids != expected_ids
        or delivered.get("object_inventory_complete") is not True
        or parsed.get("natural_language_request") not in (None, "")
        or parsed.get("visual_style_spec") is not None
        or leaked_description
        or delivered_group_roster != expected_groups
        or top_level_group_roster != expected_groups
        or leaked_group_semantics
    ):
        raise ValueError(
            "residual Placement scene program or complete object inventory "
            "was not delivered intact, or excluded theme/prompt context "
            "leaked into the model input"
        )
    return {
        "schema_version": "placement_residual_context_delivery_audit_v2",
        "scene_type": delivered_scene_type,
        "expected_object_count": len(expected_ids),
        "delivered_object_count": len(delivered_ids),
        "object_ids_complete": True,
        "group_ids": [item["group_id"] for item in expected_groups],
        "group_roster_complete": True,
        "group_semantic_metadata_delivered": False,
        "aesthetic_theme_delivered": False,
        "generation_prompt_delivered": False,
    }


def _residual_placement_canonical_scene(value: Any) -> dict[str, Any]:
    """Retain raw spatial facts while removing style/prompt descriptions."""

    source = value if isinstance(value, dict) else {}
    objects = [
        {
            key: deepcopy(item.get(key))
            for key in (
                "id",
                "category",
                "center",
                "size",
                "rotation",
            )
            if key in item
        }
        for item in source.get("objects") or []
        if isinstance(item, dict)
    ]
    return {
        key: deepcopy(source.get(key))
        for key in (
            "scene_id",
            "scene_type",
            "boundary",
            "scene_height",
            "architecture",
            "object_count",
        )
        if key in source
    } | {"objects": objects}


def _require_functional_ownership_in_model_context(
    context_text: str,
    *,
    ledger: dict[str, Any] | None,
) -> None:
    events = (
        ledger.get("events")
        if isinstance(ledger, dict)
        and isinstance(ledger.get("events"), list)
        else []
    )
    if not events:
        return
    parsed = json.loads(context_text)
    delivered_ledger = parsed.get("functional_ownership_ledger")
    delivered_events = (
        delivered_ledger.get("events")
        if isinstance(delivered_ledger, dict)
        and isinstance(delivered_ledger.get("events"), list)
        else []
    )
    expected_ids = [
        str(item.get("event_id") or "")
        for item in events
        if isinstance(item, dict)
    ]
    delivered_ids = [
        str(item.get("event_id") or "")
        for item in delivered_events
        if isinstance(item, dict)
    ]
    if delivered_ids != expected_ids:
        raise ValueError(
            "functional ownership events exceed the model context budget; "
            "partial cross-metric exclusion context is forbidden"
        )


def _budgeted_context_json(
    context: dict[str, Any],
    max_chars: int,
    *,
    priority_keys: tuple[str, ...] = (),
) -> str:
    """Serialize a valid JSON context while preserving high-value evidence.

    Raw prefix slicing can both corrupt JSON and drop a detector packet merely
    because a large scene summary appeared first.  Required/priority fields are
    therefore considered first, optional fields are added only while they fit,
    and an oversized priority value is represented by an explicit JSON-prefix
    summary instead of truncating the entire document.
    """

    limit = max(1, int(max_chars))
    full = _compact_json(context)
    if len(full) <= limit:
        return full

    marker = {"truncated": True}
    marker_key = "_benchmark_context_budget"
    priority = [key for key in priority_keys if key in context]
    priority_set = set(priority)
    # Seed every priority key before allocating the remaining budget. This
    # prevents one oversized earlier field (for example a prompt or raw
    # relation) from consuming the budget and deleting detector evidence.
    skeleton = {**{key: None for key in priority}, marker_key: marker}
    skeleton_size = len(_compact_json(skeleton))
    per_key_extra = max(0, limit - skeleton_size) // max(len(priority), 1)
    seed_limit = len("null") + per_key_extra
    result: dict[str, Any] = {
        key: _priority_seed_value(context[key], seed_limit)
        for key in priority
    }
    if len(_compact_json({**result, marker_key: marker})) > limit:
        minimal = _compact_json({marker_key: marker})
        return minimal if len(minimal) <= limit else "{}"

    # Enrich priority fields in declared order while all other priority keys
    # remain reserved in the document.
    for key in priority:
        value = context[key]
        trial = {**result, key: value, marker_key: marker}
        if len(_compact_json(trial)) <= limit:
            result[key] = value
            continue
        empty_trial = {**result, key: None, marker_key: marker}
        allowance = limit - len(_compact_json(empty_trial)) + len("null")
        compacted = _truncated_json_value(value, allowance)
        if compacted is not None:
            compacted_trial = {**result, key: compacted, marker_key: marker}
            if len(_compact_json(compacted_trial)) <= limit:
                result[key] = compacted

    # Optional fields never displace reserved evidence.
    for key, value in context.items():
        if key in priority_set:
            continue
        trial = {**result, key: value, marker_key: marker}
        if len(_compact_json(trial)) <= limit:
            result[key] = value

    rendered = _compact_json({**result, marker_key: marker})
    # ``max_context_chars`` is clamped to at least 1000 by the judge, but keep
    # this helper total for direct unit use with tiny budgets.
    if len(rendered) <= limit:
        return rendered
    minimal = _compact_json({marker_key: marker})
    return minimal if len(minimal) <= limit else "{}"


def _truncated_json_value(value: Any, max_chars: int) -> dict[str, Any] | None:
    if max_chars <= 0:
        return None
    raw = _compact_json(value)
    base = {
        "_truncated": True,
        "original_chars": len(raw),
        "json_prefix": "",
    }
    if len(_compact_json(base)) > max_chars:
        return None
    low, high = 0, len(raw)
    while low < high:
        middle = (low + high + 1) // 2
        candidate = {**base, "json_prefix": raw[:middle]}
        if len(_compact_json(candidate)) <= max_chars:
            low = middle
        else:
            high = middle - 1
    return {**base, "json_prefix": raw[:low]}


def _priority_seed_value(value: Any, max_chars: int) -> Any:
    raw = _compact_json(value)
    if len(raw) <= max_chars:
        return value
    compacted = _truncated_json_value(value, max_chars)
    if compacted is not None:
        return compacted
    marker = {"_truncated": True}
    return marker if len(_compact_json(marker)) <= max_chars else None


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))
