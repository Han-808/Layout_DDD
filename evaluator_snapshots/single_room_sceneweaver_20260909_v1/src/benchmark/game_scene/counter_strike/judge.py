"""Strict visual judge for the perceptual Counter-Strike L4 components.

The selector and judge are deliberately separate calls.  The selector may
choose at most two IDs from a frozen regional bank and is forbidden to return a
metric verdict.  The final judge sees one fixed packet; repeated calls receive
the exact same prompt and ordered image bytes. Three repeats use a strict
majority verdict and median continuous score; insufficient evidence remains
``unresolved``.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit

from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError

from benchmark.visual_judge.openai_compatible import (
    build_openai_compatible_vlm_judge,
)
from benchmark.visual_judge.openai_camera_selector import (
    build_openai_compatible_camera_selector,
)

from .evidence import (
    CounterStrikeEvidenceDescriptor,
    CounterStrikeFrozenEvidence,
)
from .loader import CounterStrikeBenchmarkConfig


COUNTER_STRIKE_VISUAL_JUDGE_VERSION = "counter_strike_visual_judge_v5"
COUNTER_STRIKE_SELECTOR_PROMPT_VERSION = "counter_strike_observation_selector_v2"
COUNTER_STRIKE_METRIC_PROMPT_VERSION = "counter_strike_static_design_judge_v4"
COUNTER_STRIKE_VERDICT_AGGREGATION = "strict_majority"
COUNTER_STRIKE_SCORE_AGGREGATION = "median"
_RETRYABLE_RESPONSE_ERROR_CODES = frozenset(
    {"model_response_invalid", "verdict_score_inconsistent"}
)
SUPPORTED_VISUAL_METRICS = frozenset(
    {"zone_clarity", "landmark_legibility", "cover_diversity"}
)
_SAFE_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


SELECTOR_SYSTEM_PROMPT = """You select visual evidence for a static 3D
Counter-Strike-like arena benchmark. You do not decide whether the arena is
valid or invalid and must not output any metric verdict. The supplied images
are frozen regional camera views captured from the original Three.js runtime.
When one is too dark, its candidate slot is deterministically replaced by a
bounded brightness-repaired copy of the same pixels and camera pose. Brightness
is not a selection decision. Select zero, one, or two camera IDs that best
resolve the stated missing visual fact. Return exactly one JSON object:
{"evidence_status":"selected","selected_view_ids":[],"reason":"..."}.
evidence_status must be selected or no_helpful_candidate. IDs must come from
the candidate list. Do not mention gameplay, bots, weapons, HUD, source code,
or hidden scene content."""


JUDGE_SYSTEM_PROMPT = """You judge one narrowly scoped property of a frozen
static 3D Counter-Strike-like arena. This is a benchmark of the 3D environment,
not gameplay. Do not evaluate weapons, controls, HUD, bot behaviour, player
skill, combat outcome, or art preference outside the named metric.

The original-runtime global views, or deterministic brightness-repaired copies
of the exact same frozen pixels when dark, are primary visual evidence.
Optional regional views were selected from a frozen bank without seeing or
returning a verdict. The neutral geometry-derived observation aid is not ground truth:
dark cells are blocking footprints and A/B mark declared spawn points. It
contains no inferred roles, routes, cover proposals, deterministic score, or
deterministic verdict.

First decide whether the packet is sufficient. Return exactly one JSON object
matching the metric-specific response_schema supplied in the user task. Do not
add fields from another metric.
evidence_status must be sufficient or insufficient. verdict must be valid,
invalid, or ambiguous. score must be null when evidence is insufficient and a
number from 0 to 1 otherwise. The user task supplies a valid_threshold:
when evidence is sufficient and the verdict is not ambiguous, return valid if
and only if score is greater than or equal to that threshold; otherwise return
invalid. If evidence is insufficient, verdict must be ambiguous,
missing_evidence must be non-empty, and defects plus the metric-specific
findings array must be empty. Invalid requires at least one explicit
significant metric-scoped defect. Confidence is intentionally not requested
and must not be invented."""


METRIC_RUBRICS = {
    "zone_clarity": (
        "Judge whether the static layout makes these five spatial roles "
        "structurally legible: both team spawn regions, preparation/transition "
        "space leaving each spawn, a main engagement region, and one or more "
        "flank regions. Apply a permissive functional-legibility policy: a role "
        "does not need a closed room, doorway, text label, or unique texture. "
        "The original-runtime views plus the neutral occupancy/spawn aid may "
        "establish it through stable spatial function. In particular, a spawn "
        "may be an identifiable perimeter-backed end region; preparation may "
        "be an open departure band between a spawn and the shared field; the "
        "main engagement region may be established by obstacle concentration "
        "or opposing-route convergence; and a flank may be a peripheral bypass "
        "around that main region without being an enclosed corridor. Do not "
        "re-score route count, spawn balance, cover diversity, or visual style. "
        "role_findings must list each required role with status clear, weak, "
        "or missing, identify its spatial_region, and cite visible/geometric "
        "evidence. The five roles must occupy distinct spatial regions: the "
        "same region cannot satisfy more than one role. Assign clear when a "
        "reasonable observer can repeatedly locate the role from those spatial "
        "cues, even if its boundary is open. Assign weak when the role is "
        "plausible but its region or connection to the other roles remains "
        "uncertain. Assign missing only when no plausible region serves the "
        "role. A "
        "valid verdict requires at least {min_clear_roles} of the five roles "
        "to be clear."
    ),
    "landmark_legibility": (
        "Judge whether at least three spatially distinct static visual cues are "
        "easy to name and reuse as location or engagement callouts. Apply a "
        "strict identity policy. Each landmark must be visible at global-view "
        "scale, have a concise appearance-based name, and possess a salient "
        "identity from shape/silhouette, color/material, or a unique compound "
        "structure. Position alone is not an identity: left box, right box, "
        "large generic box, differently rotated copies, and repeated generic "
        "walls do not become separate landmarks. Repeated instances of the "
        "same visual type count once unless their visible identities are "
        "genuinely different. A topology-only mark, coordinate, or inferred "
        "role cannot rescue a cue that is not distinguishable in the supplied "
        "original-runtime views. Do not count bots, HUD, or inferred invisible "
        "objects. landmarks must list name, view_ids, visible_cue, and "
        "spatial_region."
    ),
    "cover_diversity": (
        "Judge whether the static arena visibly provides at least four "
        "meaningfully different cover configurations. Apply a strict "
        "archetype-equivalence policy. First merge fragments belonging to one "
        "visible assembly and collapse repeated instances of the same cover "
        "archetype. Translation, rotation, color, or modest scale variation of "
        "the same generic box or wall does not create a new form. Height or "
        "width labels alone also cannot split one otherwise identical "
        "archetype. Count a separate form only when the original-runtime views "
        "show a decisive geometric or assembly difference that changes the "
        "cover profile, for example solid block versus pillar, long wall, "
        "stepped cover, or offset/compound barricade. At least four such "
        "archetypes, two height profiles, two width profiles, and two "
        "arrangement types must be represented. Density alone and the neutral "
        "occupancy diagram are insufficient to establish a visual form. "
        "cover_findings must list each distinct visible archetype with "
        "form_name, view_ids, visible_cue, spatial_region, height_profile, "
        "width_profile, and arrangement. visible_cue must state the decisive "
        "geometry or assembly feature, not merely size, color, orientation, or "
        "location. height_profile must be exactly low, waist, standing, or "
        "tall; width_profile must be exactly narrow, medium, or wide; "
        "arrangement must be exactly isolated, paired, row, cluster, or "
        "compound."
    ),
}


class CounterStrikeVisualJudgeError(RuntimeError):
    """Raised for model, schema, or evidence-contract failure."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"Counter-Strike visual judge failed [{code}]: {message}")


@dataclass(frozen=True)
class CounterStrikeVisualMetricResult:
    metric: str
    status: str
    score: float | None
    verdict: str
    evidence_phase: str
    selected_regional_ids: tuple[str, ...]
    brightness_repaired_view_ids: tuple[str, ...]
    repeats: tuple[dict[str, Any], ...]
    selector: dict[str, Any] | None
    input_contract_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "status": self.status,
            "score": self.score,
            "verdict": self.verdict,
            "evidence_phase": self.evidence_phase,
            "selected_regional_ids": list(self.selected_regional_ids),
            "brightness_repaired_view_ids": list(
                self.brightness_repaired_view_ids
            ),
            "repeat_count": len(self.repeats),
            "repeat_agreement": (
                len(
                    {
                        (
                            item["evidence_status"],
                            item["verdict"],
                        )
                        for item in self.repeats
                    }
                )
                == 1
            ),
            "verdict_aggregation": COUNTER_STRIKE_VERDICT_AGGREGATION,
            "score_aggregation": COUNTER_STRIKE_SCORE_AGGREGATION,
            "repeats": list(self.repeats),
            "selector": self.selector,
            "input_contract_sha256": self.input_contract_sha256,
            "judge_version": COUNTER_STRIKE_VISUAL_JUDGE_VERSION,
        }


class CounterStrikeVisualJudge:
    """Metric-specific VLM calls over one frozen evidence bank."""

    def __init__(
        self,
        model: Any,
        *,
        benchmark_config: CounterStrikeBenchmarkConfig,
        selector_model: Any | None = None,
        evidence_repair_dir: str | Path | None = None,
    ) -> None:
        if not hasattr(model, "chat_messages"):
            raise TypeError("model must expose chat_messages")
        if not isinstance(benchmark_config, CounterStrikeBenchmarkConfig):
            raise TypeError(
                "benchmark_config must come from "
                "load_counter_strike_benchmark_config"
            )
        self.model = model
        if selector_model is not None and not hasattr(
            selector_model,
            "chat_messages",
        ):
            raise TypeError("selector_model must expose chat_messages")
        if selector_model is model:
            raise TypeError(
                "selector_model and final judge model must be distinct runtime "
                "objects"
            )
        self.selector_model = selector_model
        self.config = benchmark_config
        self.evidence_repair_dir = (
            Path(evidence_repair_dir).expanduser().resolve()
            if evidence_repair_dir is not None
            else None
        )

    def judge_metric(
        self,
        metric: str,
        *,
        evidence: CounterStrikeFrozenEvidence,
        topology_diagram: str | Path,
        topology_context: dict[str, Any],
    ) -> CounterStrikeVisualMetricResult:
        if metric not in SUPPORTED_VISUAL_METRICS:
            raise CounterStrikeVisualJudgeError(
                "unsupported_metric",
                f"unsupported visual metric {metric!r}",
            )
        if not isinstance(evidence, CounterStrikeFrozenEvidence):
            raise CounterStrikeVisualJudgeError(
                "evidence_contract_invalid",
                "evidence must come from load_counter_strike_frozen_evidence",
            )
        diagram = Path(topology_diagram).expanduser().resolve()
        if not diagram.is_file():
            raise CounterStrikeVisualJudgeError(
                "topology_diagram_missing",
                f"topology diagram does not exist: {diagram}",
            )
        topology_diagram_sha256 = _sha256(diagram)
        frozen_topology_context = _validated_topology_context(
            topology_context,
            metric=metric,
        )
        raw_global_packet = tuple(evidence.global_views)
        # Hash validation precedes all image decoding and derived repairs, so
        # tampering is reported as a frozen-evidence contract failure rather
        # than being obscured by a PIL/luminance error.
        _verify_frozen_packet(
            (*evidence.global_views, *evidence.regional_views),
            topology_diagram=diagram,
            topology_diagram_sha256=topology_diagram_sha256,
        )
        fallback_cfg = self.config.raw["visual_evidence"]["active_fallback"]
        raw_global_diagnostics = _packet_luminance_diagnostics(
            raw_global_packet,
            repair_config=fallback_cfg["brightness_repair"],
        )
        brightness_candidates = self._brightness_repair_candidates(
            (*evidence.global_views, *evidence.regional_views),
            capture_dir=evidence.capture_dir,
        )
        repaired_by_source = {
            item.source_view_id: item
            for item in brightness_candidates
            if item.source_view_id is not None
        }
        # Brightness is a deterministic presentation repair, not a camera
        # selection decision. Replace a dark global with its bounded repaired
        # copy before the first judge call. The source ID/hash remains in the
        # descriptor; geometry and camera pose are unchanged.
        global_packet = tuple(
            repaired_by_source.get(item.id, item)
            for item in evidence.global_views
        )
        regional_candidates = tuple(
            repaired_by_source.get(item.id, item)
            for item in evidence.regional_views
        )
        global_diagnostics = _packet_luminance_diagnostics(
            global_packet,
            repair_config=fallback_cfg["brightness_repair"],
        )
        brightness_repaired_global_ids = tuple(
            item.id
            for item in global_packet
            if item.presentation == "brightness_repair"
        )
        selected: tuple[CounterStrikeEvidenceDescriptor, ...] = ()
        selector_result: dict[str, Any] | None = None
        phase = (
            "global_brightness_repaired"
            if brightness_repaired_global_ids
            else "global"
        )
        deterministic_repair_required = (
            global_diagnostics["usable_view_count"]
            < int(fallback_cfg["minimum_usable_global_views"])
        )
        # Do not spend three judge calls on a packet that the deterministic
        # evidence gate already knows has fewer than two usable global angles.
        if deterministic_repair_required:
            repeats: tuple[dict[str, Any], ...] = ()
            model_requested_repair = False
        else:
            repeats = self._repeat_judgement(
                metric,
                packet=global_packet,
                topology_diagram=diagram,
                topology_diagram_sha256=topology_diagram_sha256,
                topology_context=frozen_topology_context,
                evidence_phase=phase,
            )
            model_requested_repair = any(
                result["evidence_status"] == "insufficient"
                for result in repeats
            )
        if model_requested_repair or deterministic_repair_required:
            if not fallback_cfg["enabled"]:
                return self._combine(
                    metric,
                    repeats=repeats,
                    packet=global_packet,
                    topology_diagram_sha256=topology_diagram_sha256,
                    topology_context=frozen_topology_context,
                    evidence_phase=phase,
                    selected=selected,
                    selector=None,
                )
            missing_evidence = _missing_evidence_union(repeats)
            if deterministic_repair_required:
                missing_evidence.append(
                    "fewer_than_two_usable_global_camera_angles"
                )
            if global_diagnostics["dark_view_ids"]:
                missing_evidence.append("dark_global_camera_evidence")
            missing_evidence = sorted(set(missing_evidence))
            selector_result = self._select_regional(
                metric,
                candidates=regional_candidates,
                missing_evidence=missing_evidence,
                maximum=int(fallback_cfg["max_extra_views"]),
            )
            selector_result["raw_global_evidence_diagnostics"] = (
                raw_global_diagnostics
            )
            selector_result["effective_global_evidence_diagnostics"] = (
                global_diagnostics
            )
            selector_result["brightness_repaired_global_ids"] = list(
                brightness_repaired_global_ids
            )
            selected_ids = tuple(selector_result["selected_view_ids"])
            by_id = {item.id: item for item in regional_candidates}
            selected = tuple(by_id[view_id] for view_id in selected_ids)
            if selected:
                phase = f"{phase}_plus_regional"
                packet = global_packet + selected
                repeats = self._repeat_judgement(
                    metric,
                    packet=packet,
                    topology_diagram=diagram,
                    topology_diagram_sha256=topology_diagram_sha256,
                    topology_context=frozen_topology_context,
                    evidence_phase=phase,
                )
            else:
                packet = global_packet
                if deterministic_repair_required:
                    return self._repair_exhausted_result(
                        metric,
                        repeats=repeats,
                        packet=packet,
                        topology_diagram_sha256=topology_diagram_sha256,
                        topology_context=frozen_topology_context,
                        selector=selector_result,
                    )
        else:
            packet = global_packet
        return self._combine(
            metric,
            repeats=repeats,
            packet=packet,
            topology_diagram_sha256=topology_diagram_sha256,
            topology_context=frozen_topology_context,
            evidence_phase=phase,
            selected=selected,
            selector=selector_result,
        )

    def _brightness_repair_candidates(
        self,
        candidates: tuple[CounterStrikeEvidenceDescriptor, ...],
        *,
        capture_dir: Path,
    ) -> tuple[CounterStrikeEvidenceDescriptor, ...]:
        config = self.config.raw["visual_evidence"]["active_fallback"][
            "brightness_repair"
        ]
        if not config["enabled"]:
            return ()
        root = self.evidence_repair_dir
        if root is None:
            root = capture_dir.parent / "counter_strike_observation_repairs"
        root.mkdir(parents=True, exist_ok=True)
        repaired: list[CounterStrikeEvidenceDescriptor] = []
        for item in candidates:
            diagnostics = _image_luminance_diagnostics(item.path)
            if not _is_dark_evidence(diagnostics, config=config):
                continue
            target = float(config["target_median_luminance"])
            gain = min(
                float(config["max_gain"]),
                max(
                    1.0,
                    target / max(diagnostics["median_luminance"], 1.0 / 255.0),
                ),
            )
            fingerprint = hashlib.sha256(
                (
                    COUNTER_STRIKE_VISUAL_JUDGE_VERSION
                    + "\0"
                    + item.sha256
                    + "\0"
                    + f"{gain:.8f}"
                ).encode("utf-8")
            ).hexdigest()[:16]
            destination = root / f"brightness_{item.id}_{fingerprint}.png"
            if not destination.is_file():
                with Image.open(item.path) as source:
                    corrected = ImageEnhance.Brightness(
                        source.convert("RGB")
                    ).enhance(gain)
                    corrected.save(destination, format="PNG")
            output_diagnostics = _image_luminance_diagnostics(destination)
            repaired.append(
                CounterStrikeEvidenceDescriptor(
                    id=f"brightness__{item.id}",
                    role="brightness_repair",
                    path=destination,
                    sha256=_sha256(destination),
                    presentation="brightness_repair",
                    source_view_id=item.id,
                    luminance={
                        **output_diagnostics,
                        "gain": float(gain),
                    },
                )
            )
        return tuple(repaired)

    def _repair_exhausted_result(
        self,
        metric: str,
        *,
        repeats: tuple[dict[str, Any], ...],
        packet: tuple[CounterStrikeEvidenceDescriptor, ...],
        topology_diagram_sha256: str,
        topology_context: dict[str, Any],
        selector: dict[str, Any],
    ) -> CounterStrikeVisualMetricResult:
        contract_hash = _input_contract_sha256(
            metric=metric,
            packet=packet,
            evidence_phase="observation_repair_exhausted",
            model_id=str(getattr(self.model, "model_id", "unknown")),
            benchmark_sha256=self.config.sha256,
            topology_context=topology_context,
            topology_diagram_sha256=topology_diagram_sha256,
        )
        return CounterStrikeVisualMetricResult(
            metric=metric,
            status="unresolved",
            score=None,
            verdict="ambiguous",
            evidence_phase="observation_repair_exhausted",
            selected_regional_ids=(),
            brightness_repaired_view_ids=tuple(
                item.id
                for item in packet
                if item.presentation == "brightness_repair"
            ),
            repeats=repeats,
            selector=selector,
            input_contract_sha256=contract_hash,
        )

    def _repeat_judgement(
        self,
        metric: str,
        *,
        packet: tuple[CounterStrikeEvidenceDescriptor, ...],
        topology_diagram: Path,
        topology_diagram_sha256: str,
        topology_context: dict[str, Any],
        evidence_phase: str,
    ) -> tuple[dict[str, Any], ...]:
        _verify_frozen_packet(
            packet,
            topology_diagram=topology_diagram,
            topology_diagram_sha256=topology_diagram_sha256,
        )
        repeat_count = int(self.config.raw["visual_evidence"]["judge"]["repeats"])
        prompt = _judge_user_prompt(
            metric,
            metric_config=self.config.raw["l4_metrics"][metric],
            topology_context=topology_context,
            evidence_phase=evidence_phase,
            view_ids=[item.id for item in packet],
        )
        messages = _multimodal_messages(
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_prompt=prompt,
            images=[(item.id, item.path) for item in packet]
            + [("topology_diagram", topology_diagram)],
        )
        retry_budget = int(
            self.config.raw["visual_evidence"]["judge"][
                "response_contract_retries"
            ]
        )
        results: list[dict[str, Any]] = []
        for repeat_index in range(repeat_count):
            retry_codes: list[str] = []
            for attempt_index in range(retry_budget + 1):
                try:
                    raw = self.model.chat_messages(
                        messages,
                        response_format_json=False,
                        call_type=f"cs_static_design.{metric}",
                    )
                    result = _strict_json_object(raw)
                    normalized = _validate_metric_response(
                        result,
                        metric=metric,
                        allowed_view_ids={
                            item.id for item in packet
                        },
                        metric_config=self.config.raw["l4_metrics"][metric],
                    )
                    break
                except CounterStrikeVisualJudgeError as exc:
                    if (
                        exc.code not in _RETRYABLE_RESPONSE_ERROR_CODES
                        or attempt_index >= retry_budget
                    ):
                        raise
                    retry_codes.append(exc.code)
            normalized["repeat_index"] = repeat_index
            normalized["response_contract_retry_count"] = len(retry_codes)
            normalized["response_contract_retry_codes"] = retry_codes
            normalized["request_metadata"] = _safe_request_metadata(
                getattr(self.model, "last_request_metadata", {})
            )
            results.append(normalized)
        return tuple(results)

    def _select_regional(
        self,
        metric: str,
        *,
        candidates: tuple[CounterStrikeEvidenceDescriptor, ...],
        missing_evidence: list[str],
        maximum: int,
    ) -> dict[str, Any]:
        if self.selector_model is None:
            raise CounterStrikeVisualJudgeError(
                "selector_not_configured",
                "regional fallback requires a separately configured selector "
                "runtime object",
            )
        prompt = json.dumps(
            {
                "task": "select regional evidence only",
                "metric": metric,
                "metric_rubric": _metric_rubric(
                    metric, self.config.raw["l4_metrics"][metric]
                ),
                "missing_evidence": missing_evidence,
                "candidate_view_ids": [item.id for item in candidates],
                "candidate_views": [
                    {
                        "id": item.id,
                        "role": item.role,
                        "presentation": item.presentation,
                        "source_view_id": item.source_view_id,
                        "luminance": item.luminance,
                    }
                    for item in candidates
                ],
                "max_views": maximum,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        messages = _multimodal_messages(
            system_prompt=SELECTOR_SYSTEM_PROMPT,
            user_prompt=prompt,
            images=[(item.id, item.path) for item in candidates],
        )
        raw = self.selector_model.chat_messages(
            messages,
            response_format_json=False,
            call_type=f"cs_static_design.selector.{metric}",
        )
        parsed = _strict_json_object(raw)
        forbidden = {"verdict", "score", "valid", "invalid"}.intersection(parsed)
        if forbidden:
            raise CounterStrikeVisualJudgeError(
                "selector_returned_verdict",
                f"selector returned forbidden fields {sorted(forbidden)}",
            )
        required_fields = {"evidence_status", "selected_view_ids", "reason"}
        if set(parsed) != required_fields:
            raise CounterStrikeVisualJudgeError(
                "selector_schema_invalid",
                f"selector fields must be exactly {sorted(required_fields)}",
            )
        status = parsed.get("evidence_status")
        if status not in {"selected", "no_helpful_candidate"}:
            raise CounterStrikeVisualJudgeError(
                "selector_schema_invalid",
                "selector evidence_status must be selected or "
                "no_helpful_candidate",
            )
        selected = parsed.get("selected_view_ids")
        if not isinstance(selected, list) or any(
            not isinstance(value, str) for value in selected
        ):
            raise CounterStrikeVisualJudgeError(
                "selector_schema_invalid",
                "selector selected_view_ids must be an array of strings",
            )
        if len(selected) != len(set(selected)) or len(selected) > maximum:
            raise CounterStrikeVisualJudgeError(
                "selector_schema_invalid",
                "selector returned duplicate or over-budget view IDs",
            )
        candidate_ids = {item.id for item in candidates}
        if not set(selected).issubset(candidate_ids):
            raise CounterStrikeVisualJudgeError(
                "selector_unknown_view_id",
                "selector returned an unknown regional view ID",
            )
        by_id = {item.id: item for item in candidates}
        source_ids = [
            by_id[view_id].source_view_id or view_id for view_id in selected
        ]
        if len(source_ids) != len(set(source_ids)):
            raise CounterStrikeVisualJudgeError(
                "selector_schema_invalid",
                "selector cannot spend two slots on raw/repaired copies of "
                "the same camera angle",
            )
        if status == "selected" and not selected:
            raise CounterStrikeVisualJudgeError(
                "selector_schema_invalid",
                "selected status requires at least one regional view ID",
            )
        if status == "no_helpful_candidate" and selected:
            raise CounterStrikeVisualJudgeError(
                "selector_schema_invalid",
                "no_helpful_candidate must select zero IDs",
            )
        reason = parsed.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise CounterStrikeVisualJudgeError(
                "selector_schema_invalid",
                "selector reason must be non-empty",
            )
        return {
            "evidence_status": status,
            "selected_view_ids": selected,
            "reason": reason.strip(),
            "request_metadata": _safe_request_metadata(
                getattr(self.selector_model, "last_request_metadata", {})
            ),
            "selector_prompt_version": COUNTER_STRIKE_SELECTOR_PROMPT_VERSION,
        }

    def _combine(
        self,
        metric: str,
        *,
        repeats: tuple[dict[str, Any], ...],
        packet: tuple[CounterStrikeEvidenceDescriptor, ...],
        topology_diagram_sha256: str,
        topology_context: dict[str, Any],
        evidence_phase: str,
        selected: tuple[CounterStrikeEvidenceDescriptor, ...],
        selector: dict[str, Any] | None,
    ) -> CounterStrikeVisualMetricResult:
        contract_hash = _input_contract_sha256(
            metric=metric,
            packet=packet,
            evidence_phase=evidence_phase,
            model_id=str(getattr(self.model, "model_id", "unknown")),
            benchmark_sha256=self.config.sha256,
            topology_context=topology_context,
            topology_diagram_sha256=topology_diagram_sha256,
        )
        judge_config = self.config.raw["visual_evidence"]["judge"]
        if (
            judge_config["verdict_aggregation"]
            != COUNTER_STRIKE_VERDICT_AGGREGATION
            or judge_config["score_aggregation"]
            != COUNTER_STRIKE_SCORE_AGGREGATION
        ):
            raise CounterStrikeVisualJudgeError(
                "repeat_policy_invalid",
                "the frozen CS judge requires strict-majority verdicts and "
                "median scores",
            )
        if any(
            item["evidence_status"] == "insufficient"
            or item["verdict"] == "ambiguous"
            for item in repeats
        ):
            return CounterStrikeVisualMetricResult(
                metric=metric,
                status="unresolved",
                score=None,
                verdict="ambiguous",
                evidence_phase=evidence_phase,
                selected_regional_ids=tuple(item.id for item in selected),
                brightness_repaired_view_ids=tuple(
                    item.id
                    for item in packet
                    if item.presentation == "brightness_repair"
                ),
                repeats=repeats,
                selector=selector,
                input_contract_sha256=contract_hash,
            )
        verdict_counts = {
            verdict: sum(item["verdict"] == verdict for item in repeats)
            for verdict in ("valid", "invalid")
        }
        verdict = max(verdict_counts, key=verdict_counts.__getitem__)
        if verdict_counts[verdict] <= len(repeats) // 2:
            return CounterStrikeVisualMetricResult(
                metric=metric,
                status="unresolved",
                score=None,
                verdict="ambiguous",
                evidence_phase=evidence_phase,
                selected_regional_ids=tuple(item.id for item in selected),
                brightness_repaired_view_ids=tuple(
                    item.id
                    for item in packet
                    if item.presentation == "brightness_repair"
                ),
                repeats=repeats,
                selector=selector,
                input_contract_sha256=contract_hash,
            )
        status = "checked"
        score = float(
            statistics.median(float(item["score"]) for item in repeats)
        )
        return CounterStrikeVisualMetricResult(
            metric=metric,
            status=status,
            score=score,
            verdict=verdict,
            evidence_phase=evidence_phase,
            selected_regional_ids=tuple(item.id for item in selected),
            brightness_repaired_view_ids=tuple(
                item.id
                for item in packet
                if item.presentation == "brightness_repair"
            ),
            repeats=repeats,
            selector=selector,
            input_contract_sha256=contract_hash,
        )


def build_counter_strike_visual_judge(
    model_config: dict[str, Any],
    *,
    benchmark_config: CounterStrikeBenchmarkConfig,
    evidence_repair_dir: str | Path | None = None,
) -> CounterStrikeVisualJudge:
    """Build the CS adapter on the repository's shared safe model transport."""

    shared = build_openai_compatible_vlm_judge(model_config)
    selector = build_openai_compatible_camera_selector(model_config)
    return CounterStrikeVisualJudge(
        shared.model,
        benchmark_config=benchmark_config,
        selector_model=selector.model,
        evidence_repair_dir=evidence_repair_dir,
    )


def _metric_rubric(metric: str, metric_config: dict[str, Any]) -> str:
    """The frozen rubric, with countable bars read from the profile.

    zone_clarity's bar is interpolated rather than spelled out in the prose so
    that retuning ``min_clear_roles`` cannot leave the judge reading one number
    while validation enforces another.
    """

    rubric = METRIC_RUBRICS[metric]
    if metric == "zone_clarity":
        return rubric.format(
            min_clear_roles=int(metric_config["min_clear_roles"])
        )
    return rubric


def _judge_user_prompt(
    metric: str,
    *,
    metric_config: dict[str, Any],
    topology_context: dict[str, Any],
    evidence_phase: str,
    view_ids: list[str],
) -> str:
    context = {
        "metric": metric,
        "metric_rubric": _metric_rubric(metric, metric_config),
        "decision_contract": {
            "valid_threshold": float(metric_config["valid_threshold"]),
            "verdict_score_rule": (
                "With sufficient evidence and a non-ambiguous verdict, "
                "return valid iff score >= valid_threshold; return invalid "
                "iff score < valid_threshold."
            ),
        },
        "response_schema": _metric_response_template(
            metric,
            metric_config=metric_config,
        ),
        "scope": "frozen static 3D environment only",
        "evidence_phase": evidence_phase,
        "ordered_original_runtime_view_ids": view_ids,
        "topology_diagram_id": "topology_diagram",
        "neutral_observation_context": topology_context,
        "benchmark_instruction": (
            "Judge only significant, supportable differences. If the images "
            "and diagram cannot show the required fact, return insufficient "
            "rather than assuming validity."
        ),
    }
    return "Judge the named metric only.\n" + json.dumps(
        context,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _metric_response_template(
    metric: str,
    *,
    metric_config: dict[str, Any],
) -> dict[str, Any]:
    valid_score = max(0.8, float(metric_config["valid_threshold"]))
    common: dict[str, Any] = {
        "evidence_status": "sufficient",
        "verdict": "valid",
        "score": valid_score,
        "reason": "metric-scoped reason",
        "defects": [],
        "missing_evidence": [],
    }
    if metric == "zone_clarity":
        common["role_findings"] = [
            {
                "role": "team_a_spawn",
                "status": "clear",
                "spatial_region": "distinct team A spawn region",
                "evidence": "visible/geometric evidence",
            },
            {
                "role": "team_b_spawn",
                "status": "clear",
                "spatial_region": "distinct team B spawn region",
                "evidence": "visible/geometric evidence",
            },
            {
                "role": "preparation",
                "status": "clear",
                "spatial_region": "distinct preparation region",
                "evidence": "evidence for both teams",
            },
            {
                "role": "main_engagement",
                "status": "clear",
                "spatial_region": "distinct main engagement region",
                "evidence": "visible/geometric evidence",
            },
            {
                "role": "flank",
                "status": "clear",
                "spatial_region": "distinct flank region",
                "evidence": "visible/geometric evidence",
            },
        ]
    elif metric == "landmark_legibility":
        common["landmarks"] = [
            {
                "name": f"appearance-based callout {index + 1}",
                "view_ids": ["supplied_original_runtime_view_id"],
                "visible_cue": f"distinct visible identity cue {index + 1}",
                "spatial_region": f"distinct spatial region {index + 1}",
            }
            for index in range(int(metric_config["required_landmarks"]))
        ]
    elif metric == "cover_diversity":
        heights = ("low", "waist", "standing", "tall")
        widths = ("narrow", "medium", "wide")
        arrangements = ("isolated", "paired", "row", "cluster", "compound")
        finding_count = max(
            int(metric_config["min_cover_forms"]),
            int(metric_config["min_height_bands"]),
            int(metric_config["min_width_bands"]),
            int(metric_config["min_arrangement_types"]),
        )
        common["cover_findings"] = [
            {
                "form_name": f"distinct visible cover form {index + 1}",
                "view_ids": ["supplied_original_runtime_view_id"],
                "visible_cue": (
                    f"decisive geometric or assembly feature {index + 1}"
                ),
                "spatial_region": f"distinct spatial region {index + 1}",
                "height_profile": heights[index % len(heights)],
                "width_profile": widths[index % len(widths)],
                "arrangement": arrangements[index % len(arrangements)],
            }
            for index in range(finding_count)
        ]
    else:  # pragma: no cover - guarded by SUPPORTED_VISUAL_METRICS
        raise ValueError(f"unsupported metric {metric!r}")
    return common


def _multimodal_messages(
    *,
    system_prompt: str,
    user_prompt: str,
    images: list[tuple[str, Path]],
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {"type": "text", "text": user_prompt},
    ]
    for alias, path in images:
        content.append({"type": "text", "text": f"Image ID: {alias}"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _normalized_rgb_png_data_url(path)},
            }
        )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def _normalized_rgb_png_data_url(path: Path) -> str:
    try:
        with Image.open(path) as source:
            source.load()
            normalized = ImageOps.exif_transpose(source).convert("RGBA")
            flattened = Image.new("RGB", normalized.size, (255, 255, 255))
            flattened.paste(normalized, mask=normalized.getchannel("A"))
            output = BytesIO()
            flattened.save(output, format="PNG")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise CounterStrikeVisualJudgeError(
            "invalid_outbound_image",
            f"evidence image is not a decodable raster: {path.name}",
        ) from exc
    return "data:image/png;base64," + base64.b64encode(
        output.getvalue()
    ).decode("ascii")


def _strict_json_object(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, str):
        raise CounterStrikeVisualJudgeError(
            "model_response_invalid",
            "model response must be a JSON string",
        )
    try:
        parsed = json.loads(raw.strip())
    except json.JSONDecodeError as exc:
        raise CounterStrikeVisualJudgeError(
            "model_response_invalid",
            "model response must contain exactly one JSON object",
        ) from exc
    if not isinstance(parsed, dict):
        raise CounterStrikeVisualJudgeError(
            "model_response_invalid",
            "model response JSON must be an object",
        )
    return parsed


def _validate_metric_response(
    value: dict[str, Any],
    *,
    metric: str,
    allowed_view_ids: set[str],
    metric_config: dict[str, Any],
) -> dict[str, Any]:
    finding_field = {
        "zone_clarity": "role_findings",
        "landmark_legibility": "landmarks",
        "cover_diversity": "cover_findings",
    }.get(metric)
    if finding_field is None:
        raise CounterStrikeVisualJudgeError(
            "unsupported_metric",
            f"unsupported visual metric {metric!r}",
        )
    required = {
        "evidence_status",
        "verdict",
        "score",
        "reason",
        "defects",
        "missing_evidence",
        finding_field,
    }
    if set(value) != required:
        raise CounterStrikeVisualJudgeError(
            "metric_response_schema_invalid",
            f"response fields must be exactly {sorted(required)}",
        )
    evidence_status = value["evidence_status"]
    verdict = value["verdict"]
    if evidence_status not in {"sufficient", "insufficient"}:
        raise CounterStrikeVisualJudgeError(
            "metric_response_schema_invalid",
            "evidence_status must be sufficient or insufficient",
        )
    if verdict not in {"valid", "invalid", "ambiguous"}:
        raise CounterStrikeVisualJudgeError(
            "metric_response_schema_invalid",
            "verdict must be valid, invalid, or ambiguous",
        )
    reason = value["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise CounterStrikeVisualJudgeError(
            "metric_response_schema_invalid",
            "reason must be a non-empty string",
        )
    defects = _string_list(value["defects"], field="defects")
    missing = _string_list(
        value["missing_evidence"],
        field="missing_evidence",
    )
    if evidence_status == "insufficient":
        if verdict != "ambiguous" or value["score"] is not None or not missing:
            raise CounterStrikeVisualJudgeError(
                "metric_response_schema_invalid",
                "insufficient evidence requires ambiguous, null score, and "
                "non-empty missing_evidence",
            )
    else:
        score = value["score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise CounterStrikeVisualJudgeError(
                "metric_response_schema_invalid",
                "sufficient evidence requires a numeric score",
            )
        if not 0.0 <= float(score) <= 1.0:
            raise CounterStrikeVisualJudgeError(
                "metric_response_schema_invalid",
                "score must be between 0 and 1",
            )
    if verdict == "invalid" and not defects:
        raise CounterStrikeVisualJudgeError(
            "metric_response_schema_invalid",
            "invalid requires at least one explicit defect",
        )
    findings = value[finding_field]
    if not isinstance(findings, list):
        raise CounterStrikeVisualJudgeError(
            "metric_response_schema_invalid",
            f"{finding_field} must be an array",
        )
    if evidence_status == "insufficient":
        if defects or findings:
            raise CounterStrikeVisualJudgeError(
                "metric_response_schema_invalid",
                "insufficient evidence must not invent defects or findings",
            )
        normalized = {
            "evidence_status": evidence_status,
            "verdict": verdict,
            "score": None,
            "reason": reason.strip(),
            "defects": [],
            "missing_evidence": missing,
        }
        normalized[finding_field] = []
        return normalized
    if metric == "landmark_legibility":
        normalized_landmarks = [
            _validate_landmark(item, allowed_view_ids=allowed_view_ids)
            for item in findings
        ]
        required_landmarks = int(metric_config["required_landmarks"])
        unique_names = {
            item["name"].strip().lower() for item in normalized_landmarks
        }
        unique_regions = {
            item["spatial_region"].strip().lower()
            for item in normalized_landmarks
        }
        unique_visible_cues = {
            _normalized_finding_text(item["visible_cue"])
            for item in normalized_landmarks
        }
        if verdict == "valid" and (
            len(unique_names) < required_landmarks
            or len(unique_visible_cues) < required_landmarks
            or (
                metric_config["require_spatially_distinct_regions"]
                and len(unique_regions) < required_landmarks
            )
        ):
            raise CounterStrikeVisualJudgeError(
                "metric_response_schema_invalid",
                "valid landmark_legibility requires the configured number of "
                "distinct named landmarks, visible identity cues, and spatial "
                "regions",
            )
        normalized_findings = normalized_landmarks
    elif metric == "zone_clarity":
        roles = [_validate_role_finding(item) for item in findings]
        names = {item["role"] for item in roles}
        required_roles = {
            "team_a_spawn",
            "team_b_spawn",
            "preparation",
            "main_engagement",
            "flank",
        }
        if names != required_roles or len(roles) != len(required_roles):
            raise CounterStrikeVisualJudgeError(
                "metric_response_schema_invalid",
                "zone_clarity must return exactly one finding for each "
                "required role",
            )
        if verdict == "valid" and any(
            item["status"] == "missing" for item in roles
        ):
            raise CounterStrikeVisualJudgeError(
                "metric_response_schema_invalid",
                "valid zone_clarity cannot mark a required role missing",
            )
        if verdict == "valid" and metric_config[
            "require_non_overlapping_roles"
        ]:
            distinct_regions = {
                _normalized_finding_text(item["spatial_region"])
                for item in roles
            }
            if len(distinct_regions) != len(roles):
                raise CounterStrikeVisualJudgeError(
                    "metric_response_schema_invalid",
                    "valid zone_clarity requires a distinct non-overlapping "
                    "spatial_region for every required role",
                )
        # Without a countable bar the per-role findings constrained nothing.
        # The same arena could come back valid on one repeat and invalid on the
        # next while both responses listed most roles as weak, and validation
        # accepted both. Tying the verdict to the findings will not make the
        # judge see the same thing twice, but it forces a flip to appear as a
        # changed finding rather than as an unexplained change of score.
        clear_role_count = sum(
            item["status"] == "clear" for item in roles
        )
        minimum_clear_roles = int(metric_config["min_clear_roles"])
        if verdict == "valid" and clear_role_count < minimum_clear_roles:
            raise CounterStrikeVisualJudgeError(
                "metric_response_schema_invalid",
                "valid zone_clarity requires at least "
                f"{minimum_clear_roles} clear roles; got {clear_role_count}",
            )
        normalized_findings = roles
    elif metric == "cover_diversity":
        normalized_covers = [
            _validate_cover_finding(
                item,
                allowed_view_ids=allowed_view_ids,
            )
            for item in findings
        ]
        if verdict == "valid":
            unique_names = {
                item["form_name"].strip().lower()
                for item in normalized_covers
            }
            unique_visible_cues = {
                _normalized_finding_text(item["visible_cue"])
                for item in normalized_covers
            }
            height_profiles = {
                item["height_profile"] for item in normalized_covers
            }
            width_profiles = {
                item["width_profile"] for item in normalized_covers
            }
            arrangements = {
                item["arrangement"] for item in normalized_covers
            }
            archetype_signatures = {
                (
                    _normalized_finding_text(item["form_name"]),
                    _normalized_finding_text(item["visible_cue"]),
                    item["height_profile"],
                    item["width_profile"],
                    item["arrangement"],
                )
                for item in normalized_covers
            }
            if (
                len(unique_names) < int(metric_config["min_cover_forms"])
                or len(unique_visible_cues)
                < int(metric_config["min_cover_forms"])
                or len(archetype_signatures)
                < int(metric_config["min_cover_forms"])
                or len(height_profiles)
                < int(metric_config["min_height_bands"])
                or len(width_profiles)
                < int(metric_config["min_width_bands"])
                or len(arrangements)
                < int(metric_config["min_arrangement_types"])
            ):
                raise CounterStrikeVisualJudgeError(
                    "metric_response_schema_invalid",
                    "valid cover_diversity requires the configured number of "
                    "distinct visible archetypes and identity cues, "
                    "height/width profiles, and arrangements",
                )
        normalized_findings = normalized_covers
    else:  # pragma: no cover - guarded by SUPPORTED_VISUAL_METRICS
        raise CounterStrikeVisualJudgeError(
            "unsupported_metric",
            f"unsupported visual metric {metric!r}",
        )
    if evidence_status == "sufficient" and verdict != "ambiguous":
        threshold = float(metric_config["valid_threshold"])
        score_value = float(value["score"])
        if (verdict == "valid") != (score_value >= threshold):
            raise CounterStrikeVisualJudgeError(
                "verdict_score_inconsistent",
                "verdict is inconsistent with the frozen valid_threshold",
            )
    normalized = {
        "evidence_status": evidence_status,
        "verdict": verdict,
        "score": (
            None if value["score"] is None else float(value["score"])
        ),
        "reason": reason.strip(),
        "defects": defects,
        "missing_evidence": missing,
    }
    normalized[finding_field] = normalized_findings
    return normalized


def _validate_landmark(
    value: Any,
    *,
    allowed_view_ids: set[str],
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "name",
        "view_ids",
        "visible_cue",
        "spatial_region",
    }:
        raise CounterStrikeVisualJudgeError(
            "metric_response_schema_invalid",
            "each landmark requires name, view_ids, visible_cue, spatial_region",
        )
    view_ids = _string_list(value["view_ids"], field="landmark.view_ids")
    if (
        not view_ids
        or len(view_ids) != len(set(view_ids))
        or not set(view_ids).issubset(allowed_view_ids)
    ):
        raise CounterStrikeVisualJudgeError(
            "metric_response_schema_invalid",
            "landmark view_ids must cite supplied original-runtime views",
        )
    result = {
        "name": _non_empty(value["name"], "landmark.name"),
        "view_ids": view_ids,
        "visible_cue": _non_empty(
            value["visible_cue"],
            "landmark.visible_cue",
        ),
        "spatial_region": _non_empty(
            value["spatial_region"],
            "landmark.spatial_region",
        ),
    }
    return result


def _validate_cover_finding(
    value: Any,
    *,
    allowed_view_ids: set[str],
) -> dict[str, Any]:
    required = {
        "form_name",
        "view_ids",
        "visible_cue",
        "spatial_region",
        "height_profile",
        "width_profile",
        "arrangement",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise CounterStrikeVisualJudgeError(
            "metric_response_schema_invalid",
            "each cover finding requires form_name, view_ids, visible_cue, "
            "spatial_region, height_profile, width_profile, and arrangement",
        )
    view_ids = _string_list(
        value["view_ids"],
        field="cover_finding.view_ids",
    )
    if (
        not view_ids
        or len(view_ids) != len(set(view_ids))
        or not set(view_ids).issubset(allowed_view_ids)
    ):
        raise CounterStrikeVisualJudgeError(
            "metric_response_schema_invalid",
            "cover finding view_ids must cite supplied original-runtime views",
        )
    height = value["height_profile"]
    if height not in {"low", "waist", "standing", "tall"}:
        raise CounterStrikeVisualJudgeError(
            "metric_response_schema_invalid",
            "cover height_profile must be low, waist, standing, or tall",
        )
    width = value["width_profile"]
    if width not in {"narrow", "medium", "wide"}:
        raise CounterStrikeVisualJudgeError(
            "metric_response_schema_invalid",
            "cover width_profile must be narrow, medium, or wide",
        )
    arrangement = value["arrangement"]
    if arrangement not in {
        "isolated",
        "paired",
        "row",
        "cluster",
        "compound",
    }:
        raise CounterStrikeVisualJudgeError(
            "metric_response_schema_invalid",
            "cover arrangement must be isolated, paired, row, cluster, or "
            "compound",
        )
    return {
        "form_name": _non_empty(
            value["form_name"],
            "cover_finding.form_name",
        ),
        "view_ids": view_ids,
        "visible_cue": _non_empty(
            value["visible_cue"],
            "cover_finding.visible_cue",
        ),
        "spatial_region": _non_empty(
            value["spatial_region"],
            "cover_finding.spatial_region",
        ),
        "height_profile": height,
        "width_profile": width,
        "arrangement": arrangement,
    }


def _validate_role_finding(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "role",
        "status",
        "spatial_region",
        "evidence",
    }:
        raise CounterStrikeVisualJudgeError(
            "metric_response_schema_invalid",
            "each role finding requires role, status, spatial_region, evidence",
        )
    role = _non_empty(value["role"], "role_finding.role")
    status = value["status"]
    if status not in {"clear", "weak", "missing"}:
        raise CounterStrikeVisualJudgeError(
            "metric_response_schema_invalid",
            "role finding status must be clear, weak, or missing",
        )
    return {
        "role": role,
        "status": status,
        "spatial_region": _non_empty(
            value["spatial_region"],
            "role_finding.spatial_region",
        ),
        "evidence": _non_empty(
            value["evidence"],
            "role_finding.evidence",
        ),
    }


def _string_list(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise CounterStrikeVisualJudgeError(
            "metric_response_schema_invalid",
            f"{field} must be an array of non-empty strings",
        )
    return [item.strip() for item in value]


def _non_empty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CounterStrikeVisualJudgeError(
            "metric_response_schema_invalid",
            f"{field} must be a non-empty string",
        )
    return value.strip()


def _normalized_finding_text(value: str) -> str:
    """Normalize a model-authored identity cue for exact duplicate rejection.

    This is intentionally conservative.  The semantic equivalence decision
    remains in the frozen strict rubric; this check only prevents a response
    from satisfying a count by repeating the same cue with capitalization or
    whitespace changes.
    """

    return " ".join(value.strip().lower().split())


def _image_luminance_diagnostics(path: Path) -> dict[str, float]:
    try:
        with Image.open(path) as source:
            grayscale = ImageOps.grayscale(source.convert("RGB"))
            histogram = grayscale.histogram()
            pixel_count = max(1, sum(histogram))
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise CounterStrikeVisualJudgeError(
            "evidence_image_invalid",
            f"could not measure frozen evidence image {path.name!r}",
        ) from exc

    def percentile(fraction: float) -> float:
        target = max(1, int(math.ceil(pixel_count * fraction)))
        cumulative = 0
        for value, count in enumerate(histogram):
            cumulative += int(count)
            if cumulative >= target:
                return float(value) / 255.0
        return 1.0

    return {
        "median_luminance": percentile(0.50),
        "p90_luminance": percentile(0.90),
    }


def _is_dark_evidence(
    diagnostics: dict[str, float],
    *,
    config: dict[str, Any],
) -> bool:
    return (
        float(diagnostics["median_luminance"])
        < float(config["median_luminance_threshold"])
        and float(diagnostics["p90_luminance"])
        < float(config["p90_luminance_threshold"])
    )


def _packet_luminance_diagnostics(
    packet: tuple[CounterStrikeEvidenceDescriptor, ...],
    *,
    repair_config: dict[str, Any],
) -> dict[str, Any]:
    views: list[dict[str, Any]] = []
    for item in packet:
        diagnostics = (
            dict(item.luminance)
            if isinstance(item.luminance, dict)
            else _image_luminance_diagnostics(item.path)
        )
        dark = _is_dark_evidence(diagnostics, config=repair_config)
        views.append(
            {
                "id": item.id,
                **diagnostics,
                "dark": dark,
            }
        )
    return {
        "policy": "deterministic_luminance_gate_v1",
        "view_count": len(views),
        "usable_view_count": sum(not item["dark"] for item in views),
        "dark_view_ids": [item["id"] for item in views if item["dark"]],
        "views": views,
    }


def _missing_evidence_union(repeats: Iterable[dict[str, Any]]) -> list[str]:
    values = {
        str(item)
        for repeat in repeats
        for item in repeat.get("missing_evidence") or []
        if str(item)
    }
    return sorted(values)


def _verify_frozen_packet(
    packet: tuple[CounterStrikeEvidenceDescriptor, ...],
    *,
    topology_diagram: Path,
    topology_diagram_sha256: str,
) -> None:
    for index, item in enumerate(packet):
        if not isinstance(item, CounterStrikeEvidenceDescriptor):
            raise CounterStrikeVisualJudgeError(
                "evidence_contract_invalid",
                f"evidence packet item {index} is not a frozen descriptor",
            )
        if not item.path.is_file():
            raise CounterStrikeVisualJudgeError(
                "evidence_file_missing",
                f"frozen evidence file does not exist: {item.path}",
            )
        actual = _sha256(item.path)
        if actual != item.sha256:
            raise CounterStrikeVisualJudgeError(
                "evidence_hash_mismatch",
                f"frozen evidence hash drifted for view {item.id!r}",
            )
    if _sha256(topology_diagram) != topology_diagram_sha256:
        raise CounterStrikeVisualJudgeError(
            "topology_diagram_hash_mismatch",
            "topology diagram changed during visual adjudication",
        )


def _input_contract_sha256(
    *,
    metric: str,
    packet: tuple[CounterStrikeEvidenceDescriptor, ...],
    evidence_phase: str,
    model_id: str,
    benchmark_sha256: str,
    topology_context: dict[str, Any],
    topology_diagram_sha256: str,
) -> str:
    payload = {
        "version": COUNTER_STRIKE_VISUAL_JUDGE_VERSION,
        "prompt_version": COUNTER_STRIKE_METRIC_PROMPT_VERSION,
        "metric": metric,
        "evidence_phase": evidence_phase,
        "model_id": model_id,
        "benchmark_sha256": benchmark_sha256,
        "topology_context_sha256": hashlib.sha256(
            json.dumps(
                topology_context,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
        "ordered_evidence": [
            {"id": item.id, "sha256": item.sha256} for item in packet
        ],
        "topology_diagram_sha256": topology_diagram_sha256,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _safe_request_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    primitive_fields = {
        "model",
        "timeout_seconds",
        "response_format_json",
        "call_type",
        "message_count",
        "image_count",
        "prompt_chars",
        "authorization_configured",
        "max_tokens_field",
        "send_temperature",
        "finish_reason",
        "content_chars",
    }
    safe: dict[str, Any] = {}
    endpoint = _safe_endpoint(value.get("endpoint"))
    if endpoint is not None:
        safe["endpoint"] = endpoint
    api_key_env = value.get("api_key_env")
    if isinstance(api_key_env, str) and _SAFE_ENV_NAME.fullmatch(api_key_env):
        safe["api_key_env"] = api_key_env
    for key in sorted(primitive_fields.intersection(value)):
        item = value[key]
        if item is None or isinstance(item, (str, int, float, bool)):
            if not isinstance(item, float) or math.isfinite(item):
                safe[key] = item
    usage = value.get("usage")
    if isinstance(usage, dict):
        safe_usage = {
            key: usage[key]
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
            )
            if isinstance(usage.get(key), int)
            and not isinstance(usage.get(key), bool)
            and usage[key] >= 0
        }
        if safe_usage:
            safe["usage"] = safe_usage
    return safe


def _safe_endpoint(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return None
        host = parsed.hostname
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    except (TypeError, ValueError):
        return None


def _validated_topology_context(
    value: Any,
    *,
    metric: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CounterStrikeVisualJudgeError(
            "topology_context_invalid",
            "topology_context must be a JSON object",
        )
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        normalized = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise CounterStrikeVisualJudgeError(
            "topology_context_invalid",
            "topology_context must contain only finite JSON values",
        ) from exc
    if set(normalized) != {
        "schema_version",
        "metric",
        "scope",
        "observation_aid",
    }:
        raise CounterStrikeVisualJudgeError(
            "topology_context_invalid",
            "visual context must use the exact neutral observation schema",
        )
    if (
        normalized["schema_version"]
        != "counter_strike_neutral_visual_context_v1"
        or normalized["metric"] != metric
        or normalized["scope"] != "static_3d_environment_only"
    ):
        raise CounterStrikeVisualJudgeError(
            "topology_context_invalid",
            "visual context identity does not match the neutral contract",
        )
    aid = normalized["observation_aid"]
    if not isinstance(aid, dict) or set(aid) != {
        "shows",
        "omits",
        "ground_truth",
    }:
        raise CounterStrikeVisualJudgeError(
            "topology_context_invalid",
            "observation_aid must use the exact neutral contract",
        )
    if aid["ground_truth"] is not False:
        raise CounterStrikeVisualJudgeError(
            "topology_context_invalid",
            "neutral observation aid cannot be declared ground truth",
        )
    omitted = set(_string_list(aid["omits"], field="observation_aid.omits"))
    required_omissions = {
        "deterministic_scores",
        "deterministic_verdicts",
        "inferred_zone_roles",
        "inferred_routes",
        "cover_proposals",
        "engagement_anchor",
        "case_identity",
    }
    if not required_omissions.issubset(omitted):
        raise CounterStrikeVisualJudgeError(
            "topology_context_invalid",
            "neutral observation aid must omit all deterministic conclusions",
        )
    _string_list(aid["shows"], field="observation_aid.shows")
    return normalized


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
