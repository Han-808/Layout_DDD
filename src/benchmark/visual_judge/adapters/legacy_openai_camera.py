from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Protocol

from benchmark.models import parse_json_object
from benchmark.visual_judge.active_policy import selector_safe_proposals
from benchmark.visual_judge.contracts import (
    validate_camera_selection_response,
)
from benchmark.visual_judge.functional_evidence import (
    functional_probe_selector_context,
)
from benchmark.visual_judge.roles import (
    DecisionContract,
    VLMRole,
    vlm_audit_metadata,
)


CAMERA_SELECTION_SYSTEM_PROMPT = """You select camera evidence for a 3D scene benchmark.
Do not judge whether the metric event is valid or invalid. Select the candidate views that make the
specified event easiest to inspect. Prefer views where all target objects and the relevant contact,
gap, overlap, or room plane are visible and well framed. In highlighted previews, use the supplied
color legend and treat gray geometry as non-target context. A preview_warning_class means only
that deterministic highlight-pixel coverage was incomplete; do not infer target absence from it.
When selection_phase is active_fallback, evidence_deficiency states why the deterministic packet was
not sufficient and corrective_proposals lists the only permitted metric-specific repairs. Use it only
to choose views or one proposal that repairs the named evidence gap. Never invent pose coordinates or
an unlisted action.
When functional_probe is present, select a low or near-interaction-height view
whose camera lies in the usable-side or approach-side half-space. The target's
usable face must remain visually decodable, while a wider field preserves the
outward interaction region and enough local context to determine whether that
region is accessible. Do not choose a view
merely because it is close or low. When usable_surface_hypotheses are present,
treat their trusted local side IDs as observation hypotheses only. For an
ambiguous hypothesis, prefer complementary visibility rather than guessing one
side. For functional_correspondence, prefer one
wider view that jointly exposes the relevant objects' usable sides and their
interaction orientation. A target that occludes another only from an arbitrary
preview camera is not by itself a real user-viewing obstruction; prefer a view
that distinguishes camera parallax from the ordinary interaction direction.
Probe inclusion is not evidence of a defect and must
not be judged here. When the functional probe requires
architecture_plane_visible, choose a view that jointly exposes the decoded
usable or control side, the nearest authoritative logical boundary or visible
architecture reference, and the interior-side user approach and operating
region.
Physical wall geometry may be absent by policy; do not require or invent a
wall, and do not treat background outside the room footprint as usable
interior space.
Candidates marked render_status=blank are unusable camera evidence. Do not select them when any
render_status=ok candidate exists.
You may request at most one listed discrete
camera action when adjustment is allowed. Return exactly one JSON object:
{"selected_view_ids":["candidate_id"],"action":null,"reason":"..."}.
selected_view_ids must contain between one and max_views candidate IDs in evidence-priority order,
best view first. action must be null when
allow_adjustment is false. In active_fallback it may be null or
{"proposal_id":"one listed proposal ID"}; otherwise it may be null or
{"view_id":"candidate_id","type":"one allowed action"}. Do not return a metric verdict or score."""


class LegacyOpenAICameraModel(Protocol):
    """Minimum model transport used by the frozen selector compatibility path."""

    model_id: str
    endpoint: str
    response_format_json: bool
    last_request_metadata: dict[str, Any]

    def chat_messages(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> str: ...


class ContextJSONSerializer(Protocol):
    """Serialize bounded selector context without changing its wire shape."""

    def __call__(
        self,
        context: dict[str, Any],
        max_chars: int,
        *,
        priority_keys: tuple[str, ...] = (),
    ) -> str: ...


class OutboundImageEncoder(Protocol):
    """Encode one sanitized outbound image using a non-identifying label."""

    def __call__(self, path: Path, *, label: str) -> str: ...


class LegacyOpenAICameraSelectionAdapter:
    """Frozen query-cov/active camera-selection compatibility adapter.

    The adapter executes exactly one legacy selector call. It does not own
    cascade policy, rendering, EvidenceGate, metric judgement, or camera
    algorithms. Generic context budgeting and image sanitation are injected
    so this compatibility module does not depend back on the Judge transport.
    """

    def __init__(
        self,
        model: LegacyOpenAICameraModel,
        *,
        context_serializer: ContextJSONSerializer,
        image_encoder: OutboundImageEncoder,
        max_images: int = 6,
        max_context_chars: int = 30000,
        response_format_json: bool | None = None,
    ) -> None:
        self.model = model
        self.context_serializer = context_serializer
        self.image_encoder = image_encoder
        self.max_images = max_images
        self.max_context_chars = max_context_chars
        self.response_format_json = response_format_json

    def select(self, request: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise TypeError("camera selector request must be a JSON object")
        candidates = [
            item
            for item in request.get("candidates", [])
            if isinstance(item, dict)
        ]
        if not candidates:
            raise ValueError("camera selector requires at least one candidate")
        usable_candidates = [
            item
            for item in candidates
            if str(item.get("render_status") or "ok") != "blank"
        ]
        if not usable_candidates:
            raise ValueError(
                "camera selector received no non-blank candidate previews"
            )
        image_limit = max(1, int(self.max_images))
        if len(usable_candidates) > image_limit:
            raise ValueError(
                "camera selector candidate bank exceeds max_images; implicit "
                f"truncation is forbidden ({len(usable_candidates)} > "
                f"{image_limit})"
            )
        internal_ids = [
            str(item.get("id") or "")
            for item in usable_candidates
        ]
        if (
            any(not value for value in internal_ids)
            or len(set(internal_ids)) != len(internal_ids)
        ):
            raise ValueError(
                "camera selector candidates require unique non-empty "
                "internal IDs"
            )
        candidate_paths = [
            Path(str(item.get("image_path"))).expanduser()
            for item in usable_candidates
        ]
        missing = [
            str(path) for path in candidate_paths if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "camera selector preview evidence does not exist: "
                f"{missing}"
            )
        selected_candidates = sorted(
            usable_candidates,
            key=_selector_candidate_order_key,
        )
        paths = [
            Path(str(item.get("image_path"))).expanduser()
            for item in selected_candidates
        ]
        max_views = max(
            1,
            min(
                int(request.get("max_views") or 1),
                len(selected_candidates),
            ),
        )
        alias_to_internal = {
            f"candidate_{index:02d}": str(item.get("id"))
            for index, item in enumerate(selected_candidates)
        }
        internal_to_alias = {
            internal: alias
            for alias, internal in alias_to_internal.items()
        }
        aliases = list(alias_to_internal)
        allowed_actions = _selector_allowed_actions(
            request.get("allowed_actions")
        )
        selection_phase = str(
            request.get("selection_phase") or ""
        ).strip().lower()
        corrective_proposals: list[dict[str, Any]] = []
        proposal_lookup: dict[str, dict[str, Any]] = {}
        if selection_phase == "active_fallback":
            corrective_proposals, proposal_lookup = selector_safe_proposals(
                [
                    item
                    for item in request.get("corrective_proposals", [])
                    if isinstance(item, dict)
                ],
                internal_to_alias=internal_to_alias,
            )
        allow_adjustment = bool(request.get("allow_adjustment")) and bool(
            corrective_proposals
            if selection_phase == "active_fallback"
            else allowed_actions
        )
        audit = vlm_audit_metadata(
            VLMRole.VLM_CAMERA_SELECTOR,
            decision_contract=DecisionContract.CAMERA_SELECTION,
            judge_method="select_camera_views",
        )
        context = {
            **audit,
            "candidates": [
                {
                    "id": alias,
                    "pose": _minimal_selector_pose(item.get("pose")),
                }
                for alias, item in zip(aliases, selected_candidates)
            ],
            "max_views": max_views,
            "allow_adjustment": allow_adjustment,
            "allowed_actions": (
                allowed_actions if allow_adjustment else []
            ),
            "metric_family": _selector_metric_family(
                request.get("metric")
            ),
            "preview_role": _selector_preview_role(
                request.get("preview_role")
            ),
            "preview_warning_class": _selector_preview_warning_class(
                request.get("preview_visibility_warning")
                if request.get("preview_visibility_warning") is not None
                else request.get("preview_degradation")
            ),
            "color_legend": _sanitize_selector_legend(
                request.get("color_legend")
            ),
        }
        functional_probe = functional_probe_selector_context(
            request.get("functional_probe")
        )
        if functional_probe:
            context["functional_probe"] = functional_probe
        if selection_phase == "active_fallback":
            context["selection_phase"] = "active_fallback"
            context["evidence_deficiency"] = _sanitize_selector_deficiency(
                request.get("evidence_deficiency")
            )
            context["corrective_proposals"] = (
                corrective_proposals if allow_adjustment else []
            )
        context_text = self.context_serializer(
            context,
            max(1000, int(self.max_context_chars)),
            priority_keys=(
                "vlm_role",
                "decision_contract",
                "judge_method",
                "metric_family",
                "preview_role",
                "preview_warning_class",
                "color_legend",
                "functional_probe",
                "candidates",
                "max_views",
                "allow_adjustment",
                "allowed_actions",
                "selection_phase",
                "evidence_deficiency",
                "corrective_proposals",
            ),
        )
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Select camera evidence only; do not adjudicate the "
                    "event.\n"
                    + context_text
                ),
            }
        ]
        content.extend(
            {
                "type": "image_url",
                "image_url": {
                    "url": self.image_encoder(path, label=alias)
                },
            }
            for alias, path in zip(aliases, paths)
        )
        use_json_response = (
            bool(getattr(self.model, "response_format_json", True))
            if self.response_format_json is None
            else bool(self.response_format_json)
        )
        raw = self.model.chat_messages(
            [
                {
                    "role": "system",
                    "content": CAMERA_SELECTION_SYSTEM_PROMPT,
                },
                {"role": "user", "content": content},
            ],
            response_format_json=use_json_response,
            call_type=(
                "vlm_camera_pose.active_fallback"
                if selection_phase == "active_fallback"
                else "vlm_camera_pose.query_cov"
            ),
        )
        parsed = parse_json_object(raw)
        available = set(alias_to_internal)
        resolved_aliases = validate_camera_selection_response(
            parsed,
            available_view_ids=available,
            max_views=max_views,
        )
        action = parsed.get("action")
        resolved_action = None
        if action is not None:
            if not allow_adjustment or not isinstance(action, dict):
                raise ValueError(
                    "camera selector returned an action outside the "
                    "adjustment contract"
                )
            if selection_phase == "active_fallback":
                proposal_alias = str(action.get("proposal_id") or "")
                proposal = proposal_lookup.get(proposal_alias)
                if proposal is None:
                    raise ValueError(
                        "active camera selector action references an "
                        "unknown corrective proposal"
                    )
                resolved_action = {
                    "proposal_id": str(
                        proposal.get("proposal_id") or ""
                    ),
                    "view_id": str(
                        proposal.get("parent_view_id") or ""
                    ),
                    "type": str(
                        proposal.get("action_primitive") or ""
                    ),
                    "family": str(proposal.get("family") or ""),
                }
            else:
                if str(action.get("view_id") or "") not in available:
                    raise ValueError(
                        "camera selector action references an unknown "
                        "candidate"
                    )
                if str(action.get("type") or "") not in set(
                    allowed_actions
                ):
                    raise ValueError(
                        "camera selector requested an unsupported action"
                    )
                resolved_action = {
                    "view_id": alias_to_internal[str(action["view_id"])],
                    "type": str(action["type"]),
                }
        reason = parsed.get("reason")
        request_metadata = dict(self.model.last_request_metadata)
        request_metadata.update(
            {
                "selector_candidate_order_policy": (
                    "stable_pose_image_digest_v1"
                ),
                "selector_candidate_alias_policy": (
                    "per_request_sequential_alias_v1"
                ),
            }
        )
        result = {
            "selected_view_ids": [
                alias_to_internal[value]
                for value in resolved_aliases
            ],
            "action": resolved_action,
            "reason": (
                str(reason)[:1000] if reason is not None else ""
            ),
        }
        result.update(audit)
        result["model"] = self.model.model_id
        result["endpoint"] = self.model.endpoint
        # Preserve an auditable image count without exposing local names or
        # paths.
        result["images_used"] = aliases
        result["request_metadata"] = request_metadata
        return result


def _selector_candidate_order_key(
    candidate: dict[str, Any],
) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            _minimal_selector_pose(candidate.get("pose")),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    path = Path(str(candidate.get("image_path"))).expanduser()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selector_metric_family(value: Any) -> str:
    metric = str(value or "").strip().lower()
    allowed = {
        "collision",
        "oob",
        "support",
        "object_architecture_penetration",
        "functional_consistency",
    }
    if metric not in allowed:
        raise ValueError(
            f"camera selector does not support metric family {metric!r}"
        )
    return metric


def _selector_preview_role(value: Any) -> str:
    role = str(value or "").strip().lower()
    return (
        role
        if role in {"highlighted_focus", "rgb_fallback"}
        else "unspecified"
    )


def _selector_preview_warning_class(value: Any) -> str | None:
    if value is None or not str(value).strip():
        return None
    warning = str(value).lower()
    if "blank" in warning:
        return "blank_preview"
    if (
        "visib" in warning
        or "highlight" in warning
        or "coverage" in warning
    ):
        return "incomplete_target_visibility"
    return "preview_warning"


def _selector_allowed_actions(value: Any) -> list[str]:
    allowed = {
        "orbit_left",
        "orbit_right",
        "elevate",
        "lower",
        "dolly_in",
        "dolly_out",
    }
    if not isinstance(value, list):
        return []
    return list(
        dict.fromkeys(
            str(item) for item in value if str(item) in allowed
        )
    )


def _sanitize_selector_deficiency(value: Any) -> dict[str, Any]:
    """Allow only non-identifying, routing-level evidence deficits outbound."""

    source = value if isinstance(value, dict) else {}
    reason_allowlist = {
        "measured_local_visibility_insufficient",
        "required_local_view_count_missing",
        "required_entities_not_jointly_visible",
        "focus_region_out_of_frame",
        "target_occluded_or_too_small",
        "focus_region_too_small",
        "architecture_plane_not_visible",
        "redundant_local_views",
    }
    deficiencies = source.get("deficiencies")
    structured_reasons = (
        [
            str(item.get("code"))
            for item in deficiencies
            if isinstance(item, dict)
            and item.get("repairability") == "camera"
            and str(item.get("code") or "") in reason_allowlist
        ]
        if isinstance(deficiencies, list)
        else []
    )
    reasons = source.get("reason_codes")
    sanitized_reasons = (
        [
            str(reason)
            for reason in reasons
            if str(reason) in reason_allowlist
        ]
        if isinstance(reasons, list)
        else []
    )
    sanitized_reasons = list(
        dict.fromkeys(structured_reasons + sanitized_reasons)
    )
    result: dict[str, Any] = {
        "status": "insufficient",
        "reason_codes": sanitized_reasons,
    }
    for key in (
        "required_local_view_count",
        "measured_local_view_count",
        "usable_local_view_count",
    ):
        raw = source.get(key)
        if (
            isinstance(raw, int)
            and not isinstance(raw, bool)
            and 0 <= raw <= 8
        ):
            result[key] = raw
    utility = source.get("evidence_utility")
    if (
        isinstance(utility, (int, float))
        and not isinstance(utility, bool)
        and 0.0 <= float(utility) <= 1.0
    ):
        result["evidence_utility"] = round(float(utility), 6)
    return result


def _minimal_selector_pose(value: Any) -> dict[str, Any]:
    pose = value if isinstance(value, dict) else {}
    result: dict[str, Any] = {}
    projection = str(
        pose.get("camera_type") or "PERSP"
    ).strip().upper()
    result["projection"] = (
        "orthographic" if projection == "ORTHO" else "perspective"
    )
    for source, target in (
        ("azimuth_degrees", "azimuth_degrees"),
        ("elevation_degrees", "elevation_degrees"),
        ("lens_mm", "lens_mm"),
        ("ortho_scale", "ortho_scale"),
    ):
        raw = pose.get(source)
        if (
            isinstance(raw, (int, float))
            and not isinstance(raw, bool)
            and math.isfinite(float(raw))
        ):
            result[target] = round(float(raw), 3)
    return result


def _sanitize_selector_legend(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    for item in value[:12]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        if role.startswith("related_target_"):
            role = "related_target"
        if role not in {
            "object_a",
            "object_b",
            "primary_target",
            "related_target",
            "architecture_plane",
            "measured_support_gap",
        }:
            role = "annotation"
        entry: dict[str, Any] = {"role": role}
        color = item.get("color")
        if isinstance(color, (list, tuple)) and len(color) >= 3:
            channels: list[float] = []
            for channel in color[:3]:
                if (
                    not isinstance(channel, (int, float))
                    or isinstance(channel, bool)
                ):
                    channels = []
                    break
                numeric = float(channel)
                if not math.isfinite(numeric):
                    channels = []
                    break
                channels.append(
                    round(min(1.0, max(0.0, numeric)), 4)
                )
            if channels:
                entry["rgb"] = channels
        representation = str(
            item.get("representation") or ""
        ).lower()
        if "mesh" in representation:
            entry["representation"] = "mesh"
        elif any(
            token in representation
            for token in ("obb", "bbox", "proxy")
        ):
            entry["representation"] = "proxy"
        elif any(
            token in representation
            for token in ("plane", "boundary")
        ):
            entry["representation"] = "architecture"
        elif representation:
            entry["representation"] = "annotation"
        result.append(entry)
    return result
