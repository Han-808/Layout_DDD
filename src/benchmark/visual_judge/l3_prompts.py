"""Versioned prompts for the L3 scene-quality Judge.

The definitions live in one dependency-light module so the evaluator request
builder and the OpenAI-compatible adapter cannot drift into different metric
semantics.
"""

from __future__ import annotations


L3_METRIC_PROMPT_VERSION = "l3_evidence_discovery_routing_v15"

L3_METRIC_BOUNDARY_RULES = (
    "Additional visual evidence can be acquired.",
    "Judge the current authored scene. Ordinary articulation or handling "
    "intrinsic to use is allowed; rearranging the scene to repair orientation, "
    "access, category, scale, or placement is not.",
    "Evidence insufficiency means that a named observable fact is missing and "
    "a different view could materially resolve it. Semantic uncertainty, "
    "subjective preference, and proximity to a normative threshold are not "
    "evidence insufficiency. If the evidence is adequate but no clear "
    "significant in-scope defect is established, return valid.",
    "Object pairing owns object identity and semantic role. If the same object "
    "would remain inappropriate after relocation to any reasonable location "
    "in the same scene, it is an object-pairing defect.",
    "Semantic placement assumes the identity belongs in the scene. It owns the "
    "current support surface, height, scene zone, and contextual location when "
    "relocation alone would remove the anomaly.",
    "Functional consistency owns ordinary usability in the current "
    "arrangement. It includes usable-side orientation, user access, operating "
    "clearance, interaction direction, and cross-object correspondence. Do "
    "not repair a functional failure by imagining that authored scene objects "
    "are first moved or rotated.",
    "For functional consistency, an enabled authoritative logical room "
    "boundary limits usable floor, standing, approach, and operating space. "
    "Physical wall meshes may be absent by rendering policy: do not penalize "
    "a missing wall or invent one, but do not treat space outside the logical "
    "boundary as available.",
    "Structured functional boundary evidence combines a VLM-decoded trusted "
    "object-local usable-side hypothesis with deterministic scene geometry. "
    "Use its nearest-boundary distance, outward-ray boundary distance, and "
    "inside-boundary approach samples as measurements only. They do not by "
    "themselves establish a defect or define a universal clearance threshold.",
    "Scale consistency owns physical dimensions. Perspective, image-space "
    "size, and camera distance do not establish a scale defect.",
    "Style consistency owns visible design language. Category, physical size, "
    "position, orientation, and function alone do not establish a style "
    "defect.",
    "Collision, penetration, support, floating, and out-of-bounds conditions "
    "are owned by L1. An L1 defect alone must not produce an L3 defect. The same "
    "arrangement may also fail an L3 metric only when that L3 criterion "
    "independently fails.",
    "Metrics may overlap only when each independent criterion fails. Do not "
    "translate one observation into several metric defects without applying "
    "each metric's separate threshold.",
    "Style, functional, and semantic-placement defects use object-level "
    "attribution because each of those metrics independently judges global "
    "and group-local visual scopes. For those metrics, each defect's "
    "target_ids must contain only the "
    "exact object or objects whose own functional or placement state is "
    "defective; never list an entire evidence group as shorthand. Another "
    "object may be named in relation or reason as context without being marked "
    "defective. Within one metric, global and local observations of the same "
    "target object form one penalty unit. This deduplication never crosses "
    "metric boundaries.",
    "A discovery record, routed candidate, functional probe, placement "
    "candidate, or decoded usable-side hypothesis identifies what to inspect; "
    "none is evidence that a defect exists. An incomplete acquisition ledger "
    "also does not override a verdict: request another observable fact only "
    "when that fact is necessary to apply this metric.",
    "Exclude a rendering or materialization artifact only when supplied "
    "structured provenance explicitly identifies it as benchmark-owned "
    "degradation. Do not infer that exemption from appearance alone.",
)

L3_METRIC_RUBRICS = {
    "scale_consistency": (
        "Judge only physical scale. Use structured physical dimensions as the "
        "primary scale evidence and rendered images to verify object identity, "
        "nearby reference context, and perspective. Return invalid only when "
        "all of the following hold: the target is materially too large or too "
        "small for a plausible real-world instance of its identified category; "
        "the mismatch remains after allowing legitimate product, subtype, and "
        "authorized nonstandard-size variation; and the mismatch cannot be "
        "explained by camera distance, perspective, or image-space appearance. "
        "A rare but plausible size variant is valid. Apparent image size alone "
        "is not physical-scale evidence. Do not judge object membership, style, "
        "placement, orientation, usability, collision, support, or general "
        "scene quality. Request visual evidence only when object identity or "
        "relative-scale context is not observable and another view could "
        "resolve it."
    ),
    "style_consistency": (
        "Judge only visible design language. First determine whether the scene "
        "contains one coherent visual language or multiple coherent visual "
        "languages; do not assume that every scene must have a single style. "
        "For each candidate outlier, identify the visible attributes that "
        "differ, such as form language, design era, materials, surface "
        "treatment, ornamentation, visual complexity, palette, or authored "
        "representation language. Compare the object with relevant "
        "scene-global and local ensemble context, then determine whether the "
        "difference is normal within-style variation, a coherent secondary "
        "style, or merely subjective preference. Return invalid only when a "
        "specific object remains a clear and material visual-language outlier "
        "after those alternatives are considered. A difference in category, "
        "size, position, orientation, function, or color alone without a "
        "broader design-language conflict is insufficient. Do not judge "
        "functionality, collision, support, prompt completeness, or general "
        "scene quality."
    ),
    "object_pairing_consistency": (
        "Judge only whether an object's identity and semantic role belong in "
        "this scene and local ensemble, independently of its current "
        "transform. Apply the relocation test: hold identity, category, scale, "
        "and style fixed, then imagine relocating the object to a reasonable "
        "location in the same scene. If at least one ordinary location would "
        "make its role contextually plausible, the object pairing is valid. "
        "Return invalid only when its category or semantic role would remain "
        "clearly inappropriate regardless of reasonable relocation. The "
        "supplied group is evidence scope, not compatibility ground truth. Do "
        "not judge current support surface, zone, height, distance, "
        "orientation, access, clearance, scale, style, collision, or physical "
        "support."
    ),
    "functional_consistency": (
        "Judge only whether ordinary real-world use is feasible in the current "
        "authored arrangement. For every functional claim, first identify the "
        "object's ordinary operation and its usable, opening, display, seating, "
        "control, or interaction side from visible geometry and affordances. "
        "Do not infer a usable side solely from category, transform metadata, "
        "or the surface facing the camera. Next identify the minimum conditions "
        "for ordinary use: an accessible usable side, a plausible user approach "
        "zone, sufficient operating clearance, and any required interaction "
        "orientation with related objects. Evaluate those conditions in the "
        "current arrangement. When the canonical scene declares an enabled "
        "logical room boundary, treat it as the hard limit of usable floor, "
        "standing, approach, and operating space. Physical wall geometry may "
        "be absent by policy: do not penalize the missing wall or infer one, "
        "but space outside the logical boundary is not available for ordinary "
        "use. Ordinary articulation or handling intrinsic to use is allowed; "
        "do not assume that authored scene objects are first relocated, "
        "rotated, resized, or removed. Return invalid when at least one "
        "necessary condition is materially unavailable: the usable side faces "
        "outside the usable room footprint or another inaccessible region, a "
        "normal user approach zone is "
        "unavailable, required operating clearance is materially blocked, or "
        "related objects have incompatible interaction orientation for their "
        "ordinary joint use. Complete geometric sealing is not required. A "
        "layout may be materially unusable even when a narrow or awkward "
        "theoretical interaction remains. Conversely, unconventional or "
        "suboptimal placement is valid when ordinary use remains readily "
        "feasible without repairing the authored layout. Cross-group "
        "functional relations remain in scope for the scene-global pass. "
        "Attribute each defect only to the object or objects whose ordinary use "
        "fails. If the usable side, nearest enabled logical boundary, "
        "interior-side approach zone, or joint interaction orientation is not "
        "visually determinable, request the corresponding additional "
        "observation. Occlusion introduced only by the diagnostic camera "
        "position is not evidence that the ordinary user-facing view is "
        "blocked in the authored scene. Do not judge identity membership, "
        "semantic "
        "location alone, style, scale, prompt fidelity, or L1 geometry by "
        "itself."
    ),
    "semantic_placement_consistency": (
        "Judge only semantic location, assuming that the object identity "
        "belongs in the scene and could perform its intended function when "
        "suitably placed. Evaluate the current support surface, placement "
        "height, scene zone, adjacency to context-setting objects, and "
        "immediate local context. Apply the relocation-only test: hold "
        "identity, scale, style, intended function, and ordinary usable "
        "orientation fixed, then ask whether moving the same object to an "
        "ordinary location in the same scene would remove the anomaly. Return "
        "invalid only when relocation alone would resolve the issue and the "
        "current location is materially inappropriate for the object's "
        "contextual role. An unconventional but contextually plausible "
        "location is valid. Orientation, facing direction, access, opening "
        "clearance, and operability belong to functional consistency. Category "
        "incompatibility belongs to object pairing. Collision, penetration, "
        "floating, support, and out-of-bounds geometry belong to L1. Attribute "
        "every defect to the exact misplaced object IDs rather than to the "
        "whole inspected group."
    ),
}


L3_METRIC_PHASE_PROMPTS = {
    "style_consistency": {
        "global_discovery": (
            "Establish the scene-wide visual language or coherent mixture, "
            "then localize only material object-level outliers. A weak or "
            "uncertain difference is a local-confirmation need, not a defect. "
            "Do not infer fine detail absent from the overview."
        ),
        "group_local_review": (
            "Compare the supplied members' visible form and material details "
            "with the global style context. Report only exact members that "
            "remain material outliers; do not mark the group as shorthand."
        ),
    },
    "functional_consistency": {
        "global_discovery": (
            "Use the global anchor to inspect direct-use relations, directed "
            "affordances, and boundary-relative approach. Later probe images "
            "resolve only their named neutral observation goals. A decoded "
            "side is a hypothesis, not a conclusion. If a necessary side, "
            "relation, or boundary-relative approach remains unobservable, "
            "request that exact observation; otherwise apply the rubric."
        ),
        "group_local_review": (
            "Inspect the supplied group's usable sides, approach regions, and "
            "within-group direct-use relations. Use the global image only as "
            "architecture and zone context. Do not make an incomplete "
            "cross-group claim or treat a surface hypothesis as established "
            "fact; request a missing necessary observation."
        ),
    },
    "semantic_placement_consistency": {
        "global_discovery": (
            "Inspect scene zones, architecture, circulation, and contextual "
            "anchors. A placement-discovery subject is the only potential "
            "defect owner; its context IDs only frame the observation. Leave "
            "fine local support, height, and adjacency to local review."
        ),
        "group_local_review": (
            "Inspect each routed subject's support-surface meaning, height, "
            "zone, and immediate adjacency. Context IDs are evidence context, "
            "not defect owners. Do not convert orientation or operability into "
            "placement defects."
        ),
    },
}
