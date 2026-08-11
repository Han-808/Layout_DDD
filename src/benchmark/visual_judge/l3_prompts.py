"""Versioned prompts for the L3 scene-quality Judge.

The definitions live in one dependency-light module so the evaluator request
builder and the OpenAI-compatible adapter cannot drift into different metric
semantics.
"""

from __future__ import annotations


L3_METRIC_PROMPT_VERSION = "l3_burden_categories_v25"

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
    "An adjacency or distance relation belongs to functional consistency only "
    "when an ordinary action such as reaching, viewing, sitting, operating, "
    "opening, or cooperative use depends on it. An adjacency that only "
    "describes where an otherwise usable object contextually belongs is owned "
    "by semantic placement.",
    "Do not duplicate one observation across functional consistency and "
    "semantic placement. Both metrics may flag the same object only when two "
    "independent failures remain: the semantic-location anomaly remains after "
    "granting ordinary orientation, access, and clearance, and the usability "
    "failure remains after granting an ordinary semantic location.",
    "For functional consistency, an enabled authoritative logical room "
    "boundary constrains access and operating space even when wall meshes are "
    "hidden. Boundary measurements and usable-side hypotheses are evidence, "
    "not universal validity thresholds.",
    "For functional consistency, architecture orientation and clearance are "
    "independent predicates. Architecture orientation asks only whether a "
    "directed usable side points toward plausible accessible interior space; "
    "clearance asks whether the required approach, opening, or operating "
    "region has enough free space. A wall or room boundary may cause a "
    "clearance failure, but architecture clearance is not a separate defect. "
    "Resolve both typed checks when both are listed, while avoiding duplicate "
    "defects for one physical failure.",
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
    "none is evidence that a defect exists. Every accepted required functional "
    "check must be explicitly resolved by its owning Judge phase before any "
    "final metric verdict. A supported observation remains auditable but cannot "
    "mask unresolved required coverage.",
    "Exclude a rendering or materialization artifact only when supplied "
    "structured provenance explicitly identifies it as benchmark-owned "
    "degradation. Do not infer that exemption from appearance alone.",
    "Each defect record represents one underlying scoring event. Use "
    "attribution_mode=unary for one defective object, responsible_endpoint "
    "when one relation endpoint owns the failure, and minimum_repair_set when "
    "multiple objects form the smallest repair set. Independent failures must "
    "be separate defect records; do not duplicate one full relation penalty "
    "across every endpoint.",
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
        "scene quality. For each invalid defect, classify category as oversized, "
        "undersized, or relative_scale_mismatch and severity as noticeable or "
        "gross. Request visual evidence only when object identity or "
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
        "scene quality. For each invalid defect, classify category as "
        "style_outlier or style_cluster_conflict and severity as noticeable "
        "or gross."
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
        "support. Classify each invalid defect as out_of_context_object or "
        "incompatible_object_set. Object-pairing invalidity has no mild tier."
    ),
    "functional_consistency": (
        "Judge only whether ordinary real-world use is feasible in the current "
        "authored arrangement. Apply the supplied required functional checks; "
        "they are questions to resolve, not defect priors. A cross-object "
        "functional check is atomic: directional_correspondence tests only "
        "whether functional sides or facing directions are compatible for "
        "direct ordinary joint use; relative_use_geometry tests only whether "
        "relative position, distance, reach, coordinated operation, or an "
        "operational connection permits that joint use. Static support/contact "
        "without an action-dependent joint use belongs to L1 physical support "
        "or semantic-placement support_and_height. Do not infer a relation check from broad "
        "cooperation or semantic association. An object's own approach, "
        "opening, or operating free space belongs only to its object-level "
        "clearance check, even when another object is the visible blocker. "
        "When an object "
        "has a directed usable side, resolve architecture_orientation using "
        "the angled global context and the side-conditioned local view: test "
        "only whether that side points toward plausible accessible interior "
        "space, without imposing a free-space threshold. Separately, when "
        "ordinary use requires approach, opening, or operating room, resolve "
        "clearance by testing the amount of free space in front of the usable "
        "side for directed objects or around a non-directed object. "
        "Establish usable sides from visible geometry and affordances, not from "
        "category, transform metadata, or the side facing the camera. Treat an "
        "enabled logical room boundary as an access constraint, not as a "
        "standalone defect threshold. Ordinary articulation intrinsic to use is "
        "allowed, but do not move, rotate, resize, remove, or replace authored "
        "objects to repair the layout. Return invalid only when a necessary "
        "condition for ordinary use is materially unavailable. Unconventional "
        "or suboptimal arrangements remain valid when ordinary use is readily "
        "feasible. Use adjacency or distance only when a concrete ordinary "
        "action depends on it; a merely conventional anchor relation belongs "
        "to semantic placement. Resolve every listed check explicitly; if a required "
        "observable fact is genuinely absent and another view can resolve it, "
        "request that observation rather than assuming normality. Attribute "
        "defects only to objects whose own use fails. Do not judge category "
        "membership, semantic location alone, style, scale, or L1 geometry. "
        "Classify each invalid defect as directed_surface_unusable, "
        "functional_correspondence_failure, approach_clearance_failure, or "
        "group_function_failure. Use impaired when core use remains possible "
        "but materially degraded; use blocked only when core use requires "
        "rearranging an authored scene object."
    ),
    "semantic_placement_consistency": (
        "Judge only semantic location, assuming that the object identity "
        "belongs in the scene and could perform its intended function when "
        "suitably placed. Evaluate the current support surface, placement "
        "height, scene zone, adjacency to context-setting objects, and "
        "immediate local context as the object's current location. "
        "Apply the relocation-only test: hold "
        "identity, scale, style, intended function, and ordinary usable "
        "orientation fixed, then ask whether moving the same object to an "
        "ordinary location in the same scene would remove the anomaly. Return "
        "invalid only when relocation alone would resolve the issue. Classify "
        "the defect as semantic_surface_mismatch, zone_placement_mismatch, "
        "or local_arrangement_mismatch. These correspond only to the existing "
        "support-and-height, scene-zone, and contextual-anchor checks. Use "
        "atypical when the placement is "
        "clearly unconventional but retains a plausible interpretation; use "
        "implausible when no ordinary contextual interpretation remains and "
        "relocation would restore one. Do not use either level for aesthetic "
        "preference, a merely less optimal location, or unexplained oddness. "
        "Every placement defect must contain severity equal to exactly "
        "atypical or implausible. Contextual "
        "adjacency belongs here only when it describes where an otherwise "
        "usable object belongs; adjacency needed for reaching, viewing, "
        "sitting, operating, opening, or cooperative use belongs to functional "
        "consistency as action-required adjacency. Orientation, facing "
        "direction, access, opening "
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
            "Inspect only overall scene-level functional organization, such "
            "as scene-wide circulation or a holistic usability failure that "
            "is not one discovered object relation or one group's local "
            "arrangement. Do not decide discovered cross-group "
            "correspondences, usable frontage, local clearance, or "
            "within-group correspondence in this phase."
        ),
        "cross_group_relation_review": (
            "Inspect exactly the supplied cross-group target set and resolve "
            "every supplied atomic relation check separately using its named "
            "predicate, evidence, and observation goal. Shared images do not "
            "merge directional_correspondence and relative_use_geometry into "
            "one conclusion. "
            "Do not judge unrelated scene, architecture, frontage, clearance, "
            "or group-local issues. Any defect must stay within the supplied "
            "relation evidence scope, while target_ids must be a non-empty "
            "subset containing only objects whose own functional state fails."
        ),
        "group_local_review": (
            "Resolve every supplied group-owned functional check exactly once: "
            "architecture orientation, clearance, local affordance "
            "confirmation, and atomic within-group direct-use relations. "
            "Begin with the supplied angled global anchor, group-local view, "
            "and any reused group-owned probe views. Do not require a "
            "dedicated pre-acquired image merely because an in-group relation "
            "was discovered. If any required check cannot be resolved from "
            "the current packet, return insufficient evidence and request "
            "only the missing observations for that check's target objects. "
            "Any final verdict requires every required check to resolve. If a "
            "supported invalid row coexists with an unresolved row, return "
            "insufficient/ambiguous with no final defects and request only the "
            "missing observations; repeat the invalid row and its defect once "
            "coverage completes. "
            "The terminal budget policy will request inference under budget "
            "when acquisition is exhausted. "
            "Resolve directional_correspondence and relative_use_geometry as "
            "separate result rows even when they share evidence. Use the "
            "single angled global image as architecture and zone context and "
            "reuse the "
            "check-bound side-conditioned local image for each directed "
            "object. Treat architecture_orientation as a direction predicate "
            "and clearance as a free-space predicate; do not merge their "
            "result rows or duplicate one physical failure. Do not make an incomplete "
            "cross-group claim because the dedicated relation phase owns it. "
            "Do not treat a surface hypothesis as established fact; request "
            "a missing necessary observation."
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
            "not defect owners. Classify every accepted defect as exactly "
            "atypical or implausible. Do "
            "not convert orientation or operability, action-required "
            "adjacency, access, or clearance into placement defects. Any final "
            "verdict requires every routed check to resolve. If an invalid row "
            "coexists with an unresolved row, keep acquisition open with an "
            "insufficient/ambiguous response and no final defects."
        ),
    },
}
