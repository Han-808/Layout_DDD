"""Versioned prompts for the L3 scene-quality Judge.

The definitions live in one dependency-light module so the evaluator request
builder and the OpenAI-compatible adapter cannot drift into different metric
semantics.
"""

from __future__ import annotations


L3_METRIC_PROMPT_VERSION = "l3_metric_boundaries_v1"

L3_METRIC_BOUNDARY_RULES = (
    "Evidence insufficiency and semantic uncertainty are different. Return "
    "insufficient evidence only when another visual observation could "
    "materially resolve the decision. If the evidence is adequate but no "
    "clear significant in-scope defect is established, return valid.",
    "Object pairing concerns object identity and semantic role independently "
    "of precise transform. If an object would remain inappropriate after "
    "relocation to a reasonable position in the same scene, the defect belongs "
    "to object_pairing_consistency.",
    "Semantic placement assumes that the object identity belongs in the scene. "
    "It concerns whether the current support surface, height, scene zone, or "
    "local location is contextually appropriate. An unusual location that does "
    "not materially prevent use is not by itself a functional defect.",
    "Functional consistency concerns actual usability. It is invalid only when "
    "the current orientation, access, clearance, or ensemble arrangement makes "
    "ordinary operation impossible or materially unusable. An unusual but "
    "usable location is not a functional defect.",
    "Scale consistency concerns physical dimensions. An apparent size "
    "difference that disappears after accounting for perspective, camera "
    "distance, or legitimate size variation is not a scale defect. Size alone "
    "is not a style defect.",
    "Style consistency concerns visible design language. A difference caused "
    "only by category, physical size, position, orientation, or function is not "
    "a style defect.",
    "Collision, penetration, support, floating, and out-of-bounds conditions "
    "are owned by L1. An L1 defect alone must not produce an L3 defect. The same "
    "arrangement may also fail an L3 metric only when that L3 criterion "
    "independently fails.",
    "The metrics are not required to be mutually exclusive. A scene may fail "
    "multiple metrics when each metric's independent criterion fails. Do not "
    "duplicate one underlying observation across metrics unless each requested "
    "criterion independently fails.",
)

L3_METRIC_RUBRICS = {
    "scale_consistency": (
        "Judge only whether each target object's physical scale is plausibly "
        "consistent with its object identity, nearby reference objects, and "
        "the scene or group context. Return invalid only when an object is "
        "clearly and materially too large or too small for a plausible "
        "real-world instance of its category. Account for perspective, camera "
        "distance, legitimate product or subtype variation, and explicitly "
        "authorized nonstandard scale. Apparent image size alone is not "
        "physical-scale evidence. Do not judge object membership, style, "
        "position, orientation, functionality, collision, support, or general "
        "scene quality. If reliable relative-scale context is not visible, "
        "request additional evidence."
    ),
    "style_consistency": (
        "Judge only whether the visible design language of the target objects "
        "is coherent with the scene or local ensemble. Relevant properties may "
        "include form language, design era, materials, surface treatment, "
        "ornamentation, visual complexity, and representation style. Return "
        "invalid only when an object is a clear and material visual-style "
        "outlier. Allow normal within-style variation, coherent mixed styles, "
        "and differences that amount only to subjective preference. Evaluate "
        "style independently of object category, scale, position, orientation, "
        "functionality, collision, support, and prompt completeness."
    ),
    "object_pairing_consistency": (
        "Judge only whether each target object's category and semantic role "
        "plausibly belong in the supplied scene and local group context. The "
        "supplied group is a visual-evidence scope, not compatibility ground "
        "truth. Evaluate each supplied target against both the broader scene "
        "type and the local ensemble. Apply the relocation counterfactual: if "
        "the object were moved to a reasonable location within the same scene "
        "without changing its identity, would its category or role still be "
        "clearly inappropriate? Return invalid only when the incompatibility "
        "remains. Do not judge the current support surface, zone, height, "
        "distance, orientation, access, clearance, scale, style, collision, or "
        "physical support."
    ),
    "functional_consistency": (
        "Judge only whether the target object or local ensemble can perform its "
        "ordinary real-world function in its current arrangement. Consider "
        "whether the required interaction side is accessible, the functional "
        "orientation is usable, necessary opening or operating clearance "
        "exists, and participating objects can operate together as an "
        "ensemble. Return invalid only when the current arrangement makes "
        "ordinary use impossible or materially unusable. A merely "
        "unconventional or suboptimal arrangement is valid when ordinary use "
        "remains feasible. Evaluate functionality independently of category "
        "membership, semantic location alone, style, scale, prompt fidelity, "
        "and unrelated exact spatial relations. An L1 condition may support a "
        "functional defect only when ordinary usability independently fails."
    ),
    "semantic_placement_consistency": (
        "Judge only whether the current location of each target object is "
        "semantically plausible, assuming that the object category belongs in "
        "the scene. Relevant properties include the semantic suitability of "
        "its support surface, placement height, scene zone, and immediate local "
        "context. Apply the relocation counterfactual: if the same object were "
        "moved to an ordinary nearby location without changing its identity, "
        "scale, style, or intended function, would the anomaly disappear? "
        "Return invalid only when it would and the current location "
        "independently fails semantic appropriateness. Do not judge collision, "
        "penetration, out-of-bounds geometry, floating, physical support, "
        "category "
        "compatibility, scale, style, functional orientation, access, "
        "clearance, operability, or prompt fidelity. An L1 condition alone is "
        "not a semantic-placement defect."
    ),
}
