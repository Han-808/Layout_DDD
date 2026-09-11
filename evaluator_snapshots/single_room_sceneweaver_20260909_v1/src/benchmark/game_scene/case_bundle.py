"""Author a trusted case bundle for one exported game level.

The canonical L0-L4 workflow needs three things from a game case that the
evaluation profile cannot supply on its own, and getting any of them wrong fails
in a way that is hard to read back from the report:

* ``asset_policy`` -- without it every L3 metric is conservatively ``pending``,
  so Scene Quality never scores and the benchmark score comes back ``null``.
  That looks like a broken judge rather than a missing declaration.
* ``scene_request.room`` -- the official path checks the submitted scene against
  the room the case declares. A game level's room is its own play volume, so the
  room has to be read off the exported bounds instead of assumed.
* an empty frozen ``specification_contract`` -- a game level carries no
  scene-level prompt and the track uses no benchmark asset database, so there
  are no claims. Stating that explicitly lets L2 resolve to ``no_claims``
  instead of staying unresolved.

This module writes all three from the exported scene so a game case cannot be
assembled with one of them silently missing.
"""

from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any

from benchmark.evaluator.profile import resolve_evaluation_profile
from benchmark.utils.io import write_json


CASE_BUNDLE_VERSION = "benchmark_case_bundle_v1"
SPECIFICATION_CONTRACT_VERSION = "specification_contract_v1"

# A browser game authors its own geometry and materials outright and draws on no
# benchmark asset database, so the generator owns every asset role. This is what
# makes L3 style_consistency 'relevant' rather than 'pending'.
GAME_ASSET_POLICY: dict[str, str] = {
    "mode": "generated_or_open_assets",
    "identity_owner": "generator",
    "category_selection_owner": "generator",
    "scale_owner": "generator",
    "appearance_owner": "generator",
    "arrangement_owner": "generator",
}

# Games have no scene-level prompt to score against, so the prompt granularity
# is metadata only. Coarse-grained is the honest label for "nothing specified".
GAME_PROMPT_GRANULARITY = "coarse_grained"


class GameCaseBundleError(ValueError):
    """Raised when a game case bundle cannot be authored from the inputs."""


def game_scene_request(
    scene: dict[str, Any],
    *,
    instruction: str,
) -> dict[str, Any]:
    """Build a scene request whose room is the exported level's own volume."""

    request_id = scene.get("request_id")
    if not isinstance(request_id, str) or not request_id.strip():
        raise GameCaseBundleError("exported scene is missing a request_id")
    boundary = scene.get("boundary")
    if not isinstance(boundary, list) or len(boundary) < 3:
        raise GameCaseBundleError(
            "exported scene is missing a boundary; the case room cannot be derived"
        )
    height = scene.get("scene_height")
    if not isinstance(height, (int, float)) or isinstance(height, bool) or height <= 0.0:
        raise GameCaseBundleError(
            "exported scene is missing a positive scene_height; the case room "
            "cannot be derived"
        )
    return {
        "request_id": request_id,
        "instruction": instruction,
        "scene_type": scene.get("scene_type") or "game_level",
        "structure": False,
        "prompt_granularity": GAME_PROMPT_GRANULARITY,
        "room": {
            "boundary": deepcopy(boundary),
            "height": float(height),
            "unit": "meter",
        },
    }


def empty_specification_contract(request_id: str) -> dict[str, Any]:
    """A frozen contract that states there are no claims, rather than omitting them."""

    return {
        "contract_version": SPECIFICATION_CONTRACT_VERSION,
        "source": "trusted_case_bundle",
        "frozen": True,
        "request_id": request_id,
        "claims": {"oor": [], "oar": [], "functional_semantic_fidelity": []},
    }


def build_game_case_bundle(
    scene: dict[str, Any],
    *,
    out_dir: str | Path,
    case_id: str,
    evaluation_profile: dict[str, Any],
    instruction: str = "Evaluate the exported game level.",
    visual_style_spec: dict[str, Any] | None = None,
    camera_evidence: dict[str, Any] | None = None,
) -> Path:
    """Write a complete trusted case bundle for one exported game scene.

    ``evaluation_profile`` must already be canonical and fully resolved; the
    checked-in ``configs/evaluation/metric_profile_game_canonical_v1.yaml`` is
    the intended input. Returns the bundle root.
    """

    resolved_profile = resolve_evaluation_profile(deepcopy(evaluation_profile))
    if resolved_profile != evaluation_profile:
        raise GameCaseBundleError(
            "evaluation_profile must be fully resolved before it enters a trusted "
            "bundle; implicit code defaults are not allowed"
        )

    request = game_scene_request(scene, instruction=instruction)
    root = Path(out_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "scene_request": request,
        "specification_contract": empty_specification_contract(request["request_id"]),
        "asset_policy": deepcopy(GAME_ASSET_POLICY),
        "evaluation_profile": resolved_profile,
    }
    if visual_style_spec is not None:
        artifacts["visual_style_spec"] = deepcopy(visual_style_spec)

    records: dict[str, dict[str, str]] = {}
    for name, payload in artifacts.items():
        path = write_json(root / f"{name}.json", payload)
        records[name] = {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    manifest = {
        "bundle_version": CASE_BUNDLE_VERSION,
        "case_id": case_id,
        "task": {"evaluator_output_type": "o1_object_state"},
        "artifacts": records,
        "evaluation": {
            "workflow": "canonical_l0_l4",
            "p0b_official_mode": True,
            "camera_evidence": deepcopy(camera_evidence)
            if camera_evidence is not None
            else {
                # A browser capture has no .blend scene behind it, so the local
                # camera modes Blender provides are unavailable here.
                "mode": None,
                "metric_modes": {},
                "max_views": 2,
                "max_steps": 0,
                "collision_overlay": False,
                "collision_contour": False,
            },
        },
    }
    write_json(root / "case_bundle.json", manifest)
    return root
