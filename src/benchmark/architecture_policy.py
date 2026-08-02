from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Iterable, Mapping


ARCHITECTURE_POLICY_VERSION = "explicit_physical_walls_v1"
ARCHITECTURE_CONTRACT_ID = "bounded_room_explicit_walls_v1"
DEFAULT_PHYSICAL_WALL_POLICY = "explicit_only"
PHYSICAL_WALL_POLICIES = ("explicit_only", "always_enclosed")
CANONICAL_WALL_IDS = (
    "north_wall",
    "south_wall",
    "east_wall",
    "west_wall",
)
CANONICAL_CORNER_IDS = (
    "northeast_corner",
    "northwest_corner",
    "southeast_corner",
    "southwest_corner",
)
CORNER_WALL_IDS = {
    "northeast_corner": ("north_wall", "east_wall"),
    "northwest_corner": ("north_wall", "west_wall"),
    "southeast_corner": ("south_wall", "east_wall"),
    "southwest_corner": ("south_wall", "west_wall"),
}
WALL_RELATION_TYPES = {
    "against_wall",
    "near_wall",
    "along_wall",
    "mounted_on_wall",
    "attached_to_wall",
    "on_wall",
}
CORNER_RELATION_TYPES = {
    "at_corner",
    "near_corner",
    "in_corner",
}

_WALL_ALIASES = {
    "north": "north_wall",
    "north_wall": "north_wall",
    "back": "north_wall",
    "back_wall": "north_wall",
    "rear": "north_wall",
    "rear_wall": "north_wall",
    "south": "south_wall",
    "south_wall": "south_wall",
    "front": "south_wall",
    "front_wall": "south_wall",
    "east": "east_wall",
    "east_wall": "east_wall",
    "right": "east_wall",
    "right_wall": "east_wall",
    "west": "west_wall",
    "west_wall": "west_wall",
    "left": "west_wall",
    "left_wall": "west_wall",
}
_CORNER_ALIASES = {
    "northeast": "northeast_corner",
    "northeast_corner": "northeast_corner",
    "north_east": "northeast_corner",
    "north_east_corner": "northeast_corner",
    "back_right": "northeast_corner",
    "back_right_corner": "northeast_corner",
    "northwest": "northwest_corner",
    "northwest_corner": "northwest_corner",
    "north_west": "northwest_corner",
    "north_west_corner": "northwest_corner",
    "back_left": "northwest_corner",
    "back_left_corner": "northwest_corner",
    "southeast": "southeast_corner",
    "southeast_corner": "southeast_corner",
    "south_east": "southeast_corner",
    "south_east_corner": "southeast_corner",
    "front_right": "southeast_corner",
    "front_right_corner": "southeast_corner",
    "southwest": "southwest_corner",
    "southwest_corner": "southwest_corner",
    "south_west": "southwest_corner",
    "south_west_corner": "southwest_corner",
    "front_left": "southwest_corner",
    "front_left_corner": "southwest_corner",
}
_NEGATED_WALL_PATTERNS = (
    r"\bno\s+(?:physical\s+)?walls?\b",
    r"\bwithout\s+(?:any\s+)?(?:physical\s+)?walls?\b",
    r"\bwall[- ]free\b",
    r"\bopen[- ]sided\b",
)
_GENERIC_WALL_PATTERNS = (
    r"\bagainst\s+(?:a|any|the)\s+wall\b",
    r"\b(?:near|along|on|onto)\s+(?:a|any|the)\s+wall\b",
    r"\b(?:mounted|attached|hung)\s+(?:on|to)\s+(?:a|any|the)\s+wall\b",
    r"\bwalls?\b",
)


def resolve_architecture_activation(
    room: dict,
    *,
    instruction: str = "",
    specification_contract: dict | None = None,
    reference_annotation: dict | None = None,
    object_plan: dict | None = None,
    visual_style_spec: dict | None = None,
    physical_wall_policy: str = DEFAULT_PHYSICAL_WALL_POLICY,
    policy_source: str = "canonical_default",
) -> dict[str, Any]:
    """Resolve physical architecture once from benchmark-owned inputs.

    The resolver intentionally never accepts generated scene output or scene
    type. Its result is the authority consumed by generation, rendering and
    evaluation.
    """

    policy = validate_physical_wall_policy(physical_wall_policy)
    activation: dict[str, list[dict[str, Any]]] = {
        wall_id: [] for wall_id in CANONICAL_WALL_IDS
    }
    claims: list[dict[str, Any]] = []

    if policy == "always_enclosed":
        for wall_id in CANONICAL_WALL_IDS:
            activation[wall_id].append(
                {
                    "source": "compatibility_policy",
                    "wall_id": wall_id,
                    "claim": "always_enclosed",
                }
            )
        claims.append(
            {
                "source": "compatibility_policy",
                "claim": "always_enclosed",
                "active_wall_ids": list(CANONICAL_WALL_IDS),
            }
        )
    else:
        source_payloads = (
            ("specification_contract", _specification_relations(specification_contract)),
            ("reference_annotation", _reference_relations(reference_annotation)),
            ("public_object_plan", _object_plan_relations(object_plan)),
        )
        for source_name, relations in source_payloads:
            for relation in relations:
                walls = _walls_for_relation(relation)
                if not walls:
                    continue
                claim = _activation_claim(source_name, relation, walls)
                claims.append(claim)
                for wall_id in walls:
                    activation[wall_id].append(deepcopy(claim))

        prompt_claims = _walls_from_text(instruction)
        for item in prompt_claims:
            claim = {
                "source": "natural_language_prompt",
                "claim": item["claim"],
                "active_wall_ids": list(item["active_wall_ids"]),
            }
            claims.append(claim)
            for wall_id in item["active_wall_ids"]:
                activation[wall_id].append(deepcopy(claim))

        if visual_style_spec is not None:
            style_text = " ".join(_iter_string_values(visual_style_spec))
            for item in _walls_from_text(style_text, wall_appearance_only=True):
                claim = {
                    "source": "visual_style_spec",
                    "claim": item["claim"],
                    "active_wall_ids": list(item["active_wall_ids"]),
                }
                claims.append(claim)
                for wall_id in item["active_wall_ids"]:
                    activation[wall_id].append(deepcopy(claim))

    active_wall_ids = tuple(
        wall_id for wall_id in CANONICAL_WALL_IDS if activation[wall_id]
    )
    activation_sources = tuple(
        dict.fromkeys(
            str(item["source"])
            for item in claims
            if str(item.get("source") or "")
        )
    )
    return build_architecture_contract(
        room,
        physical_wall_policy=policy,
        requested_policy=physical_wall_policy,
        policy_source=policy_source,
        active_wall_ids=active_wall_ids,
        activation_sources=activation_sources,
        activation_claims=claims,
    )


def build_architecture_contract(
    room: dict,
    *,
    physical_wall_policy: str = DEFAULT_PHYSICAL_WALL_POLICY,
    requested_policy: str | None = None,
    policy_source: str = "canonical_default",
    active_wall_ids: Iterable[str] = (),
    activation_sources: Iterable[str] = (),
    activation_claims: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    policy = validate_physical_wall_policy(physical_wall_policy)
    walls = validate_active_wall_ids(active_wall_ids)
    if policy == "always_enclosed" and tuple(walls) != CANONICAL_WALL_IDS:
        raise ValueError("always_enclosed requires all four canonical wall IDs")
    allowed_corners = [
        corner_id
        for corner_id, required_walls in CORNER_WALL_IDS.items()
        if all(wall_id in walls for wall_id in required_walls)
    ]
    allowed_tokens = ["floor", "ceiling", *walls, *allowed_corners]
    contract = {
        "id": ARCHITECTURE_CONTRACT_ID,
        "source": "benchmark_task_contract",
        "architecture_policy_version": ARCHITECTURE_POLICY_VERSION,
        "room": deepcopy(room),
        "logical_boundary": {
            "enabled": True,
            "boundary": deepcopy(room.get("boundary")),
        },
        "floor": {
            "enabled": True,
            "policy": "always",
            "z": float(room.get("floor_z", 0.0)),
        },
        "ceiling": {
            "enabled": True,
            "policy": "preserve_existing_behavior",
            "z": (
                float(room.get("height"))
                if room.get("height") is not None
                else None
            ),
        },
        "physical_walls": {
            "requested_policy": str(
                requested_policy
                if requested_policy is not None
                else physical_wall_policy
            ),
            "policy": policy,
            "effective_policy": policy,
            "policy_source": str(policy_source),
            "active_wall_ids": list(walls),
            "activation_sources": list(
                dict.fromkeys(str(value) for value in activation_sources if str(value))
            ),
            "activation_claims": [deepcopy(dict(item)) for item in activation_claims],
            "compatibility_mode": policy == "always_enclosed",
        },
        "allowed_architecture_tokens": allowed_tokens,
        "elements": ["floor", "ceiling", *walls],
        "wall_count": len(walls),
        "floor_z": float(room.get("floor_z", 0.0)),
    }
    return validate_architecture_contract(contract)


def validate_architecture_contract(contract: Any) -> dict[str, Any]:
    if not isinstance(contract, dict):
        raise ValueError("architecture contract must be a JSON object")
    if contract.get("id") != ARCHITECTURE_CONTRACT_ID:
        raise ValueError(
            f"architecture contract id must be {ARCHITECTURE_CONTRACT_ID!r}"
        )
    if contract.get("architecture_policy_version") != ARCHITECTURE_POLICY_VERSION:
        raise ValueError(
            "architecture contract version must be "
            f"{ARCHITECTURE_POLICY_VERSION!r}"
        )
    logical = contract.get("logical_boundary")
    if not isinstance(logical, dict) or logical.get("enabled") is not True:
        raise ValueError("architecture logical_boundary.enabled must be true")
    if not isinstance(logical.get("boundary"), list):
        raise ValueError("architecture logical_boundary.boundary must be a list")
    room = contract.get("room")
    if not isinstance(room, dict) or logical.get("boundary") != room.get(
        "boundary"
    ):
        raise ValueError(
            "architecture logical boundary must match architecture room boundary"
        )
    floor = contract.get("floor")
    if not isinstance(floor, dict) or floor.get("enabled") is not True:
        raise ValueError("architecture floor.enabled must be true")
    physical = contract.get("physical_walls")
    if not isinstance(physical, dict):
        raise ValueError("architecture physical_walls must be a JSON object")
    policy = validate_physical_wall_policy(physical.get("policy"))
    validate_physical_wall_policy(physical.get("requested_policy"))
    if physical.get("effective_policy") != policy:
        raise ValueError(
            "architecture physical_walls.effective_policy must match policy"
        )
    if not isinstance(physical.get("policy_source"), str) or not str(
        physical.get("policy_source")
    ).strip():
        raise ValueError(
            "architecture physical_walls.policy_source must be non-empty"
        )
    if not isinstance(physical.get("activation_sources"), list) or any(
        not isinstance(value, str) or not value
        for value in physical.get("activation_sources")
    ):
        raise ValueError(
            "architecture physical_walls.activation_sources must contain strings"
        )
    if not isinstance(physical.get("activation_claims"), list) or any(
        not isinstance(value, dict)
        for value in physical.get("activation_claims")
    ):
        raise ValueError(
            "architecture physical_walls.activation_claims must contain objects"
        )
    if physical.get("compatibility_mode") is not (
        policy == "always_enclosed"
    ):
        raise ValueError(
            "architecture physical_walls.compatibility_mode conflicts with policy"
        )
    active = validate_active_wall_ids(physical.get("active_wall_ids") or ())
    if policy == "always_enclosed" and active != CANONICAL_WALL_IDS:
        raise ValueError("always_enclosed architecture must activate all four walls")
    tokens = contract.get("allowed_architecture_tokens")
    if not isinstance(tokens, list) or any(
        not isinstance(value, str) or not value for value in tokens
    ):
        raise ValueError(
            "architecture allowed_architecture_tokens must be non-empty strings"
        )
    allowed_corners = [
        corner_id
        for corner_id, walls in CORNER_WALL_IDS.items()
        if all(wall in active for wall in walls)
    ]
    expected_tokens = ["floor", "ceiling", *active, *allowed_corners]
    if tokens != expected_tokens:
        raise ValueError(
            "architecture allowed_architecture_tokens must exactly match active "
            f"architecture: expected {expected_tokens}, got {tokens}"
        )
    expected_elements = ["floor", "ceiling", *active]
    if contract.get("elements") != expected_elements:
        raise ValueError(
            "architecture elements must exactly match active physical "
            f"architecture: expected {expected_elements}"
        )
    for index, claim in enumerate(physical["activation_claims"]):
        claim_walls = claim.get("active_wall_ids")
        if claim_walls is None:
            continue
        try:
            normalized_claim_walls = validate_active_wall_ids(claim_walls)
        except ValueError as exc:
            raise ValueError(
                f"architecture activation_claims[{index}] is invalid: {exc}"
            ) from exc
        if any(wall_id not in active for wall_id in normalized_claim_walls):
            raise ValueError(
                f"architecture activation_claims[{index}] references inactive walls"
            )
    if int(contract.get("wall_count", -1)) != len(active):
        raise ValueError("architecture wall_count conflicts with active_wall_ids")
    return contract


def validate_physical_wall_policy(value: Any) -> str:
    policy = str(value or "").strip().lower()
    if policy not in PHYSICAL_WALL_POLICIES:
        raise ValueError(
            "physical wall policy must be one of "
            f"{list(PHYSICAL_WALL_POLICIES)}, got {value!r}"
        )
    return policy


def validate_active_wall_ids(values: Iterable[Any]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise ValueError("active_wall_ids must be a list of canonical wall IDs")
    normalized: list[str] = []
    for value in values:
        raw = str(value).strip().lower()
        wall_id = canonical_wall_id(raw)
        if wall_id is None or raw != wall_id:
            raise ValueError(f"unknown physical wall ID {value!r}")
        if wall_id not in normalized:
            normalized.append(wall_id)
    return tuple(
        wall_id for wall_id in CANONICAL_WALL_IDS if wall_id in normalized
    )


def active_wall_ids_from_contract(contract: Mapping[str, Any] | None) -> tuple[str, ...]:
    if not isinstance(contract, Mapping):
        return ()
    physical = contract.get("physical_walls")
    if not isinstance(physical, Mapping):
        return ()
    return validate_active_wall_ids(physical.get("active_wall_ids") or ())


def architecture_contract_from_scene(
    scene: Mapping[str, Any],
    *,
    room: dict | None = None,
) -> dict[str, Any]:
    metadata = scene.get("metadata") if isinstance(scene, Mapping) else None
    existing = (
        metadata.get("architecture_contract")
        if isinstance(metadata, Mapping)
        else None
    )
    if isinstance(existing, dict):
        return validate_architecture_contract(existing)
    resolved_room = room or {
        "boundary": deepcopy(
            scene.get("boundary")
            or [[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0]]
        ),
        "height": (
            scene.get("scene_height")
            if scene.get("scene_height") is not None
            else 2.8
        ),
        "floor_z": 0.0,
    }
    return build_architecture_contract(resolved_room)


def require_generated_architecture_targets_active(
    relations: Iterable[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> None:
    validated = validate_architecture_contract(dict(contract))
    allowed = set(validated["allowed_architecture_tokens"])
    for index, relation in enumerate(relations):
        if not isinstance(relation, Mapping):
            continue
        family = str(relation.get("family") or "oar").strip().lower()
        if family != "oar":
            continue
        relation_type = _compact(relation.get("type") or relation.get("predicate"))
        target = _relation_target(relation)
        if not target:
            if relation_type in WALL_RELATION_TYPES | CORNER_RELATION_TYPES:
                raise ValueError(
                    f"generated OAR relation {index} has no explicit architecture target"
                )
            continue
        canonical_target = canonical_architecture_target(target)
        if canonical_target is None or canonical_target not in allowed:
            raise ValueError(
                "generated OAR relation targets inactive architecture: "
                f"{target!r} is not in {sorted(allowed)}"
            )


def canonical_wall_id(value: Any) -> str | None:
    return _WALL_ALIASES.get(_compact(value))


def canonical_corner_id(value: Any) -> str | None:
    return _CORNER_ALIASES.get(_compact(value))


def canonical_architecture_target(value: Any) -> str | None:
    compact = _compact(value)
    if compact in {"floor", "ceiling"}:
        return compact
    return canonical_wall_id(compact) or canonical_corner_id(compact)


def walls_required_for_architecture_target(value: Any) -> tuple[str, ...]:
    wall_id = canonical_wall_id(value)
    if wall_id is not None:
        return (wall_id,)
    corner_id = canonical_corner_id(value)
    if corner_id is not None:
        return CORNER_WALL_IDS[corner_id]
    return ()


def _specification_relations(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        return []
    claims = value.get("claims")
    if not isinstance(claims, dict) or not isinstance(claims.get("oar"), list):
        return []
    return [
        deepcopy(item)
        for item in claims["oar"]
        if isinstance(item, dict) and item.get("required") is not False
    ]


def _reference_relations(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or not isinstance(value.get("oar_relations"), list):
        return []
    return [
        deepcopy(item)
        for item in value["oar_relations"]
        if isinstance(item, dict)
        and str(item.get("claim_state") or "confirmed").strip().lower()
        == "confirmed"
    ]


def _object_plan_relations(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        return []
    relations = value.get("relations")
    if not isinstance(relations, list):
        return []
    return [
        deepcopy(item)
        for item in relations
        if isinstance(item, dict)
        and str(item.get("family") or "").strip().lower() == "oar"
    ]


def _walls_for_relation(relation: Mapping[str, Any]) -> tuple[str, ...]:
    relation_type = _compact(
        relation.get("relation_type")
        or relation.get("type")
        or relation.get("predicate")
        or relation.get("relation")
    )
    target = _relation_target(relation)
    walls = walls_required_for_architecture_target(target)
    if walls:
        return walls
    if relation_type not in WALL_RELATION_TYPES | CORNER_RELATION_TYPES:
        return ()
    compact_target = _compact(target)
    if compact_target in {"wall", "any_wall", "walls", ""}:
        return CANONICAL_WALL_IDS
    if compact_target in {"corner", "any_corner", "corners"}:
        return CANONICAL_WALL_IDS
    raise ValueError(
        "explicit wall/corner claim cannot be mapped to canonical physical "
        f"architecture: type={relation_type!r}, target={target!r}"
    )


def _relation_target(relation: Mapping[str, Any]) -> Any:
    for key in (
        "architectural_element",
        "target",
        "object",
        "wall",
        "corner",
    ):
        if relation.get(key) is not None:
            return relation[key]
    return None


def _activation_claim(
    source_name: str,
    relation: Mapping[str, Any],
    walls: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "source": source_name,
        "claim": {
            key: deepcopy(relation[key])
            for key in (
                "claim_id",
                "relation_id",
                "relation_type",
                "type",
                "predicate",
                "architectural_element",
                "target",
                "wall",
                "corner",
            )
            if relation.get(key) is not None
        },
        "active_wall_ids": list(walls),
    }


def _walls_from_text(
    value: Any,
    *,
    wall_appearance_only: bool = False,
) -> list[dict[str, Any]]:
    text = str(value or "").strip().lower()
    if not text or any(re.search(pattern, text) for pattern in _NEGATED_WALL_PATTERNS):
        return []
    claims: list[dict[str, Any]] = []
    for alias, corner_id in _CORNER_ALIASES.items():
        phrase = alias.replace("_", r"[\s_-]*")
        if re.search(rf"\b{phrase}\b", text):
            claims.append(
                {
                    "claim": corner_id,
                    "active_wall_ids": CORNER_WALL_IDS[corner_id],
                }
            )
    for direction in ("north", "south", "east", "west"):
        if re.search(rf"\b{direction}[\s_-]+wall\b", text):
            wall_id = f"{direction}_wall"
            claims.append(
                {"claim": wall_id, "active_wall_ids": (wall_id,)}
            )
    appearance = bool(
        re.search(
            r"\b(?:white|black|gray|grey|brick|glass|stone|wood(?:en)?|"
            r"concrete|painted|exposed|textured|colored|colour(?:ed)?)\s+walls?\b",
            text,
        )
        or re.search(r"\bwall\s+(?:material|appearance|finish|color|colour|texture)\b", text)
    )
    if appearance:
        claims.append(
            {
                "claim": "explicit_wall_appearance",
                "active_wall_ids": CANONICAL_WALL_IDS,
            }
        )
    if not wall_appearance_only and not claims:
        if any(re.search(pattern, text) for pattern in _GENERIC_WALL_PATTERNS):
            claims.append(
                {
                    "claim": "generic_wall_requirement",
                    "active_wall_ids": CANONICAL_WALL_IDS,
                }
            )
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for claim in claims:
        walls = tuple(claim["active_wall_ids"])
        if walls in seen:
            continue
        seen.add(walls)
        unique.append(claim)
    return unique


def _iter_string_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in {"scene_type", "room_type"}:
                continue
            yield from _iter_string_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_string_values(item)


def _compact(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
