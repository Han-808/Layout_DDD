from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


# This is intentionally smaller and less heuristic than SceneCritic's alias
# registry.  Layout_DDD categories are generator-owned free strings, so only
# explicit, reviewable aliases are safe at the evaluator boundary.
DEFAULT_CATEGORY_ALIASES: dict[str, str] = {
    "arm_chair": "armchair",
    "bedside_table": "nightstand",
    "bookcase": "bookshelf",
    "carpet": "rug",
    "closet": "wardrobe",
    "couch": "sofa",
    "dining_chair": "chair",
    "end_table": "side_table",
    "fridge": "refrigerator",
    "houseplant": "plant",
    "love_seat": "sofa",
    "loveseat": "sofa",
    "media_console": "tv_stand",
    "night_stand": "nightstand",
    "office_chair": "office_chair",
    "potted_plant": "plant",
    "settee": "sofa",
    "side_table": "side_table",
    "television": "tv",
    "tv_console": "tv_stand",
}

_METADATA_KEYS = {
    "metadata",
    "_metadata",
    "schema_version",
    "ontology_version",
    "version",
    "provenance",
    "source",
    "categories",
}
_COOCCURRENCE_RESERVED_KEYS = {
    "global",
    "by_room",
    "per_room",
    "room_conditioned",
    "rooms",
    "metadata",
}


@dataclass(frozen=True)
class CategoryResolution:
    raw_category: str
    normalized_category: str
    ontology_category: str | None
    mapping_source: str

    @property
    def known(self) -> bool:
        return self.ontology_category is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw_category": self.raw_category,
            "normalized_category": self.normalized_category,
            "ontology_category": self.ontology_category,
            "mapping_source": self.mapping_source,
            "known": self.known,
        }


@dataclass(frozen=True)
class CooccurrenceRecord:
    probability: float
    pair_count: int | None
    anchor_observation_count: int | None
    npmi: float | None
    context: str
    probability_source: str
    raw: Any

    def as_dict(self) -> dict[str, Any]:
        return {
            "p_other_given_anchor": self.probability,
            "pair_count": self.pair_count,
            # SceneOnto's category ``count`` is an object-observation count,
            # not a distinct-scene denominator.  Keep that distinction
            # explicit even though it is still useful as a support proxy.
            "anchor_observation_count": self.anchor_observation_count,
            "npmi": self.npmi,
            "context": self.context,
            "probability_source": self.probability_source,
        }


class OntologyIndex:
    """Validated, normalization-aware view over a SceneOnto-style mapping."""

    def __init__(
        self,
        raw: Mapping[str, Any] | None,
        *,
        aliases: Mapping[str, str] | None = None,
        identity: Mapping[str, Any] | None = None,
        storage_semantics: str = "sparse_top_k",
    ) -> None:
        self.raw = dict(raw or {})
        self.categories = _category_mapping(self.raw)
        self._normalized_to_key = _normalized_category_index(self.categories)
        self.aliases = _validated_aliases(
            {**DEFAULT_CATEGORY_ALIASES, **dict(aliases or {})},
            normalized_to_key=self._normalized_to_key,
        )
        self.storage_semantics = str(storage_semantics or "sparse_top_k")
        self.identity = {
            **dict(identity or {}),
            "schema_version": _ontology_schema_version(self.raw),
            "storage_semantics": self.storage_semantics,
            "category_count": len(self.categories),
        }

    @property
    def available(self) -> bool:
        return bool(self.categories)

    def resolve(self, raw_category: object) -> CategoryResolution:
        raw_text = str(raw_category or "").strip()
        normalized = normalize_category_label(raw_text)
        if normalized in self._normalized_to_key:
            return CategoryResolution(
                raw_category=raw_text,
                normalized_category=normalized,
                ontology_category=self._normalized_to_key[normalized],
                mapping_source="ontology_exact_normalized",
            )
        alias_target = self.aliases.get(normalized)
        if alias_target is not None:
            ontology_key = self._normalized_to_key.get(normalize_category_label(alias_target))
            if ontology_key is not None:
                return CategoryResolution(
                    raw_category=raw_text,
                    normalized_category=normalized,
                    ontology_category=ontology_key,
                    mapping_source="explicit_alias",
                )
        return CategoryResolution(
            raw_category=raw_text,
            normalized_category=normalized,
            ontology_category=None,
            mapping_source="unmapped",
        )

    def entry(self, category: str) -> dict[str, Any] | None:
        value = self.categories.get(category)
        return dict(value) if isinstance(value, Mapping) else None

    def dimension_stats(self, category: str, axis: str) -> dict[str, float] | None:
        entry = self.entry(category)
        if entry is None:
            return None
        dimensions = entry.get("dimensions")
        if not isinstance(dimensions, Mapping):
            return None
        candidates = (
            dimensions.get(f"{axis}_m"),
            dimensions.get(axis),
        )
        stats = next((value for value in candidates if isinstance(value, Mapping)), None)
        if stats is None:
            return None
        p5 = _optional_finite(stats.get("p5"), f"{category}.dimensions.{axis}.p5")
        p95 = _optional_finite(stats.get("p95"), f"{category}.dimensions.{axis}.p95")
        if p5 is None or p95 is None:
            return None
        if p5 <= 0.0 or p95 <= 0.0 or p5 > p95:
            raise ValueError(
                f"ontology dimension interval for {category}.{axis} must satisfy 0 < p5 <= p95"
            )
        result = {"p5": p5, "p95": p95}
        median = _optional_finite(stats.get("median"), f"{category}.dimensions.{axis}.median")
        if median is not None:
            if median <= 0.0:
                raise ValueError(f"ontology dimension median for {category}.{axis} must be positive")
            result["median"] = median
        for key in ("n", "count", "n_samples"):
            if stats.get(key) is not None:
                result["n_samples"] = _required_count(
                    stats[key],
                    f"{category}.dimensions.{axis}.{key}",
                )
                break
        if "n_samples" not in result and dimensions.get(f"n_{axis}") is not None:
            result["n_samples"] = _required_count(
                dimensions[f"n_{axis}"],
                f"{category}.dimensions.n_{axis}",
            )
        return result

    def observation_count(self, category: str, *, room_type: str | None = None) -> int | None:
        entry = self.entry(category)
        if entry is None:
            return None
        if room_type:
            room_key = normalize_category_label(room_type)
            for key in ("room_associations", "room_counts", "rooms"):
                by_room = entry.get(key)
                if not isinstance(by_room, Mapping):
                    continue
                room_value = _mapping_value_by_normalized_key(by_room, room_key)
                count = _count_from_value(room_value)
                if count is not None:
                    return count
        for key in ("count", "total_scene_count", "scene_count", "n_scenes"):
            count = _optional_count(entry.get(key), f"{category}.{key}")
            if count is not None:
                return count
        return None

    def cooccurrence_record(
        self,
        anchor: str,
        other: str,
        *,
        room_type: str | None = None,
    ) -> CooccurrenceRecord | None:
        entry = self.entry(anchor)
        if entry is None:
            return None
        context, mapping = _cooccurrence_mapping(entry, room_type=room_type)
        if mapping is None:
            return None
        raw_record = _mapping_value_for_category(mapping, other, self)
        if raw_record is None:
            return None
        anchor_count = self.observation_count(
            anchor,
            room_type=room_type if context == "room_conditioned" else None,
        )
        return _parse_cooccurrence_record(
            raw_record,
            anchor_observation_count=anchor_count,
            path=f"{anchor}.cooccurrence.{other}",
            context=context,
        )


def load_ontology(
    value: Mapping[str, Any] | str | Path | None,
    *,
    aliases: Mapping[str, str] | None = None,
    storage_semantics: str = "sparse_top_k",
) -> OntologyIndex:
    if value is None:
        return OntologyIndex(
            {},
            aliases=aliases,
            identity={"source": None, "sha256": None, "available": False},
            storage_semantics=storage_semantics,
        )
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser().resolve()
        payload = path.read_bytes()
        try:
            raw = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"spatial ontology must be a UTF-8 JSON object: {path}") from exc
        if not isinstance(raw, Mapping):
            raise ValueError("spatial ontology root must be a JSON object")
        identity = {
            "source": path.as_posix(),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "available": True,
        }
        return OntologyIndex(
            raw,
            aliases=aliases,
            identity=identity,
            storage_semantics=storage_semantics,
        )
    if not isinstance(value, Mapping):
        raise TypeError("spatial ontology must be a mapping, path, or None")
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return OntologyIndex(
        value,
        aliases=aliases,
        identity={
            "source": "in_memory",
            "sha256": hashlib.sha256(canonical).hexdigest(),
            "available": True,
        },
        storage_semantics=storage_semantics,
    )


def normalize_category_label(value: object) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _category_mapping(raw: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    explicit = raw.get("categories")
    source = explicit if isinstance(explicit, Mapping) else raw
    result: dict[str, dict[str, Any]] = {}
    for key, value in source.items():
        if str(key) in _METADATA_KEYS or not isinstance(value, Mapping):
            continue
        if explicit is None and not _looks_like_category_entry(value):
            continue
        result[str(key)] = dict(value)
    return result


def _looks_like_category_entry(value: Mapping[str, Any]) -> bool:
    return bool(
        set(value)
        & {
            "count",
            "total_scene_count",
            "scene_count",
            "dimensions",
            "cooccurrence",
            "room_associations",
            "support_surfaces",
            "orientation_relationships",
        }
    )


def _normalized_category_index(categories: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_key in categories:
        normalized = normalize_category_label(raw_key)
        if not normalized:
            raise ValueError("ontology category names must contain letters or digits")
        previous = result.get(normalized)
        if previous is not None and previous != raw_key:
            raise ValueError(
                f"ontology categories {previous!r} and {raw_key!r} normalize to the same key"
            )
        result[normalized] = str(raw_key)
    return result


def _validated_aliases(
    aliases: Mapping[str, str],
    *,
    normalized_to_key: Mapping[str, str],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_alias, raw_target in aliases.items():
        alias = normalize_category_label(raw_alias)
        target = normalize_category_label(raw_target)
        if not alias or not target:
            raise ValueError("category aliases and targets must be non-empty")
        previous = result.get(alias)
        if previous is not None and previous != target:
            raise ValueError(f"category alias {raw_alias!r} has conflicting targets")
        # Keeping aliases whose target is not present is useful when the same
        # config is shared by ontology subsets. They remain unmapped coverage.
        result[alias] = normalized_to_key.get(target, target)
    return result


def _ontology_schema_version(raw: Mapping[str, Any]) -> str | None:
    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), Mapping) else {}
    for value in (
        raw.get("schema_version"),
        raw.get("ontology_version"),
        raw.get("version"),
        metadata.get("schema_version"),
        metadata.get("ontology_version"),
        metadata.get("version"),
    ):
        if value is not None and str(value).strip():
            return str(value)
    return None


def _cooccurrence_mapping(
    entry: Mapping[str, Any],
    *,
    room_type: str | None,
) -> tuple[str, Mapping[str, Any] | None]:
    cooccurrence = entry.get("cooccurrence")
    room_key = normalize_category_label(room_type) if room_type else ""
    if room_key:
        room_sources: list[Any] = []
        if isinstance(cooccurrence, Mapping):
            room_sources.extend(
                cooccurrence.get(key)
                for key in ("by_room", "per_room", "room_conditioned", "rooms")
            )
        room_sources.extend(
            entry.get(key)
            for key in ("cooccurrence_by_room", "room_cooccurrence", "room_conditioned_cooccurrence")
        )
        for source in room_sources:
            if not isinstance(source, Mapping):
                continue
            candidate = _mapping_value_by_normalized_key(source, room_key)
            if isinstance(candidate, Mapping):
                return "room_conditioned", candidate
    if isinstance(cooccurrence, Mapping):
        global_mapping = cooccurrence.get("global")
        if isinstance(global_mapping, Mapping):
            return "global", global_mapping
        direct = {
            str(key): value
            for key, value in cooccurrence.items()
            if str(key) not in _COOCCURRENCE_RESERVED_KEYS
        }
        if direct:
            return "global", direct
    global_fallback = entry.get("global_cooccurrence")
    return ("global", global_fallback) if isinstance(global_fallback, Mapping) else ("global", None)


def _mapping_value_for_category(
    mapping: Mapping[str, Any],
    category: str,
    ontology: OntologyIndex,
) -> Any:
    if category in mapping:
        return mapping[category]
    target = normalize_category_label(category)
    for raw_key, value in mapping.items():
        resolution = ontology.resolve(raw_key)
        resolved = normalize_category_label(resolution.ontology_category or raw_key)
        if resolved == target:
            return value
    return None


def _mapping_value_by_normalized_key(mapping: Mapping[str, Any], normalized_key: str) -> Any:
    for key, value in mapping.items():
        if normalize_category_label(key) == normalized_key:
            return value
    return None


def _parse_cooccurrence_record(
    raw: Any,
    *,
    anchor_observation_count: int | None,
    path: str,
    context: str,
) -> CooccurrenceRecord:
    pair_count: int | None = None
    npmi: float | None = None
    if isinstance(raw, bool):
        raise ValueError(f"{path} must not be boolean")
    if isinstance(raw, (int, float)):
        pair_count = _required_count(raw, path)
        if anchor_observation_count is None or anchor_observation_count <= 0:
            raise ValueError(f"{path} count requires a positive anchor observation count")
        probability = float(pair_count) / float(anchor_observation_count)
        probability_source = "legacy_count_over_observation_count_approximation"
    elif isinstance(raw, Mapping):
        for key in ("count", "pair_count", "cooccurrence_count", "n_scenes"):
            if raw.get(key) is not None:
                pair_count = _required_count(raw[key], f"{path}.{key}")
                break
        probability = None
        for key in ("p_b_given_a", "probability", "fraction", "conditional_probability"):
            if raw.get(key) is not None:
                probability = _required_finite(raw[key], f"{path}.{key}")
                break
        if probability is None and pair_count is not None and anchor_observation_count:
            probability = float(pair_count) / float(anchor_observation_count)
            probability_source = "count_over_observation_count_approximation"
        elif probability is not None:
            probability_source = "recorded_conditional_probability"
        if probability is None:
            raise ValueError(f"{path} requires p_b_given_a/probability/fraction or count")
        if raw.get("npmi") is not None:
            npmi = _required_finite(raw["npmi"], f"{path}.npmi")
            if not -1.0 <= npmi <= 1.0:
                raise ValueError(f"{path}.npmi must be between -1 and 1")
        for key in ("anchor_count", "anchor_scene_count", "count_a", "total_scene_count"):
            if raw.get(key) is not None:
                anchor_observation_count = _required_count(raw[key], f"{path}.{key}")
                break
    else:
        raise ValueError(f"{path} must be a number or JSON object")
    if not 0.0 <= float(probability) <= 1.0:
        raise ValueError(f"{path} conditional probability must be between 0 and 1")
    return CooccurrenceRecord(
        probability=float(probability),
        pair_count=pair_count,
        anchor_observation_count=anchor_observation_count,
        npmi=npmi,
        context=context,
        probability_source=probability_source,
        raw=raw,
    )


def _count_from_value(value: Any) -> int | None:
    if isinstance(value, Mapping):
        for key in ("count", "scene_count", "n_scenes"):
            count = _optional_count(value.get(key), key)
            if count is not None:
                return count
        return None
    return _optional_count(value, "count")


def _optional_count(value: Any, path: str) -> int | None:
    if value is None:
        return None
    return _required_count(value, path)


def _required_count(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a non-negative integer")
    number = float(value)
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        raise ValueError(f"{path} must be a non-negative integer")
    return int(number)


def _optional_finite(value: Any, path: str) -> float | None:
    if value is None:
        return None
    return _required_finite(value, path)


def _required_finite(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{path} must be a finite number")
    return number
