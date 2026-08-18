"""Metric-scoped public context for L3 Judge prompts.

The generator may receive a public task brief, but that does not authorize the
L3 Judge to consume every requested relation as a validity prior.  This module
keeps the interface explicit: public values are frozen once, and each metric
selects only the short fields it needs.  The full generation instruction is
available to callers as an opt-in field but is not selected by the benchmark
defaults.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Mapping

from benchmark.evaluator.scene_quality.definitions import (
    SUPPORTED_SCENE_QUALITY_METRICS,
)


METRIC_PROMPT_CONTEXT_VERSION = "l3_metric_prompt_context_v1"
_FIELD_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_MAX_CONTEXT_VALUE_CHARS = 1200
_MAX_METRIC_INSTRUCTION_CHARS = 1200

DEFAULT_METRIC_CONTEXT_FIELDS: dict[str, tuple[str, ...]] = {
    "scale_consistency": (),
    "style_consistency": ("room_type",),
    "object_pairing_consistency": ("room_type",),
    "functional_consistency": (),
    "semantic_placement_consistency": (),
}

DEFAULT_METRIC_CONTEXT_INSTRUCTIONS = {
    "style_consistency": (
        "Treat room type as context, not as a mandatory aesthetic. Do not "
        "require stereotypical contents or a stereotypical visual style. "
        "Only an explicit style descriptor, when supplied, defines an "
        "intended visual style."
    ),
    "object_pairing_consistency": (
        "Use room type only to disambiguate ordinary semantic compatibility. "
        "Do not require stereotypical contents and do not penalize a "
        "plausible multi-purpose use."
    ),
}


def resolve_scene_quality_prompt_context(
    *,
    scene: Mapping[str, Any],
    original_prompt: str | None,
    override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve a validated, auditable public-context bank.

    ``override`` may contain ``values``, ``metric_fields``, and
    ``metric_instructions``.  Values are short strings; every metric field is
    an explicit reference into that bank.  This allows experiments to expose
    more or less public context without changing prompt-building code.
    """

    if override is not None and not isinstance(override, Mapping):
        raise TypeError("scene_quality_prompt_context must be a JSON object")
    patch = dict(override or {})
    unknown_top = sorted(
        set(patch)
        - {
            "schema_version",
            "values",
            "metric_fields",
            "metric_instructions",
            "source",
        }
    )
    if unknown_top:
        raise ValueError(
            "scene_quality_prompt_context contains unknown fields: "
            f"{unknown_top}"
        )
    schema_version = patch.get("schema_version")
    if schema_version not in {None, METRIC_PROMPT_CONTEXT_VERSION}:
        raise ValueError(
            "scene_quality_prompt_context.schema_version must be "
            f"{METRIC_PROMPT_CONTEXT_VERSION!r}"
        )

    values: dict[str, str] = {}
    room_type = scene.get("scene_type") or scene.get("room_type")
    if isinstance(room_type, str) and room_type.strip():
        values["room_type"] = room_type.strip()
    if isinstance(original_prompt, str) and original_prompt.strip():
        values["original_prompt"] = original_prompt.strip()
    for key in ("style_descriptor", "visual_style", "style"):
        value = scene.get(key)
        if isinstance(value, str) and value.strip():
            values["style_descriptor"] = value.strip()
            break

    supplied_values = patch.get("values")
    if supplied_values is not None:
        if not isinstance(supplied_values, Mapping):
            raise TypeError(
                "scene_quality_prompt_context.values must be a JSON object"
            )
        for raw_name, raw_value in supplied_values.items():
            name = _context_field_name(raw_name)
            if raw_value is None:
                values.pop(name, None)
                continue
            values[name] = _short_context_value(
                raw_value,
                where=f"scene_quality_prompt_context.values.{name}",
            )

    metric_fields = {
        metric: list(fields)
        for metric, fields in DEFAULT_METRIC_CONTEXT_FIELDS.items()
    }
    supplied_fields = patch.get("metric_fields")
    if supplied_fields is not None:
        if not isinstance(supplied_fields, Mapping):
            raise TypeError(
                "scene_quality_prompt_context.metric_fields must be a JSON "
                "object"
            )
        unknown_metrics = sorted(
            set(str(name) for name in supplied_fields)
            - set(SUPPORTED_SCENE_QUALITY_METRICS)
        )
        if unknown_metrics:
            raise ValueError(
                "scene_quality_prompt_context.metric_fields contains unknown "
                f"metrics: {unknown_metrics}"
            )
        for metric, raw_fields in supplied_fields.items():
            if not isinstance(raw_fields, list):
                raise TypeError(
                    "scene_quality_prompt_context.metric_fields."
                    f"{metric} must be a list"
                )
            fields = [_context_field_name(item) for item in raw_fields]
            if len(fields) != len(set(fields)):
                raise ValueError(
                    "scene_quality_prompt_context.metric_fields."
                    f"{metric} contains duplicate fields"
                )
            missing = sorted(set(fields) - set(values))
            if missing:
                raise ValueError(
                    "scene_quality_prompt_context.metric_fields."
                    f"{metric} references missing values: {missing}"
                )
            metric_fields[str(metric)] = fields

    metric_instructions = dict(DEFAULT_METRIC_CONTEXT_INSTRUCTIONS)
    supplied_instructions = patch.get("metric_instructions")
    if supplied_instructions is not None:
        if not isinstance(supplied_instructions, Mapping):
            raise TypeError(
                "scene_quality_prompt_context.metric_instructions must be a "
                "JSON object"
            )
        unknown_metrics = sorted(
            set(str(name) for name in supplied_instructions)
            - set(SUPPORTED_SCENE_QUALITY_METRICS)
        )
        if unknown_metrics:
            raise ValueError(
                "scene_quality_prompt_context.metric_instructions contains "
                f"unknown metrics: {unknown_metrics}"
            )
        for metric, raw_instruction in supplied_instructions.items():
            if raw_instruction is None:
                metric_instructions.pop(str(metric), None)
                continue
            instruction = _short_context_value(
                raw_instruction,
                where=(
                    "scene_quality_prompt_context.metric_instructions."
                    f"{metric}"
                ),
                max_chars=_MAX_METRIC_INSTRUCTION_CHARS,
            )
            metric_instructions[str(metric)] = instruction

    source = patch.get("source")
    if source is None:
        source = "scene_and_public_request"
    source = _short_context_value(
        source,
        where="scene_quality_prompt_context.source",
        max_chars=160,
    )
    return {
        "schema_version": METRIC_PROMPT_CONTEXT_VERSION,
        "source": source,
        "values": values,
        "metric_fields": metric_fields,
        "metric_instructions": metric_instructions,
        "full_instruction_default": "excluded",
    }


def metric_prompt_context(
    resolved: Mapping[str, Any],
    metric: str,
) -> dict[str, Any]:
    """Select and render one metric's deterministic short context."""

    metric_name = str(metric)
    if metric_name not in SUPPORTED_SCENE_QUALITY_METRICS:
        raise ValueError(f"unsupported scene-quality metric {metric_name!r}")
    fields = list((resolved.get("metric_fields") or {}).get(metric_name) or [])
    values_bank = resolved.get("values") or {}
    selected_values = {
        field: str(values_bank[field])
        for field in fields
        if field in values_bank
    }
    instruction = (resolved.get("metric_instructions") or {}).get(metric_name)
    lines = [
        "Public task context supplied to the generator; it is not a defect "
        "prior or a required answer."
    ]
    lines.extend(
        f"{field}: {selected_values[field]}" for field in fields
        if field in selected_values
    )
    if instruction:
        lines.append(str(instruction))
    rendered = "\n".join(lines) if selected_values or instruction else None
    return {
        "schema_version": METRIC_PROMPT_CONTEXT_VERSION,
        "source": resolved.get("source"),
        "metric": metric_name,
        "selected_fields": fields,
        "values": selected_values,
        "metric_instruction": instruction,
        "rendered_prompt": rendered,
        "original_prompt_included": "original_prompt" in fields,
    }


def prompt_context_manifest(resolved: Mapping[str, Any]) -> dict[str, Any]:
    """Return a defensive audit record without changing the frozen bank."""

    return deepcopy(dict(resolved))


def _context_field_name(value: Any) -> str:
    if not isinstance(value, str) or not _FIELD_RE.fullmatch(value):
        raise ValueError(
            "prompt-context field names must match "
            "^[a-z][a-z0-9_]{0,63}$"
        )
    return value


def _short_context_value(
    value: Any,
    *,
    where: str,
    max_chars: int = _MAX_CONTEXT_VALUE_CHARS,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{where} must be a non-empty string")
    normalized = " ".join(value.split())
    if len(normalized) > max_chars:
        raise ValueError(f"{where} exceeds {max_chars} characters")
    return normalized


__all__ = [
    "DEFAULT_METRIC_CONTEXT_FIELDS",
    "DEFAULT_METRIC_CONTEXT_INSTRUCTIONS",
    "METRIC_PROMPT_CONTEXT_VERSION",
    "metric_prompt_context",
    "prompt_context_manifest",
    "resolve_scene_quality_prompt_context",
]
