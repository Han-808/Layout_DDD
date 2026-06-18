from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class ModelResponseError(ValueError):
    """Raised when a model response cannot be converted into layout JSON."""


def parse_json_object(raw: str | dict) -> dict:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        raise ModelResponseError(f"Expected dict or JSON string, got {type(raw).__name__}.")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise ModelResponseError("Model response does not contain a JSON object.") from None
        parsed = json.loads(raw[start : end + 1])
    if not isinstance(parsed, dict):
        raise ModelResponseError("Model response JSON must be an object.")
    return parsed


@dataclass
class BaseLayoutModel(ABC):
    name: str

    @abstractmethod
    def generate_layout(self, bm_instance: dict, layout_schema: dict) -> dict:
        """Generate a full layout JSON object for a benchmark instance."""

    @abstractmethod
    def repair_layout(
        self,
        bm_instance: dict,
        current_layout: dict,
        feedback: dict,
        layout_schema: dict,
    ) -> dict:
        """Repair a full layout JSON object from deterministic feedback."""


def build_generation_prompt(bm_instance: dict, layout_schema: dict) -> str:
    return (
        "Generate an explicit 3D scene layout as JSON only.\n"
        "Use bbox objects with center [x, y, z], size [width, depth, height], "
        "and yaw in degrees around z/up. Do not include validity fields.\n\n"
        "Requirements:\n"
        "- Return exactly one JSON object and no Markdown.\n"
        "- Use the input case_id/task_id as scene_id.\n"
        "- For each input object in bm_instance.objects, create exactly one layout object.\n"
        "- Preserve each input object id as layout object_id exactly.\n"
        "- Preserve each input object category exactly.\n"
        "- If an input object has bbox_size, use it as layout size.\n"
        "- Relation outputs must use keys source and target, not subject/object or source_category.\n"
        "- Keep all object centers inside the room boundary when a boundary is provided.\n\n"
        f"Benchmark instance:\n{json.dumps(bm_instance, indent=2)}\n\n"
        f"Layout JSON schema:\n{json.dumps(layout_schema, indent=2)}\n"
    )


def build_repair_prompt(
    bm_instance: dict,
    current_layout: dict,
    feedback: dict,
    layout_schema: dict,
) -> str:
    return (
        "Repair the explicit 3D scene layout. Return full corrected layout JSON only, "
        "with no explanation.\n"
        "Preserve object IDs. Preserve valid objects unless necessary. Keep the room "
        "boundary unchanged. Fix only listed violations.\n\n"
        f"Benchmark instance:\n{json.dumps(bm_instance, indent=2)}\n\n"
        f"Current layout:\n{json.dumps(current_layout, indent=2)}\n\n"
        f"Deterministic feedback:\n{json.dumps(feedback, indent=2)}\n\n"
        f"Layout JSON schema:\n{json.dumps(layout_schema, indent=2)}\n"
    )
