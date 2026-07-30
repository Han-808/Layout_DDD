from __future__ import annotations

import json


class ModelResponseError(ValueError):
    """Raised when a model response cannot be converted into a JSON object."""


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
        try:
            parsed = json.loads(raw[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ModelResponseError(f"Model response contains malformed JSON: {exc.msg} at char {exc.pos}.") from exc
    if not isinstance(parsed, dict):
        raise ModelResponseError("Model response JSON must be an object.")
    return parsed
