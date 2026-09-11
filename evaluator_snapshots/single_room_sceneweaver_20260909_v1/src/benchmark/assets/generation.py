from __future__ import annotations

import asyncio
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
from typing import Any, Protocol


class AssetGenerationError(RuntimeError):
    """Raised when an asset-generation tool cannot produce a usable asset."""


class AssetGenerationTool(Protocol):
    """Synchronous benchmark-facing contract for generated-asset providers."""

    def generate_asset(self, request: dict[str, Any]) -> dict[str, Any]:
        """Generate one asset and return its metadata/geometry references."""


class MCPAssetGenerationTool:
    """Thin adapter around an MCP client exposing ``call_tool``.

    The MCP SDK is intentionally not a package dependency. Callers inject an
    already configured client so local, remote, and test clients share the same
    benchmark-facing interface.
    """

    def __init__(self, client: Any, *, tool_name: str = "generate_asset") -> None:
        if not hasattr(client, "call_tool"):
            raise TypeError("MCP asset-generation client must provide call_tool(name, arguments)")
        self.client = client
        self.tool_name = str(tool_name or "generate_asset")

    def generate_asset(self, request: dict[str, Any]) -> dict[str, Any]:
        result = self.client.call_tool(self.tool_name, request)
        return _coerce_tool_result(_resolve_awaitable(result))


def invoke_asset_generation_tool(tool: Any, request: dict[str, Any]) -> dict[str, Any]:
    """Invoke a callable, native tool object, or MCP-style tool object."""

    if tool is None:
        raise AssetGenerationError(
            "The VLM requested asset generation, but no asset_generation_tool was configured."
        )
    if hasattr(tool, "generate_asset"):
        result = tool.generate_asset(request)
    elif callable(tool):
        result = tool(request)
    else:
        raise AssetGenerationError(
            "asset_generation_tool must be callable or provide generate_asset(request)"
        )
    return _coerce_tool_result(_resolve_awaitable(result))


def load_asset_generation_tool(spec: str | None) -> Any | None:
    """Load ``module:attribute`` or ``/path/to/module.py:attribute`` plugins."""

    if not spec:
        return None
    module_ref, separator, attribute = str(spec).partition(":")
    if not separator or not module_ref or not attribute:
        raise AssetGenerationError(
            "Asset generator plugin must use module:attribute or /path/module.py:attribute"
        )
    path = Path(module_ref).expanduser()
    if path.exists():
        module_spec = importlib.util.spec_from_file_location("asset_generation_plugin", path)
        if module_spec is None or module_spec.loader is None:
            raise AssetGenerationError(f"Cannot import asset generator plugin from {path}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_ref)
    tool = getattr(module, attribute, None)
    if tool is None:
        raise AssetGenerationError(f"Asset generator plugin has no attribute {attribute!r}")
    return tool() if inspect.isclass(tool) else tool


def _coerce_tool_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        payload = result.get("asset") if isinstance(result.get("asset"), dict) else result
        return dict(payload)
    structured = getattr(result, "structured_content", None)
    if structured is None:
        structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        payload = structured.get("asset") if isinstance(structured.get("asset"), dict) else structured
        return dict(payload)
    if isinstance(result, str):
        return _parse_json_result(result)
    content = getattr(result, "content", None)
    if isinstance(content, list):
        for item in content:
            text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
            if isinstance(text, str) and text.strip():
                return _parse_json_result(text)
    raise AssetGenerationError(
        "Asset-generation tool must return an asset JSON object or MCP text/structured content."
    )


def _parse_json_result(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AssetGenerationError(f"Asset-generation tool returned invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise AssetGenerationError("Asset-generation tool JSON result must be an object")
    payload = parsed.get("asset") if isinstance(parsed.get("asset"), dict) else parsed
    return dict(payload)


def _resolve_awaitable(result: Any) -> Any:
    if not inspect.isawaitable(result):
        return result
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(result)
    raise AssetGenerationError(
        "Async asset-generation tools cannot be awaited from the synchronous harness while an event loop is running; "
        "provide a synchronous adapter."
    )
