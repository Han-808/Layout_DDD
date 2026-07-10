from __future__ import annotations

import base64
import importlib
import importlib.util
import json
import mimetypes
from pathlib import Path
from typing import Any

from benchmark.assets.generation import AssetGenerationError, invoke_asset_generation_tool
from benchmark.nl_scene.converter import call_chat_model, parse_json_object_from_text


class AssetRetrievalError(RuntimeError):
    """Raised when asset retrieval cannot run."""


def retrieve_assets_for_object_plan(
    object_plan: dict,
    *,
    asset_index_path: str,
    retrieval_k: int = 5,
    retriever_module_path: str | None = None,
    use_vlm_selector: bool = False,
    model_config: dict | None = None,
    asset_generation_tool: Any | None = None,
) -> dict:
    """Resolve object requirements through top-k retrieval and VLM routing.

    The VLM returns an explicit ``select`` or ``generate`` action. Generation is
    executed by the workflow through an injected callable/tool/MCP adapter; the
    model never receives implicit authority to execute arbitrary code.
    """

    index_path = Path(asset_index_path).expanduser()
    if not index_path.with_suffix(".json").exists() and not index_path.exists():
        raise FileNotFoundError(f"Asset index path does not exist: {asset_index_path}")
    retriever_cls = _load_asset_retriever_class(retriever_module_path)
    retriever = retriever_cls(index_path=str(index_path))
    k = max(1, int(retrieval_k))
    generation_enabled = asset_generation_tool is not None
    objects = object_plan.get("objects") if isinstance(object_plan, dict) else None
    if not isinstance(objects, list):
        raise AssetRetrievalError("object_plan must contain an objects list")
    output_objects = []
    for index, object_spec in enumerate(objects):
        if not isinstance(object_spec, dict):
            continue
        candidates = retriever.retrieve(
            description=str(object_spec.get("description") or object_spec.get("category") or "object"),
            category=object_spec.get("category"),
            size_constraint=object_spec.get("estimated_size"),
            top_k=k,
        )
        normalized_candidates = [_compact_candidate(item) for item in list(candidates or [])[:k]]
        selected_asset: dict[str, Any] | None = None
        selection_action = "unresolved"
        selection_decision: dict[str, Any] = {
            "action": "unresolved",
            "selected_jid": None,
            "reason": "no candidates returned",
            "generation_request": None,
        }
        selection_reason = "no candidates returned"
        object_id = str(object_spec.get("id") or f"obj_{index:03d}")
        if normalized_candidates and not use_vlm_selector:
            selected_asset = normalized_candidates[0]
            selection_action = "select"
            selection_reason = "top retrieval result; VLM selector disabled"
            selection_decision = {
                "action": "select",
                "selected_jid": selected_asset.get("jid"),
                "reason": selection_reason,
                "generation_request": None,
            }
        elif use_vlm_selector:
            selection_decision = _select_asset_with_vlm(
                object_spec,
                normalized_candidates,
                model_config=model_config,
                generation_available=generation_enabled,
            )
            selection_action = selection_decision["action"]
            selection_reason = selection_decision["reason"]
            if selection_action == "select":
                selected_jid = str(selection_decision.get("selected_jid") or "")
                selected_asset = _candidate_by_jid(normalized_candidates, selected_jid)
                if selected_asset is None and normalized_candidates:
                    selected_asset = normalized_candidates[0]
                    selection_decision["selected_jid"] = selected_asset.get("jid")
                    selection_reason = (
                        f"selector returned unavailable jid {selected_jid!r}; selected top retrieval result"
                    )
                    selection_decision["reason"] = selection_reason
                elif selected_asset is None:
                    raise AssetRetrievalError(
                        "Asset selector chose a database asset, but retrieval returned no candidates."
                    )
            elif selection_action == "generate":
                if not generation_enabled:
                    raise AssetGenerationError(
                        "The VLM found no suitable retrieved asset, but asset generation is not enabled/configured."
                    )
                generation_request = _asset_generation_request(
                    object_plan=object_plan,
                    object_id=object_id,
                    object_spec=object_spec,
                    candidates=normalized_candidates,
                    decision=selection_decision,
                )
                selection_decision["generation_request"] = generation_request["generation_request"]
                generated = invoke_asset_generation_tool(asset_generation_tool, generation_request)
                selected_asset = _normalize_generated_asset(
                    generated,
                    request_id=str(object_plan.get("request_id") or "request_001"),
                    object_id=object_id,
                    object_spec=object_spec,
                    decision=selection_decision,
                )
                selection_reason = selection_reason or "generated because no retrieved candidate was suitable"
        elif not normalized_candidates:
            raise AssetRetrievalError("Asset retrieval returned no candidates and the VLM selector is disabled.")
        output_objects.append(
            {
                "object_id": object_id,
                "object_spec": {
                    "category": object_spec.get("category"),
                    "description": object_spec.get("description"),
                    "estimated_size": object_spec.get("estimated_size"),
                },
                "selected_asset": selected_asset,
                "candidates": normalized_candidates,
                "selection_action": selection_action,
                "selection_decision": selection_decision,
                "selection_reason": selection_reason,
            }
        )
    return {"request_id": str(object_plan.get("request_id") or "request_001"), "objects": output_objects}


def retrieve_assets_for_scene_spec(scene_spec: dict, **kwargs: Any) -> dict:
    """Temporary alias that returns canonical asset_selection, not asset_retrieval."""

    return retrieve_assets_for_object_plan(scene_spec, **kwargs)


def _load_asset_retriever_class(module_path: str | None) -> type:
    if module_path:
        path = Path(module_path).expanduser()
        if path.exists():
            spec = importlib.util.spec_from_file_location("dynamic_asset_retriever", path)
            if spec is None or spec.loader is None:
                raise AssetRetrievalError(f"Cannot import retriever module from {module_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            module = importlib.import_module(module_path)
    else:
        module = importlib.import_module("benchmark.assets.retriever")
    retriever_cls = getattr(module, "AssetRetriever", None)
    if retriever_cls is None:
        raise AssetRetrievalError("AssetRetriever class was not found in the retriever module")
    return retriever_cls


def _select_asset_with_vlm(
    object_spec: dict,
    candidates: list[dict[str, Any]],
    *,
    model_config: dict | None,
    generation_available: bool,
) -> dict[str, Any]:
    config = model_config or {}
    selector = config.get("asset_selector") or config.get("selector")
    if callable(selector):
        selected = selector(object_spec, candidates)
        return _normalize_selector_decision(
            selected,
            generation_available=generation_available,
            default_reason="selected by callable asset selector",
        )
    selector_response = config.get("selector_response") or config.get("mock_selector_response")
    if isinstance(selector_response, dict):
        return _normalize_selector_decision(
            selector_response,
            generation_available=generation_available,
            default_reason="selected by configured asset selector",
        )
    prompt = {
        "object_spec": object_spec,
        "candidates": candidates,
        "asset_generation_available": generation_available,
        "instructions": (
            "Judge whether any candidate faithfully matches the requested object category, description, visual attributes, "
            "and approximate dimensions. Return action=select with one listed selected_jid, or action=generate with "
            "selected_jid=null and a generation_request when no candidate is suitable. Never invent a candidate jid."
        ),
        "output_schema": {
            "action": "select | generate",
            "selected_jid": "listed jid or null",
            "reason": "short decision rationale",
            "generation_request": {
                "prompt": "asset-generation description",
                "category": "object category",
                "target_size": ["width", "depth", "height"],
            },
        },
    }
    response = call_chat_model(
        [
            {
                "role": "system",
                "content": (
                    "You are a conservative 3D asset selector. Select only a genuinely suitable retrieved asset. "
                    "When generation is available and none is suitable, authorize asset generation. Return JSON only."
                ),
            },
            {"role": "user", "content": _selector_user_content(prompt, candidates)},
        ],
        model_config=model_config,
        response_format_json=True,
        call_type="asset_selector",
    )
    parsed = parse_json_object_from_text(response)
    return _normalize_selector_decision(
        parsed,
        generation_available=generation_available,
        default_reason="selected by VLM asset selector",
    )


def _normalize_selector_decision(
    value: Any,
    *,
    generation_available: bool,
    default_reason: str,
) -> dict[str, Any]:
    if isinstance(value, str):
        value = {"action": "select", "selected_jid": value}
    if not isinstance(value, dict):
        raise AssetRetrievalError("Asset selector must return a JSON object or candidate jid")
    selected_jid = value.get("selected_jid") or value.get("jid")
    action = str(value.get("action") or value.get("decision") or "").strip().lower()
    if not action:
        action = "select" if selected_jid else "generate"
    if action not in {"select", "generate"}:
        raise AssetRetrievalError("Asset selector action must be 'select' or 'generate'")
    generation_request = value.get("generation_request")
    if not isinstance(generation_request, dict):
        generation_request = None
    reason = str(value.get("reason") or value.get("selection_reason") or default_reason)
    if action == "select" and not selected_jid:
        raise AssetRetrievalError("Asset selector action 'select' requires selected_jid")
    if action == "generate" and not generation_available:
        reason = reason or "no suitable candidate; asset generation unavailable"
    return {
        "action": action,
        "selected_jid": str(selected_jid) if selected_jid is not None else None,
        "reason": reason,
        "generation_request": generation_request,
    }


def _asset_generation_request(
    *,
    object_plan: dict,
    object_id: str,
    object_spec: dict,
    candidates: list[dict[str, Any]],
    decision: dict[str, Any],
) -> dict[str, Any]:
    generation_request = decision.get("generation_request")
    if not isinstance(generation_request, dict):
        generation_request = {}
    generation_request = dict(generation_request)
    generation_request.setdefault(
        "prompt",
        str(object_spec.get("description") or object_spec.get("category") or "3D object"),
    )
    generation_request.setdefault("category", object_spec.get("category"))
    generation_request.setdefault("target_size", object_spec.get("estimated_size"))
    return {
        "request_id": str(object_plan.get("request_id") or "request_001"),
        "object_id": object_id,
        "object_spec": {
            "category": object_spec.get("category"),
            "description": object_spec.get("description"),
            "estimated_size": object_spec.get("estimated_size"),
        },
        "rejected_candidates": candidates,
        "selector_reason": decision.get("reason"),
        "generation_request": generation_request,
    }


def _normalize_generated_asset(
    asset: dict[str, Any],
    *,
    request_id: str,
    object_id: str,
    object_spec: dict,
    decision: dict[str, Any],
) -> dict[str, Any]:
    raw = dict(asset)
    jid = raw.get("jid") or raw.get("asset_id") or raw.get("id") or f"generated:{request_id}:{object_id}"
    raw["jid"] = str(jid)
    raw.setdefault("category", object_spec.get("category") or "object")
    raw.setdefault("description", object_spec.get("description") or object_spec.get("category") or "generated object")
    raw.setdefault("short_desc", raw.get("description"))
    raw.setdefault("size", raw.get("dimensions") or object_spec.get("estimated_size"))
    asset_ref = raw.get("asset_ref") if isinstance(raw.get("asset_ref"), dict) else {}
    asset_ref = dict(asset_ref)
    asset_ref["source_db"] = "generated"
    asset_ref.setdefault("asset_key", str(jid))
    for source_key, target_key in [
        ("mesh_path", "mesh_uri"),
        ("pointcloud_path", "pointcloud_uri"),
        ("metadata_path", "metadata_uri"),
    ]:
        if raw.get(source_key) is not None and not asset_ref.get(target_key):
            asset_ref[target_key] = raw.get(source_key)
    asset_ref.setdefault("mesh_uri", raw.get("mesh_uri"))
    asset_ref.setdefault("pointcloud_uri", raw.get("pointcloud_uri"))
    asset_ref.setdefault("metadata_uri", raw.get("metadata_uri"))
    raw["asset_ref"] = asset_ref
    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    metadata = dict(metadata)
    metadata.update(
        {
            "generated": True,
            "generation_reason": decision.get("reason"),
            "generation_request": decision.get("generation_request"),
        }
    )
    raw["metadata"] = metadata
    compact = _compact_candidate(raw)
    size = compact.get("size")
    if not isinstance(size, list) or len(size) != 3:
        raise AssetGenerationError(
            "Generated asset must provide size/dimensions, or the object plan must provide estimated_size."
        )
    if not any(asset_ref.get(key) for key in ["mesh_uri", "pointcloud_uri", "metadata_uri"]):
        raise AssetGenerationError(
            "Generated asset must provide mesh_uri, pointcloud_uri, metadata_uri, or the corresponding *_path field."
        )
    return compact


def _selector_user_content(prompt: dict[str, Any], candidates: list[dict[str, Any]]) -> str | list[dict[str, Any]]:
    text = json.dumps(prompt, ensure_ascii=False)
    media: list[tuple[str, str]] = []
    for candidate in candidates:
        image_url = _candidate_image_url(candidate)
        if image_url:
            media.append((str(candidate.get("jid") or "candidate"), image_url))
    if not media:
        return text
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for jid, image_url in media:
        content.append({"type": "text", "text": f"Visual preview for candidate {jid}:"})
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    return content


def _candidate_image_url(candidate: dict[str, Any]) -> str | None:
    asset_ref = candidate.get("asset_ref") if isinstance(candidate.get("asset_ref"), dict) else {}
    value = (
        candidate.get("preview_uri")
        or candidate.get("thumbnail_uri")
        or candidate.get("render_uri")
        or asset_ref.get("preview_uri")
    )
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    if value.startswith(("http://", "https://", "data:image/")):
        return value
    path = Path(value).expanduser()
    if not path.is_file():
        return None
    mime = mimetypes.guess_type(path.name)[0]
    if not mime or not mime.startswith("image/"):
        return None
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    jid = candidate.get("jid") or candidate.get("asset_id") or candidate.get("id")
    size = candidate.get("size") or candidate.get("dimensions")
    desc = candidate.get("desc") or candidate.get("description") or candidate.get("caption_en") or candidate.get("short_desc") or ""
    short_desc = candidate.get("short_desc") or desc
    compact = {
        "jid": jid,
        "category": candidate.get("category") or candidate.get("class_en") or "",
        "retrieval_category": candidate.get("retrieval_category") or candidate.get("retrieval_class_en") or candidate.get("category") or "",
        "desc": desc,
        "short_desc": short_desc,
        "size": size,
        "asset_ref": _asset_ref(candidate, jid),
        "asset_proxy": candidate.get("asset_proxy") if isinstance(candidate.get("asset_proxy"), dict) else {
            "type": "obb_from_metadata_or_csv",
            "bbox_center_local": [0, 0, 0],
            "bbox_size": size,
        },
        "metadata": _asset_metadata(candidate),
    }
    if candidate.get("score") is not None:
        compact["score"] = float(candidate.get("score"))
    for key in ["preview_uri", "thumbnail_uri", "render_uri"]:
        if candidate.get(key):
            compact[key] = candidate.get(key)
    return compact


def _asset_ref(candidate: dict[str, Any], jid: object) -> dict[str, Any]:
    if isinstance(candidate.get("asset_ref"), dict):
        ref = dict(candidate["asset_ref"])
    else:
        ref = {"source_db": "imaginarium", "asset_key": jid}
    ref.setdefault("source_db", ref.pop("source", "imaginarium"))
    ref.setdefault("asset_key", jid or ref.get("asset_id"))
    for source_key, target_key in [("mesh_path", "mesh_uri"), ("pointcloud_path", "pointcloud_uri"), ("metadata_path", "metadata_uri")]:
        if candidate.get(source_key) is not None and not ref.get(target_key):
            ref[target_key] = candidate.get(source_key)
    ref.setdefault("mesh_uri", candidate.get("mesh_uri"))
    ref.setdefault("pointcloud_uri", candidate.get("pointcloud_uri"))
    ref.setdefault("metadata_uri", candidate.get("metadata_uri"))
    return ref


def _asset_metadata(candidate: dict[str, Any]) -> dict[str, Any]:
    metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
    normalized = dict(metadata)
    normalized["interactive"] = bool(metadata.get("interactive") or candidate.get("interactive") or False)
    normalized["inner_placement"] = bool(metadata.get("inner_placement") or candidate.get("inner_placement") or False)
    normalized["align_to_wall_normal"] = bool(
        metadata.get("align_to_wall_normal")
        or metadata.get("alignToWallNormal")
        or candidate.get("alignToWallNormal")
        or False
    )
    normalized["scaling_strategy"] = metadata.get("scaling_strategy") or candidate.get("scaling_strategy")
    return normalized


def _candidate_by_jid(candidates: list[dict[str, Any]], jid: str) -> dict[str, Any] | None:
    for candidate in candidates:
        if str(candidate.get("jid")) == str(jid):
            return candidate
    return None
