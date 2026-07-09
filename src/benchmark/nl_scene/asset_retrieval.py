from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
from typing import Any

from benchmark.nl_scene.converter import call_chat_model, parse_json_object_from_text


class AssetRetrievalError(RuntimeError):
    """Raised when asset retrieval cannot run."""


def retrieve_assets_for_scene_spec(
    scene_spec: dict,
    *,
    asset_index_path: str,
    retrieval_k: int = 1,
    retriever_module_path: str | None = None,
    use_vlm_selector: bool = True,
    model_config: dict | None = None,
) -> dict:
    """Attach retrieval candidates and selected assets to a converter scene spec."""

    index_path = Path(asset_index_path).expanduser()
    if not index_path.with_suffix(".json").exists() and not index_path.exists():
        raise FileNotFoundError(f"Asset index path does not exist: {asset_index_path}")
    retriever_cls = _load_asset_retriever_class(retriever_module_path)
    retriever = retriever_cls(index_path=str(index_path))
    k = max(1, int(retrieval_k))
    objects = scene_spec.get("objects") if isinstance(scene_spec, dict) else None
    if not isinstance(objects, list):
        raise AssetRetrievalError("scene_spec must contain an objects list")
    output_objects = []
    for object_spec in objects:
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
        selection_reason = "no candidates returned"
        if normalized_candidates and k == 1:
            selected_asset = normalized_candidates[0]
            selection_reason = "selected top-1 retriever result"
        elif normalized_candidates:
            if use_vlm_selector:
                selected_jid, reason = _select_asset_with_vlm(object_spec, normalized_candidates, model_config=model_config)
                selected_asset = _candidate_by_jid(normalized_candidates, selected_jid) or normalized_candidates[0]
                selection_reason = reason or "selected by VLM asset selector"
                if selected_asset.get("jid") != selected_jid:
                    selection_reason = f"selector returned unavailable jid {selected_jid!r}; selected top retriever result"
            else:
                selected_asset = normalized_candidates[0]
                selection_reason = "selected top retriever result because VLM selector is disabled"
        output_objects.append(
            {
                "object_spec": object_spec,
                "candidates": normalized_candidates,
                "selected_asset": selected_asset,
                "selection_reason": selection_reason,
            }
        )
    return {"scene_spec": scene_spec, "retrieval_k": k, "objects": output_objects}


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


def _select_asset_with_vlm(object_spec: dict, candidates: list[dict[str, Any]], *, model_config: dict | None) -> tuple[str, str]:
    config = model_config or {}
    selector = config.get("asset_selector") or config.get("selector")
    if callable(selector):
        selected = selector(object_spec, candidates)
        if isinstance(selected, dict):
            return str(selected.get("selected_jid") or selected.get("jid") or ""), str(selected.get("reason") or selected.get("selection_reason") or "")
        return str(selected), "selected by callable asset selector"
    selector_response = config.get("selector_response") or config.get("mock_selector_response")
    if isinstance(selector_response, dict):
        return str(selector_response.get("selected_jid") or selector_response.get("jid") or ""), str(selector_response.get("reason") or "")
    prompt = {
        "object_spec": object_spec,
        "candidates": candidates,
        "instructions": "Choose exactly one candidate for this object. Return JSON with selected_jid and reason.",
    }
    response = call_chat_model(
        [
            {"role": "system", "content": "You select the best 3D asset candidate. Return JSON only."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        model_config=model_config,
        response_format_json=True,
        call_type="asset_selector",
    )
    parsed = parse_json_object_from_text(response)
    return str(parsed.get("selected_jid") or parsed.get("jid") or ""), str(parsed.get("reason") or "")


def _compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "jid": candidate.get("jid") or candidate.get("asset_id") or candidate.get("id"),
        "short_desc": candidate.get("short_desc") or candidate.get("description") or "",
        "category": candidate.get("category") or "",
        "size": candidate.get("size") or candidate.get("dimensions"),
    }
    if candidate.get("description") is not None:
        compact["description"] = candidate.get("description")
    if candidate.get("score") is not None:
        compact["score"] = float(candidate.get("score"))
    for key in ["asset_ref", "mesh_path", "pointcloud_path", "metadata_path"]:
        if candidate.get(key) is not None:
            compact[key] = candidate.get(key)
    return compact


def _candidate_by_jid(candidates: list[dict[str, Any]], jid: str) -> dict[str, Any] | None:
    for candidate in candidates:
        if str(candidate.get("jid")) == str(jid):
            return candidate
    return None
