from __future__ import annotations

from benchmark.models.base_model import BaseLayoutModel
from benchmark.models.langchain_model import LangChainModel
from benchmark.models.mock_model import MockModel
from benchmark.models.openai_compatible_model import OpenAICompatibleModel


def create_model(model_name: str, model_config: dict | None = None) -> BaseLayoutModel:
    config = model_config or {}
    model_defs = config.get("models", {})
    selected = model_defs.get(model_name, {"provider": model_name, "name": model_name})
    provider = selected.get("provider", model_name)

    if provider == "mock" or model_name == "mock":
        return MockModel(
            name=selected.get("name") or model_name,
            behavior=selected.get("behavior", "valid"),
        )

    if provider in {"openai_compatible", "vllm"}:
        endpoint = selected.get("endpoint")
        model_id = selected.get("model") or selected.get("model_id")
        if not endpoint:
            raise ValueError(f"Model '{model_name}' requires an OpenAI-compatible endpoint.")
        if not model_id:
            raise ValueError(f"Model '{model_name}' requires a concrete model/model_id in configs/model_config.yaml.")
        return OpenAICompatibleModel(
            name=selected.get("name") or model_id,
            endpoint=endpoint,
            model_id=model_id,
            api_key_env=selected.get("api_key_env"),
            api_key=selected.get("api_key"),
            temperature=float(selected.get("temperature", 0.0)),
            max_tokens=selected.get("max_tokens"),
            timeout_seconds=int(selected.get("timeout_seconds", 180)),
            response_format_json=bool(selected.get("response_format_json", False)),
        )

    if provider in {"anthropic_compatible", "gemini_compatible", "langchain"}:
        return LangChainModel(name=selected.get("name") or model_name, runnable=selected.get("runnable"))

    raise ValueError(f"Unsupported model provider '{provider}' for model '{model_name}'.")
