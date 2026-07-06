from __future__ import annotations

from benchmark.models.base_model import BaseLayoutModel
from benchmark.models.langchain_model import LangChainModel
from benchmark.models.mock_model import MockModel
from benchmark.models.openai_compatible_model import OpenAICompatibleModel


MODEL_ADAPTERS = {
    "mock": MockModel,
    "openai_compatible": OpenAICompatibleModel,
    "vllm": OpenAICompatibleModel,
}


def create_model(model_name: str, model_config: dict | None = None) -> BaseLayoutModel:
    config = model_config or {}
    api_config = config.get("api", {}) if isinstance(config.get("api"), dict) else {}
    model_defs = config.get("models", {})
    selected = model_defs.get(model_name, {"provider": model_name, "name": model_name})
    provider = selected.get("provider", model_name)

    if provider == "mock" or model_name == "mock":
        adapter_cls = MODEL_ADAPTERS["mock"]
        return _attach_model_runtime_config(
            adapter_cls(
                name=selected.get("name") or model_name,
                behavior=selected.get("behavior", "valid"),
                judge_evidence_budgeting=_judge_evidence_budgeting(selected),
            ),
            selected,
        )

    if provider in {"openai_compatible", "vllm"}:
        adapter_cls = MODEL_ADAPTERS[provider]
        endpoint = selected.get("endpoint")
        model_id = selected.get("model") or selected.get("model_id")
        if not endpoint:
            raise ValueError(f"Model '{model_name}' requires an OpenAI-compatible endpoint.")
        if not model_id:
            raise ValueError(f"Model '{model_name}' requires a concrete model/model_id in configs/model_config.yaml.")
        return _attach_model_runtime_config(
            adapter_cls(
                name=selected.get("name") or model_id,
                endpoint=endpoint,
                model_id=model_id,
                api_key_env=selected.get("api_key_env"),
                api_key=selected.get("api_key"),
                temperature=float(selected.get("temperature", 0.0)),
                max_tokens=selected.get("max_tokens"),
                generation_max_tokens=selected.get("generation_max_tokens"),
                repair_max_tokens=selected.get("repair_max_tokens"),
                judge_max_tokens=selected.get("judge_max_tokens"),
                context_length=selected.get("context_length", selected.get("context_limit_tokens")),
                prompt_safety_margin_tokens=int(selected.get("prompt_safety_margin_tokens", 4096)),
                fail_fast_prompt_budget=bool(selected.get("fail_fast_prompt_budget", True)),
                timeout_seconds=int(selected.get("timeout_seconds", 180)),
                response_format_json=bool(selected.get("response_format_json", False)),
                max_retries=int(selected.get("max_retries", api_config.get("max_retries", 0))),
                retry_backoff_seconds=float(selected.get("retry_backoff_seconds", api_config.get("retry_backoff_seconds", 1.0))),
                retry_on_status=list(selected.get("retry_on_status", api_config.get("retry_on_status", [429, 500, 502, 503, 504]))),
                runtime_profile=selected.get("runtime_profile"),
                judge_evidence_budgeting=_judge_evidence_budgeting(selected),
            ),
            selected,
        )

    if provider in {"anthropic_compatible", "gemini_compatible", "langchain"}:
        return _attach_model_runtime_config(
            LangChainModel(name=selected.get("name") or model_name, runnable=selected.get("runnable")),
            selected,
        )

    raise ValueError(f"Unsupported model provider '{provider}' for model '{model_name}'.")


def _judge_evidence_budgeting(model_def: dict) -> bool:
    return bool(model_def.get("judge_evidence_budgeting", False)) if isinstance(model_def, dict) else False


def _attach_model_runtime_config(model: BaseLayoutModel, model_def: dict) -> BaseLayoutModel:
    setattr(model, "judge_evidence_budgeting", _judge_evidence_budgeting(model_def))
    return model
