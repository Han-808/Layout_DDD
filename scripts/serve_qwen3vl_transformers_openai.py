from __future__ import annotations

import argparse
import base64
import io
import time
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor


app = FastAPI()
processor: Qwen3VLProcessor | None = None
model: Qwen3VLForConditionalGeneration | None = None
model_id = ""


@app.get("/v1/models")
def list_models() -> dict:
    return {"object": "list", "data": [{"id": model_id, "object": "model"}]}


@app.post("/v1/chat/completions")
def chat_completions(payload: dict[str, Any]) -> dict:
    if processor is None or model is None:
        raise HTTPException(status_code=503, detail="model is not loaded")

    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="messages must be a list")

    max_tokens = int(payload.get("max_tokens") or 1024)
    qwen_messages, images = _convert_messages(messages)

    try:
        prompt = processor.apply_chat_template(qwen_messages, tokenize=False, add_generation_prompt=True)
        processor_kwargs: dict[str, Any] = {"text": [prompt], "return_tensors": "pt"}
        if images:
            processor_kwargs["images"] = images
        inputs = processor(**processor_kwargs).to(model.device)
        input_len = int(inputs["input_ids"].shape[-1])
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        new_tokens = output_ids[:, input_len:]
        content = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
    except torch.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=f"CUDA out of memory: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    now = int(time.time())
    return {
        "id": f"chatcmpl-{now}",
        "object": "chat.completion",
        "created": now,
        "model": payload.get("model") or model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": input_len,
            "completion_tokens": int(new_tokens.shape[-1]),
            "total_tokens": int(input_len + new_tokens.shape[-1]),
        },
    }


def _convert_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    qwen_messages: list[dict[str, Any]] = []
    images: list[Image.Image] = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = message.get("content", "")
        if isinstance(content, str):
            qwen_messages.append({"role": role, "content": [{"type": "text", "text": content}]})
            continue
        if not isinstance(content, list):
            qwen_messages.append({"role": role, "content": [{"type": "text", "text": str(content)}]})
            continue

        converted_content: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                converted_content.append({"type": "text", "text": str(item)})
                continue
            item_type = item.get("type")
            if item_type == "text":
                converted_content.append({"type": "text", "text": str(item.get("text") or "")})
            elif item_type == "image_url":
                image = _image_from_openai_item(item)
                if image is not None:
                    images.append(image)
                    converted_content.append({"type": "image"})
            elif item_type == "image":
                image = _image_from_openai_item(item)
                if image is not None:
                    images.append(image)
                    converted_content.append({"type": "image"})
            else:
                converted_content.append({"type": "text", "text": str(item)})
        qwen_messages.append({"role": role, "content": converted_content})
    return qwen_messages, images


def _image_from_openai_item(item: dict[str, Any]) -> Image.Image | None:
    image_url = item.get("image_url")
    url = image_url.get("url") if isinstance(image_url, dict) else image_url
    if url is None:
        url = item.get("image")
    if not isinstance(url, str):
        return None
    if not url.startswith("data:image/"):
        return None
    _, _, encoded = url.partition(",")
    if not encoded:
        return None
    raw = base64.b64decode(encoded)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny OpenAI-compatible Qwen3-VL transformers server.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "eager"])
    return parser.parse_args()


def main() -> None:
    global processor, model, model_id

    args = parse_args()
    model_id = args.model_path
    processor = Qwen3VLProcessor.from_pretrained(args.model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=torch.float32,
        device_map="auto",
        attn_implementation=args.attn_implementation,
    ).eval()
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
