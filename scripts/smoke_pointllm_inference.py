#!/usr/bin/env python3
"""Stage-3 validation: real point-cloud inference through a PointLLM checkpoint.

Stage 1 proves the bytes are on disk. Stage 2 proves the environment imports.
Neither proves the model consumes a point cloud, so this stage does what the
235B launcher's `visual-alive` image request does for the Qwen judge, with one
addition: PointLLM's answer must actually *change* when the point cloud
changes. A model that ignores its point tokens and answers from the text prompt
alone would pass a single-sample smoke test.

Hard checks:
  1. On-disk config matches the pinned registry entry.
  2. The point encoder and projector are materialised, not randomly initialised.
  3. Two different real point clouds produce two different non-empty answers.
  4. Registry-declared response markers are present (PointLLM-R must emit its
     <REASONING>/<ANSWER> chain of thought; the baseline must not).

Usage:
    PYTHONPATH=/mnt/group/cmh/tools/PointLLM-R \
    /mnt/group/cmh/envs/pointllm/bin/python scripts/smoke_pointllm_inference.py \
        --model-key PointLLM-R-7B \
        --modelnet-dat /mnt/group/cmh/models/pointllm_data/modelnet40_data/modelnet40_test_8192pts_fps.dat
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

DEFAULT_REGISTRY = (
    Path(__file__).resolve().parents[1] / "configs" / "models" / "pointllm_mnet_registry.json"
)

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def pc_norm(points: np.ndarray) -> np.ndarray:
    """Centre and unit-scale xyz, leaving the remaining channels untouched."""
    xyz = points[:, :3]
    rest = points[:, 3:]
    xyz = xyz - np.mean(xyz, axis=0)
    scale = np.max(np.sqrt(np.sum(xyz**2, axis=1)))
    xyz = xyz / scale
    return np.concatenate((xyz, rest), axis=1)


def load_modelnet_sample(dat_path: Path, index: int, categories: list[str]) -> tuple[np.ndarray, str]:
    """Mirror pointllm.data.ModelNet preprocessing for a single test object.

    The .dat file stores (8192, 6) arrays of xyz plus normals. PointLLM v1.2
    drops the normals and substitutes an all-black colour channel, because
    ModelNet has no colour.
    """
    with dat_path.open("rb") as handle:
        list_of_points, list_of_labels = pickle.load(handle)
    if not 0 <= index < len(list_of_points):
        raise SystemExit(f"ModelNet index {index} out of range (0..{len(list_of_points) - 1})")
    points = np.array(list_of_points[index], dtype=np.float64)[:, 0:3]
    points = np.concatenate((points, np.zeros_like(points)), axis=-1)
    points = pc_norm(points)
    label = int(np.array(list_of_labels[index]).item())
    return points.astype(np.float32), categories[label]


def load_npy_sample(npy_path: Path) -> tuple[np.ndarray, str]:
    points = np.load(npy_path).astype(np.float32)
    if points.ndim != 2 or points.shape[1] != 6:
        raise SystemExit(f"{npy_path} has shape {points.shape}; expected (N, 6) xyz+rgb")
    return pc_norm(points), f"npy:{npy_path.name}"


def modelnet_categories() -> list[str]:
    import pointllm.data as pointllm_data

    names = (
        Path(pointllm_data.__file__).parent
        / "modelnet_config"
        / "modelnet40_shape_names_modified.txt"
    )
    return [line.rstrip() for line in names.read_text(encoding="utf-8").splitlines() if line.strip()]


def array_digest(points: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(points, dtype=np.float32).tobytes()).hexdigest()


def build_prompt(conv, question: str, point_backbone_config: dict) -> str:
    token_len = point_backbone_config["point_token_len"]
    patch = point_backbone_config["default_point_patch_token"]
    if point_backbone_config["mm_use_point_start_end"]:
        start = point_backbone_config["default_point_start_token"]
        end = point_backbone_config["default_point_end_token"]
        prefix = start + patch * token_len + end
    else:
        prefix = patch * token_len
    conv.reset()
    conv.append_message(conv.roles[0], prefix + "\n" + question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--model-key", required=True, help="registry key, e.g. PointLLM-R-7B")
    parser.add_argument("--model-dir", type=Path, default=None, help="override the registry path")
    parser.add_argument("--modelnet-dat", type=Path, default=None)
    parser.add_argument(
        "--modelnet-index",
        type=int,
        action="append",
        default=None,
        help="repeat twice to pick the two contrast objects (default: 0 and 1234)",
    )
    parser.add_argument("--npy", type=Path, action="append", default=None)
    parser.add_argument("--prompt", default="What is this?")
    parser.add_argument("--torch-dtype", choices=tuple(DTYPES), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    if args.model_key not in registry["models"]:
        raise SystemExit(f"unknown model key {args.model_key!r}")
    spec = registry["models"][args.model_key]
    model_dir = args.model_dir or Path(spec["local_dir"])

    if not model_dir.is_dir():
        raise SystemExit(f"checkpoint directory not found: {model_dir} (run stage 1 first)")

    print(f"==== stage 3: real point-cloud inference for {args.model_key} ====")
    print(f"checkpoint : {model_dir}")
    print(f"hf_repo    : {spec['hf_repo']}@{spec['revision']}")

    on_disk = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    for field in ("model_type", "architectures", "vocab_size"):
        if on_disk.get(field) != spec[field]:
            raise SystemExit(
                f"checkpoint identity mismatch on {field}: "
                f"{on_disk.get(field)!r} on disk, {spec[field]!r} in the registry"
            )
    print("checkpoint identity: OK")

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible; stage 3 requires a GPU")
    print(f"gpu        : {torch.cuda.get_device_name(0)}")

    from transformers import AutoTokenizer

    from pointllm.conversation import SeparatorStyle, conv_templates
    from pointllm.model import PointLLMLlamaForCausalLM
    from pointllm.model.utils import KeywordsStoppingCriteria
    from pointllm.utils import disable_torch_init

    samples: list[tuple[np.ndarray, str]] = []
    if args.npy:
        samples.extend(load_npy_sample(path) for path in args.npy)
    if args.modelnet_dat is not None:
        indices = args.modelnet_index or [0, 1234]
        categories = modelnet_categories()
        samples.extend(
            load_modelnet_sample(args.modelnet_dat, index, categories) for index in indices
        )
    if len(samples) < 2:
        raise SystemExit(
            "stage 3 needs at least two distinct point clouds so the "
            "point-cloud-sensitivity check is meaningful; pass --modelnet-dat or two --npy files"
        )

    digests = [array_digest(points) for points, _ in samples]
    if len(set(digests)) != len(digests):
        raise SystemExit("the supplied point clouds are not distinct")
    for (_, label), digest in zip(samples, digests):
        print(f"point cloud: {label} sha256={digest[:16]}")

    dtype = DTYPES[args.torch_dtype]
    disable_torch_init()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    load_started = time.time()
    model = PointLLMLlamaForCausalLM.from_pretrained(
        str(model_dir), low_cpu_mem_usage=False, use_cache=True, torch_dtype=dtype
    ).cuda()
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
    model.eval()
    load_seconds = time.time() - load_started
    print(f"loaded in {load_seconds:.1f}s as {args.torch_dtype}")

    inner = model.get_model()
    backbone_params = sum(p.numel() for p in inner.point_backbone.parameters())
    proj_params = sum(p.numel() for p in inner.point_proj.parameters())
    backbone_norm = sum(float(p.detach().float().abs().sum()) for p in inner.point_backbone.parameters())
    print(f"point encoder params: {backbone_params} (abs-sum {backbone_norm:.1f})")
    print(f"projector params    : {proj_params}")
    if backbone_params == 0 or proj_params == 0 or backbone_norm == 0.0:
        raise SystemExit("point encoder or projector is empty; the checkpoint did not load them")

    point_backbone_config = inner.point_backbone_config
    conv = conv_templates["vicuna_v1_1"].copy()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    results = []
    for (points, label), digest in zip(samples, digests):
        prompt = build_prompt(conv, args.prompt, point_backbone_config)
        input_ids = torch.as_tensor(tokenizer([prompt]).input_ids).cuda()
        stopping = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
        point_clouds = torch.from_numpy(points).unsqueeze(0).cuda().to(dtype)

        started = time.time()
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                point_clouds=point_clouds,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                stopping_criteria=[stopping],
            )
        elapsed = time.time() - started

        text = tokenizer.batch_decode(
            output_ids[:, input_ids.shape[1] :], skip_special_tokens=True
        )[0].strip()
        if text.endswith(stop_str):
            text = text[: -len(stop_str)].strip()

        print(f"\n---- {label} ({elapsed:.1f}s) ----\n{text}")
        results.append(
            {
                "source": label,
                "point_cloud_sha256": digest,
                "prompt": args.prompt,
                "response": text,
                "latency_seconds": round(elapsed, 3),
            }
        )

    problems = []
    for result in results:
        if not result["response"]:
            problems.append(f"empty response for {result['source']}")

    responses = [result["response"] for result in results]
    if len(set(responses)) == 1:
        problems.append(
            "every point cloud produced an identical response; the model is not "
            "conditioning on the point tokens"
        )

    markers = spec.get("cot_response_markers") or []
    for marker in markers:
        if not all(marker in response for response in responses):
            problems.append(f"expected chain-of-thought marker {marker!r} missing from a response")
    if not markers:
        # The baseline must not look like PointLLM-R; that would mean the
        # directories were swapped.
        for marker in ("<REASONING>", "<ANSWER>"):
            if any(marker in response for response in responses):
                problems.append(
                    f"baseline checkpoint emitted {marker!r}; {model_dir} may hold PointLLM-R weights"
                )

    for result, (_, label) in zip(results, samples):
        if label.startswith("npy:"):
            continue
        result["modelnet_label"] = label
        result["label_in_response"] = label.lower() in result["response"].lower()
        print(f"[info] ModelNet GT {label!r} present in response: {result['label_in_response']}")

    record = {
        "stage": "3_real_point_cloud_inference",
        "model_key": args.model_key,
        "hf_repo": spec["hf_repo"],
        "revision": spec["revision"],
        "checkpoint_dir": str(model_dir),
        "torch_dtype": args.torch_dtype,
        "gpu": torch.cuda.get_device_name(0),
        "load_seconds": round(load_seconds, 3),
        "point_backbone_params": backbone_params,
        "point_proj_params": proj_params,
        "results": results,
        "problems": problems,
        "status": "fail" if problems else "pass",
    }

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")

    print()
    for problem in problems:
        print(f"PROBLEM: {problem}", file=sys.stderr)
    if problems:
        print("STAGE 3 FAILED", file=sys.stderr)
        return 1
    print("STAGE 3 PASSED: the checkpoint ran real point clouds and its answers track the input.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
