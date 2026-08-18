from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmark.scene_generation import LocalSceneEvalHy4Generator, SceneGenerator


RUNNER_ROOT = (
    Path(__file__).resolve().parents[1] / "tools" / "sceneeval_hy4_runner_v050"
)
if str(RUNNER_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNNER_ROOT))

import sceneeval_local_multimodel as local_multimodel  # noqa: E402


def test_frozen_runner_and_human100_input_are_available() -> None:
    generator = LocalSceneEvalHy4Generator()

    verified = generator.validate_installation()
    batch = generator.validate_input(generator.default_input_path)

    assert len(verified) == 16
    assert batch.scene_count == 100
    assert batch.first_scene_id == 0
    assert batch.last_scene_id == 99
    assert batch.sha256 == "4f632141aa5539e33a6e5f9b63c26f7d66a1882c48c07ba0fd5036af74fa892f"
    assert isinstance(generator, SceneGenerator)


def test_single_scene_interface_delegates_to_frozen_runner(tmp_path: Path) -> None:
    generator = LocalSceneEvalHy4Generator()
    modules = generator._modules()
    frozen_result = SimpleNamespace(
        scene_id=7,
        status="captured",
        attempt_count=2,
        stop_batch=False,
    )

    with (
        patch.object(modules["client_config"], "load_run_config", return_value="config"),
        patch.object(modules["runner"], "run_scene", return_value=frozen_result) as run_scene,
    ):
        result = generator.run_scene(
            scene_id=7,
            description="Place a chair in the room.",
            output_root=tmp_path / "single",
        )

    scene = run_scene.call_args.args[0]
    assert scene.scene_id == 7
    assert scene.description == "Place a chair in the room."
    assert run_scene.call_args.args[2] == "config"
    assert result.scene_id == 7
    assert result.status == "captured"
    assert result.attempt_count == 2
    assert result.stop_batch is False


def test_batch_interface_delegates_to_frozen_runner(tmp_path: Path) -> None:
    generator = LocalSceneEvalHy4Generator()
    modules = generator._modules()
    output_root = tmp_path / "batch"

    with (
        patch.object(modules["client_config"], "load_run_config", return_value="config"),
        patch.object(
            modules["runner"],
            "run_batch",
            return_value=({"captured": 100}, False),
        ) as run_batch,
    ):
        result = generator.run_batch(
            input_jsonl=generator.default_input_path,
            output_root=output_root,
            resume=True,
        )

    assert len(run_batch.call_args.args[0].scenes) == 100
    assert run_batch.call_args.args[1] == output_root.resolve()
    assert run_batch.call_args.args[2] == "config"
    assert run_batch.call_args.kwargs == {"resume": True}
    assert result.summary == {"captured": 100}
    assert result.stopped is False
    assert result.output_root == output_root.resolve()


def test_local_multimodel_order_and_exact_input() -> None:
    batch = local_multimodel.validate_exact_input()

    assert local_multimodel.MODEL_ORDER == (
        "claude-opus-4-8",
        "kimi-k3",
        "gpt-5.6-luna",
    )
    assert len(batch.scenes) == 100
    assert batch.sha256 == local_multimodel.EXPECTED_INPUT_SHA256


def test_local_multimodel_request_reuses_exact_prompts_and_is_minimal() -> None:
    description = "Place a chair in the room."
    request = json.loads(
        local_multimodel.request_preview(
            "claude-opus-4-8",
            description,
        )
    )

    assert set(request) == {
        "model",
        "messages",
        "max_tokens",
        "output_config",
        "stream",
        "thinking",
    }
    assert request["model"] == "claude-opus-4-8"
    assert request["messages"][0] == {
        "role": "system",
        "content": local_multimodel.build_system_prompt(),
    }
    assert request["messages"][1] == {
        "role": "user",
        "content": description,
    }
    assert request["thinking"] == {"type": "adaptive"}
    assert request["output_config"] == {"effort": "high"}
    assert "temperature" not in request
    assert "top_k" not in request
    assert "chat_template_kwargs" not in request

    for model in ("kimi-k3", "gpt-5.6-luna"):
        standard_request = json.loads(
            local_multimodel.request_preview(model, description)
        )
        assert standard_request["reasoning_effort"] == "high"
        assert "thinking" not in standard_request
        assert "output_config" not in standard_request


def test_runtime_api_key_is_only_in_wire_headers() -> None:
    secret = "unit-test-secret-never-persist"
    captured: dict[str, object] = {}

    def fake_wire_post_once(
        endpoint: str,
        request_body: bytes,
        *,
        connect_timeout: float,
        read_timeout: float,
        request_headers: dict[str, str],
    ) -> str:
        captured.update(
            endpoint=endpoint,
            request_body=request_body,
            request_headers=request_headers,
        )
        return "ok"

    config = local_multimodel._make_config(
        model="kimi-k3",
        base_url=local_multimodel.DEFAULT_BASE_URL,
        timeout_seconds=1800.0,
        max_retries=2,
        max_tokens=65536,
    )
    artifact_headers = local_multimodel._redacted_request_headers(
        config,
        "local-attempt-id",
    )
    with patch.object(local_multimodel, "wire_post_once", fake_wire_post_once):
        transport = local_multimodel._transport_with_runtime_secret(secret)
        assert (
            transport(
                config.endpoint,
                b"{}",
                connect_timeout=1.0,
                read_timeout=1.0,
                request_headers=artifact_headers,
            )
            == "ok"
        )

    assert artifact_headers["Authorization"] == local_multimodel.REDACTED_BEARER
    assert secret not in json.dumps(config.public_dict())
    wire_headers = captured["request_headers"]
    assert isinstance(wire_headers, dict)
    assert wire_headers["Authorization"] == f"Bearer {secret}"
    assert "SessionID" not in wire_headers


def test_standard_usage_without_reasoning_details_is_supported() -> None:
    assert local_multimodel._visible_output_tokens(
        {"usage": {"completion_tokens": 37}}
    ) == (37, 37, 0)


def test_ordered_suite_invokes_models_serially_and_never_serializes_key(
    tmp_path: Path,
) -> None:
    secret = "suite-secret-never-write"
    seen: list[str] = []
    original_request_value = local_multimodel.frozen_runner._request_value

    def fake_run_batch(batch, output_root, config, *, resume):
        assert len(batch.scenes) == 100
        assert resume is False
        assert config.api_key == "<runtime-secret-redacted>"
        assert local_multimodel.frozen_runner._request_value is local_multimodel._request_value
        seen.append(config.wire_model)
        return ({"captured": 100}, False)

    output_root = tmp_path / "suite"
    with patch.object(
        local_multimodel.frozen_runner,
        "run_batch",
        side_effect=fake_run_batch,
    ):
        results = local_multimodel.run_ordered_suite(
            api_key=secret,
            output_root=output_root,
        )

    assert tuple(seen) == local_multimodel.MODEL_ORDER
    assert tuple(results) == local_multimodel.MODEL_ORDER
    assert local_multimodel.frozen_runner._request_value is original_request_value
    artifact_text = "\n".join(
        path.read_text("utf-8") for path in output_root.rglob("*.json")
    )
    assert secret not in artifact_text


def test_smoke_probe_uses_one_exact_scene_for_each_model(tmp_path: Path) -> None:
    secret = "probe-secret-never-write"
    seen: list[tuple[str, int]] = []

    def fake_initialize(output_root, batch, config):
        output_root.mkdir(parents=True)

    def fake_run_scene(scene, output_root, config):
        seen.append((config.wire_model, scene.scene_id))
        return SimpleNamespace(
            scene_id=scene.scene_id,
            status="captured",
            attempt_count=1,
            stop_batch=False,
        )

    output_root = tmp_path / "probe"
    with (
        patch.object(
            local_multimodel.frozen_runner,
            "initialize_run",
            side_effect=fake_initialize,
        ),
        patch.object(
            local_multimodel.frozen_runner,
            "run_scene",
            side_effect=fake_run_scene,
        ),
    ):
        results = local_multimodel.run_smoke_probe(
            api_key=secret,
            output_root=output_root,
            scene_id=7,
        )

    assert seen == [(model, 7) for model in local_multimodel.MODEL_ORDER]
    assert tuple(results) == local_multimodel.MODEL_ORDER
    assert all(item["status"] == "captured" for item in results.values())
    artifact_text = "\n".join(
        path.read_text("utf-8") for path in output_root.rglob("*.json")
    )
    assert secret not in artifact_text
