from __future__ import annotations

import json
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LITELLM_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "inference"
    / "litellm_fable5_gpt56sol_hy3_local.yaml"
)
JUDGE_CONFIGS = {
    "claude-fable-5": PROJECT_ROOT
    / "configs"
    / "models"
    / "claude_fable5_litellm_local_judge.json",
    "gpt-5.6-sol": PROJECT_ROOT
    / "configs"
    / "models"
    / "gpt5_6_sol_litellm_local_judge.json",
    "hy3": PROJECT_ROOT
    / "configs"
    / "models"
    / "hy3_litellm_local_judge.json",
}
LOCAL_LAUNCHER = (
    PROJECT_ROOT / "Support" / "bash" / "local" / "run_litellm_judge_proxy.sh"
)


def test_local_litellm_config_has_only_the_three_requested_models() -> None:
    config = yaml.safe_load(LITELLM_CONFIG.read_text())
    deployments = config["model_list"]
    assert [item["model_name"] for item in deployments] == list(JUDGE_CONFIGS)
    assert config["litellm_settings"]["telemetry"] is False
    assert config["general_settings"]["master_key"] == "os.environ/LITELLM_MASTER_KEY"


def test_local_litellm_config_references_credentials_only_by_environment() -> None:
    config = yaml.safe_load(LITELLM_CONFIG.read_text())
    expected_key_envs = {
        "claude-fable-5": "os.environ/OPENAPI_API_KEY",
        "gpt-5.6-sol": "os.environ/OPENAPI_GPT_KEY",
        "hy3": "os.environ/HY3_RESOLVED_API_KEY",
    }
    for deployment in config["model_list"]:
        name = deployment["model_name"]
        assert deployment["litellm_params"]["api_key"] == expected_key_envs[name]

    raw = LITELLM_CONFIG.read_text()
    assert "provider=" not in raw
    assert "x-api-key:" not in raw.lower()


def test_layout_judge_configs_match_local_litellm_aliases() -> None:
    for model, path in JUDGE_CONFIGS.items():
        config = json.loads(path.read_text())
        assert config["provider"] == "openai_compatible"
        assert config["endpoint"] == "http://127.0.0.1:4000/v1"
        assert config["model"] == model
        assert config["api_key_env"] == "LITELLM_MASTER_KEY"
        assert "api_key" not in config
        assert config["send_temperature"] is False
        assert config["response_format_json"] is False
        assert config["max_images"] == 4


def test_local_launcher_disables_unrelated_remote_cost_map_fetch() -> None:
    launcher = LOCAL_LAUNCHER.read_text()
    assert "LITELLM_MODE=PRODUCTION" in launcher
    assert "LITELLM_LOCAL_MODEL_COST_MAP=True" in launcher
    assert '--host "$HOST"' in launcher


def test_local_launcher_keeps_hy3_gateway_and_tokenhub_credentials_separate() -> None:
    launcher = LOCAL_LAUNCHER.read_text()
    assert 'HY3_ROUTE="${HY3_ROUTE:-tokenhub}"' in launcher
    assert 'HY3_RESOLVED_API_KEY="$OPENAPI_HY3_KEY"' in launcher
    assert 'HY3_RESOLVED_API_KEY="$TOKENHUB_API_KEY"' in launcher
    assert 'HY3_RESOLVED_API_BASE="${OPENAPI_BASE_URL%/}"' in launcher
    assert 'HY3_RESOLVED_API_BASE="https://tokenhub.tencentmaas.com"' in launcher
