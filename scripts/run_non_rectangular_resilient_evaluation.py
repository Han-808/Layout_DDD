#!/usr/bin/env python3
"""Materialize and evaluate completed nonrect model cohorts resiliently."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import threading
import time
import urllib.request


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmark.non_rectangular.resilient import (  # noqa: E402
    NoAPIMockEvaluatorFactory,
    NoAPIMockMaterializer,
    ResilientCampaignConfig,
    ResilientCampaignError,
    run_resilient_nonrect_campaign,
)
from benchmark.non_rectangular.runtime import (  # noqa: E402
    DefaultNonRectangularRuntimeFactory,
)


DEFAULT_BLENDER = Path("/Applications/Blender.app/Contents/MacOS/Blender")
DEFAULT_API2_PORT = 4010
DEFAULT_API2_PROXY_LAUNCHER = Path(
    "/Users/han_mohan/Desktop/Layout_DDD/Support/bash/local/"
    "run_litellm_gpt56sol_standard_proxy.sh"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-root",
        action="append",
        required=True,
        metavar="MODEL=PATH",
        help=(
            "Completed or partial generation model root. Repeat in desired "
            "model order. Each root contains scene directories."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help=(
            "Evaluation-only output root, conventionally "
            "Support/artifacts/outputs/non_rectangular_evaluation/<campaign-id>."
        ),
    )
    parser.add_argument("--asset-csv", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--catalog-snapshot-id", required=True)
    parser.add_argument("--blender-bin", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument(
        "--api2-gpt56",
        action="store_true",
        help=(
            "Own an API2 Azure-standard GPT-5.6-Sol proxy, prompting for the "
            "APP_ID:APP_KEY credential when STANDARD_API_CREDENTIAL is unset."
        ),
    )
    parser.add_argument("--port", type=_positive_int, default=DEFAULT_API2_PORT)
    parser.add_argument(
        "--proxy-launcher",
        type=Path,
        default=DEFAULT_API2_PROXY_LAUNCHER,
    )
    parser.add_argument(
        "--mock-no-api",
        action="store_true",
        help="Run materialization/evaluation with deterministic test doubles.",
    )
    parser.add_argument("--max-workers", type=_positive_int, default=3)
    parser.add_argument(
        "--max-materialization-attempts",
        type=_positive_int,
        default=3,
    )
    parser.add_argument("--max-room-attempts", type=_positive_int, default=3)
    parser.add_argument(
        "--api-max-retries",
        type=int,
        choices=(5,),
        default=5,
        help="Fixed exact-request retries after the initial logical API attempt.",
    )
    parser.add_argument(
        "--materialization-timeout-seconds",
        type=_positive_int,
        default=900,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--recover-interrupted",
        action="store_true",
        help=(
            "Explicitly allow fresh immutable attempts after ambiguous partial "
            "attempt directories; partials remain preserved."
        ),
    )
    args = parser.parse_args(argv)
    runtime_modes = sum(
        bool(value)
        for value in (
            args.mock_no_api,
            args.api2_gpt56,
            args.runtime_config is not None,
        )
    )
    if runtime_modes != 1:
        parser.error(
            "select exactly one of --api2-gpt56, --runtime-config, or "
            "--mock-no-api"
        )
    return args


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _model_roots(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for raw in values:
        model, separator, path = raw.partition("=")
        model = model.strip()
        path = path.strip()
        if separator != "=" or not model or not path:
            raise ResilientCampaignError(
                "--model-root values must use MODEL=PATH"
            )
        if model in result:
            raise ResilientCampaignError(f"duplicate model root: {model}")
        result[model] = Path(path)
    return result


def _load_runtime(path: Path) -> dict:
    try:
        value = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ResilientCampaignError("cannot read runtime config JSON") from exc
    if not isinstance(value, dict):
        raise ResilientCampaignError("runtime config must be a JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_roots = _model_roots(args.model_root)
    config = ResilientCampaignConfig.create(
        model_roots=model_roots,
        output_root=args.output_root,
        asset_csv=args.asset_csv,
        asset_root=args.asset_root,
        catalog_snapshot_id=args.catalog_snapshot_id,
        blender_bin=args.blender_bin,
        max_workers=args.max_workers,
        max_materialization_attempts=args.max_materialization_attempts,
        max_room_attempts=args.max_room_attempts,
        api_max_retries=args.api_max_retries,
        materialization_timeout_seconds=args.materialization_timeout_seconds,
        resume=args.resume,
        recover_interrupted=args.recover_interrupted,
    )
    if args.mock_no_api:
        factory = NoAPIMockEvaluatorFactory()
        materializer = NoAPIMockMaterializer()
        result = run_resilient_nonrect_campaign(
            config,
            evaluator_factory=factory,
            materializer_backend=materializer,
        )
    elif args.api2_gpt56:
        proxy = _OwnedAPI2GPT56Proxy(
            port=args.port,
            launcher=args.proxy_launcher,
        )
        try:
            proxy.start()
            factory = _ProxyEnsuringFactory(
                proxy=proxy,
                delegate=DefaultNonRectangularRuntimeFactory(
                    proxy.runtime_config(blender_bin=args.blender_bin)
                ),
            )
            result = run_resilient_nonrect_campaign(
                config,
                evaluator_factory=factory,
            )
        finally:
            proxy.stop()
    else:
        runtime = _load_runtime(args.runtime_config)
        runtime.setdefault("blender_bin", str(args.blender_bin.expanduser().resolve()))
        factory = DefaultNonRectangularRuntimeFactory(runtime)
        result = run_resilient_nonrect_campaign(
            config,
            evaluator_factory=factory,
        )
    print(json.dumps(result.public_dict(), indent=2, sort_keys=True))
    return 0 if result.status == "complete" else 2


class _ProxyEnsuringFactory:
    def __init__(self, *, proxy, delegate) -> None:
        self.proxy = proxy
        self.delegate = delegate

    def identity(self):
        return self.delegate.identity()

    def usage_totals(self):
        return self.delegate.usage_totals()

    def build(self, context):
        self.proxy.ensure()
        return self.delegate.build(context)


class _OwnedAPI2GPT56Proxy:
    def __init__(self, *, port: int, launcher: Path) -> None:
        self.port = int(port)
        self.launcher = launcher.expanduser().resolve()
        self.endpoint = f"http://127.0.0.1:{self.port}/v1"
        self.master_key = secrets.token_hex(32)
        self.credential = ""
        self.process: subprocess.Popen[bytes] | None = None
        self.lock = threading.Lock()
        self.log_path = Path(
            f"/private/tmp/layoutddd-nonrect-api2-gpt56-{self.port}.log"
        )
        self._prior_master_key = os.environ.get("LITELLM_MASTER_KEY")

    def start(self) -> None:
        if not self.launcher.is_file():
            raise ResilientCampaignError(
                f"API2 proxy launcher is unavailable: {self.launcher}"
            )
        credential = os.environ.get("STANDARD_API_CREDENTIAL", "").strip()
        if not credential:
            credential = getpass.getpass(
                "API2 Azure-standard credential (APP_ID:APP_KEY; hidden): "
            ).strip()
        credential = credential.split("?", 1)[0]
        if not credential or ":" not in credential:
            raise ResilientCampaignError(
                "API2 credential must have APP_ID:APP_KEY form"
            )
        self.credential = credential
        os.environ["LITELLM_MASTER_KEY"] = self.master_key
        self.ensure()

    def ensure(self) -> None:
        with self.lock:
            if (
                self.process is not None
                and self.process.poll() is None
                and self._ready()
            ):
                return
            self._terminate()
            if not _port_available(self.port):
                raise ResilientCampaignError(
                    f"API2 proxy port {self.port} is occupied"
                )
            child_env = os.environ.copy()
            child_env.update(
                {
                    "STANDARD_API_BASE_URL": "http://llm-api.model-eval.woa.com",
                    "STANDARD_API_BASE_URL_V1": (
                        "http://llm-api.model-eval.woa.com/v1"
                    ),
                    "STANDARD_API_CREDENTIAL": self.credential,
                    "LITELLM_MASTER_KEY": self.master_key,
                    "LITELLM_GPT56_STANDARD_PORT": str(self.port),
                }
            )
            with self.log_path.open("ab") as log:
                self.process = subprocess.Popen(
                    [str(self.launcher)],
                    cwd=self.launcher.parents[3],
                    env=child_env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            for _ in range(90):
                if self.process.poll() is not None:
                    raise ResilientCampaignError(
                        f"API2 proxy exited; inspect {self.log_path}"
                    )
                if self._ready():
                    return
                time.sleep(1.0)
            self._terminate()
            raise ResilientCampaignError(
                f"API2 proxy did not become ready; inspect {self.log_path}"
            )

    def stop(self) -> None:
        with self.lock:
            self._terminate()
        self.credential = ""
        if self._prior_master_key is None:
            os.environ.pop("LITELLM_MASTER_KEY", None)
        else:
            os.environ["LITELLM_MASTER_KEY"] = self._prior_master_key

    def runtime_config(self, *, blender_bin: Path) -> dict:
        return {
            "provider": "api2",
            "blender_bin": str(blender_bin.expanduser().resolve()),
            "judge": {
                "name": "nonrect-api2-gpt56-judge",
                "endpoint": self.endpoint,
                "model": "gpt-5.6-sol",
                "api_key_env": "LITELLM_MASTER_KEY",
                "timeout_seconds": 3000,
                "max_tokens": 8192,
                "context_length": 114688,
                "max_images": 8,
                "response_format_json": False,
                "send_temperature": False,
                "retry_backoff_seconds": 30.0,
                "min_request_interval_seconds": 1.0,
            },
            "camera": {
                "mode": "visibility_ranked",
                "max_views": 4,
                "max_steps": 3,
                "candidate_count": 6,
                "active_repair": False,
                "metric_modes": {},
            },
            "renderer": {
                "timeout_seconds": 1800,
                "width": 768,
                "height": 768,
                "render_engine": "BLENDER_EEVEE_NEXT",
                "preview_width": 256,
                "preview_height": 256,
                "require_asset_mesh": True,
            },
            "l1_config": {},
            "scene_quality_config": {},
            "evaluation_profile": {},
            "api2_proxy": {
                "launcher_sha256": _sha256_file(self.launcher),
                "port": self.port,
                "upstream": "api_azure_openai_gpt-5.6-sol",
            },
        }

    def _ready(self) -> bool:
        request = urllib.request.Request(
            self.endpoint.rstrip("/") + "/models",
            headers={"Authorization": f"Bearer {self.master_key}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=2.0) as response:
                if response.status != 200:
                    return False
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            return False
        rows = payload.get("data") if isinstance(payload, dict) else None
        return isinstance(rows, list) and any(
            str(item.get("id") or "") == "gpt-5.6-sol"
            for item in rows
            if isinstance(item, dict)
        )

    def _terminate(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)


def _port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ResilientCampaignError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
