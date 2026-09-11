"""Thin local interface over the frozen SceneEval HY4 v0.5.0 runner.

The implementation deliberately delegates prompt construction, request bytes,
retry behavior, transport semantics, and artifact persistence to the vendored
runner. This module provides an importable interface; it is not a second runner.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from benchmark.scene_generation.interfaces import (
    BatchGenerationResult,
    InputBatchInfo,
    SceneGenerationResult,
)


class FrozenRunnerError(RuntimeError):
    """Raised when the local frozen runner is missing or has changed."""


FROZEN_FILE_SHA256 = {
    "run.py": "288f688433dc0a8f860e79b19e0bf86f6ce5d66b1ac953ba89b32aaf3f2546d9",
    "sceneeval_hy4/__init__.py": "188e9c6beacbfeba7c729da634be85a54b7f644fe60812d348c4524812a365ad",
    "sceneeval_hy4/artifacts.py": "d9784a6a780ef187026ba4b64da9d978b80ed271fd0b2b0e0c4c4b5a6ddd4738",
    "sceneeval_hy4/cli.py": "bbc844c0acee593156e7001481c7312a986dccd5b1a26831ed178686b7028f2b",
    "sceneeval_hy4/client_config.py": "6b5321d28b1da4581c31c67a59ba5d98766f19da09bc5217dee3ea16ae0a7799",
    "sceneeval_hy4/constants.py": "c5282d8522d9545c408413bf8384ca4a293313bb3915677c378f7ffa42ec6e02",
    "sceneeval_hy4/inputs.py": "7aead9e95ed83607b51a563eee4dab3d22ea99b6820ccfe418e2001c190e2385",
    "sceneeval_hy4/layout_schema.py": "339b6e23405f245e15368e432b2b0385674f722003106fd95aa61834c6d0b98f",
    "sceneeval_hy4/prompt.py": "64ffb72951730a5a919e7d16173a5dc8c7bab98dcea17684c3eebefc25e69517",
    "sceneeval_hy4/runner.py": "95326c44ec4a483d1f2185a46236675e8008182a0bd6ef28d1d1e693d3f83b0e",
    "sceneeval_hy4/strict_json.py": "c2e0947c6fbd40f96606311545444489197dcbf34bfeeaab58fb783e23647b40",
    "sceneeval_hy4/transport.py": "26203ee4293824ab393668fbc6e45bfba10f94bb0f017f697a9c33954cc01632",
    "prompt_protocol.txt": "a3106dbbc611a44b8eee39f7f90ce693ed6096bffecb1f420c9b96a97f964ee9",
    "openai_clients.yaml": "a127bd55896fbefb0063fff2e726b625912cdeab7b8dd816f3074e7200c25e61",
    "schema/layout.schema.json": "5b31966b64a0d1244b66c75a204967b74792fd69d4d1ecfa7ab1d47311840816",
    "data/sceneeval_human100_generation.jsonl": "4f632141aa5539e33a6e5f9b63c26f7d66a1882c48c07ba0fd5036af74fa892f",
}


def _default_runner_root() -> Path:
    return Path(__file__).resolve().parents[3] / "tools" / "sceneeval_hy4_runner_v050"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LocalSceneEvalHy4Generator:
    """Importable facade for the exact frozen SceneEval HY4 runner."""

    def __init__(self, runner_root: str | Path | None = None) -> None:
        self.runner_root = Path(runner_root or _default_runner_root()).resolve()

    @property
    def default_input_path(self) -> Path:
        return self.runner_root / "data" / "sceneeval_human100_generation.jsonl"

    def validate_installation(self) -> dict[str, str]:
        """Verify every behavior-bearing source/config/input byte."""

        verified: dict[str, str] = {}
        for relative, expected in FROZEN_FILE_SHA256.items():
            path = self.runner_root / relative
            if not path.is_file():
                raise FrozenRunnerError(f"frozen runner file is missing: {path}")
            actual = _sha256(path)
            if actual != expected:
                raise FrozenRunnerError(
                    f"frozen runner file changed: {relative}; "
                    f"expected {expected}, received {actual}"
                )
            verified[relative] = actual
        return verified

    def validate_input(self, input_jsonl: str | Path) -> InputBatchInfo:
        modules = self._modules()
        path = Path(input_jsonl).expanduser().resolve()
        batch = modules["inputs"].load_human100_jsonl(path)
        return InputBatchInfo(
            path=path,
            sha256=batch.sha256,
            scene_count=len(batch.scenes),
            first_scene_id=batch.scenes[0].scene_id,
            last_scene_id=batch.scenes[-1].scene_id,
        )

    def run_scene(
        self,
        *,
        scene_id: int,
        description: str,
        output_root: str | Path,
    ) -> SceneGenerationResult:
        modules = self._modules()
        if isinstance(scene_id, bool) or not isinstance(scene_id, int):
            raise TypeError("scene_id must be an integer")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("description must be non-empty text")
        root = Path(output_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        scene = modules["inputs"].SceneInput(
            scene_id=scene_id,
            description=description,
        )
        config = modules["client_config"].load_run_config()
        result = modules["runner"].run_scene(scene, root, config)
        return SceneGenerationResult(
            scene_id=result.scene_id,
            status=result.status,
            attempt_count=result.attempt_count,
            stop_batch=result.stop_batch,
            output_root=root,
        )

    def run_batch(
        self,
        *,
        input_jsonl: str | Path,
        output_root: str | Path,
        resume: bool = False,
    ) -> BatchGenerationResult:
        modules = self._modules()
        input_path = Path(input_jsonl).expanduser().resolve()
        root = Path(output_root).expanduser().resolve()
        batch = modules["inputs"].load_human100_jsonl(input_path)
        config = modules["client_config"].load_run_config()
        summary, stopped = modules["runner"].run_batch(
            batch,
            root,
            config,
            resume=bool(resume),
        )
        return BatchGenerationResult(
            summary=summary,
            stopped=stopped,
            output_root=root,
        )

    def _modules(self) -> dict[str, ModuleType]:
        self.validate_installation()
        root_text = str(self.runner_root)
        existing = sys.modules.get("sceneeval_hy4")
        if existing is None:
            sys.path.insert(0, root_text)
            importlib.invalidate_caches()
            package = importlib.import_module("sceneeval_hy4")
        else:
            package = existing
        package_root = Path(package.__file__ or "").resolve().parent.parent
        if package_root != self.runner_root:
            raise FrozenRunnerError(
                "a different sceneeval_hy4 package is already imported: "
                f"{package_root}"
            )
        return {
            "inputs": importlib.import_module("sceneeval_hy4.inputs"),
            "client_config": importlib.import_module("sceneeval_hy4.client_config"),
            "runner": importlib.import_module("sceneeval_hy4.runner"),
        }
