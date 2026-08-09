from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_vlm_evidence_viewer.py"
SPEC = importlib.util.spec_from_file_location(
    "run_vlm_evidence_viewer",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_viewer_runner_uses_a_loopback_url_and_scene_fragment() -> None:
    assert RUNNER.viewer_url(
        "127.0.0.1",
        8765,
        scene="N021",
    ) == "http://127.0.0.1:8765/index.html#N021"
    assert RUNNER.viewer_url(
        "::1",
        8765,
        scene=None,
    ) == "http://[::1]:8765/index.html"


def test_viewer_runner_rejects_non_loopback_host() -> None:
    parser = argparse.ArgumentParser()
    with pytest.raises(SystemExit):
        RUNNER.validate_loopback_host(parser, "0.0.0.0")


def test_viewer_runner_rebuilds_when_persisted_reports_change(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    manifest = run_root / "run_manifest.json"
    manifest.write_text(
        json.dumps({"status": "running"}),
        encoding="utf-8",
    )
    bundle_dir = run_root / "viewer_bundle"
    runtime = RUNNER.ViewerRuntime(
        run_root=run_root,
        bundle_dir=bundle_dir,
        watch=True,
    )

    assert runtime.rebuild(force=True) is True
    assert (bundle_dir / "index.html").is_file()
    assert runtime.rebuild() is False

    manifest.write_text(
        json.dumps({"status": "complete", "cases": []}),
        encoding="utf-8",
    )

    assert runtime.rebuild() is True
    document = (bundle_dir / "index.html").read_text(encoding="utf-8")
    assert "<span>complete</span>" in document
