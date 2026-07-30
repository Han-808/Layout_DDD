from __future__ import annotations

from pathlib import Path

from benchmark.models import OpenAICompatibleModel


ROOT = Path(__file__).resolve().parents[1]


def test_current_runtime_does_not_import_legend_modules() -> None:
    current_files = [ROOT / "generate.py", ROOT / "evaluate.py"]
    current_files.extend(
        path
        for path in (ROOT / "scripts").glob("*.py")
        if path.is_file()
    )
    current_files.extend(
        path
        for path in (ROOT / "src" / "benchmark").rglob("*.py")
        if "legend" not in path.relative_to(ROOT / "src" / "benchmark").parts
    )

    violations = []
    for path in current_files:
        if "benchmark.legend" in path.read_text(encoding="utf-8"):
            violations.append(path.relative_to(ROOT).as_posix())

    assert violations == []


def test_retired_main_namespace_surfaces_are_absent() -> None:
    retired_paths = [
        "schemas/layout.schema.json",
        "schemas/bm_instance.schema.json",
        "schemas/feedback.schema.json",
        "src/benchmark/data",
        "src/benchmark/datasets",
        "src/benchmark/pipeline.py",
        "src/benchmark/workflow",
        "src/benchmark/input_modes.py",
        "src/benchmark/models/base_model.py",
        "src/benchmark/metrics",
        "src/benchmark/visualization",
        "src/benchmark/adapters/manual",
        "src/benchmark/adapters/passthrough",
        "src/benchmark/evaluator/evaluator.py",
        "scripts/run_benchmark.py",
        "scripts/run_single_case.py",
        "scripts/generate_scene.py",
    ]

    present = []
    for path in retired_paths:
        candidate = ROOT / path
        if candidate.is_file() or (candidate.is_dir() and any(candidate.rglob("*.py"))):
            present.append(path)
    assert present == []


def test_current_openai_client_has_no_legend_layout_methods() -> None:
    assert not hasattr(OpenAICompatibleModel, "generate_layout")
    assert not hasattr(OpenAICompatibleModel, "repair_layout")
