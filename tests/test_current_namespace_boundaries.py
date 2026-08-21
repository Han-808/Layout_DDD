from __future__ import annotations

import ast
from pathlib import Path

from benchmark.models import OpenAICompatibleModel


ROOT = Path(__file__).resolve().parents[1]


def _runtime_import_targets(tree: ast.AST) -> set[str]:
    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
            continue
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            targets.add(node.module)
            continue
        if not isinstance(node, ast.Call) or not node.args:
            continue
        function = node.func
        is_dynamic_import = (
            isinstance(function, ast.Name)
            and function.id in {"__import__", "import_module"}
        ) or (
            isinstance(function, ast.Attribute)
            and function.attr == "import_module"
        )
        first_argument = node.args[0]
        if (
            is_dynamic_import
            and isinstance(first_argument, ast.Constant)
            and isinstance(first_argument.value, str)
        ):
            targets.add(first_argument.value)
    return targets


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

    violations: list[str] = []
    for path in current_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if any(
            module == "benchmark.legend"
            or module.startswith("benchmark.legend.")
            for module in _runtime_import_targets(tree)
        ):
            violations.append(path.relative_to(ROOT).as_posix())

    assert violations == []


def test_runtime_import_scanner_catches_constant_dynamic_imports() -> None:
    tree = ast.parse(
        "\n".join(
            [
                "import benchmark.legend.data",
                "from benchmark.legend import workflow",
                "importlib.import_module('benchmark.legend.judge')",
                "import_module('benchmark.legend.metrics')",
                "__import__('benchmark.legend.adapters')",
                "importlib.util.find_spec('benchmark.legend')",
            ]
        )
    )
    assert _runtime_import_targets(tree) == {
        "benchmark.legend.data",
        "benchmark.legend",
        "benchmark.legend.judge",
        "benchmark.legend.metrics",
        "benchmark.legend.adapters",
    }


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
