from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import date
from pathlib import Path

import pytest

from scripts.check_component_lifecycle import (
    EXPECTED_SCHEMA_VERSION,
    LIFECYCLE_STATES,
    LifecycleError,
    load_registry,
    tracked_test_import_problems,
    validate_registry,
    validate_index_snapshot,
    _tree_content_sha256,
)


def _write(root: Path, relative: str, text: str = "# source\n") -> str:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return relative


def _component(
    component_id: str,
    root: str,
    *,
    state: str = "active",
    tracked_policy: str = "all_tracked",
    entrypoints: list[str] | None = None,
    sources: list[str] | None = None,
    dependencies: list[str] | None = None,
    replacement: str | None = None,
    quarantine_evidence: list[str] | None = None,
    quarantine_started_at: str | None = None,
    minimum_quarantine_days: int | None = None,
    content_sha256: str | None = None,
    deletion_approval_evidence: list[str] | None = None,
) -> dict[str, object]:
    default_source = f"{root}/run.py"
    if state in {"quarantine", "deletion_candidate", "deleted"}:
        quarantine_started_at = quarantine_started_at or "2024-01-01"
        minimum_quarantine_days = (
            7 if minimum_quarantine_days is None else minimum_quarantine_days
        )
    result = {
        "component_id": component_id,
        "root": root,
        "state": state,
        "owner": "test-owner",
        "replacement": replacement,
        "reason": f"Lifecycle reason for {component_id}.",
        "tracked_policy": tracked_policy,
        "entrypoints": entrypoints if entrypoints is not None else [default_source],
        "sources": sources if sources is not None else [default_source],
        "dependencies": dependencies if dependencies is not None else [],
        "quarantine_evidence": (
            quarantine_evidence if quarantine_evidence is not None else []
        ),
    }
    if quarantine_started_at is not None:
        result["quarantine_started_at"] = quarantine_started_at
    if minimum_quarantine_days is not None:
        result["minimum_quarantine_days"] = minimum_quarantine_days
    if content_sha256 is not None:
        result["content_sha256"] = content_sha256
    if deletion_approval_evidence is not None:
        result["deletion_approval_evidence"] = deletion_approval_evidence
    return result


def _registry(*components: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "components": list(components),
        "artifacts": [
            {
                "artifact_id": "registry_v1",
                "path": "configs/repository/component_lifecycle_v1.json",
                "state": "active",
                "owner": "test-owner",
                "reason": "Synthetic lifecycle registry under test.",
            },
            {
                "artifact_id": "active_bundles_v1",
                "path": "configs/runners/active_generation_bundles_v1.json",
                "state": "active",
                "owner": "test-owner",
                "reason": "Synthetic active bundle manifest under test.",
            }
        ],
    }


def _active_manifest(root: str = "pkg", entrypoint: str = "run.py") -> dict[str, object]:
    return {
        "schema_version": "active_generation_bundles_v1",
        "bundles": [
            {
                "bundle_id": "pkg",
                "root": root,
                "entrypoints": [entrypoint],
            }
        ],
    }


def _validate(
    repo: Path,
    registry: dict[str, object],
    tracked: set[str],
    *,
    active_manifest: dict[str, object] | None = None,
    index_modes: dict[str, str] | None = None,
    baseline_registry: dict[str, object] | None = None,
) -> dict[str, object]:
    registry_path = repo / "configs/repository/component_lifecycle_v1.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    resolved_active_manifest = active_manifest or _active_manifest()
    active_path = repo / "configs/runners/active_generation_bundles_v1.json"
    active_path.parent.mkdir(parents=True, exist_ok=True)
    active_path.write_text(json.dumps(resolved_active_manifest), encoding="utf-8")
    tracked = {
        *tracked,
        "configs/repository/component_lifecycle_v1.json",
        "configs/runners/active_generation_bundles_v1.json",
    }
    if index_modes is not None:
        index_modes = {
            **{path: "100644" for path in tracked},
            **index_modes,
        }
    return validate_registry(
        registry,
        repo_root=repo,
        tracked=tracked,
        index_modes=index_modes,
        active_manifest=resolved_active_manifest,
        registry_path=registry_path,
        active_manifest_path=active_path,
        baseline_registry=baseline_registry,
        today=date(2024, 2, 1),
    )


def test_valid_registry_covers_active_bundle_and_reports_every_state(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/run.py")
    summary = _validate(
        tmp_path,
        _registry(_component("pkg", "pkg")),
        {"pkg/run.py"},
    )

    assert summary["component_count"] == 1
    assert summary["active_bundle_count"] == 1
    assert set(summary["state_counts"]) == LIFECYCLE_STATES
    assert summary["state_counts"]["active"] == 1


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("state", "retired", "unknown lifecycle state"),
        ("tracked_policy", "sometimes", "unknown tracked policy"),
        ("root", "pkg/*", "exact normalized repo path"),
        ("owner", "", "non-empty string"),
        ("reason", "", "non-empty string"),
    ],
)
def test_schema_rejects_unknown_or_nonexact_lifecycle_values(
    tmp_path: Path, field: str, value: str, match: str
) -> None:
    _write(tmp_path, "pkg/run.py")
    component = _component("pkg", "pkg")
    component[field] = value
    with pytest.raises(LifecycleError, match=match):
        _validate(tmp_path, _registry(component), {"pkg/run.py"})


def test_component_roots_must_not_overlap(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/run.py")
    _write(tmp_path, "pkg/child/run.py")
    with pytest.raises(LifecycleError, match="component roots overlap"):
        _validate(
            tmp_path,
            _registry(
                _component("parent", "pkg", state="compatibility"),
                _component("child", "pkg/child"),
            ),
            {"pkg/run.py", "pkg/child/run.py"},
            active_manifest=_active_manifest("pkg/child"),
        )


def test_sources_and_entrypoints_must_exist_under_exact_root(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/run.py")
    component = _component(
        "pkg", "pkg", sources=["pkg/missing.py"], entrypoints=["pkg/run.py"]
    )
    with pytest.raises(LifecycleError, match="source/entrypoint does not exist"):
        _validate(tmp_path, _registry(component), {"pkg/run.py"})


def test_tracked_policy_uses_git_index_view_not_file_presence(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/run.py")
    with pytest.raises(LifecycleError, match="requires all files tracked"):
        _validate(tmp_path, _registry(_component("pkg", "pkg")), set())

    local = _component(
        "pkg",
        "pkg",
        state="local_only",
        tracked_policy="no_tracked_files",
    )
    with pytest.raises(LifecycleError, match="requires no tracked files"):
        _validate(
            tmp_path,
            _registry(local),
            {"pkg/run.py"},
            active_manifest={"bundles": []},
        )


def test_absent_local_only_component_is_valid_in_clean_checkout(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/run.py")
    local = _component(
        "local",
        "tools/local_runner",
        state="local_only",
        tracked_policy="no_tracked_files",
        entrypoints=["tools/local_runner/run.py"],
        sources=["tools/local_runner/run.py"],
    )
    summary = _validate(
        tmp_path,
        _registry(_component("pkg", "pkg"), local),
        {"pkg/run.py"},
    )
    assert summary["state_counts"]["local_only"] == 1


def test_active_bundle_requires_exact_root_and_all_declared_entrypoints(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "pkg/run.py")
    component = _component("pkg", "pkg", entrypoints=[])
    with pytest.raises(LifecycleError, match="lack lifecycle coverage"):
        _validate(tmp_path, _registry(component), {"pkg/run.py"})

    with pytest.raises(LifecycleError, match="absent from registry"):
        _validate(
            tmp_path,
            _registry(_component("pkg", "pkg")),
            {"pkg/run.py"},
            active_manifest=_active_manifest("another"),
        )


def test_tracked_test_cannot_import_untracked_tools_module(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/run.py")
    _write(tmp_path, "tools/local_runner/__init__.py")
    _write(tmp_path, "tools/local_runner/runner.py")
    _write(
        tmp_path,
        "tests/test_local_runner.py",
        "from tools.local_runner.runner import run\n",
    )
    local = _component(
        "local",
        "tools/local_runner",
        state="local_only",
        tracked_policy="no_tracked_files",
        entrypoints=["tools/local_runner/runner.py"],
        sources=["tools/local_runner/runner.py"],
    )
    tracked = {"pkg/run.py", "tests/test_local_runner.py"}

    problems = tracked_test_import_problems(repo_root=tmp_path, tracked=tracked)
    assert problems == [
        "tests/test_local_runner.py: tools.local_runner.runner resolves only to "
        "untracked files ['tools/local_runner/runner.py']"
    ]
    with pytest.raises(LifecycleError, match="tracked tests depend on untracked"):
        _validate(
            tmp_path,
            _registry(_component("pkg", "pkg"), local),
            tracked,
        )


def test_tracked_test_cannot_load_untracked_tools_path_dynamically(tmp_path: Path) -> None:
    _write(tmp_path, "tools/local_runner/runner.py")
    _write(
        tmp_path,
        "tests/test_dynamic.py",
        'from pathlib import Path\nTARGET = Path("tools") / "local_runner" / "runner.py"\n',
    )
    problems = tracked_test_import_problems(
        repo_root=tmp_path, tracked={"tests/test_dynamic.py"}
    )
    assert problems == [
        "tests/test_dynamic.py: tools path exists only outside Git index: "
        "tools/local_runner/runner.py"
    ]


def test_active_component_cannot_reach_unsafe_state_transitively(tmp_path: Path) -> None:
    for path in ("pkg/run.py", "compat/run.py", "local/run.py"):
        _write(tmp_path, path)
    registry = _registry(
        _component("pkg", "pkg", dependencies=["compat"]),
        _component(
            "compat",
            "compat",
            state="compatibility",
            replacement="pkg",
            dependencies=["local"],
        ),
        _component(
            "local",
            "local",
            state="local_only",
            tracked_policy="no_tracked_files",
        ),
    )
    with pytest.raises(LifecycleError, match="reaches unsafe local_only dependency"):
        _validate(
            tmp_path,
            registry,
            {"pkg/run.py", "compat/run.py"},
        )


def test_compatibility_component_is_runnable_and_cannot_reach_local_only(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "pkg/run.py")
    _write(tmp_path, "compat/run.py")
    _write(tmp_path, "local/run.py")
    with pytest.raises(LifecycleError, match="runnable component compat"):
        _validate(
            tmp_path,
            _registry(
                _component("pkg", "pkg"),
                _component(
                    "compat",
                    "compat",
                    state="compatibility",
                    replacement="pkg",
                    dependencies=["local"],
                ),
                _component(
                    "local",
                    "local",
                    state="local_only",
                    tracked_policy="no_tracked_files",
                ),
            ),
            {"pkg/run.py", "compat/run.py"},
        )


def test_runnable_literal_path_dependency_is_checked(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "compat/run.py",
        'from pathlib import Path\nTARGET = Path("tools") / "local" / "run.py"\n',
    )
    _write(tmp_path, "successor/run.py")
    _write(tmp_path, "tools/local/run.py")
    with pytest.raises(LifecycleError, match="undeclared path dependency local"):
        _validate(
            tmp_path,
            _registry(
                _component(
                    "compat",
                    "compat",
                    state="compatibility",
                    replacement="successor",
                ),
                _component("successor", "successor"),
                _component(
                    "local",
                    "tools/local",
                    state="local_only",
                    tracked_policy="no_tracked_files",
                ),
            ),
            {"compat/run.py", "successor/run.py"},
            active_manifest=_active_manifest("compat"),
        )


def test_active_component_import_must_have_declared_component_dependency(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "pkg/run.py", "from tools.shared import helper\n")
    _write(tmp_path, "tools/shared/__init__.py")
    registry = _registry(
        _component("pkg", "pkg"),
        _component(
            "shared",
            "tools/shared",
            state="compatibility",
            replacement="pkg",
            entrypoints=["tools/shared/__init__.py"],
            sources=["tools/shared/__init__.py"],
        ),
    )
    with pytest.raises(LifecycleError, match="undeclared import dependency shared"):
        _validate(
            tmp_path,
            registry,
            {"pkg/run.py", "tools/shared/__init__.py"},
        )


def test_frozen_replay_content_is_hash_pinned(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/run.py")
    _write(tmp_path, "legacy/run.py", "VALUE = 1\n")
    tracked = {"pkg/run.py", "legacy/run.py"}
    digest = _tree_content_sha256(
        root="legacy", repo_root=tmp_path, tracked=tracked
    )
    registry = _registry(
        _component("pkg", "pkg"),
        _component(
            "legacy",
            "legacy",
            state="frozen_replay",
            content_sha256=digest,
        ),
    )
    _validate(tmp_path, registry, tracked)
    (tmp_path / "legacy/run.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(LifecycleError, match="content hash mismatch"):
        _validate(tmp_path, registry, tracked)


def test_frozen_component_identity_includes_git_mode(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/run.py")
    _write(tmp_path, "legacy/run.py", "VALUE = 1\n")
    tracked = {"pkg/run.py", "legacy/run.py"}
    digest = _tree_content_sha256(
        root="legacy",
        repo_root=tmp_path,
        tracked=tracked,
        index_modes={"legacy/run.py": "100644"},
    )
    registry = _registry(
        _component("pkg", "pkg"),
        _component(
            "legacy",
            "legacy",
            state="frozen_replay",
            content_sha256=digest,
        ),
    )
    _validate(
        tmp_path,
        registry,
        tracked,
        index_modes={"legacy/run.py": "100644"},
    )
    with pytest.raises(LifecycleError, match="content hash mismatch"):
        _validate(
            tmp_path,
            registry,
            tracked,
            index_modes={"legacy/run.py": "100755"},
        )


def test_frozen_transition_rejects_bytes_and_hash_update(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/run.py")
    _write(tmp_path, "legacy/run.py", "VALUE = 1\n")
    tracked = {"pkg/run.py", "legacy/run.py"}
    original_digest = _tree_content_sha256(
        root="legacy", repo_root=tmp_path, tracked=tracked
    )
    baseline = _registry(
        _component("pkg", "pkg"),
        _component(
            "legacy",
            "legacy",
            state="frozen_replay",
            content_sha256=original_digest,
        ),
    )
    (tmp_path / "legacy/run.py").write_text("VALUE = 2\n", encoding="utf-8")
    changed_digest = _tree_content_sha256(
        root="legacy", repo_root=tmp_path, tracked=tracked
    )
    current = json.loads(json.dumps(baseline))
    changed = current["components"][1]
    assert isinstance(changed, dict)
    changed["content_sha256"] = changed_digest

    with pytest.raises(LifecycleError, match="frozen_replay lifecycle record is immutable"):
        _validate(
            tmp_path,
            current,
            tracked,
            baseline_registry=baseline,
        )


def test_frozen_artifact_identity_includes_git_mode(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/run.py")
    _write(tmp_path, "configs/example_v1.json", "{}\n")
    tracked = {"pkg/run.py", "configs/example_v1.json"}
    registry = _registry(_component("pkg", "pkg"))
    registry["artifacts"].append(
        {
            "artifact_id": "example_v1",
            "path": "configs/example_v1.json",
            "state": "frozen_replay",
            "owner": "test-owner",
            "reason": "Synthetic frozen artifact.",
            "content_sha256": hashlib.sha256(
                (tmp_path / "configs/example_v1.json").read_bytes()
            ).hexdigest(),
            "git_mode": "100644",
        }
    )
    _validate(
        tmp_path,
        registry,
        tracked,
        index_modes={"configs/example_v1.json": "100644"},
    )
    with pytest.raises(LifecycleError, match="git mode mismatch"):
        _validate(
            tmp_path,
            registry,
            tracked,
            index_modes={"configs/example_v1.json": "100755"},
        )


def test_compatibility_component_requires_active_replacement(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/run.py")
    _write(tmp_path, "compat/run.py")
    tracked = {"pkg/run.py", "compat/run.py"}

    missing = _component("compat", "compat", state="compatibility")
    with pytest.raises(LifecycleError, match="requires an active replacement"):
        _validate(
            tmp_path,
            _registry(_component("pkg", "pkg"), missing),
            tracked,
        )

    frozen = _component(
        "pkg",
        "pkg",
        state="frozen_replay",
        content_sha256=_tree_content_sha256(
            root="pkg", repo_root=tmp_path, tracked=tracked
        ),
    )
    nonactive = _component(
        "compat",
        "compat",
        state="compatibility",
        replacement="pkg",
    )
    with pytest.raises(LifecycleError, match="replacement must be active"):
        _validate(
            tmp_path,
            _registry(frozen, nonactive),
            tracked,
            active_manifest=_active_manifest("compat"),
        )

    active = _component("pkg", "pkg")
    valid = _component(
        "compat",
        "compat",
        state="compatibility",
        replacement="pkg",
    )
    summary = _validate(
        tmp_path,
        _registry(active, valid),
        tracked,
    )
    assert summary["state_counts"]["compatibility"] == 1


def test_compatibility_artifact_requires_active_replacement(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/run.py")
    _write(tmp_path, "configs/example_v1.json", "{}\n")
    tracked = {"pkg/run.py", "configs/example_v1.json"}
    registry = _registry(_component("pkg", "pkg"))
    registry["artifacts"].append(
        {
            "artifact_id": "example_v1",
            "path": "configs/example_v1.json",
            "state": "compatibility",
            "owner": "test-owner",
            "reason": "Synthetic compatibility artifact.",
        }
    )
    with pytest.raises(LifecycleError, match="requires an active replacement"):
        _validate(tmp_path, registry, tracked)

    artifact = registry["artifacts"][-1]
    assert isinstance(artifact, dict)
    artifact["replacement"] = "registry_v1"
    summary = _validate(tmp_path, registry, tracked)
    assert summary["artifact_state_counts"]["compatibility"] == 1


def test_versioned_file_requires_artifact_or_component_classification(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "pkg/run.py")
    _write(tmp_path, "configs/example_v1.json", "{}\n")
    tracked = {"pkg/run.py", "configs/example_v1.json"}
    registry = _registry(_component("pkg", "pkg"))
    with pytest.raises(LifecycleError, match="lack lifecycle classification"):
        _validate(tmp_path, registry, tracked)
    registry["artifacts"].append(
        {
            "artifact_id": "example_v1",
            "path": "configs/example_v1.json",
            "state": "active",
            "owner": "test-owner",
            "reason": "Synthetic versioned artifact.",
        }
    )
    _validate(tmp_path, registry, tracked)


def test_packaged_artifact_mirror_must_be_byte_identical(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/run.py")
    _write(tmp_path, "configs/example_v1.json", "{\"value\":1}\n")
    _write(tmp_path, "src/pkg/example_v1.json", "{\"value\":1}\n")
    tracked = {
        "pkg/run.py",
        "configs/example_v1.json",
        "src/pkg/example_v1.json",
    }
    registry = _registry(_component("pkg", "pkg"))
    registry["artifacts"].extend(
        [
            {
                "artifact_id": "example_v1",
                "path": "configs/example_v1.json",
                "state": "active",
                "owner": "test-owner",
                "reason": "Synthetic source artifact.",
            },
            {
                "artifact_id": "packaged_example_v1",
                "path": "src/pkg/example_v1.json",
                "state": "active",
                "owner": "test-owner",
                "reason": "Synthetic packaged mirror.",
                "mirror_of": "configs/example_v1.json",
            },
        ]
    )
    _validate(tmp_path, registry, tracked)
    (tmp_path / "src/pkg/example_v1.json").write_text(
        "{\"value\":2}\n", encoding="utf-8"
    )
    with pytest.raises(LifecycleError, match="packaged artifact drift"):
        _validate(tmp_path, registry, tracked)


def test_runnable_component_rejects_symlink_or_gitlink_mode(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/run.py")
    tracked = {"pkg/run.py"}
    with pytest.raises(LifecycleError, match="unsupported git modes"):
        _validate(
            tmp_path,
            _registry(_component("pkg", "pkg")),
            tracked,
            index_modes={
                "pkg/run.py": "120000",
                "configs/repository/component_lifecycle_v1.json": "100644",
            },
        )


def _deletion_registry(tmp_path: Path) -> tuple[dict[str, object], set[str]]:
    _write(tmp_path, "pkg/run.py")
    _write(tmp_path, "old/run.py")
    _write(
        tmp_path,
        "docs/quarantine/old.md",
        "Quarantine evidence for old_component at old.\n",
    )
    candidate = _component(
        "old_component",
        "old",
        state="deletion_candidate",
        quarantine_evidence=["docs/quarantine/old.md"],
        content_sha256=_tree_content_sha256(
            root="old",
            repo_root=tmp_path,
            tracked={"old/run.py"},
        ),
    )
    return (
        _registry(_component("pkg", "pkg"), candidate),
        {"pkg/run.py", "old/run.py", "docs/quarantine/old.md"},
    )


def test_deletion_candidate_requires_identifying_tracked_quarantine_evidence(
    tmp_path: Path,
) -> None:
    registry, tracked = _deletion_registry(tmp_path)
    candidate = registry["components"][1]
    assert isinstance(candidate, dict)
    candidate["quarantine_evidence"] = []
    with pytest.raises(LifecycleError, match="requires quarantine evidence"):
        _validate(tmp_path, registry, tracked)

    registry, tracked = _deletion_registry(tmp_path)
    (tmp_path / "docs/quarantine/old.md").write_text(
        "Unrelated quarantine note.\n", encoding="utf-8"
    )
    with pytest.raises(LifecycleError, match="does not identify"):
        _validate(tmp_path, registry, tracked)


def test_deletion_candidate_requires_zero_inbound_references(tmp_path: Path) -> None:
    registry, tracked = _deletion_registry(tmp_path)
    _write(tmp_path, "pkg/reference.py", 'LEGACY = "old"\n')
    tracked.add("pkg/reference.py")
    with pytest.raises(LifecycleError, match=r"inbound references: \['pkg/reference.py'\]"):
        _validate(tmp_path, registry, tracked)


def test_deletion_candidate_passes_only_with_evidence_and_no_inbound_refs(
    tmp_path: Path,
) -> None:
    registry, tracked = _deletion_registry(tmp_path)
    summary = _validate(tmp_path, registry, tracked)
    assert summary["deletion_candidate_count"] == 1


def test_candidate_transition_requires_unchanged_quarantine_baseline(
    tmp_path: Path,
) -> None:
    candidate_registry, tracked = _deletion_registry(tmp_path)
    active_baseline = _registry(
        _component("pkg", "pkg"),
        _component("old_component", "old"),
    )
    with pytest.raises(LifecycleError, match="only from quarantine"):
        _validate(
            tmp_path,
            candidate_registry,
            tracked,
            baseline_registry=active_baseline,
        )

    quarantine_baseline = json.loads(json.dumps(candidate_registry))
    quarantined = quarantine_baseline["components"][1]
    assert isinstance(quarantined, dict)
    quarantined["state"] = "quarantine"
    summary = _validate(
        tmp_path,
        candidate_registry,
        tracked,
        baseline_registry=quarantine_baseline,
    )
    assert summary["deletion_candidate_count"] == 1

    changed_candidate = json.loads(json.dumps(candidate_registry))
    changed = changed_candidate["components"][1]
    assert isinstance(changed, dict)
    changed["reason"] = "Changed during destructive transition."
    with pytest.raises(LifecycleError, match="must preserve quarantined metadata"):
        _validate(
            tmp_path,
            changed_candidate,
            tracked,
            baseline_registry=quarantine_baseline,
        )


def test_deletion_requires_candidate_transition_and_retained_tombstone(
    tmp_path: Path,
) -> None:
    baseline, baseline_tracked = _deletion_registry(tmp_path)
    candidate = baseline["components"][1]
    assert isinstance(candidate, dict)
    content_sha256 = candidate["content_sha256"]
    assert isinstance(content_sha256, str)

    _write(
        tmp_path,
        "docs/quarantine/old-approval.md",
        "Deletion approved for old_component at old.\n",
    )
    (tmp_path / "old/run.py").unlink()
    (tmp_path / "old").rmdir()
    current_tracked = baseline_tracked - {"old/run.py"}
    current_tracked.add("docs/quarantine/old-approval.md")
    tombstone = _component(
        "old_component",
        "old",
        state="deleted",
        tracked_policy="no_tracked_files",
        quarantine_evidence=["docs/quarantine/old.md"],
        content_sha256=content_sha256,
        deletion_approval_evidence=["docs/quarantine/old-approval.md"],
    )
    current = _registry(_component("pkg", "pkg"), tombstone)

    summary = _validate(
        tmp_path,
        current,
        current_tracked,
        baseline_registry=baseline,
    )
    assert summary["state_counts"]["deleted"] == 1

    with pytest.raises(LifecycleError, match="disappeared without a deleted tombstone"):
        _validate(
            tmp_path,
            _registry(_component("pkg", "pkg")),
            current_tracked,
            baseline_registry=baseline,
        )

    frozen_baseline = _registry(
        _component("pkg", "pkg"),
        _component(
            "old_component",
            "old",
            state="frozen_replay",
            content_sha256=content_sha256,
        ),
    )
    with pytest.raises(LifecycleError, match="only to quarantine"):
        _validate(
            tmp_path,
            current,
            current_tracked,
            baseline_registry=frozen_baseline,
        )

    changed_tombstone = json.loads(json.dumps(current))
    changed = changed_tombstone["components"][1]
    assert isinstance(changed, dict)
    changed["reason"] = "A changed tombstone is forbidden."
    with pytest.raises(LifecycleError, match="tombstone is immutable"):
        _validate(
            tmp_path,
            changed_tombstone,
            current_tracked,
            baseline_registry=current,
        )


def test_deleted_artifact_requires_candidate_transition_and_approval(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "pkg/run.py")
    _write(tmp_path, "configs/old_v1.json", "{}\n")
    _write(
        tmp_path,
        "docs/quarantine/old-artifact.md",
        "Quarantine evidence for old_artifact_v1 at configs/old_v1.json.\n",
    )
    digest = hashlib.sha256(
        (tmp_path / "configs/old_v1.json").read_bytes()
    ).hexdigest()
    baseline = _registry(_component("pkg", "pkg"))
    baseline["artifacts"].append(
        {
            "artifact_id": "old_artifact_v1",
            "path": "configs/old_v1.json",
            "state": "deletion_candidate",
            "owner": "test-owner",
            "reason": "Synthetic artifact awaiting deletion.",
            "content_sha256": digest,
            "git_mode": "100644",
            "quarantine_evidence": ["docs/quarantine/old-artifact.md"],
            "quarantine_started_at": "2024-01-01",
            "minimum_quarantine_days": 7,
        }
    )
    _write(
        tmp_path,
        "docs/quarantine/old-artifact-approval.md",
        "Deletion approved for old_artifact_v1.\n",
    )
    (tmp_path / "configs/old_v1.json").unlink()
    current = _registry(_component("pkg", "pkg"))
    current["artifacts"].append(
        {
            "artifact_id": "old_artifact_v1",
            "path": "configs/old_v1.json",
            "state": "deleted",
            "owner": "test-owner",
            "reason": "Synthetic artifact awaiting deletion.",
            "content_sha256": digest,
            "git_mode": "100644",
            "quarantine_evidence": ["docs/quarantine/old-artifact.md"],
            "quarantine_started_at": "2024-01-01",
            "minimum_quarantine_days": 7,
            "deletion_approval_evidence": [
                "docs/quarantine/old-artifact-approval.md"
            ],
        }
    )
    tracked = {
        "pkg/run.py",
        "docs/quarantine/old-artifact.md",
        "docs/quarantine/old-artifact-approval.md",
    }
    summary = _validate(
        tmp_path,
        current,
        tracked,
        baseline_registry=baseline,
    )
    assert summary["artifact_state_counts"]["deleted"] == 1


def test_deletion_candidate_requires_elapsed_quarantine_period(
    tmp_path: Path,
) -> None:
    registry, tracked = _deletion_registry(tmp_path)
    candidate = registry["components"][1]
    assert isinstance(candidate, dict)
    candidate["quarantine_started_at"] = "2024-01-30"
    candidate["minimum_quarantine_days"] = 7
    with pytest.raises(LifecycleError, match="period has not elapsed"):
        _validate(tmp_path, registry, tracked)


def test_in_progress_quarantine_is_valid_before_deletion_eligibility(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "pkg/run.py")
    _write(tmp_path, "old/run.py")
    _write(
        tmp_path,
        "docs/quarantine/old.md",
        "Quarantine evidence for old_component at old.\n",
    )
    quarantine = _component(
        "old_component",
        "old",
        state="quarantine",
        quarantine_evidence=["docs/quarantine/old.md"],
        quarantine_started_at="2024-01-30",
        minimum_quarantine_days=7,
        content_sha256=_tree_content_sha256(
            root="old",
            repo_root=tmp_path,
            tracked={"old/run.py"},
        ),
    )
    summary = _validate(
        tmp_path,
        _registry(_component("pkg", "pkg"), quarantine),
        {"pkg/run.py", "old/run.py", "docs/quarantine/old.md"},
    )
    assert summary["state_counts"]["quarantine"] == 1
    assert summary["deletion_candidate_count"] == 0


def test_index_snapshot_ignores_unstaged_worktree_bytes(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write(repo, "pkg/run.py", "VALUE = 'indexed'\n")
    active_path = repo / "configs/runners/active_generation_bundles_v1.json"
    active_path.parent.mkdir(parents=True)
    active_path.write_text(json.dumps(_active_manifest()), encoding="utf-8")
    registry = _registry(_component("pkg", "pkg"))
    registry_path = repo / "configs/repository/component_lifecycle_v1.json"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    for args in (
        ("init", "--quiet"),
        ("config", "user.email", "lifecycle@example.invalid"),
        ("config", "user.name", "Lifecycle Test"),
        ("add", "--force", "--all"),
    ):
        subprocess.run(["git", *args], cwd=repo, check=True)

    (repo / "pkg/run.py").write_text("this is not valid python !!!\n", encoding="utf-8")
    summary = validate_index_snapshot(
        repo_root=repo,
        registry_path=registry_path,
        active_manifest_path=active_path,
        today=date(2024, 2, 1),
    )
    assert summary["active_bundle_count"] == 1


def test_index_snapshot_ignores_repository_smudge_filters(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write(repo, "pkg/run.py")
    _write(repo, "legacy/run.py", "VALUE = 'INDEXED'\n")
    _write(repo, ".gitattributes", "legacy/run.py filter=lifecycle-review\n")
    active_path = repo / "configs/runners/active_generation_bundles_v1.json"
    active_path.parent.mkdir(parents=True)
    active_path.write_text(json.dumps(_active_manifest()), encoding="utf-8")
    tracked = {
        "pkg/run.py",
        "legacy/run.py",
        ".gitattributes",
        "configs/runners/active_generation_bundles_v1.json",
        "configs/repository/component_lifecycle_v1.json",
    }
    digest = _tree_content_sha256(
        root="legacy", repo_root=repo, tracked=tracked
    )
    registry = _registry(
        _component("pkg", "pkg"),
        _component(
            "legacy",
            "legacy",
            state="frozen_replay",
            content_sha256=digest,
        ),
    )
    registry_path = repo / "configs/repository/component_lifecycle_v1.json"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    for args in (
        ("init", "--quiet"),
        ("config", "user.email", "lifecycle@example.invalid"),
        ("config", "user.name", "Lifecycle Test"),
        ("config", "filter.lifecycle-review.smudge", "sed s/INDEXED/SMUDGED/"),
        ("add", "--force", "--all"),
    ):
        subprocess.run(["git", *args], cwd=repo, check=True)

    summary = validate_index_snapshot(
        repo_root=repo,
        registry_path=registry_path,
        active_manifest_path=active_path,
        today=date(2024, 2, 1),
    )
    assert summary["active_bundle_count"] == 1


def test_checked_in_registry_names_all_states_and_local_only_hy34_versions() -> None:
    registry = load_registry()
    components = registry["components"]
    assert isinstance(components, list)
    by_id = {component["component_id"]: component for component in components}
    assert set(LIFECYCLE_STATES) == {
        "active",
        "compatibility",
        "frozen_replay",
        "local_only",
        "quarantine",
        "deletion_candidate",
        "deleted",
    }
    assert by_id["hy34_retrieval_conditioned_runner_v1_local"]["state"] == "local_only"
    assert by_id["hy34_retrieval_conditioned_runner_v2_local"]["state"] == "local_only"
    assert by_id["metric_profile_canonical_v1"]["state"] == "frozen_replay"
