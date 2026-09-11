"""CLI for static checking, dataset identity, and campaign execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from benchmark.evaluation_campaign.config import (
    load_campaign,
    load_local_bindings,
    load_profile_registry,
    resolve_profile_binding,
)
from benchmark.evaluation_campaign.dataset_identity import (
    inspect_evaluation_dataset,
)
from benchmark.evaluation_campaign.orchestrator import (
    EvaluationCampaignOrchestrator,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BINDINGS = (
    REPO_ROOT
    / "configs/evaluation/campaigns/evaluation_bindings.local.json"
)


def _run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "run"):
        child = subparsers.add_parser(name)
        child.add_argument("--config", type=Path, required=True)
        if name == "run":
            child.add_argument("--bindings", type=Path, default=DEFAULT_BINDINGS)
    identity = subparsers.add_parser("dataset-identity")
    identity.add_argument("--dataset-root", type=Path, required=True)
    identity.add_argument("--case-id", action="append", required=True)
    args = parser.parse_args(argv)

    if args.command == "dataset-identity":
        result = inspect_evaluation_dataset(
            args.dataset_root,
            expected_case_ids=args.case_id,
        )
        print(json.dumps(result.public_dict(), indent=2, ensure_ascii=False))
        return 0

    campaign = load_campaign(args.config, repo_root=REPO_ROOT)
    profiles = load_profile_registry(campaign.profile_registry)
    profile = profiles[campaign.judge_profile_id]
    missing_prior_profiles = sorted(
        {
            prior.judge_profile_id
            for prior in campaign.case_plan.prior_attempt_roots
            if prior.judge_profile_id not in profiles
        }
    )
    if missing_prior_profiles:
        raise SystemExit(
            "prior attempt Judge profiles are absent from the registry: "
            + ",".join(missing_prior_profiles)
        )
    binding = None
    if args.command == "run":
        _require_private_binding_path(args.bindings)
        bindings = load_local_bindings(args.bindings)
        binding = resolve_profile_binding(profile, bindings)
    orchestrator = EvaluationCampaignOrchestrator(
        campaign,
        profile,
        binding,
        repo_root=REPO_ROOT,
        python_executable=REPO_ROOT / ".venv/bin/python",
    )
    if args.command == "check":
        result = orchestrator.check()
    else:
        result = orchestrator.run()
        result = {
            "status": result.status,
            "selected_case_ids": list(result.selected_case_ids),
            "unresolved_case_ids": list(result.unresolved_case_ids),
            "final_root": (
                result.final_root.resolve().relative_to(REPO_ROOT).as_posix()
                if result.final_root
                else None
            ),
        }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return _run(argv)
    except Exception as exc:
        # Dependency and deployment errors can contain local paths, endpoints,
        # credential environment names, or child-process details. Keep the
        # public terminal surface stable and path-free.
        print(
            f"error: {type(exc).__name__}: evaluation campaign command failed",
            file=sys.stderr,
        )
        return 3


def _require_private_binding_path(path: Path) -> None:
    selected = path.expanduser().absolute()
    if selected.is_symlink():
        raise SystemExit("local evaluation binding must not be a symlink")
    resolved = selected.resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if tracked.returncode == 0:
        raise SystemExit("local evaluation binding must never be Git-tracked")
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "--", relative],
        cwd=REPO_ROOT,
        check=False,
    )
    if ignored.returncode != 0:
        raise SystemExit(
            "repository-local evaluation binding must be covered by .gitignore"
        )
