#!/usr/bin/env python3
"""Query the real frozen shared DB from a disposable isolated workspace."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

from arena import create_episode
from database_host import EpisodeDatabase
from isolated_exec import run_isolated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-bindings", required=True)
    args = parser.parse_args()
    episode = create_episode(
        agent_id="trusted-real-db-smoke",
        scene_id="scene_012121",
        run_id=f"pid-{os.getpid()}",
    )
    try:
        with EpisodeDatabase(
            episode=episode,
            resource_bindings=args.resource_bindings,
        ) as database_service:
            result = run_isolated(
                workspace=episode.workspace,
                runtime_root="/usr/bin",
                command=[
                    str(episode.workspace / "sieve-agent-tool"),
                    "search-assets",
                    "--query",
                    "wooden dining chair",
                    "--size",
                    "0.5",
                    "0.5",
                    "0.9",
                    "--top-k",
                    "4",
                ],
                tool_socket=database_service.socket_path,
                tool_token=database_service.capability_token,
                stdout_path=episode.host / "search.stdout.json",
                stderr_path=episode.host / "search.stderr.log",
                stdin_text="",
                timeout_seconds=180.0,
            )
            if result.returncode != 0:
                raise RuntimeError("isolated real-DB query failed")
            envelope = json.loads(
                (episode.host / "search.stdout.json").read_text(encoding="utf-8")
            )
            payload = envelope.get("result", {})
            if envelope.get("ok") is not True or payload.get("result_count") != 4:
                raise RuntimeError("real-DB search result contract differs")
            if payload.get("catalog_snapshot_id") != database_service.database.snapshot_id:
                raise RuntimeError("real-DB search snapshot identity differs")
            if len({item["asset_id"] for item in payload.get("results", [])}) != 4:
                raise RuntimeError("real-DB search returned duplicate assets")
            print(
                json.dumps(
                    {
                        "status": "valid",
                        "database_snapshot_id": payload["catalog_snapshot_id"],
                        "result_count": payload["result_count"],
                        "query_executed_inside_isolated_workspace": True,
                        "raw_database_mounted_into_workspace": False,
                        "model_or_generation_started": False,
                    },
                    sort_keys=True,
                )
            )
            return 0
    finally:
        scene_root = episode.root.parent
        agent_root = scene_root.parent
        shutil.rmtree(episode.root, ignore_errors=True)
        if scene_root.is_dir() and not any(scene_root.iterdir()):
            scene_root.rmdir()
        if agent_root.is_dir() and not any(agent_root.iterdir()):
            agent_root.rmdir()


if __name__ == "__main__":
    raise SystemExit(main())
