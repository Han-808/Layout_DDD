#!/usr/bin/env python3
"""Materialize one fresh, single-scene Agent workspace."""

from __future__ import annotations

import argparse
import json

from arena import create_episode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    episode = create_episode(
        agent_id=args.agent_id,
        scene_id=args.scene_id,
        run_id=args.run_id,
    )
    print(
        json.dumps(
            {
                "status": "prepared",
                "episode_root": str(episode.root),
                "agent_workspace": str(episode.workspace),
                "scene_id": episode.case.scene_id,
                "model_or_agent_started": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
