"""Run the game probe across a corpus and tabulate what individualization did.

The object definition frozen in ``configs/game/game_mode_canonical_v1.yaml`` is
argued to be independent of how a generated implementation happens to organise
its scene graph. This script is how that argument gets checked against real
implementations rather than against synthetic fixtures: it captures every
ingestible game under a corpus root and reports, per implementation, how many
meshes the probe saw, how many survived each rule, and how many object pairs a
collision run would then call invalid.

Usage::

    PYTHONPATH=src python scripts/survey_game_individualization.py \
        --corpus "Support/datasets/game_corpus/20260720_190732 2" \
        --out Support/reports/individualization

Implementations whose three.js comes from a CDN are skipped unless a local
replacement is supplied, because a page that cannot load three has no scene to
probe and would otherwise be indistinguishable from a genuine probe failure.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmark.api.evaluation import run_evaluate  # noqa: E402
from benchmark.evaluator.profile import resolve_evaluation_profile  # noqa: E402
from benchmark.game_scene.harness import rewrite_entry_html  # noqa: E402
from benchmark.rendering import HeadlessBrowserRenderer  # noqa: E402
from benchmark.utils.io import load_yaml, write_json  # noqa: E402

THREE_D_CATEGORIES = (
    "billiards",
    "cs_fps",
    "gta",
    "mario_kart",
    "subway_surfers",
    "programming_puzzle",
)
_USES_THREE = re.compile(r"THREE\.|from ['\"]three|WebGLRenderer")
_SKIP_DIRS = {"node_modules", "dist", "vendor", "lib"}


def find_entry(impl: Path) -> Path | None:
    candidates = [
        f
        for f in impl.rglob("index.html")
        if not _SKIP_DIRS.intersection(f.relative_to(impl).parts)
    ]
    return min(candidates, key=lambda f: len(f.parts)) if candidates else None


def uses_three(impl: Path, entry: Path) -> bool:
    sources = [entry] + [
        f
        for f in impl.rglob("*.js")
        if "three" not in f.name.lower() and "node_modules" not in str(f)
    ]
    return any(_USES_THREE.search(f.read_text(errors="ignore")) for f in sources[:60])


def capture(impl: Path, entry: Path, out_dir: Path, three_replacement: Path | None) -> dict[str, Any]:
    renderer = HeadlessBrowserRenderer(
        entry_html=entry,
        game_root=impl,
        three_replacement=three_replacement,
        timeout_seconds=90,
    )
    return renderer.capture_game_source(
        out_dir=out_dir,
        scene_id=f"{impl.parent.name}__{impl.name}",
        request_id=f"{impl.parent.name}__{impl.name}",
        require_probe=True,
    )


def collision_summary(scene: dict[str, Any], geometry: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Run collision alone and report how many pairs it would call invalid."""

    profile = resolve_evaluation_profile(
        load_yaml(REPO_ROOT / "configs/evaluation/metric_profile_game_canonical_v1.yaml", default={})
    )
    report = run_evaluate(
        scene=scene,
        out=out_dir / "evaluate",
        eval_generic_validity=True,
        collision_geometry=geometry,
        evaluation_profile=profile,
    )
    collision = report["reports"]["generic_validity"]["metrics"]["collision"]
    pairs = collision.get("pairs") or []
    routes: dict[str, int] = {}
    for pair in pairs:
        route = str(pair.get("route") or "unknown")
        routes[route] = routes.get(route, 0) + 1
    return {
        "status": collision.get("status"),
        "score": collision.get("score"),
        "pair_count": len(pairs),
        "invalid_count": collision.get("collision_count"),
        "routes": routes,
    }


def survey_one(
    impl: Path,
    out_root: Path,
    three_replacement: Path | None,
    collision_object_cap: int,
) -> dict[str, Any]:
    record: dict[str, Any] = {"category": impl.parent.name, "implementation": impl.name}
    entry = find_entry(impl)
    if entry is None:
        return {**record, "status": "no_entry"}
    if not uses_three(impl, entry):
        return {**record, "status": "not_three_js"}

    _, rewrite = rewrite_entry_html(entry.read_text(errors="ignore"))
    record["instrumentation"] = rewrite.instrumentation
    if rewrite.network_required and three_replacement is None:
        return {**record, "status": "needs_three_replacement", "urls": rewrite.external_three_urls}

    out_dir = out_root / impl.parent.name / impl.name
    try:
        manifest = capture(impl, entry, out_dir, three_replacement)
    except Exception as exc:  # noqa: BLE001 - a survey must not stop at one bad page
        return {**record, "status": "capture_failed", "error": f"{type(exc).__name__}: {exc}"}

    if not manifest.get("probe_available"):
        return {**record, "status": "no_scene_found"}

    scene = json.loads(Path(manifest["exported_scene"]).read_text())
    individualization = scene["metadata"]["game_scene_import"]["individualization"]
    probe = individualization.get("probe") or {}
    record.update(
        {
            "status": "ok",
            "top_level_children": probe.get("top_level_child_count"),
            "max_graph_depth": probe.get("max_graph_depth"),
            "meshes_visited": (probe.get("counts") or {}).get("meshes_visited"),
            "probe_objects": individualization["probe_objects"],
            "dropped_non_physical": individualization["dropped_non_physical"],
            "absorbed": individualization["absorbed_into_container"],
            "objects_exported": individualization["objects_exported"],
            "declared_categories": probe.get("declared_category_count"),
            "warnings": [w["code"] for w in individualization["warnings"]],
        }
    )
    exported = individualization["objects_exported"]
    if exported > collision_object_cap:
        # Collision is quadratic in the object count and was sized for rooms of a
        # few dozen objects. A city level is minutes of work and millions of pair
        # records, which is a finding in its own right rather than a run to wait
        # out during a survey.
        record["collision"] = {
            "status": "skipped_object_count_over_cap",
            "cap": collision_object_cap,
            "would_be_pairs": exported * (exported - 1) // 2,
        }
        return record
    try:
        record["collision"] = collision_summary(scene, manifest["collision_geometry"], out_dir)
    except Exception as exc:  # noqa: BLE001
        record["collision"] = {"error": f"{type(exc).__name__}: {exc}"}
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--categories", nargs="*", default=list(THREE_D_CATEGORIES))
    parser.add_argument("--three-replacement", type=Path, default=None)
    parser.add_argument("--only", nargs="*", default=None, help="limit to these implementations")
    parser.add_argument(
        "--collision-object-cap",
        type=int,
        default=600,
        help="skip the collision run above this object count; it is quadratic",
    )
    args = parser.parse_args()

    corpus = args.corpus.expanduser().resolve()
    out_root = args.out.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for name in args.categories:
        category = corpus / name
        if not category.is_dir():
            continue
        for impl in sorted(p for p in category.iterdir() if p.is_dir()):
            if args.only and impl.name not in args.only:
                continue
            print(f"[{name}/{impl.name}] ...", flush=True)
            try:
                record = survey_one(
                    impl, out_root, args.three_replacement, args.collision_object_cap
                )
            except Exception:  # noqa: BLE001
                record = {
                    "category": name,
                    "implementation": impl.name,
                    "status": "survey_crashed",
                    "error": traceback.format_exc(limit=3),
                }
            records.append(record)
            print(f"    {record['status']}", flush=True)

    write_json(out_root / "individualization_survey.json", records)
    print_table(records)
    return 0


def print_table(records: list[dict[str, Any]]) -> None:
    header = (
        f"{'category':<20}{'impl':<18}{'顶层':>5}{'深度':>5}{'mesh':>6}"
        f"{'丢弃':>5}{'吸收':>5}{'物体':>6}{'对':>6}{'无效':>6}{'分':>7}  警告"
    )
    print("\n" + header)
    print("-" * len(header))
    for record in records:
        if record["status"] != "ok":
            print(f"{record['category']:<20}{record['implementation']:<18}{record['status']}")
            continue
        collision = record.get("collision") or {}
        score = collision.get("score")
        print(
            f"{record['category']:<20}{record['implementation']:<18}"
            f"{record['top_level_children'] or 0:>5}{record['max_graph_depth'] or 0:>5}"
            f"{record['meshes_visited'] or 0:>6}{record['dropped_non_physical']:>5}"
            f"{record['absorbed']:>5}{record['objects_exported']:>6}"
            f"{collision.get('pair_count', 0):>6}{collision.get('invalid_count') or 0:>6}"
            f"{(f'{score:.3f}' if isinstance(score, float) else str(score)):>7}  "
            f"{','.join(record['warnings'])}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
