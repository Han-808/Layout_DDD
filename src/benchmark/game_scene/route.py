"""Executable browser-Game route into the canonical L0--L4 evaluator.

The route performs exactly one browser capture.  That capture authors the
canonical scene, world-baked collision meshes, and appearance frames together.
The resulting scene is then placed in a trusted Game case bundle and evaluated
through :func:`benchmark.api.submission.evaluate_submission`.  A frozen
renderer adapter makes the evaluator reuse the same capture instead of opening
the game a second time.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
from typing import Any

from benchmark.api.submission import evaluate_submission
from benchmark.evaluator.visual_style_spec import validate_visual_style_spec
from benchmark.game_scene.case_bundle import build_game_case_bundle
from benchmark.game_scene.mode import GameModeConfig, load_game_mode_config
from benchmark.rendering.browser import (
    FrozenBrowserCaptureRenderer,
    HeadlessBrowserRenderer,
    UnsupportedBrowserRenderPipelineError,
)
from benchmark.utils.io import read_json, write_json
from benchmark.visual_judge import build_openai_compatible_vlm_judge


GAME_MODE_RUN_VERSION = "game_mode_run_v1"


class GameModeRunError(RuntimeError):
    """Raised when a browser game cannot complete the configured route."""


class GameNotIngestableError(GameModeRunError):
    """Raised when the source has no supported Three.js scene runtime."""


def run_game_mode(
    *,
    game_mode_config: GameModeConfig | str | Path,
    game_root: str | Path,
    out_dir: str | Path,
    case_id: str,
    entry_html: str | Path | None = None,
    scene_id: str | None = None,
    request_id: str | None = None,
    three_replacement: str | Path | None = None,
    visual_style_spec: dict[str, Any] | str | Path | None = None,
    vlm_judge: Any | None = None,
    official_mode: bool = True,
    renderer: HeadlessBrowserRenderer | None = None,
) -> dict[str, Any]:
    """Capture and evaluate one Three.js game through the canonical workflow.

    ``renderer`` is a test seam.  Production callers leave it unset so every
    capture setting comes from the strict ``game_mode`` config.
    """

    mode = (
        game_mode_config
        if isinstance(game_mode_config, GameModeConfig)
        else load_game_mode_config(game_mode_config)
    )
    root = Path(game_root).expanduser().resolve()
    destination = Path(out_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "game_mode_run_manifest.json"
    normalized_case_id = _non_empty(case_id, "case_id")
    normalized_scene_id = _non_empty(
        scene_id or f"{normalized_case_id}_scene",
        "scene_id",
    )
    normalized_request_id = _non_empty(
        request_id or f"{normalized_case_id}_request",
        "request_id",
    )

    entry = _resolve_entry(root, entry_html or mode.entry_html)
    replacement = _resolve_optional_file(three_replacement, "three_replacement")
    style_spec_source: dict[str, Any] | str | Path = (
        visual_style_spec
        if visual_style_spec is not None
        else mode.default_visual_style_spec_path
    )
    style_spec = _load_style_spec(style_spec_source)
    validate_visual_style_spec(
        style_spec,
        require_trusted_source=official_mode,
    )
    if official_mode and vlm_judge is None:
        raise GameModeRunError("official game evaluation requires a VLM judge")
    source_compatibility = _classify_source_runtime(root)

    run_record: dict[str, Any] = {
        "run_version": GAME_MODE_RUN_VERSION,
        "status": "running",
        "mode": "game",
        "workflow": "canonical_l0_l4",
        "case_id": normalized_case_id,
        "official_mode": bool(official_mode),
        "route_config": {
            "path": mode.path.as_posix(),
            "sha256": _sha256(mode.path),
        },
        "evaluation_profile": {
            "path": mode.evaluation_profile_path.as_posix(),
            "sha256": _sha256(mode.evaluation_profile_path),
            "active_metrics": sorted(
                mode.raw["evaluation"]["active_metrics"]
            ),
        },
        "source": {
            "game_root": root.as_posix(),
            "entry_html": entry.as_posix(),
            "entry_html_sha256": _sha256(entry),
            "three_replacement": replacement.as_posix() if replacement else None,
            "three_replacement_sha256": _sha256(replacement) if replacement else None,
            "compatibility": source_compatibility,
        },
        "capture": {
            "performed_once": False,
            "directory": (destination / "renders").as_posix(),
        },
        "credentials": {
            "embedded": False,
            "source": "judge_configuration_environment_only",
        },
    }
    write_json(manifest_path, run_record)
    if not source_compatibility["ingestable"]:
        run_record.update(
            {
                "status": "not_ingestable",
                "failure": {"error_type": "GameNotIngestableError"},
            }
        )
        write_json(manifest_path, run_record)
        raise GameNotIngestableError(
            "game_mode accepts only a source that references Three.js and "
            "constructs a Scene plus WebGLRenderer"
        )

    try:
        active_renderer = renderer or HeadlessBrowserRenderer(
            entry_html=entry,
            game_root=root,
            three_replacement=replacement,
            **mode.renderer_kwargs,
        )
        capture_manifest = active_renderer.capture_game_source(
            out_dir=destination / "renders",
            scene_id=normalized_scene_id,
            request_id=normalized_request_id,
            scene_type="game_level",
            require_probe=True,
        )
        exported_scene_path = Path(str(capture_manifest["exported_scene"])).resolve()
        exported_scene = read_json(exported_scene_path)
        run_record["capture"] = {
            "performed_once": True,
            "directory": (destination / "renders").as_posix(),
            "manifest": (destination / "renders" / "render_manifest.json").as_posix(),
            "manifest_sha256": _sha256(
                destination / "renders" / "render_manifest.json"
            ),
            "exported_scene": exported_scene_path.as_posix(),
            "exported_scene_sha256": _sha256(exported_scene_path),
            "probe_available": bool(capture_manifest.get("probe_available")),
            "object_count": len(exported_scene.get("objects", [])),
        }
        write_json(manifest_path, run_record)

        bundle_root = build_game_case_bundle(
            exported_scene,
            out_dir=destination / "case_bundle",
            case_id=normalized_case_id,
            evaluation_profile=mode.evaluation_profile,
            instruction=mode.instruction,
            visual_style_spec=style_spec,
        )
        frozen_renderer = FrozenBrowserCaptureRenderer(
            capture_dir=destination / "renders"
        )
        result = evaluate_submission(
            scene=exported_scene_path,
            case_bundle=bundle_root,
            out_dir=destination,
            renderer=frozen_renderer,
            vlm_judge=vlm_judge,
            official_mode=official_mode,
        )
        report = result["evaluation_report"]
        run_record.update(
            {
                "status": (
                    "complete"
                    if report.get("benchmark_score_status") == "complete"
                    else "incomplete"
                ),
                "case_bundle": {
                    "path": bundle_root.as_posix(),
                    "manifest_sha256": _sha256(bundle_root / "case_bundle.json"),
                },
                "evaluation": {
                    "report": (destination / "evaluation_report.json").as_posix(),
                    "report_sha256": _sha256(
                        destination / "evaluation_report.json"
                    ),
                    "benchmark_score": report.get("benchmark_score"),
                    "benchmark_score_status": report.get(
                        "benchmark_score_status"
                    ),
                    "submission_manifest": (
                        destination / "submission_run_manifest.json"
                    ).as_posix(),
                },
            }
        )
        write_json(manifest_path, run_record)
        result["game_mode_manifest"] = run_record
        return result
    except UnsupportedBrowserRenderPipelineError as exc:
        run_record.update(
            {
                "status": "not_ingestable",
                "failure": {
                    "error_type": "UnsupportedBrowserRenderPipelineError",
                    "classification": "unsupported_threejs_render_pipeline",
                },
            }
        )
        write_json(manifest_path, run_record)
        raise GameNotIngestableError(
            "the Three.js source does not expose the required direct "
            "WebGLRenderer camera-control capability"
        ) from exc
    except Exception as exc:
        # Do not serialize an exception message here. Provider exceptions can
        # contain echoed HTTP data; the durable manifest records only the safe
        # failure class while the original exception still propagates.
        run_record.update(
            {
                "status": "failed",
                "failure": {"error_type": type(exc).__name__},
            }
        )
        write_json(manifest_path, run_record)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Capture one instrumentable Three.js game and evaluate it through "
            "the canonical L0-L4 Game profile."
        )
    )
    parser.add_argument(
        "--game-mode-config",
        default="configs/game/game_mode_canonical_v1.yaml",
    )
    parser.add_argument("--game-root", required=True)
    parser.add_argument("--entry-html", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--request-id", default=None)
    parser.add_argument("--three-replacement", default=None)
    parser.add_argument("--visual-style-spec", default=None)
    parser.add_argument("--vlm-judge-config", default=None)
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()

    judge = None
    if args.vlm_judge_config:
        judge_config = read_json(args.vlm_judge_config)
        if not isinstance(judge_config, dict):
            parser.error("--vlm-judge-config must point to a JSON object")
        judge = build_openai_compatible_vlm_judge(judge_config)
    if not args.diagnostic and judge is None:
        parser.error("official game evaluation requires --vlm-judge-config")

    result = run_game_mode(
        game_mode_config=args.game_mode_config,
        game_root=args.game_root,
        entry_html=args.entry_html,
        out_dir=args.out_dir,
        case_id=args.case_id,
        scene_id=args.scene_id,
        request_id=args.request_id,
        three_replacement=args.three_replacement,
        visual_style_spec=args.visual_style_spec,
        vlm_judge=judge,
        official_mode=not args.diagnostic,
    )
    report = result["evaluation_report"]
    print(f"benchmark_score: {report.get('benchmark_score')}")
    print(f"benchmark_score_status: {report.get('benchmark_score_status')}")
    print(f"game_mode_manifest: {Path(args.out_dir).resolve() / 'game_mode_run_manifest.json'}")


def _resolve_entry(root: Path, value: str | Path) -> Path:
    if not root.is_dir():
        raise GameModeRunError(f"game_root does not exist: {root}")
    supplied = Path(value).expanduser()
    entry = (root / supplied).resolve() if not supplied.is_absolute() else supplied.resolve()
    try:
        entry.relative_to(root)
    except ValueError as exc:
        raise GameModeRunError("entry_html must remain inside game_root") from exc
    if not entry.is_file():
        raise GameModeRunError(f"game entry HTML does not exist: {entry}")
    return entry


def _load_style_spec(
    value: dict[str, Any] | str | Path | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    loaded = read_json(value)
    if not isinstance(loaded, dict):
        raise GameModeRunError("visual_style_spec must be a JSON object")
    return loaded


def _resolve_optional_file(
    value: str | Path | None,
    label: str,
) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise GameModeRunError(f"{label} does not exist: {path}")
    return path


_SOURCE_SUFFIXES = frozenset({".html", ".htm", ".js", ".mjs", ".cjs", ".ts", ".tsx"})
_SOURCE_EXCLUDED_PARTS = frozenset(
    {".git", "node_modules", "__pycache__", "outputs", "dist", "build"}
)
_THREE_REFERENCE = re.compile(
    r"(?:\bTHREE\b|(?:from|import)[^;\n]{0,200}[\"']three(?:/[^\"']*)?[\"']|"
    r"three(?:\.min|\.module)?\.js)",
    re.IGNORECASE,
)
_SCENE_CONSTRUCTOR = re.compile(
    r"\bnew\s+(?:THREE\s*\.\s*)?Scene\s*\(",
    re.IGNORECASE,
)
_RENDERER_CONSTRUCTOR = re.compile(
    r"\bnew\s+(?:THREE\s*\.\s*)?WebGLRenderer\s*\(",
    re.IGNORECASE,
)
_EFFECT_COMPOSER_CONSTRUCTOR = re.compile(
    r"\bnew\s+(?:THREE\s*\.\s*)?EffectComposer\s*\(",
    re.IGNORECASE,
)


def _classify_source_runtime(root: Path) -> dict[str, Any]:
    """Apply the frozen corpus-ingestibility rule without executing the page."""

    has_three = False
    has_scene = False
    has_renderer = False
    has_effect_composer = False
    scanned = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _SOURCE_SUFFIXES:
            continue
        relative = path.relative_to(root)
        if any(part in _SOURCE_EXCLUDED_PARTS for part in relative.parts):
            continue
        try:
            # Generated games are small. Capping individual source reads avoids
            # treating a vendored/minified library as the application itself.
            if path.stat().st_size > 5 * 1024 * 1024:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        scanned += 1
        has_three = has_three or bool(_THREE_REFERENCE.search(text))
        has_scene = has_scene or bool(_SCENE_CONSTRUCTOR.search(text))
        has_renderer = has_renderer or bool(_RENDERER_CONSTRUCTOR.search(text))
        has_effect_composer = has_effect_composer or bool(
            _EFFECT_COMPOSER_CONSTRUCTOR.search(text)
        )
    ingestable = bool(
        has_three
        and has_scene
        and has_renderer
        and not has_effect_composer
    )
    return {
        "classification": (
            "threejs_direct_webgl_renderer"
            if ingestable
            else (
                "unsupported_threejs_render_pipeline"
                if has_effect_composer
                else "unsupported_browser_runtime"
            )
        ),
        "ingestable": ingestable,
        "source_files_scanned": scanned,
        "has_three_reference": has_three,
        "has_scene_constructor": has_scene,
        "has_webgl_renderer_constructor": has_renderer,
        "has_effect_composer": has_effect_composer,
    }


def _non_empty(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise GameModeRunError(f"{label} must be a non-empty string")
    return text


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
