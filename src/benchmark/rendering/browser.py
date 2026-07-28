"""Headless-browser renderer for generator-authored browser game scenes.

``evaluate_submission`` accepts any object exposing ``render_scene``; it never
type-checks against :class:`~benchmark.rendering.blender.BlenderRenderer`. This
renderer uses that seam to make a browser game a first-class evidence source.

One ``render_scene`` call fills all three evaluator channels: it steps the game
to a fixed tick under the deterministic harness, screenshots the configured
camera presets (appearance evidence), and runs the scene probe to emit baked
triangle meshes (``collision_geometry_v1``). Rendering happens inside the runner
and writes beneath ``out_dir``, so the evidence stays benchmark-generated and
``submitted_evidence_accepted`` remains false.

The game is served over loopback HTTP rather than opened as ``file://``. A
file-origin page cannot honour an import map or load ES modules, which is how
most observed games pull in three.js, and owning the origin is also what lets
the harness instrument the ``Scene`` constructor before any game code runs. See
:mod:`benchmark.game_scene.harness`.

Playwright is an optional dependency and is imported lazily, so importing this
module never requires a browser. The ``page_factory`` seam exists so the
orchestration can be tested without one.
"""

from __future__ import annotations

import base64
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, UnidentifiedImageError

from benchmark.scene_io.object_normalization import rotation_matrix_from_euler
from benchmark.utils.io import read_json, write_json


BROWSER_RENDER_BACKEND = "headless_browser_game_v1"

DEFAULT_VIEWS = (
    {"name": "gameplay", "step_frames": 0},
    {"name": "gameplay_settled", "step_frames": 120},
)
CONTROLLED_CAMERA_VIEW_FAMILY = "canonical_high_oblique_pair_v1"
CONTROLLED_STYLE_LOCAL_VIEW_FAMILY = "canonical_style_region_quadrants_v1"
CONTROLLED_CAMERA_APPEARANCE_FIDELITY = "original_runtime_direct_webgl"
DEFAULT_CONTROLLED_CAMERA = {
    "enabled": False,
    "required": False,
    "view_family": CONTROLLED_CAMERA_VIEW_FAMILY,
    "image_budget": 2,
    "style_local_fallback_enabled": False,
    "style_local_view_family": CONTROLLED_STYLE_LOCAL_VIEW_FAMILY,
    "style_local_image_budget": 4,
    "canvas_only": True,
    "include_authored_camera_diagnostics": True,
    "unsupported_render_pipeline": "fail_not_ingestable",
}

_ASSET_DIR = Path(__file__).resolve().parents[1] / "game_scene"


class BrowserRenderError(RuntimeError):
    """Raised when the headless browser cannot produce trusted evidence."""


class UnsupportedBrowserRenderPipelineError(BrowserRenderError):
    """Raised when a Three.js page has no controllable direct render context."""


class HeadlessBrowserRenderer:
    """Render and probe a browser game under a deterministic replay harness."""

    def __init__(
        self,
        *,
        entry_html: str | Path,
        game_root: str | Path | None = None,
        three_replacement: str | Path | None = None,
        width: int = 768,
        height: int = 768,
        seed: int = 20260727,
        step_ms: float = 1000.0 / 60.0,
        warmup_frames: int = 60,
        views: tuple[dict[str, Any], ...] = DEFAULT_VIEWS,
        max_vertices_per_object: int = 200000,
        unit_scale: float = 1.0,
        exclude_camera_descendants: bool = True,
        drop_non_physical_meshes: bool = True,
        collapse_contained_meshes: bool = True,
        controlled_camera: dict[str, Any] | None = None,
        timeout_seconds: int = 120,
        page_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.entry_html = Path(entry_html).expanduser().resolve()
        # Everything the page may fetch has to be reachable from the served root,
        # so a game whose assets sit above the entry document needs it widened.
        self.game_root = (
            Path(game_root).expanduser().resolve() if game_root is not None else self.entry_html.parent
        )
        self.three_replacement = (
            Path(three_replacement).expanduser().resolve() if three_replacement is not None else None
        )
        self.width = int(width)
        self.height = int(height)
        self.seed = int(seed)
        self.step_ms = float(step_ms)
        self.warmup_frames = int(warmup_frames)
        self.views = tuple(views)
        self.max_vertices_per_object = int(max_vertices_per_object)
        self.unit_scale = float(unit_scale)
        self.exclude_camera_descendants = bool(exclude_camera_descendants)
        # Individualization happens in the exporter, not in the page: telling an
        # outline shell from a level boundary needs the fitted boxes, which only
        # exist on the Python side.
        self.drop_non_physical_meshes = bool(drop_non_physical_meshes)
        self.collapse_contained_meshes = bool(collapse_contained_meshes)
        self.controlled_camera = _controlled_camera_config(controlled_camera)
        self.timeout_seconds = int(timeout_seconds)
        self._page_factory = page_factory

    def render_scene(
        self,
        *,
        scene_path: str | Path,
        out_dir: str | Path,
        asset_root: str | Path | None = None,
    ) -> dict[str, Any]:
        """Produce overview renders plus collision geometry beneath ``out_dir``."""

        destination = Path(out_dir).expanduser().resolve()
        submitted = read_json(Path(scene_path))
        return self.capture_game_source(
            out_dir=destination,
            scene_id=str(submitted.get("scene_id") or "game_scene"),
            request_id=str(submitted.get("request_id") or "game_request"),
            scene_type=str(submitted.get("scene_type") or "game_level"),
            require_probe=False,
        )

    def capture_game_source(
        self,
        *,
        out_dir: str | Path,
        scene_id: str,
        request_id: str,
        scene_type: str = "game_level",
        require_probe: bool = True,
    ) -> dict[str, Any]:
        """Capture one frozen browser state and export its canonical artifacts.

        Unlike :meth:`render_scene`, this entry point does not require a
        pre-existing canonical scene.  It is the bootstrap used by the Game
        ingestion route: the browser source is probed first, and the resulting
        canonical scene then becomes the exact scene passed to the evaluator.
        """

        destination = Path(out_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        exported_scene: dict[str, Any] | None = None
        collision_geometry: dict[str, Any] | None = None
        probe_path: Path | None = None
        geometry_manifest_path: Path | None = None
        exported_scene_path: Path | None = None
        with self._open_page() as page:
            authored_views = self._capture_views(page, destination)
            probe_payload = page.probe(
                _read_asset("probe.js"),
                {
                    "maxVerticesPerObject": self.max_vertices_per_object,
                    "unitScale": self.unit_scale,
                    "tick": self._total_frames(),
                    "seed": self.seed,
                    "excludeCameraDescendants": self.exclude_camera_descendants,
                },
            )
            rewrite_report = getattr(page, "rewrite_report", None)
            if probe_payload is not None:
                probe_path = Path(
                    write_json(destination / "probe_payload.json", probe_payload)
                )
                # Imported here because the exporter reaches into the evaluator
                # package, which imports benchmark.rendering in turn.
                from benchmark.game_scene.exporter import build_scene_and_collision_geometry

                exported_scene, collision_geometry = build_scene_and_collision_geometry(
                    probe_payload,
                    scene_id=scene_id,
                    request_id=request_id,
                    scene_type=scene_type,
                    mesh_dir=destination / "collision_geometry",
                    drop_non_physical=self.drop_non_physical_meshes,
                    collapse_contained=self.collapse_contained_meshes,
                )
                geometry_manifest_path = Path(
                    write_json(
                        destination / "collision_geometry_manifest.json",
                        collision_geometry,
                    )
                )
                collision_geometry["manifest_path"] = geometry_manifest_path.as_posix()
                exported_scene_path = Path(
                    write_json(
                        destination / "probe_exported_scene.json",
                        exported_scene,
                    )
                )
            views, controlled_camera_report = self._resolve_evidence_views(
                page=page,
                destination=destination,
                authored_views=authored_views,
                exported_scene=exported_scene,
            )
        if require_probe and probe_payload is None:
            raise BrowserRenderError(
                "game_mode requires an instrumented Three.js scene, but the "
                "browser probe returned no scene"
            )

        manifest: dict[str, Any] = {
            "backend": BROWSER_RENDER_BACKEND,
            "entry_html": self.entry_html.as_posix(),
            "game_root": self.game_root.as_posix(),
            "rotation_unit": "degree",
            "render_config": {
                "width": self.width,
                "height": self.height,
                "seed": self.seed,
                "step_ms": self.step_ms,
                "warmup_frames": self.warmup_frames,
                "views": [dict(view) for view in self.views],
                "max_vertices_per_object": self.max_vertices_per_object,
                "unit_scale": self.unit_scale,
                "exclude_camera_descendants": self.exclude_camera_descendants,
                "drop_non_physical_meshes": self.drop_non_physical_meshes,
                "collapse_contained_meshes": self.collapse_contained_meshes,
                "controlled_camera": dict(self.controlled_camera),
            },
            "source_scene_identity": {
                "scene_id": scene_id,
                "request_id": request_id,
                "scene_type": scene_type,
            },
            "views": views,
            "authored_camera_views": authored_views,
            "controlled_camera": controlled_camera_report,
            "probe_available": probe_payload is not None,
        }
        if rewrite_report is not None:
            manifest["page_instrumentation"] = rewrite_report
        if exported_scene is not None and collision_geometry is not None:
            assert probe_path is not None
            assert exported_scene_path is not None
            assert geometry_manifest_path is not None
            manifest["collision_geometry"] = collision_geometry
            manifest["exported_scene"] = exported_scene_path.as_posix()
            manifest["probe_payload"] = probe_path.as_posix()
            manifest["capture_artifacts"] = _capture_artifact_hashes(
                destination=destination,
                views=views,
                supplemental_views=(
                    controlled_camera_report.get("style_local_fallback", {}).get(
                        "views", []
                    )
                ),
                authored_views=authored_views,
                probe_path=probe_path,
                exported_scene_path=exported_scene_path,
                collision_geometry=collision_geometry,
                collision_manifest_path=geometry_manifest_path,
            )
        write_json(destination / "render_manifest.json", manifest)
        return manifest

    def _total_frames(self) -> int:
        return self.warmup_frames + sum(int(view.get("step_frames", 0)) for view in self.views)

    def _capture_views(self, page: Any, destination: Path) -> list[dict[str, Any]]:
        page.step(self.warmup_frames)
        views: list[dict[str, Any]] = []
        elapsed_frames = self.warmup_frames
        for view in self.views:
            frames = int(view.get("step_frames", 0))
            if frames:
                page.step(frames)
                elapsed_frames += frames
            name = str(view.get("name") or f"view_{len(views):02d}")
            path = destination / f"standardized_{name}.png"
            page.screenshot(path)
            if not path.is_file():
                raise BrowserRenderError(f"headless browser produced no image for view {name!r}")
            views.append({"name": name, "path": path.as_posix(), "tick": elapsed_frames})
        if not views:
            raise BrowserRenderError("headless browser renderer produced no overview views")
        return views

    def _resolve_evidence_views(
        self,
        *,
        page: Any,
        destination: Path,
        authored_views: list[dict[str, Any]],
        exported_scene: dict[str, Any] | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        config = self.controlled_camera
        if not config["enabled"]:
            return authored_views, {
                "enabled": False,
                "status": "authored_camera_fallback",
                "appearance_fidelity": "original_runtime_authored_camera",
            }
        if exported_scene is None:
            if config["required"]:
                raise UnsupportedBrowserRenderPipelineError(
                    "controlled camera requires a canonical scene exported from "
                    "an instrumented Three.js runtime"
                )
            return authored_views, {
                "enabled": True,
                "status": "unavailable_authored_camera_fallback",
                "reason": "canonical_scene_unavailable",
                "appearance_fidelity": "original_runtime_authored_camera",
            }
        poses = _controlled_global_camera_poses(
            exported_scene,
            image_budget=int(config["image_budget"]),
            view_family=str(config["view_family"]),
        )
        try:
            views = page.render_camera_views(
                _read_asset("render_camera_view.js"),
                poses,
                destination,
            )
        except Exception as exc:
            if config["required"]:
                if isinstance(exc, UnsupportedBrowserRenderPipelineError):
                    raise
                raise UnsupportedBrowserRenderPipelineError(
                    "original Three.js runtime cannot provide benchmark-controlled "
                    f"camera evidence: {type(exc).__name__}"
                ) from exc
            return authored_views, {
                "enabled": True,
                "status": "unavailable_authored_camera_fallback",
                "reason": type(exc).__name__,
                "appearance_fidelity": "original_runtime_authored_camera",
            }
        if len(views) != int(config["image_budget"]):
            raise UnsupportedBrowserRenderPipelineError(
                "controlled camera returned an incomplete view set: "
                f"expected {config['image_budget']}, got {len(views)}"
            )
        style_local_views: list[dict[str, Any]] = []
        style_local_poses: list[dict[str, Any]] = []
        if config["style_local_fallback_enabled"]:
            style_local_poses = _controlled_style_local_camera_poses(
                exported_scene,
                image_budget=int(config["style_local_image_budget"]),
                view_family=str(config["style_local_view_family"]),
            )
            try:
                style_local_views = page.render_camera_views(
                    _read_asset("render_camera_view.js"),
                    style_local_poses,
                    destination,
                )
            except Exception as exc:
                if isinstance(exc, UnsupportedBrowserRenderPipelineError):
                    raise
                raise UnsupportedBrowserRenderPipelineError(
                    "original Three.js runtime cannot provide benchmark-controlled "
                    f"style-local evidence: {type(exc).__name__}"
                ) from exc
            if len(style_local_views) != int(config["style_local_image_budget"]):
                raise UnsupportedBrowserRenderPipelineError(
                    "controlled style-local camera returned an incomplete view set: "
                    f"expected {config['style_local_image_budget']}, "
                    f"got {len(style_local_views)}"
                )
            for view, pose in zip(style_local_views, style_local_poses):
                view["scope"] = "object_local"
                view["role"] = "style_local_fallback"
                view["target_object_ids"] = list(
                    pose.get("target_object_ids") or []
                )
        return views, {
            "enabled": True,
            "status": "ready",
            "renderer_backend": "threejs_original_runtime",
            "view_family": config["view_family"],
            "image_budget": config["image_budget"],
            "canvas_only": config["canvas_only"],
            "appearance_fidelity": CONTROLLED_CAMERA_APPEARANCE_FIDELITY,
            "poses": poses,
            "style_local_fallback": {
                "enabled": bool(config["style_local_fallback_enabled"]),
                "status": (
                    "ready"
                    if config["style_local_fallback_enabled"]
                    else "disabled"
                ),
                "view_family": config["style_local_view_family"],
                "image_budget": (
                    config["style_local_image_budget"]
                    if config["style_local_fallback_enabled"]
                    else 0
                ),
                "consumption_policy": (
                    "global_screen_suspicious_or_insufficient_only"
                ),
                "views": style_local_views,
                "poses": style_local_poses,
            },
        }

    def _open_page(self) -> Any:
        if self._page_factory is not None:
            return self._page_factory(renderer=self)
        return _PlaywrightPage(self)


def _read_asset(name: str) -> str:
    return (_ASSET_DIR / name).read_text(encoding="utf-8")


class _PlaywrightPage:
    """Context-managed Playwright page wired to the deterministic harness."""

    def __init__(self, renderer: HeadlessBrowserRenderer) -> None:
        self._renderer = renderer
        self._playwright = None
        self._browser = None
        self._page = None
        self._cdp_session = None
        self._server: Any | None = None

    def __enter__(self) -> "_PlaywrightPage":
        try:
            from playwright.sync_api import sync_playwright  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise BrowserRenderError(
                "HeadlessBrowserRenderer requires the 'browser' extra: pip install '3d-layout-benchmark[browser]' "
                "and then 'playwright install chromium'"
            ) from exc
        # Imported here because benchmark.game_scene reaches into the evaluator,
        # which imports benchmark.rendering in turn; a module-level import would
        # close that cycle and leave evaluator modules partially initialised.
        from benchmark.game_scene.harness import GameSourceServer

        renderer = self._renderer
        if not renderer.entry_html.is_file():
            raise BrowserRenderError(f"game entry HTML does not exist: {renderer.entry_html}")
        # The server rewrites the entry document, so the determinism harness and
        # the Scene-constructor trap are already the page's first script. No
        # driver-level init script is needed, and installing one would double up.
        self._server = GameSourceServer(
            renderer.game_root,
            entry=renderer.entry_html,
            seed=renderer.seed,
            step_ms=renderer.step_ms,
            three_replacement=renderer.three_replacement,
        ).__enter__()
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(args=["--use-gl=swiftshader"])
        self._page = self._browser.new_page(
            viewport={"width": renderer.width, "height": renderer.height}
        )
        self._cdp_session = self._page.context.new_cdp_session(self._page)
        self._page.set_default_timeout(renderer.timeout_seconds * 1000)
        self._page.goto(self._server.entry_url, wait_until="load")
        return self

    def __exit__(self, *exc_info: object) -> None:
        # Teardown must never mask the failure that is already propagating.
        for close in (
            lambda: self._browser.close() if self._browser is not None else None,
            lambda: self._playwright.stop() if self._playwright is not None else None,
            lambda: self._server.__exit__() if self._server is not None else None,
        ):
            try:
                close()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass

    @property
    def rewrite_report(self) -> dict[str, Any] | None:
        if self._server is None or self._server.rewrite_report is None:
            return None
        return self._server.rewrite_report.to_dict()

    def step(self, frames: int) -> None:
        if frames <= 0:
            return
        self._evaluate(f"window.__benchmarkStep({int(frames)})")

    def screenshot(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._page.screenshot(path=str(path))

    def probe(self, script: str, options: dict[str, Any]) -> dict[str, Any] | None:
        return self._evaluate(f"({script})({json.dumps(options)})")

    def render_camera_views(
        self,
        script: str,
        poses: list[dict[str, Any]],
        destination: Path,
    ) -> list[dict[str, Any]]:
        views: list[dict[str, Any]] = []
        image_hashes: set[str] = set()
        for pose in poses:
            view_id = str(pose["id"])
            result: dict[str, Any] | None = None
            png_bytes: bytes | None = None
            # SwiftShader can very occasionally return a blank drawing buffer
            # during sequential browser startup. Retry the exact same render
            # once, then fail loudly; never score a blank view as evidence.
            for attempt in range(2):
                response = self._evaluate(
                    f"({script})"
                    f"({json.dumps({'view_id': view_id, 'pose': pose['source_pose']})})"
                )
                if not isinstance(response, dict) or response.get("ok") is not True:
                    code = (
                        str(response.get("code"))
                        if isinstance(response, dict)
                        else "invalid_controlled_camera_response"
                    )
                    raise UnsupportedBrowserRenderPipelineError(
                        f"controlled Three.js camera is unavailable: {code}"
                    )
                data_url = str(response.get("image_data_url") or "")
                prefix = "data:image/png;base64,"
                if not data_url.startswith(prefix):
                    raise UnsupportedBrowserRenderPipelineError(
                        "controlled Three.js camera returned no PNG canvas readback"
                    )
                try:
                    candidate_bytes = base64.b64decode(
                        data_url[len(prefix) :],
                        validate=True,
                    )
                except (ValueError, TypeError) as exc:
                    raise UnsupportedBrowserRenderPipelineError(
                        "controlled Three.js camera returned malformed PNG data"
                    ) from exc
                if not candidate_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
                    raise UnsupportedBrowserRenderPipelineError(
                        "controlled Three.js camera readback is not a PNG"
                    )
                if _png_is_exactly_uniform(candidate_bytes):
                    if attempt == 0:
                        self._reload_for_controlled_camera()
                        continue
                    raise BrowserRenderError(
                        f"controlled Three.js camera produced a uniform image for {view_id!r}"
                    )
                result = response
                result["capture_retry_count"] = attempt
                png_bytes = candidate_bytes
                break
            assert result is not None
            assert png_bytes is not None
            digest = hashlib.sha256(png_bytes).hexdigest()
            if digest in image_hashes:
                raise BrowserRenderError(
                    "controlled Three.js camera produced duplicate evidence views"
                )
            image_hashes.add(digest)
            scope = str(pose.get("scope") or "global")
            prefix = "local" if scope != "global" else "global"
            path = destination / f"{prefix}_{view_id}.png"
            path.write_bytes(png_bytes)
            if not path.is_file():
                raise BrowserRenderError(
                    f"controlled Three.js camera produced no image for {view_id!r}"
                )
            views.append(
                {
                    "id": view_id,
                    "name": view_id,
                    "path": path.as_posix(),
                    "scope": scope,
                    "role": pose.get("role"),
                    "presentation": "raw",
                    "backend": "threejs_original_runtime",
                    "appearance_fidelity": result["appearance_fidelity"],
                    "camera_pose_canonical": pose["canonical_pose"],
                    "camera_pose_source": pose["source_pose"],
                    "target_object_ids": list(
                        pose.get("target_object_ids") or []
                    ),
                    "runtime_diagnostics": {
                        "registered_renderer_count": result.get(
                            "registered_renderer_count"
                        ),
                        "direct_render_call_count": result.get(
                            "direct_render_call_count"
                        ),
                        "canvas_width": result.get("canvas_width"),
                        "canvas_height": result.get("canvas_height"),
                        "capture_retry_count": result.get(
                            "capture_retry_count",
                            0,
                        ),
                        "suppressed_runtime_mesh_count": int(
                            result.get("suppressed_runtime_mesh_count") or 0
                        ),
                    },
                }
            )
        return views

    def _reload_for_controlled_camera(self) -> None:
        """Rebuild one deterministic page after a transient blank WebGL frame."""

        if self._page is None:
            raise BrowserRenderError("Chromium page is unavailable for render retry")
        self._page.reload(wait_until="load")
        # The source capture has already advanced through every authored
        # diagnostic state before controlled rendering. Replaying the same
        # frozen number of frames restores the same scene state.
        self.step(self._renderer._total_frames())

    def _evaluate(self, expression: str) -> Any:
        """Evaluate through CDP without depending on page-overwritable builtins.

        Generated games sometimes assign domain objects to names such as
        ``window.Map``. Playwright's main-world argument/result serializer uses
        that builtin and then fails before our script executes. CDP transports
        the return value directly and remains valid without altering the page.
        """

        if self._cdp_session is None:
            raise BrowserRenderError("Chromium CDP session is unavailable")
        response = self._cdp_session.send(
            "Runtime.evaluate",
            {
                "expression": expression,
                "returnByValue": True,
                "awaitPromise": True,
            },
        )
        if response.get("exceptionDetails") is not None:
            details = response["exceptionDetails"]
            safe_name = (
                details.get("exception", {}).get("className")
                if isinstance(details, dict)
                else None
            )
            raise BrowserRenderError(
                "browser runtime evaluation failed"
                + (f": {safe_name}" if safe_name else "")
            )
        result = response.get("result") or {}
        return result.get("value")


class FrozenBrowserCaptureRenderer:
    """Expose one already-captured browser observation as a trusted renderer.

    The Game route must probe before it can author the canonical scene and case
    bundle.  Re-running the browser during evaluation would create a second
    observation and break the identity between the scored scene, collision
    geometry, and pixels.  This renderer validates and reuses the first capture
    only when the evaluator asks for the same output directory and geometry.
    """

    def __init__(self, *, capture_dir: str | Path) -> None:
        self.capture_dir = Path(capture_dir).expanduser().resolve()
        self.manifest_path = self.capture_dir / "render_manifest.json"
        if not self.manifest_path.is_file():
            raise BrowserRenderError(
                f"frozen browser capture manifest does not exist: {self.manifest_path}"
            )
        self.manifest = read_json(self.manifest_path)
        if self.manifest.get("backend") != BROWSER_RENDER_BACKEND:
            raise BrowserRenderError("frozen browser capture has an unexpected backend")
        exported_path = Path(str(self.manifest.get("exported_scene") or "")).expanduser().resolve()
        if not exported_path.is_file():
            raise BrowserRenderError("frozen browser capture has no exported canonical scene")
        self.exported_scene = read_json(exported_path)
        self._validate_artifacts()

    def render_scene(
        self,
        *,
        scene_path: str | Path,
        out_dir: str | Path,
        asset_root: str | Path | None = None,
    ) -> dict[str, Any]:
        del asset_root
        requested_dir = Path(out_dir).expanduser().resolve()
        if requested_dir != self.capture_dir:
            raise BrowserRenderError(
                "frozen browser evidence can only be reused from its original "
                "trusted output directory"
            )
        requested_scene = read_json(Path(scene_path))
        if _scene_geometry_signature(requested_scene) != _scene_geometry_signature(
            self.exported_scene
        ):
            raise BrowserRenderError(
                "frozen browser capture does not match the canonical scene being evaluated"
            )
        self._validate_artifacts()
        return read_json(self.manifest_path)

    def provide_scene_quality_evidence(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        """Return frozen local style views without making a metric decision."""

        if request.get("metric") != "style_consistency":
            return {
                "status": "insufficient",
                "reason": "frozen_browser_provider_supports_style_only",
                "render_evidence_items": [],
            }
        if request.get("evidence_scope") != "object_local":
            return {
                "status": "insufficient",
                "reason": "unsupported_frozen_browser_evidence_scope",
                "render_evidence_items": [],
            }
        controlled = self.manifest.get("controlled_camera")
        fallback = (
            controlled.get("style_local_fallback")
            if isinstance(controlled, dict)
            else None
        )
        if not isinstance(fallback, dict) or fallback.get("status") != "ready":
            return {
                "status": "insufficient",
                "reason": "style_local_fallback_bank_unavailable",
                "render_evidence_items": [],
            }
        views = [
            dict(item)
            for item in fallback.get("views") or []
            if isinstance(item, dict) and item.get("path")
        ]
        target_ids = {
            str(item)
            for item in request.get("object_ids") or []
            if str(item)
        }
        if target_ids:
            views.sort(
                key=lambda item: (
                    -len(
                        target_ids.intersection(
                            str(value)
                            for value in item.get("target_object_ids") or []
                        )
                    ),
                    str(item.get("id") or item.get("name") or ""),
                )
            )
        policy = (
            request.get("evidence_policy")
            if isinstance(request.get("evidence_policy"), dict)
            else {}
        )
        budget = max(
            1,
            min(
                int(
                    policy.get("image_budget")
                    or fallback.get("image_budget")
                    or len(views)
                    or 1
                ),
                len(views),
            ),
        )
        selected = views[:budget]
        if not selected:
            return {
                "status": "insufficient",
                "reason": "style_local_fallback_bank_empty",
                "render_evidence_items": [],
            }
        self._validate_artifacts()
        return {
            "status": "available",
            "selection_role": "visual_evidence_only_do_not_judge_metric",
            "render_evidence_items": [
                {
                    "path": item["path"],
                    "role": "object_local",
                    "target_object_ids": list(
                        item.get("target_object_ids") or []
                    ),
                }
                for item in selected
            ],
        }

    def _validate_artifacts(self) -> None:
        hashes = self.manifest.get("capture_artifacts")
        if not isinstance(hashes, dict) or not hashes:
            raise BrowserRenderError("frozen browser capture is missing artifact hashes")
        for relative, expected in hashes.items():
            path = (self.capture_dir / str(relative)).resolve()
            try:
                path.relative_to(self.capture_dir)
            except ValueError as exc:
                raise BrowserRenderError("frozen browser artifact escaped its capture root") from exc
            if not path.is_file():
                raise BrowserRenderError(f"frozen browser artifact does not exist: {path}")
            if _sha256(path) != expected:
                raise BrowserRenderError(f"frozen browser artifact hash mismatch: {path}")


def _capture_artifact_hashes(
    *,
    destination: Path,
    views: list[dict[str, Any]],
    supplemental_views: list[dict[str, Any]],
    authored_views: list[dict[str, Any]],
    probe_path: Path,
    exported_scene_path: Path,
    collision_geometry: dict[str, Any],
    collision_manifest_path: Path,
) -> dict[str, str]:
    paths = [
        *(Path(str(view["path"])) for view in views),
        *(Path(str(view["path"])) for view in supplemental_views),
        *(Path(str(view["path"])) for view in authored_views),
        probe_path,
        exported_scene_path,
        collision_manifest_path,
    ]
    for entry in collision_geometry.get("objects", {}).values():
        if not isinstance(entry, dict):
            continue
        raw_path = entry.get("geometry_path")
        if raw_path:
            paths.append(Path(str(raw_path)))
    records: dict[str, str] = {}
    for path in paths:
        resolved = path.expanduser().resolve()
        try:
            relative = resolved.relative_to(destination)
        except ValueError as exc:
            raise BrowserRenderError("captured browser artifact escaped its output directory") from exc
        if not resolved.is_file():
            raise BrowserRenderError(f"captured browser artifact does not exist: {resolved}")
        records[relative.as_posix()] = _sha256(resolved)
    return records


def _png_is_exactly_uniform(png_bytes: bytes) -> bool:
    """Reject only a truly constant frame, not a merely dark or simple scene."""

    try:
        with Image.open(BytesIO(png_bytes)) as image:
            extrema = image.convert("RGB").getextrema()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise UnsupportedBrowserRenderPipelineError(
            "controlled Three.js camera returned an unreadable PNG"
        ) from exc
    return all(low == high for low, high in extrema)


def _controlled_camera_config(value: dict[str, Any] | None) -> dict[str, Any]:
    config = dict(DEFAULT_CONTROLLED_CAMERA)
    if value is not None:
        if not isinstance(value, dict):
            raise TypeError("controlled_camera must be a mapping")
        unknown = set(value) - set(config)
        if unknown:
            raise ValueError(
                f"controlled_camera has unsupported keys {sorted(unknown)}"
            )
        config.update(value)
    if config["view_family"] != CONTROLLED_CAMERA_VIEW_FAMILY:
        raise ValueError(
            "controlled_camera.view_family must be "
            f"{CONTROLLED_CAMERA_VIEW_FAMILY!r}"
        )
    if (
        isinstance(config["image_budget"], bool)
        or not isinstance(config["image_budget"], int)
        or not 1 <= config["image_budget"] <= 2
    ):
        raise ValueError("controlled_camera.image_budget must be 1 or 2")
    if not isinstance(config["style_local_fallback_enabled"], bool):
        raise TypeError(
            "controlled_camera.style_local_fallback_enabled must be boolean"
        )
    if config["style_local_view_family"] != CONTROLLED_STYLE_LOCAL_VIEW_FAMILY:
        raise ValueError(
            "controlled_camera.style_local_view_family must be "
            f"{CONTROLLED_STYLE_LOCAL_VIEW_FAMILY!r}"
        )
    if (
        isinstance(config["style_local_image_budget"], bool)
        or not isinstance(config["style_local_image_budget"], int)
        or not 1 <= config["style_local_image_budget"] <= 4
    ):
        raise ValueError(
            "controlled_camera.style_local_image_budget must be between 1 and 4"
        )
    for key in (
        "enabled",
        "required",
        "canvas_only",
        "include_authored_camera_diagnostics",
    ):
        if not isinstance(config[key], bool):
            raise TypeError(f"controlled_camera.{key} must be boolean")
    if config["unsupported_render_pipeline"] != "fail_not_ingestable":
        raise ValueError(
            "controlled_camera.unsupported_render_pipeline must be "
            "'fail_not_ingestable'"
        )
    if config["enabled"] is False and config["required"] is True:
        raise ValueError("controlled_camera.required requires enabled=true")
    return config


def _controlled_global_camera_poses(
    scene: dict[str, Any],
    *,
    image_budget: int,
    view_family: str,
) -> list[dict[str, Any]]:
    boundary = scene.get("boundary")
    if not isinstance(boundary, list) or len(boundary) < 3:
        raise BrowserRenderError(
            "controlled global camera requires canonical scene boundary"
        )
    boundary_xs = [float(point[0]) for point in boundary]
    boundary_ys = [float(point[1]) for point in boundary]
    scene_floor_z = 0.0
    scene_ceiling_z = float(scene.get("scene_height") or 0.0)
    if scene_ceiling_z <= scene_floor_z:
        raise BrowserRenderError(
            "controlled global camera requires positive scene_height"
        )
    focus = _game_visual_focus_bounds(scene)
    if focus is None:
        min_x, max_x = min(boundary_xs), max(boundary_xs)
        min_y, max_y = min(boundary_ys), max(boundary_ys)
        floor_z, ceiling_z = scene_floor_z, scene_ceiling_z
        framing_bounds_source = "canonical_scene_envelope"
    else:
        minimum, maximum = focus
        min_x, min_y, floor_z = minimum
        max_x, max_y, ceiling_z = maximum
        framing_bounds_source = "non_flat_visual_structure_envelope"
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    width = max_x - min_x
    depth = max_y - min_y
    span = max(width, depth, ceiling_z, 0.5)
    target = [
        center_x,
        center_y,
        min(ceiling_z * 0.38, floor_z + max(1.0, ceiling_z * 0.55)),
    ]
    camera_z = max(ceiling_z + span * 0.55, target[2] + span * 0.70)
    canonical_locations = [
        [center_x + span * 0.82, center_y - span * 0.92, camera_z],
        [
            center_x - span * 0.92,
            center_y - span * 0.42,
            target[2] + (camera_z - target[2]) * 0.96,
        ],
    ]
    metadata = scene.get("metadata") if isinstance(scene.get("metadata"), dict) else {}
    imported = (
        metadata.get("game_scene_import")
        if isinstance(metadata.get("game_scene_import"), dict)
        else {}
    )
    unit_scale = float(imported.get("unit_scale") or 1.0)
    translation = imported.get("translation_applied")
    source_up_axis = str(imported.get("source_up_axis") or "y").lower()
    if (
        not isinstance(translation, list)
        or len(translation) != 3
        or unit_scale <= 0
        or source_up_axis not in {"y", "z"}
    ):
        raise BrowserRenderError(
            "controlled camera requires invertible game_scene_import transform"
        )
    vertical_fov = 50.0
    near_m = 0.02
    full_scene_span = max(
        max(boundary_xs) - min(boundary_xs),
        max(boundary_ys) - min(boundary_ys),
        scene_ceiling_z,
    )
    far_m = max(100.0, full_scene_span * 12.0, scene_ceiling_z * 12.0)
    poses: list[dict[str, Any]] = []
    for index, location in enumerate(canonical_locations[:image_budget]):
        canonical_pose = {
            "camera_type": "PERSP",
            "location": location,
            "target": target,
            "vertical_fov_degrees": vertical_fov,
            "near_m": near_m,
            "far_m": far_m,
            "framing_bounds_source": framing_bounds_source,
            "framing_bounds": [
                [min_x, min_y, floor_z],
                [max_x, max_y, ceiling_z],
            ],
        }
        source_pose = {
            "camera_type": "PERSP",
            "location": _canonical_point_to_source(
                location,
                translation=translation,
                unit_scale=unit_scale,
                source_up_axis=source_up_axis,
            ),
            "target": _canonical_point_to_source(
                target,
                translation=translation,
                unit_scale=unit_scale,
                source_up_axis=source_up_axis,
            ),
            "up": [0.0, 1.0, 0.0] if source_up_axis == "y" else [0.0, 0.0, 1.0],
            "vertical_fov_degrees": vertical_fov,
            "near": near_m / unit_scale,
            "far": far_m / unit_scale,
        }
        poses.append(
            {
                "id": f"global_oblique_{index:02d}",
                "view_family": view_family,
                "canonical_pose": canonical_pose,
                "source_pose": source_pose,
            }
        )
    return poses


def _controlled_style_local_camera_poses(
    scene: dict[str, Any],
    *,
    image_budget: int,
    view_family: str,
) -> list[dict[str, Any]]:
    """Build a small scene-agnostic regional bank for conditional style review.

    The bank is rendered during the same frozen browser capture as the two
    global views, but is not shown to the final judge unless the global style
    screen is suspicious or insufficient. Regions are defined only from
    canonical geometry; this is one reusable Three.js capability, not a
    Counter-Strike scene adapter.
    """

    focus = _game_visual_focus_bounds(scene)
    boundary = scene.get("boundary")
    if not isinstance(boundary, list) or len(boundary) < 3:
        raise BrowserRenderError(
            "controlled style-local camera requires canonical scene boundary"
        )
    if focus is None:
        xs = [float(point[0]) for point in boundary]
        ys = [float(point[1]) for point in boundary]
        minimum = [min(xs), min(ys), 0.0]
        maximum = [
            max(xs),
            max(ys),
            float(scene.get("scene_height") or 0.0),
        ]
    else:
        minimum, maximum = focus
    min_x, min_y, floor_z = minimum
    max_x, max_y, ceiling_z = maximum
    if max_x <= min_x or max_y <= min_y or ceiling_z <= floor_z:
        raise BrowserRenderError(
            "controlled style-local camera requires non-degenerate scene bounds"
        )

    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    x_edges = [min_x, center_x, max_x]
    y_edges = [min_y, center_y, max_y]
    region_indices = [(0, 0), (1, 0), (0, 1), (1, 1)]
    metadata = (
        scene.get("metadata")
        if isinstance(scene.get("metadata"), dict)
        else {}
    )
    imported = (
        metadata.get("game_scene_import")
        if isinstance(metadata.get("game_scene_import"), dict)
        else {}
    )
    unit_scale = float(imported.get("unit_scale") or 1.0)
    translation = imported.get("translation_applied")
    source_up_axis = str(imported.get("source_up_axis") or "y").lower()
    if (
        not isinstance(translation, list)
        or len(translation) != 3
        or unit_scale <= 0
        or source_up_axis not in {"y", "z"}
    ):
        raise BrowserRenderError(
            "controlled camera requires invertible game_scene_import transform"
        )

    full_span = max(max_x - min_x, max_y - min_y, ceiling_z - floor_z, 1.0)
    regional_span = max((max_x - min_x) / 2.0, (max_y - min_y) / 2.0, 1.0)
    vertical_span = ceiling_z - floor_z
    target_z = min(
        max(floor_z + vertical_span * 0.32, floor_z + 0.05),
        ceiling_z - 0.05,
    )
    far_m = max(100.0, full_span * 12.0)
    offsets = (
        (0.62, -0.74),
        (-0.72, -0.56),
        (0.70, 0.58),
        (-0.58, 0.74),
    )
    objects = [
        item
        for item in scene.get("objects") or []
        if isinstance(item, dict)
        and isinstance(item.get("center"), list)
        and len(item["center"]) == 3
    ]
    poses: list[dict[str, Any]] = []
    for index, (x_index, y_index) in enumerate(region_indices[:image_budget]):
        region_min_x, region_max_x = x_edges[x_index], x_edges[x_index + 1]
        region_min_y, region_max_y = y_edges[y_index], y_edges[y_index + 1]
        target = [
            (region_min_x + region_max_x) / 2.0,
            (region_min_y + region_max_y) / 2.0,
            target_z,
        ]
        offset_x, offset_y = offsets[index]
        location = [
            target[0] + regional_span * offset_x,
            target[1] + regional_span * offset_y,
            target[2] + max(regional_span * 0.72, (ceiling_z - floor_z) * 0.55),
        ]
        target_object_ids = [
            str(item.get("id"))
            for item in objects
            if region_min_x <= float(item["center"][0]) <= region_max_x
            and region_min_y <= float(item["center"][1]) <= region_max_y
        ]
        canonical_pose = {
            "camera_type": "PERSP",
            "location": location,
            "target": target,
            "vertical_fov_degrees": 48.0,
            "near_m": 0.02,
            "far_m": far_m,
            "region_bounds": [
                [region_min_x, region_min_y, floor_z],
                [region_max_x, region_max_y, ceiling_z],
            ],
        }
        source_pose = {
            "camera_type": "PERSP",
            "location": _canonical_point_to_source(
                location,
                translation=translation,
                unit_scale=unit_scale,
                source_up_axis=source_up_axis,
            ),
            "target": _canonical_point_to_source(
                target,
                translation=translation,
                unit_scale=unit_scale,
                source_up_axis=source_up_axis,
            ),
            "up": (
                [0.0, 1.0, 0.0]
                if source_up_axis == "y"
                else [0.0, 0.0, 1.0]
            ),
            "vertical_fov_degrees": 48.0,
            "near": 0.02 / unit_scale,
            "far": far_m / unit_scale,
        }
        poses.append(
            {
                "id": f"style_region_{index:02d}",
                "scope": "object_local",
                "role": "style_local_fallback",
                "view_family": view_family,
                "target_object_ids": target_object_ids,
                "canonical_pose": canonical_pose,
                "source_pose": source_pose,
            }
        )
    return poses


def _game_visual_focus_bounds(
    scene: dict[str, Any],
) -> tuple[list[float], list[float]] | None:
    """Frame 3D structure without letting an oversized ground plane dominate.

    Browser games commonly use one enormous PlaneGeometry as a ground/sky
    backdrop. It remains in the rendered scene, but its almost-zero thickness
    makes it a presentation surface rather than a useful camera-framing target.
    This geometry-only rule is identical for every game and never reads names,
    categories, or scene-specific constants.
    """

    bounds: list[tuple[list[float], list[float]]] = []
    for item in scene.get("objects") or []:
        if not isinstance(item, dict):
            continue
        center = item.get("center")
        size = item.get("size")
        if (
            not isinstance(center, list)
            or len(center) != 3
            or not isinstance(size, list)
            or len(size) != 3
        ):
            continue
        rotation = item.get("rotation")
        if not isinstance(rotation, list) or len(rotation) != 3:
            rotation = [0.0, 0.0, 0.0]
        values = [float(value) for value in [*center, *size, *rotation]]
        if not all(math.isfinite(value) for value in values):
            continue
        sx, sy, sz = values[3:6]
        if min(sx, sy, sz) <= 0:
            continue
        center_vector = np.asarray(values[:3], dtype=float)
        half = np.asarray([sx, sy, sz], dtype=float) / 2.0
        rotation_matrix = rotation_matrix_from_euler(values[6:9])
        local_corners = np.asarray(
            [
                [sign_x * half[0], sign_y * half[1], sign_z * half[2]]
                for sign_x in (-1.0, 1.0)
                for sign_y in (-1.0, 1.0)
                for sign_z in (-1.0, 1.0)
            ],
            dtype=float,
        )
        world_corners = center_vector + local_corners @ rotation_matrix.T
        world_min = world_corners.min(axis=0)
        world_max = world_corners.max(axis=0)
        world_extent = world_max - world_min
        # Canonical size is expressed in the object's local OBB axes. A
        # PlaneGeometry floor can therefore look tall in local Z after export.
        # Classify presentation planes only after rotating into world space.
        if world_extent[2] <= max(
            0.02,
            0.01 * max(float(world_extent[0]), float(world_extent[1])),
        ):
            continue
        bounds.append(
            (
                [float(value) for value in world_min],
                [float(value) for value in world_max],
            )
        )
    if not bounds:
        return None
    minimum = [min(entry[0][axis] for entry in bounds) for axis in range(3)]
    maximum = [max(entry[1][axis] for entry in bounds) for axis in range(3)]
    if any(maximum[axis] <= minimum[axis] for axis in range(3)):
        return None
    return minimum, maximum


def _canonical_point_to_source(
    point: list[float],
    *,
    translation: list[Any],
    unit_scale: float,
    source_up_axis: str,
) -> list[float]:
    shifted = [
        float(point[index]) - float(translation[index])
        for index in range(3)
    ]
    if source_up_axis == "y":
        # Forward basis: (source_x, source_y, source_z) ->
        # (canonical_x, -canonical_y, canonical_z) = (x, -z, y).
        source = [shifted[0], shifted[2], -shifted[1]]
    else:
        source = shifted
    result = [value / unit_scale for value in source]
    if not all(math.isfinite(value) for value in result):
        raise BrowserRenderError("controlled camera transform is non-finite")
    return result


def _scene_geometry_signature(scene: dict[str, Any]) -> dict[str, Any]:
    objects = []
    for obj in scene.get("objects", []):
        if not isinstance(obj, dict):
            continue
        objects.append(
            {
                "id": obj.get("id"),
                "center": obj.get("center"),
                "size": obj.get("size"),
                "rotation": obj.get("rotation"),
                "geometry_provenance": obj.get("geometry_provenance"),
            }
        )
    return {
        "scene_id": scene.get("scene_id"),
        "request_id": scene.get("request_id"),
        "scene_type": scene.get("scene_type"),
        "boundary": scene.get("boundary"),
        "scene_height": scene.get("scene_height"),
        "objects": sorted(objects, key=lambda item: str(item["id"])),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
