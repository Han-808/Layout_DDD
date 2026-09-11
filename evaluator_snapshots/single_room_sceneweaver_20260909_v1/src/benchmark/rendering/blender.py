from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat

from benchmark.architecture_policy import (
    architecture_contract_from_scene,
    validate_architecture_contract,
)

RENDER_ENGINES = (
    "BLENDER_WORKBENCH",
    "BLENDER_EEVEE_NEXT",
    "CYCLES",
)
CYCLES_DEVICES = (
    "CPU",
    "CUDA",
    "OPTIX",
    "AUTO",
)
DEFAULT_COLLISION_MAX_VERTICES_PER_OBJECT = 50_000
DEFAULT_COLLISION_MAX_FACES_PER_OBJECT = 100_000
DEFAULT_COLLISION_MAX_TOTAL_VERTICES = 200_000
DEFAULT_COLLISION_MAX_TOTAL_FACES = 400_000


class BlenderRenderError(RuntimeError):
    """Raised when the configured Blender process cannot produce evidence."""


class BlenderRenderer:
    """Render canonical scene JSON through a trusted headless Blender worker."""

    def __init__(
        self,
        *,
        blender_bin: str | Path,
        timeout_seconds: int = 900,
        width: int = 768,
        height: int = 768,
        render_engine: str = "BLENDER_EEVEE_NEXT",
        cycles_device: str = "CPU",
        cycles_samples: int = 16,
        cycles_denoising: bool = False,
        preview_render_engine: str | None = None,
        preview_width: int = 256,
        preview_height: int = 256,
        preview_cycles_samples: int = 1,
        require_asset_mesh: bool = False,
        collision_max_vertices_per_object: int = DEFAULT_COLLISION_MAX_VERTICES_PER_OBJECT,
        collision_max_faces_per_object: int = DEFAULT_COLLISION_MAX_FACES_PER_OBJECT,
        collision_max_total_vertices: int = DEFAULT_COLLISION_MAX_TOTAL_VERTICES,
        collision_max_total_faces: int = DEFAULT_COLLISION_MAX_TOTAL_FACES,
    ) -> None:
        self.blender_bin = Path(blender_bin).expanduser()
        self.timeout_seconds = max(1, int(timeout_seconds))
        self.width = max(64, int(width))
        self.height = max(64, int(height))
        if render_engine not in RENDER_ENGINES:
            raise ValueError(f"render_engine must be one of {list(RENDER_ENGINES)}, got {render_engine!r}")
        normalized_cycles_device = str(cycles_device).upper()
        if normalized_cycles_device not in CYCLES_DEVICES:
            raise ValueError(
                f"cycles_device must be one of {list(CYCLES_DEVICES)}, got {cycles_device!r}"
            )
        self.render_engine = render_engine
        resolved_preview_engine = preview_render_engine or render_engine
        if resolved_preview_engine not in RENDER_ENGINES:
            raise ValueError(
                "preview_render_engine must be one of "
                f"{list(RENDER_ENGINES)}, got {resolved_preview_engine!r}"
            )
        self.cycles_device = normalized_cycles_device
        self.cycles_samples = max(1, int(cycles_samples))
        self.cycles_denoising = bool(cycles_denoising)
        self.preview_render_engine = resolved_preview_engine
        self.preview_width = max(64, int(preview_width))
        self.preview_height = max(64, int(preview_height))
        self.preview_cycles_samples = max(1, int(preview_cycles_samples))
        self.require_asset_mesh = bool(require_asset_mesh)
        self.collision_max_vertices_per_object = _non_negative_int(
            collision_max_vertices_per_object,
            "collision_max_vertices_per_object",
        )
        self.collision_max_faces_per_object = _non_negative_int(
            collision_max_faces_per_object,
            "collision_max_faces_per_object",
        )
        self.collision_max_total_vertices = _non_negative_int(
            collision_max_total_vertices,
            "collision_max_total_vertices",
        )
        self.collision_max_total_faces = _non_negative_int(
            collision_max_total_faces,
            "collision_max_total_faces",
        )

    def render_scene(
        self,
        *,
        scene_path: str | Path,
        out_dir: str | Path,
        asset_root: str | Path | None = None,
    ) -> dict[str, Any]:
        scene = Path(scene_path).expanduser().resolve()
        destination = Path(out_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        self._preflight(scene)
        try:
            scene_data = json.loads(scene.read_text(encoding="utf-8"))
            architecture = architecture_contract_from_scene(scene_data)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise BlenderRenderError(
                f"invalid scene architecture contract: {exc}"
            ) from exc
        architecture_path = destination / "architecture_contract.json"
        architecture_path.write_text(
            json.dumps(architecture, indent=2),
            encoding="utf-8",
        )
        worker = Path(__file__).with_name("blender_worker.py").resolve()
        command = [
            str(self.blender_bin),
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python",
            str(worker),
            "--",
            "--scene-json",
            str(scene),
            "--out-dir",
            str(destination),
            "--architecture-contract",
            str(architecture_path),
            "--width",
            str(self.width),
            "--height",
            str(self.height),
            "--render-engine",
            self.render_engine,
            "--cycles-device",
            self.cycles_device,
            "--cycles-samples",
            str(self.cycles_samples),
            "--collision-max-vertices-per-object",
            str(self.collision_max_vertices_per_object),
            "--collision-max-faces-per-object",
            str(self.collision_max_faces_per_object),
            "--collision-max-total-vertices",
            str(self.collision_max_total_vertices),
            "--collision-max-total-faces",
            str(self.collision_max_total_faces),
        ]
        if self.cycles_denoising:
            command.append("--cycles-denoising")
        if asset_root is not None:
            command.extend(["--asset-root", str(Path(asset_root).expanduser().resolve())])

        manifest_path = destination / "render_manifest.json"
        timed_out = False
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = _subprocess_stream_text(exc.stdout)
            stderr = _subprocess_stream_text(exc.stderr)
            (destination / "blender.stdout.log").write_text(stdout, encoding="utf-8")
            (destination / "blender.stderr.log").write_text(stderr, encoding="utf-8")
            if not manifest_path.is_file():
                stage = _last_worker_stage(destination)
                raise BlenderRenderError(
                    f"Blender timed out after {self.timeout_seconds}s before producing render evidence; "
                    f"last_worker_stage={stage!r}"
                ) from exc
            completed = None
        else:
            (destination / "blender.stdout.log").write_text(completed.stdout, encoding="utf-8")
            (destination / "blender.stderr.log").write_text(completed.stderr, encoding="utf-8")
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout)[-2000:]
                raise BlenderRenderError(f"Blender exited with code {completed.returncode}: {detail}")

        if not manifest_path.is_file():
            raise BlenderRenderError(f"Blender completed without writing {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BlenderRenderError(f"Invalid Blender render manifest: {manifest_path}") from exc
        try:
            rendered_architecture = validate_architecture_contract(
                manifest.get("architecture")
            )
        except ValueError as exc:
            raise BlenderRenderError(
                f"Blender render manifest has an invalid architecture contract: {exc}"
            ) from exc
        if rendered_architecture != architecture:
            raise BlenderRenderError(
                "Blender render architecture contract differs from the canonical "
                "scene architecture contract"
            )
        if timed_out:
            collision_export = manifest.get("collision_geometry_export")
            if not isinstance(collision_export, dict) or collision_export.get("status") == "pending":
                manifest["collision_geometry_export"] = {
                    **(collision_export if isinstance(collision_export, dict) else {}),
                    "status": "timed_out_after_render",
                    "timeout_seconds": self.timeout_seconds,
                }
            manifest["worker_completion"] = {
                "status": "timed_out_after_base_manifest",
                "timeout_seconds": self.timeout_seconds,
                "last_worker_stage": _last_worker_stage(destination),
                "render_evidence_preserved": True,
            }
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        views = manifest.get("views")
        if not isinstance(views, list) or not views:
            raise BlenderRenderError("Blender render manifest contains no views")
        missing = [str(item.get("path")) for item in views if not Path(str(item.get("path"))).is_file()]
        if missing:
            raise BlenderRenderError(f"Blender reported missing render files: {missing}")
        blank = []
        for item in views:
            if not isinstance(item, dict):
                continue
            stats = _saved_image_stats(Path(str(item.get("path"))))
            item["pixel_stats"] = stats
            maximum = float(stats.get("max_luminance", 0.0))
            luminance_range = float(stats.get("luminance_range", 0.0))
            luminance_stddev = float(stats.get("luminance_stddev", 0.0))
            if maximum < 0.01 or luminance_range < 0.005 or luminance_stddev < 0.005:
                blank.append(str(item.get("name") or item.get("path")))
        manifest["render_validation"] = {
            "pixel_stats_source": "saved_png_pillow",
            "blank_views": blank,
        }
        objects = [item for item in manifest.get("objects", []) if isinstance(item, dict)]
        identity_validation = _validate_identity_render(
            manifest,
            objects=objects,
        )
        if identity_validation is not None:
            manifest["render_validation"][
                "identity_map"
            ] = identity_validation
        asset_mesh_count = sum(item.get("representation") == "asset_mesh" for item in objects)
        proxy_count = sum(item.get("representation") == "bbox_proxy" for item in objects)
        manifest["asset_coverage"] = {
            "object_count": len(objects),
            "asset_mesh_count": asset_mesh_count,
            "bbox_proxy_count": proxy_count,
            "asset_mesh_rate": asset_mesh_count / len(objects) if objects else 0.0,
            "required": self.require_asset_mesh,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        geometry_manifest = manifest.get("collision_geometry_manifest")
        if isinstance(geometry_manifest, str) and Path(geometry_manifest).is_file():
            manifest["collision_geometry"] = json.loads(Path(geometry_manifest).read_text(encoding="utf-8"))
            manifest["collision_geometry"]["manifest_path"] = geometry_manifest
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if blank:
            raise BlenderRenderError(
                f"Blender produced blank or near-uniform render evidence for views {blank}; "
                f"engine={self.render_engine}"
            )
        if self.require_asset_mesh and asset_mesh_count == 0:
            raise BlenderRenderError(
                "Asset-backed rendering was requested, but Blender rendered zero asset meshes; "
                "inspect object asset bindings, asset_root, and import warnings in render_manifest.json"
            )
        return manifest

    def render_prepared_scene(
        self,
        *,
        blend_file: str | Path,
        normalized_scene_path: str | Path,
        out_dir: str | Path,
    ) -> dict[str, Any]:
        """Render a previously verified blend without rematerializing it.

        Geometry comes exclusively from ``blend_file``.  The normalized JSON is
        used only to verify registered identity and trusted architecture.  The
        worker opens the blend with auto-execution disabled, adds ephemeral
        benchmark cameras/lights in memory, and never saves the source scene.
        """

        source = Path(blend_file).expanduser().resolve()
        normalized = Path(normalized_scene_path).expanduser().resolve()
        destination = Path(out_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        self._preflight_blend(source)
        self._preflight(normalized)
        if source == normalized:
            raise BlenderRenderError(
                "trusted blend and normalized scene paths must be distinct"
            )
        try:
            normalized_data = json.loads(normalized.read_text(encoding="utf-8"))
            architecture = architecture_contract_from_scene(normalized_data)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise BlenderRenderError(
                f"invalid prepared normalized scene contract: {exc}"
            ) from exc

        # These are deliberately uncached full-file hashes.  A read-only worker
        # must not be able to hide a same-size/same-mtime source modification.
        source_hash_before = _sha256_file(source)
        normalized_hash_before = _sha256_file(normalized)
        worker = Path(__file__).with_name(
            "blender_prepared_worker.py"
        ).resolve()
        command = [
            str(self.blender_bin),
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            str(source),
            "--python-exit-code",
            "1",
            "--python",
            str(worker),
            "--",
            "--normalized-scene-json",
            str(normalized),
            "--out-dir",
            str(destination),
            "--width",
            str(self.width),
            "--height",
            str(self.height),
            "--render-engine",
            self.render_engine,
            "--cycles-device",
            self.cycles_device,
            "--cycles-samples",
            str(self.cycles_samples),
            "--collision-max-vertices-per-object",
            str(self.collision_max_vertices_per_object),
            "--collision-max-faces-per-object",
            str(self.collision_max_faces_per_object),
            "--collision-max-total-vertices",
            str(self.collision_max_total_vertices),
            "--collision-max-total-faces",
            str(self.collision_max_total_faces),
        ]
        if self.cycles_denoising:
            command.append("--cycles-denoising")

        completed = None
        timeout_error: subprocess.TimeoutExpired | None = None
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            timeout_error = exc
            stdout = _subprocess_stream_text(exc.stdout)
            stderr = _subprocess_stream_text(exc.stderr)
            (destination / "prepared_blender.stdout.log").write_text(
                stdout,
                encoding="utf-8",
            )
            (destination / "prepared_blender.stderr.log").write_text(
                stderr,
                encoding="utf-8",
            )
        else:
            (destination / "prepared_blender.stdout.log").write_text(
                completed.stdout,
                encoding="utf-8",
            )
            (destination / "prepared_blender.stderr.log").write_text(
                completed.stderr,
                encoding="utf-8",
            )
        finally:
            source_hash_after = _sha256_file(source)
            normalized_hash_after = _sha256_file(normalized)

        if source_hash_after != source_hash_before:
            raise BlenderRenderError(
                "read-only prepared renderer modified the trusted source blend"
            )
        if normalized_hash_after != normalized_hash_before:
            raise BlenderRenderError(
                "read-only prepared renderer modified the normalized scene"
            )
        if timeout_error is not None:
            raise BlenderRenderError(
                f"Blender prepared render timed out after {self.timeout_seconds}s"
            ) from timeout_error
        assert completed is not None
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout)[-4000:]
            raise BlenderRenderError(
                "Blender prepared worker exited with code "
                f"{completed.returncode}: {detail}"
            )

        manifest_path = destination / "prepared_render_manifest.json"
        manifest = _load_render_manifest(
            manifest_path,
            label="prepared render",
        )
        rendered_source = Path(
            str(manifest.get("blend_file") or "")
        ).expanduser().resolve()
        if rendered_source != source:
            raise BlenderRenderError(
                "prepared render manifest does not identify the trusted source blend"
            )
        rendered_normalized = Path(
            str(manifest.get("normalized_scene_path") or "")
        ).expanduser().resolve()
        if rendered_normalized != normalized:
            raise BlenderRenderError(
                "prepared render manifest does not identify the normalized scene"
            )
        if manifest.get("source_scene_saved") is not False:
            raise BlenderRenderError(
                "prepared render worker did not attest read-only source handling"
            )
        try:
            rendered_architecture = validate_architecture_contract(
                manifest.get("architecture")
            )
        except ValueError as exc:
            raise BlenderRenderError(
                f"prepared render manifest has invalid architecture: {exc}"
            ) from exc
        if rendered_architecture != architecture:
            raise BlenderRenderError(
                "prepared render architecture differs from normalized scene"
            )

        blank = _validate_render_views(manifest)
        objects = [
            item for item in manifest.get("objects", []) if isinstance(item, dict)
        ]
        identity_validation = _validate_identity_render(
            manifest,
            objects=objects,
        )
        if identity_validation is not None:
            manifest.setdefault("render_validation", {})[
                "identity_map"
            ] = identity_validation
        asset_mesh_count = sum(
            item.get("representation") == "asset_mesh" for item in objects
        )
        manifest["asset_coverage"] = {
            "object_count": len(objects),
            "asset_mesh_count": asset_mesh_count,
            "bbox_proxy_count": 0,
            "asset_mesh_rate": (
                asset_mesh_count / len(objects) if objects else 0.0
            ),
            "required": True,
        }
        geometry_manifest = manifest.get("collision_geometry_manifest")
        if (
            isinstance(geometry_manifest, str)
            and Path(geometry_manifest).is_file()
        ):
            manifest["collision_geometry"] = json.loads(
                Path(geometry_manifest).read_text(encoding="utf-8")
            )
            manifest["collision_geometry"][
                "manifest_path"
            ] = geometry_manifest
            manifest["collision_geometry"]["summary"] = dict(
                manifest["collision_geometry"].get("export_summary") or {}
            )
        manifest.update(
            {
                "blend_file": source.as_posix(),
                "normalized_scene_path": normalized.as_posix(),
                "source_blend_sha256_before": source_hash_before,
                "source_blend_sha256_after": source_hash_after,
                "source_blend_modified": False,
                "normalized_scene_sha256_before": normalized_hash_before,
                "normalized_scene_sha256_after": normalized_hash_after,
                "normalized_scene_modified": False,
                "auto_execution_disabled": True,
                "trusted_source_verified": True,
            }
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        if blank:
            raise BlenderRenderError(
                "Blender produced blank or near-uniform prepared views "
                f"{blank}; engine={self.render_engine}"
            )
        if objects and asset_mesh_count != len(objects):
            raise BlenderRenderError(
                "prepared renderer encountered a non-catalog proxy representation"
            )
        return manifest

    def render_camera_views(
        self,
        *,
        blend_file: str | Path,
        out_dir: str | Path,
        camera_views: list[dict[str, Any]],
        preview: bool = False,
        allow_blank_views: bool = False,
    ) -> dict[str, Any]:
        """Render ephemeral Track-To cameras against an existing scene.

        The worker opens the saved benchmark ``.blend`` with auto-execution
        disabled and never saves it. Only temporary cameras and their tracking
        targets are added to the in-memory scene.
        """

        source = Path(blend_file).expanduser().resolve()
        destination = Path(out_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        self._preflight_blend(source)
        source_stat_before = _file_stat_signature(source)
        source_hash_before = _sha256_file(source)
        if not isinstance(camera_views, list) or not camera_views:
            raise ValueError("camera_views must be a non-empty list")
        poses_path = destination / "camera_views.json"
        poses_path.write_text(json.dumps(camera_views, indent=2), encoding="utf-8")
        worker = Path(__file__).with_name("blender_camera_worker.py").resolve()
        render_engine = self.preview_render_engine if preview else self.render_engine
        width = min(self.width, self.preview_width) if preview else self.width
        height = min(self.height, self.preview_height) if preview else self.height
        cycles_samples = self.preview_cycles_samples if preview else self.cycles_samples
        command = [
            str(self.blender_bin),
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            str(source),
            "--python-exit-code",
            "1",
            "--python",
            str(worker),
            "--",
            "--camera-views-json",
            str(poses_path),
            "--out-dir",
            str(destination),
            "--width",
            str(width),
            "--height",
            str(height),
            "--render-engine",
            render_engine,
            "--cycles-device",
            self.cycles_device,
            "--cycles-samples",
            str(cycles_samples),
        ]
        if self.cycles_denoising and not preview:
            command.append("--cycles-denoising")
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = _subprocess_stream_text(exc.stdout)
            stderr = _subprocess_stream_text(exc.stderr)
            (destination / "camera_blender.stdout.log").write_text(stdout, encoding="utf-8")
            (destination / "camera_blender.stderr.log").write_text(stderr, encoding="utf-8")
            raise BlenderRenderError(
                f"Blender camera evidence timed out after {self.timeout_seconds}s"
            ) from exc
        (destination / "camera_blender.stdout.log").write_text(completed.stdout, encoding="utf-8")
        (destination / "camera_blender.stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout)[-2000:]
            raise BlenderRenderError(f"Blender camera worker exited with code {completed.returncode}: {detail}")
        source_stat_after = _file_stat_signature(source)
        source_hash_after = _sha256_file(source)
        if source_hash_after != source_hash_before:
            raise BlenderRenderError("Read-only camera worker modified the source Blender scene")
        manifest_path = destination / "camera_render_manifest.json"
        manifest = _load_render_manifest(manifest_path, label="camera render")
        blank = _validate_render_views(manifest)
        manifest["camera_evidence"] = {
            "preview": bool(preview),
            "blank_view_policy": "record" if allow_blank_views else "reject",
            "source_blend": str(source),
            "source_blend_modified": False,
            "source_blend_sha256_before": source_hash_before,
            "source_blend_sha256_after": source_hash_after,
            "source_blend_stat_before": list(source_stat_before[1:]),
            "source_blend_stat_after": list(source_stat_after[1:]),
            "render_engine": render_engine,
            "cycles_samples": cycles_samples if render_engine == "CYCLES" else None,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if blank and not allow_blank_views:
            raise BlenderRenderError(
                f"Blender produced blank or near-uniform camera evidence for views {blank}; "
                f"engine={render_engine}"
            )
        return manifest

    def render_collision_overlay_views(
        self,
        *,
        blend_file: str | Path,
        out_dir: str | Path,
        camera_views: list[dict[str, Any]],
        overlay_spec: dict[str, Any],
        preview: bool = False,
        allow_blank_views: bool = False,
    ) -> dict[str, Any]:
        """Render the same-pose collision diagnostic overlay, read-only.

        The overlay is a deterministic object-color pass: object A is
        highlighted red, object B cyan, and other objects dim, with both OBB
        wireframes, a legend, and yellow closest-point markers driven by
        ``overlay_spec``. Like the RGB camera pass it opens the saved ``.blend``
        with auto-execution disabled, only adds ephemeral cameras/markers, and
        never saves over the source scene.
        """

        source = Path(blend_file).expanduser().resolve()
        destination = Path(out_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        self._preflight_blend(source)
        if not isinstance(camera_views, list) or not camera_views:
            raise ValueError("camera_views must be a non-empty list")
        if not isinstance(overlay_spec, dict) or not overlay_spec:
            raise ValueError("overlay_spec must be a non-empty JSON object")
        source_stat_before = _file_stat_signature(source)
        source_hash_before = _sha256_file(source)
        poses_path = destination / "camera_views.json"
        poses_path.write_text(json.dumps(camera_views, indent=2), encoding="utf-8")
        overlay_path = destination / "collision_overlay_spec.json"
        overlay_path.write_text(json.dumps(overlay_spec, indent=2), encoding="utf-8")
        worker = Path(__file__).with_name("blender_collision_overlay_worker.py").resolve()
        render_engine = self.preview_render_engine if preview else self.render_engine
        width = min(self.width, self.preview_width) if preview else self.width
        height = min(self.height, self.preview_height) if preview else self.height
        cycles_samples = self.preview_cycles_samples if preview else self.cycles_samples
        command = [
            str(self.blender_bin),
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            str(source),
            "--python-exit-code",
            "1",
            "--python",
            str(worker),
            "--",
            "--camera-views-json",
            str(poses_path),
            "--overlay-spec-json",
            str(overlay_path),
            "--out-dir",
            str(destination),
            "--width",
            str(width),
            "--height",
            str(height),
            "--render-engine",
            render_engine,
            "--cycles-device",
            self.cycles_device,
            "--cycles-samples",
            str(cycles_samples),
        ]
        if self.cycles_denoising and not preview:
            command.append("--cycles-denoising")
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = _subprocess_stream_text(exc.stdout)
            stderr = _subprocess_stream_text(exc.stderr)
            (destination / "overlay_blender.stdout.log").write_text(stdout, encoding="utf-8")
            (destination / "overlay_blender.stderr.log").write_text(stderr, encoding="utf-8")
            raise BlenderRenderError(
                f"Blender collision overlay timed out after {self.timeout_seconds}s"
            ) from exc
        (destination / "overlay_blender.stdout.log").write_text(completed.stdout, encoding="utf-8")
        (destination / "overlay_blender.stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout)[-2000:]
            raise BlenderRenderError(f"Blender collision overlay worker exited with code {completed.returncode}: {detail}")
        source_stat_after = _file_stat_signature(source)
        source_hash_after = _sha256_file(source)
        if source_hash_after != source_hash_before:
            raise BlenderRenderError("Read-only collision overlay worker modified the source Blender scene")
        manifest_path = destination / "collision_overlay_manifest.json"
        manifest = _load_render_manifest(manifest_path, label="collision overlay")
        blank = _validate_render_views(manifest)
        manifest["camera_evidence"] = {
            "preview": bool(preview),
            "blank_view_policy": "record" if allow_blank_views else "reject",
            "role": "collision_pair_overlay",
            "source_blend": str(source),
            "source_blend_modified": False,
            "source_blend_sha256_before": source_hash_before,
            "source_blend_sha256_after": source_hash_after,
            "render_engine": render_engine,
            "cycles_samples": cycles_samples if render_engine == "CYCLES" else None,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if blank and not allow_blank_views:
            raise BlenderRenderError(
                f"Blender produced blank or near-uniform collision overlay evidence for views {blank}"
            )
        return manifest

    def render_target_id_masks(
        self,
        *,
        blend_file: str | Path,
        out_dir: str | Path,
        camera_views: list[dict[str, Any]],
        overlay_spec: dict[str, Any],
        preview: bool = True,
        respect_occlusion: bool = False,
    ) -> dict[str, Any]:
        """Render per-candidate, per-target binary identity masks, read-only.

        Each target's mask is a binary object-identity (Object Index / ID)
        image whose foreground pixels are only that canonical object's visible
        surface. Masks are independent of materials, lighting, color management,
        wireframes, markers, and legends, so they - not the decorative RGB
        render - drive candidate visibility ranking. Rendering is per-candidate
        lenient: a blank or failed candidate is recorded with a ``status`` and
        never aborts the batch (that is the caller's ranking concern). Only a
        worker crash or a modified source ``.blend`` raises.
        """

        source = Path(blend_file).expanduser().resolve()
        destination = Path(out_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        self._preflight_blend(source)
        if not isinstance(camera_views, list) or not camera_views:
            raise ValueError("camera_views must be a non-empty list")
        if not isinstance(overlay_spec, dict) or not overlay_spec:
            raise ValueError("overlay_spec must be a non-empty JSON object")
        source_stat_before = _file_stat_signature(source)
        source_hash_before = _sha256_file(source)
        poses_path = destination / "camera_views.json"
        poses_path.write_text(json.dumps(camera_views, indent=2), encoding="utf-8")
        overlay_path = destination / "mask_overlay_spec.json"
        overlay_path.write_text(json.dumps(overlay_spec, indent=2), encoding="utf-8")
        worker = Path(__file__).with_name("blender_collision_mask_worker.py").resolve()
        # Candidate ranking uses cheap preview masks.  Opt-in presentation
        # variants may request full-resolution masks so a 2D segmentation
        # contour aligns exactly with the final RGB evidence.
        width = min(self.width, self.preview_width) if preview else self.width
        height = min(self.height, self.preview_height) if preview else self.height
        command = [
            str(self.blender_bin),
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            str(source),
            "--python-exit-code",
            "1",
            "--python",
            str(worker),
            "--",
            "--camera-views-json",
            str(poses_path),
            "--overlay-spec-json",
            str(overlay_path),
            "--out-dir",
            str(destination),
            "--width",
            str(width),
            "--height",
            str(height),
        ]
        if respect_occlusion:
            command.append("--respect-occlusion")
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            (destination / "mask_blender.stdout.log").write_text(_subprocess_stream_text(exc.stdout), encoding="utf-8")
            (destination / "mask_blender.stderr.log").write_text(_subprocess_stream_text(exc.stderr), encoding="utf-8")
            raise BlenderRenderError(f"Blender identity-mask pass timed out after {self.timeout_seconds}s") from exc
        (destination / "mask_blender.stdout.log").write_text(completed.stdout, encoding="utf-8")
        (destination / "mask_blender.stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout)[-2000:]
            raise BlenderRenderError(f"Blender identity-mask worker exited with code {completed.returncode}: {detail}")
        source_stat_after = _file_stat_signature(source)
        source_hash_after = _sha256_file(source)
        if source_hash_after != source_hash_before:
            raise BlenderRenderError("Read-only identity-mask worker modified the source Blender scene")
        manifest_path = destination / "target_id_mask_manifest.json"
        manifest = _load_render_manifest(manifest_path, label="identity mask")
        # Deliberately no batch blank-rejection: per-candidate status lives in the
        # manifest so a single blank candidate cannot poison the others.
        manifest["camera_evidence"] = {
            "preview": bool(preview),
            "role": "target_id_masks",
            "occlusion_policy": (
                "respect_scene_occlusion"
                if respect_occlusion
                else "ignore_scene_occlusion_projection_proxy"
            ),
            "source_blend": str(source),
            "source_blend_modified": False,
            "source_blend_sha256_before": source_hash_before,
            "source_blend_sha256_after": source_hash_after,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    def render_focus_evidence_bundle(
        self,
        *,
        blend_file: str | Path,
        out_dir: str | Path,
        local_camera_views: list[dict[str, Any]],
        global_camera_views: list[dict[str, Any]],
        overlay_spec: dict[str, Any],
    ) -> dict[str, Any]:
        """Render RGB locals plus highlighted local/global views in one process.

        The visibility preview still runs separately because its pixels decide
        which local poses are selected. Once selected, this bundled pass opens
        the source ``.blend`` only once, avoiding three repeated Blender/Cycles
        initializations per metric event.
        """

        source = Path(blend_file).expanduser().resolve()
        destination = Path(out_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        self._preflight_blend(source)
        if not isinstance(local_camera_views, list) or not local_camera_views:
            raise ValueError("local_camera_views must be a non-empty list")
        if not isinstance(global_camera_views, list):
            raise TypeError("global_camera_views must be a list")
        if not isinstance(overlay_spec, dict) or not overlay_spec:
            raise ValueError("overlay_spec must be a non-empty JSON object")

        source_stat_before = _file_stat_signature(source)
        source_hash_before = _sha256_file(source)
        request_path = destination / "focus_bundle_request.json"
        request_path.write_text(
            json.dumps(
                {
                    "local_camera_views": local_camera_views,
                    "global_camera_views": global_camera_views,
                    "overlay_spec": overlay_spec,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        worker = Path(__file__).with_name("blender_focus_bundle_worker.py").resolve()
        command = [
            str(self.blender_bin),
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            str(source),
            "--python-exit-code",
            "1",
            "--python",
            str(worker),
            "--",
            "--request-json",
            str(request_path),
            "--out-dir",
            str(destination),
            "--width",
            str(self.width),
            "--height",
            str(self.height),
            "--render-engine",
            self.render_engine,
            "--cycles-device",
            self.cycles_device,
            "--cycles-samples",
            str(self.cycles_samples),
        ]
        if self.cycles_denoising:
            command.append("--cycles-denoising")
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = _subprocess_stream_text(exc.stdout)
            stderr = _subprocess_stream_text(exc.stderr)
            (destination / "bundle_blender.stdout.log").write_text(stdout, encoding="utf-8")
            (destination / "bundle_blender.stderr.log").write_text(stderr, encoding="utf-8")
            raise BlenderRenderError(
                f"Blender focus evidence bundle timed out after {self.timeout_seconds}s"
            ) from exc
        (destination / "bundle_blender.stdout.log").write_text(completed.stdout, encoding="utf-8")
        (destination / "bundle_blender.stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout)[-2000:]
            raise BlenderRenderError(
                f"Blender focus evidence bundle worker exited with code {completed.returncode}: {detail}"
            )

        source_stat_after = _file_stat_signature(source)
        source_hash_after = _sha256_file(source)
        if source_hash_after != source_hash_before:
            raise BlenderRenderError("Read-only focus evidence worker modified the source Blender scene")
        manifest_path = destination / "focus_bundle_manifest.json"
        manifest = _load_render_manifest(manifest_path, label="focus evidence bundle")
        blank = _validate_render_views(manifest)
        manifest["camera_evidence"] = {
            "source_blend": str(source),
            "source_blend_modified": False,
            "source_blend_sha256_before": source_hash_before,
            "source_blend_sha256_after": source_hash_after,
            "render_engine": self.render_engine,
            "bundled_process": True,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if blank:
            raise BlenderRenderError(
                f"Blender produced blank or near-uniform bundled evidence for views {blank}; "
                f"engine={self.render_engine}"
            )
        return manifest

    def render_focus_overlay_views(
        self,
        *,
        blend_file: str | Path,
        out_dir: str | Path,
        camera_views: list[dict[str, Any]],
        overlay_spec: dict[str, Any],
        preview: bool = False,
        allow_blank_views: bool = False,
    ) -> dict[str, Any]:
        """Render a generic read-only target-highlight pass.

        Collision keeps its historical method name and manifest contract. OOB,
        Support, and future metric evidence use this semantic alias over the
        same restricted engine-independent worker.
        """

        return self.render_collision_overlay_views(
            blend_file=blend_file,
            out_dir=out_dir,
            camera_views=camera_views,
            overlay_spec=overlay_spec,
            preview=preview,
            allow_blank_views=allow_blank_views,
        )

    def _preflight(self, scene_path: Path) -> None:
        if not self.blender_bin.is_file():
            raise BlenderRenderError(f"Blender executable does not exist: {self.blender_bin}")
        if not os.access(self.blender_bin, os.X_OK):
            raise BlenderRenderError(f"Blender path is not executable: {self.blender_bin}")
        if not scene_path.is_file():
            raise BlenderRenderError(f"Canonical scene JSON does not exist: {scene_path}")

    def _preflight_blend(self, blend_file: Path) -> None:
        if not self.blender_bin.is_file():
            raise BlenderRenderError(f"Blender executable does not exist: {self.blender_bin}")
        if not os.access(self.blender_bin, os.X_OK):
            raise BlenderRenderError(f"Blender path is not executable: {self.blender_bin}")
        if not blend_file.is_file():
            raise BlenderRenderError(f"Blender scene does not exist: {blend_file}")

def _saved_image_stats(path: Path) -> dict[str, Any]:
    try:
        with Image.open(path) as source:
            rgb = source.convert("RGB")
            grayscale = rgb.convert("L")
            minimum, maximum = grayscale.getextrema()
            image_stat = ImageStat.Stat(grayscale)
            mean = float(image_stat.mean[0])
            stddev = float(image_stat.stddev[0])
            return {
                "source": "saved_png_pillow",
                "size": [int(rgb.width), int(rgb.height)],
                "channel_extrema": [list(values) for values in rgb.getextrema()],
                "min_luminance": float(minimum) / 255.0,
                "max_luminance": float(maximum) / 255.0,
                "luminance_range": float(maximum - minimum) / 255.0,
                "mean_luminance": mean / 255.0,
                "luminance_stddev": stddev / 255.0,
            }
    except OSError as exc:
        raise BlenderRenderError(f"Cannot inspect rendered evidence image {path}: {exc}") from exc


def _validate_identity_render(
    manifest: dict[str, Any],
    *,
    objects: list[dict[str, Any]],
) -> dict[str, Any] | None:
    identity_render = manifest.get("identity_render")
    if not isinstance(identity_render, dict):
        # Older renderer manifests remain an explicit degraded grouping input.
        return None
    expected_ids = {
        str(item.get("id") or "").strip()
        for item in objects
        if str(item.get("id") or "").strip()
    }
    status = str(identity_render.get("status") or "")
    if status == "not_applicable":
        if expected_ids:
            raise BlenderRenderError(
                "identity render cannot be not_applicable when renderable "
                "canonical objects exist"
            )
        return {
            "status": "not_applicable",
            "expected_object_count": 0,
            "legend_object_count": 0,
            "visible_exact_identity_count": 0,
        }
    if status != "available":
        raise BlenderRenderError(
            "identity_render.status must be available or not_applicable"
        )
    legend = manifest.get("identity_legend")
    if not isinstance(legend, dict) or not legend:
        raise BlenderRenderError(
            "available identity render requires identity_legend"
        )
    legend_ids = [str(value).strip() for value in legend.values()]
    if (
        any(not value for value in legend_ids)
        or len(legend_ids) != len(set(legend_ids))
        or set(legend_ids) != expected_ids
    ):
        raise BlenderRenderError(
            "identity legend must map exactly once to every rendered "
            "canonical object ID"
        )
    views = [
        item
        for item in manifest.get("views", [])
        if isinstance(item, dict)
        and item.get("name") == "identity_map"
    ]
    if len(views) != 1:
        raise BlenderRenderError(
            "available identity render requires exactly one identity_map view"
        )
    path = Path(str(views[0].get("path") or ""))
    try:
        with Image.open(path) as source:
            raw_pixels = source.convert("RGB").tobytes()
            pixels = set(
                zip(
                    raw_pixels[0::3],
                    raw_pixels[1::3],
                    raw_pixels[2::3],
                )
            )
    except OSError as exc:
        raise BlenderRenderError(
            f"cannot decode identity-map image {path}: {exc}"
        ) from exc
    palette: set[tuple[int, int, int]] = set()
    for alias in legend:
        text = str(alias).strip()
        if (
            len(text) != 7
            or not text.startswith("#")
        ):
            raise BlenderRenderError(
                "identity legend aliases must be #RRGGBB colors"
            )
        try:
            palette.add(tuple(bytes.fromhex(text[1:])))
        except ValueError as exc:
            raise BlenderRenderError(
                "identity legend contains an invalid #RRGGBB color"
            ) from exc
    visible_exact = len(palette & pixels)
    if expected_ids and visible_exact == 0:
        raise BlenderRenderError(
            "identity-map PNG contains no exact identity-legend color; "
            "check Blender color-management settings"
        )
    return {
        "status": "verified",
        "expected_object_count": len(expected_ids),
        "legend_object_count": len(legend_ids),
        "visible_exact_identity_count": visible_exact,
        "color_encoding": identity_render.get("color_encoding"),
    }


def _load_render_manifest(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise BlenderRenderError(f"Blender completed without writing {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BlenderRenderError(f"Invalid Blender {label} manifest: {path}") from exc
    if not isinstance(value, dict):
        raise BlenderRenderError(f"Invalid Blender {label} manifest root: {path}")
    return value


def _validate_render_views(manifest: dict[str, Any]) -> list[str]:
    views = manifest.get("views")
    if not isinstance(views, list) or not views:
        raise BlenderRenderError("Blender render manifest contains no views")
    missing = [str(item.get("path")) for item in views if not Path(str(item.get("path"))).is_file()]
    if missing:
        raise BlenderRenderError(f"Blender reported missing render files: {missing}")
    blank = []
    for item in views:
        if not isinstance(item, dict):
            continue
        stats = _saved_image_stats(Path(str(item.get("path"))))
        item["pixel_stats"] = stats
        maximum = float(stats.get("max_luminance", 0.0))
        luminance_range = float(stats.get("luminance_range", 0.0))
        luminance_stddev = float(stats.get("luminance_stddev", 0.0))
        if maximum < 0.01 or luminance_range < 0.005 or luminance_stddev < 0.005:
            blank.append(str(item.get("name") or item.get("path")))
    manifest["render_validation"] = {
        "pixel_stats_source": "saved_png_pillow",
        "blank_views": blank,
    }
    return blank


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_stat_signature(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return str(path), int(stat.st_size), int(stat.st_mtime_ns)


def _non_negative_int(value: int, label: str) -> int:
    resolved = int(value)
    if resolved < 0:
        raise ValueError(f"{label} must be non-negative")
    return resolved


def _subprocess_stream_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _last_worker_stage(destination: Path) -> str | None:
    progress_path = destination / "blender_worker_progress.jsonl"
    if not progress_path.is_file():
        return None
    try:
        lines = [line for line in progress_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            return None
        record = json.loads(lines[-1])
        return str(record.get("stage")) if isinstance(record, dict) and record.get("stage") else None
    except (OSError, json.JSONDecodeError):
        return None
