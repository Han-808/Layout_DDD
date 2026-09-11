"""Non-semantic inspection defaults for saved Blender artifacts.

Benchmark workers run headlessly, so Blender otherwise persists the factory
``SOLID`` viewport even when the scene has fully materialized textures.  This
module changes only saved editor presentation state and optionally embeds
already-loaded shader images.  It does not add scene objects, alter materials,
or change render settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


INSPECTION_VIEW_VERSION = "textured_material_preview_v1"


def configure_textured_inspection_view(bpy_module: Any) -> dict[str, Any]:
    """Persist Material Preview in every saved 3D viewport.

    Material Preview deliberately uses Blender's studio illumination rather
    than scene lights/world.  This also works for the sanitized build-only
    materialization artifact, whose scientific contract forbids cameras and
    lights.
    """

    configured: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for screen in list(bpy_module.data.screens):
        for area_index, area in enumerate(screen.areas):
            if str(area.type) != "VIEW_3D":
                continue
            try:
                space = area.spaces.active
                shading = space.shading
                shading.type = "MATERIAL"
                if hasattr(shading, "use_scene_lights"):
                    shading.use_scene_lights = False
                if hasattr(shading, "use_scene_world"):
                    shading.use_scene_world = False
                configured.append(
                    {
                        "screen": str(screen.name),
                        "area_index": area_index,
                        "shading": str(shading.type),
                    }
                )
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                failures.append(
                    {
                        "screen": str(screen.name),
                        "area_index": str(area_index),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    return {
        "version": INSPECTION_VIEW_VERSION,
        "status": "configured" if configured and not failures else (
            "partial" if configured else "unavailable"
        ),
        "viewport_shading": "MATERIAL",
        "lighting": "studio",
        "use_scene_lights": False,
        "use_scene_world": False,
        "configured_viewport_count": len(configured),
        "configured_viewports": configured,
        "failures": failures,
        "scene_semantics_modified": False,
    }


def embed_available_shader_images(bpy_module: Any) -> dict[str, Any]:
    """Pack loaded external images that are actually referenced by shaders.

    Missing/unloadable images remain visible as Blender diagnostics and are
    reported rather than turning an otherwise usable render artifact into a
    worker failure.  The fixed-catalog materializer keeps its stricter
    all-images packing validation; this helper primarily makes renderer-owned
    ``scene.blend`` files portable.
    """

    referenced: dict[int, Any] = {}
    node_trees = []
    for material in list(bpy_module.data.materials):
        tree = getattr(material, "node_tree", None)
        if tree is not None:
            node_trees.append(tree)
    for tree in list(getattr(bpy_module.data, "node_groups", ())):
        node_trees.append(tree)
    for tree in node_trees:
        for node in list(tree.nodes):
            image = getattr(node, "image", None)
            if image is not None:
                referenced[id(image)] = image

    packed_now: list[str] = []
    already_packed: list[str] = []
    unresolved: list[dict[str, Any]] = []
    for image in sorted(referenced.values(), key=lambda value: str(value.name)):
        name = str(image.name)
        if bool(image.packed_files):
            already_packed.append(name)
            continue
        source = str(getattr(image, "source", ""))
        if source not in {"FILE", "SEQUENCE", "TILED"}:
            unresolved.append(
                {"image": name, "source": source, "reason": "not_external_file"}
            )
            continue
        if not bool(getattr(image, "has_data", False)):
            unresolved.append(
                {"image": name, "source": source, "reason": "image_data_not_loaded"}
            )
            continue
        try:
            image.pack()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raw_path = str(getattr(image, "filepath", ""))
            try:
                resolved_path = str(Path(bpy_module.path.abspath(raw_path)).resolve())
            except (OSError, RuntimeError, TypeError, ValueError):
                resolved_path = raw_path
            unresolved.append(
                {
                    "image": name,
                    "source": source,
                    "path": resolved_path,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        if bool(image.packed_files):
            packed_now.append(name)
        else:
            unresolved.append(
                {"image": name, "source": source, "reason": "pack_not_persisted"}
            )

    return {
        "referenced_shader_image_count": len(referenced),
        "already_packed_count": len(already_packed),
        "packed_now_count": len(packed_now),
        "unresolved_count": len(unresolved),
        "packed_now": packed_now,
        "unresolved": unresolved,
    }
