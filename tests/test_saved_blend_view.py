from __future__ import annotations

from types import SimpleNamespace

from benchmark.rendering.saved_blend_view import (
    INSPECTION_VIEW_VERSION,
    configure_textured_inspection_view,
    embed_available_shader_images,
)


def test_saved_blend_uses_material_preview_without_scene_lighting() -> None:
    shading = SimpleNamespace(
        type="SOLID",
        use_scene_lights=True,
        use_scene_world=True,
    )
    view = SimpleNamespace(
        type="VIEW_3D",
        spaces=SimpleNamespace(active=SimpleNamespace(shading=shading)),
    )
    non_view = SimpleNamespace(type="OUTLINER")
    screen = SimpleNamespace(name="Layout", areas=[view, non_view])
    bpy = SimpleNamespace(data=SimpleNamespace(screens=[screen]))

    report = configure_textured_inspection_view(bpy)

    assert shading.type == "MATERIAL"
    assert shading.use_scene_lights is False
    assert shading.use_scene_world is False
    assert report == {
        "version": INSPECTION_VIEW_VERSION,
        "status": "configured",
        "viewport_shading": "MATERIAL",
        "lighting": "studio",
        "use_scene_lights": False,
        "use_scene_world": False,
        "configured_viewport_count": 1,
        "configured_viewports": [
            {"screen": "Layout", "area_index": 0, "shading": "MATERIAL"}
        ],
        "failures": [],
        "scene_semantics_modified": False,
    }


def test_saved_blend_reports_headless_screen_without_view3d() -> None:
    screen = SimpleNamespace(
        name="Rendering",
        areas=[SimpleNamespace(type="OUTLINER")],
    )
    bpy = SimpleNamespace(data=SimpleNamespace(screens=[screen]))

    report = configure_textured_inspection_view(bpy)

    assert report["status"] == "unavailable"
    assert report["configured_viewport_count"] == 0
    assert report["scene_semantics_modified"] is False


def test_saved_blend_embeds_only_loaded_shader_images() -> None:
    class Image:
        def __init__(self, name: str, *, has_data: bool) -> None:
            self.name = name
            self.source = "FILE"
            self.has_data = has_data
            self.filepath = f"/{name}.png"
            self.packed_files: list[object] = []
            self.pack_calls = 0

        def pack(self) -> None:
            self.pack_calls += 1
            self.packed_files.append(object())

    loaded = Image("loaded", has_data=True)
    unloaded = Image("unloaded", has_data=False)
    nodes = [SimpleNamespace(image=loaded), SimpleNamespace(image=unloaded)]
    material = SimpleNamespace(node_tree=SimpleNamespace(nodes=nodes))
    bpy = SimpleNamespace(
        data=SimpleNamespace(materials=[material], node_groups=[]),
        path=SimpleNamespace(abspath=lambda value: value),
    )

    report = embed_available_shader_images(bpy)

    assert loaded.pack_calls == 1
    assert unloaded.pack_calls == 0
    assert report["packed_now"] == ["loaded"]
    assert report["unresolved"] == [
        {
            "image": "unloaded",
            "source": "FILE",
            "reason": "image_data_not_loaded",
        }
    ]
