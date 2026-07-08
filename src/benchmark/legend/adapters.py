from __future__ import annotations

from benchmark.data.scene_adapters import layout_to_scene, scene_to_layout


def legend_layout_to_scene(layout: dict, case: dict | None = None) -> dict:
    return layout_to_scene(layout, case)


def scene_to_legend_layout(scene: dict) -> dict:
    return scene_to_layout(scene)
