"""Observable room geometry; not a generator-authored architecture contract."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from benchmark.architecture_policy import architecture_contract_from_scene


def observable_architecture_from_scene(
    scene: dict[str, Any],
) -> dict[str, Any]:
    """Return observable architecture without generator activation claims."""

    from benchmark.non_rectangular.geometry import polygon_geometry_from_scene

    geometry = polygon_geometry_from_scene(scene)
    if geometry is None:
        return deepcopy(architecture_contract_from_scene(scene))
    thicknesses = [wall.thickness_m for wall in geometry.walls]
    return {
        "geometry_type": "non_rectangular_polygon",
        "logical_boundary": {
            "enabled": True,
            "boundary": [list(point) for point in geometry.floor_polygon_xy],
        },
        "floor": {"enabled": True, "z": geometry.floor_z_m},
        "ceiling": {"enabled": geometry.ceiling_in_scope, "z": geometry.ceiling_z_m if geometry.ceiling_in_scope else None},
        "physical_walls": {
            "active_wall_ids": [wall.wall_id for wall in geometry.walls],
            "wall_thickness_m": (
                max(thicknesses) if thicknesses else None
            ),
            "wall_segments": [
                wall.public_dict() for wall in geometry.walls
            ],
        },
    }
