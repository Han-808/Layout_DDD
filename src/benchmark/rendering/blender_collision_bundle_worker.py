"""One-load Collision raw/annotation/mask orchestration; never saves the scene.

The existing pass entry points remain authoritative. This worker is deliberately
single-pair/single-pose, with no resident service, scene pruning, or hash cache.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import bpy

WORKER_DIR = Path(__file__).resolve().parent
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

import blender_camera_worker as raw_worker
import blender_collision_overlay_worker as annotation_worker
import blender_collision_mask_worker as mask_worker


def _settings_snapshot(owner):
    values = {}
    for prop in owner.bl_rna.properties:
        if prop.is_readonly or prop.type not in {"BOOLEAN", "INT", "FLOAT", "STRING", "ENUM"}:
            continue
        value = getattr(owner, prop.identifier)
        values[prop.identifier] = tuple(value) if getattr(prop, "is_array", False) else value
    return owner, values


def _snapshot_scene():
    scene = bpy.context.scene
    collections = ["objects", "cameras", "lights", "curves", "meshes", "materials"]
    return {
        "data": {name: {item.as_pointer() for item in getattr(bpy.data, name)} for name in collections},
        "camera": scene.camera,
        "active": bpy.context.view_layer.objects.active,
        "selected": list(bpy.context.selected_objects),
        "world_color": tuple(scene.world.color),
        "settings": [_settings_snapshot(owner) for owner in [
            scene.render, scene.render.image_settings, scene.display.shading, scene.cycles,
        ]],
    }


def _restore_scene(snapshot):
    # Only delete data created by this attempt, never source-scene objects.
    bpy.context.scene.camera = snapshot["camera"]
    for name, original in snapshot["data"].items():
        collection = getattr(bpy.data, name)
        for item in list(collection):
            if item.as_pointer() not in original:
                if name == "objects":
                    collection.remove(item, do_unlink=True)
                elif item.users == 0:
                    collection.remove(item)
    for owner, values in snapshot["settings"]:
        for name, value in values.items():
            if owner == bpy.context.scene.render and name == "engine":
                # Every original pass explicitly chooses its own engine.
                # Avoid an unused round trip through the saved scene's engine.
                continue
            current = getattr(owner, name)
            if isinstance(value, tuple):
                current = tuple(current)
            # Older saved scenes can expose unset enum values which cannot be
            # assigned in a newer Blender. Do not touch unchanged settings.
            if current != value:
                setattr(owner, name, value)
    bpy.context.scene.world.color = snapshot["world_color"]
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    for obj in snapshot["selected"]:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = snapshot["active"]
    bpy.context.view_layer.update()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-json", required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    request = json.loads(Path(args.request_json).read_text(encoding="utf-8"))
    source = Path(request["source_blend"])
    if Path(bpy.data.filepath).resolve() != source.resolve():
        raise RuntimeError("Collision bundle loaded a different source scene")
    expected = request["source_sha256"]
    audit_path = Path(request["audit_path"])
    audit = {"schema_version": "collision_final_bundle_v1", "raw_complete": False,
             "status": "running", "hash_checks": [{"boundary": "raw_before", "sha256": expected, "location": "parent"}]}

    def save():
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    def check(boundary):
        handshake = Path(request["hash_handshake_dir"])
        (handshake / f"{boundary}.request").touch(exist_ok=False)
        response = handshake / f"{boundary}.response.json"
        deadline = time.monotonic() + request["timeout_seconds"]
        while not response.is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Parent integrity check timed out: {boundary}")
            time.sleep(0.01)
        result = json.loads(response.read_text(encoding="utf-8"))
        if result.get("error"):
            raise RuntimeError(result["error"])
        if result.get("boundary") != boundary:
            raise RuntimeError("Parent integrity response boundary mismatch")
        actual = result["sha256"]
        audit["hash_checks"].append(result)
        if actual != expected:
            audit["source_modified"] = True
            save()
            raise RuntimeError("Read-only Collision bundle detected a modified source scene")
        save()

    original_argv = sys.argv
    save()
    try:
        snapshot = _snapshot_scene()
        for stage, worker in [("raw", raw_worker), ("annotation", annotation_worker), ("mask", mask_worker)]:
            audit["stage"] = stage
            save()
            if stage != "raw":
                check(stage + "_before")
            sys.argv = [str(Path(worker.__file__)), "--", *request["stages"][stage]]
            worker.main()
            # The final mask-after hash is performed by the parent, including
            # on a process crash/timeout. Six full reads across three passes.
            if stage != "mask":
                check(stage + "_after")
            if stage == "raw":
                audit["raw_complete"] = True
                save()
            if stage != "mask":
                _restore_scene(snapshot)
        audit["status"] = "complete"
        save()
    except BaseException as exc:
        audit.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        raise
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
