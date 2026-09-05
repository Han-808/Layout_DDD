"""No-Blender tests of the exact input factory contract, not native loop mocks."""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def factory_module():
    path = Path(__file__).resolve().parents[1] / "scripts/external_harness_bridges/scene_weaver_frozen_assets.py"
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location("frozen_input_test_module", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.pop(0)


@pytest.fixture
def binding(tmp_path):
    mesh = tmp_path / "asset.glb"
    mesh.write_bytes(b"identity-only CI fixture, not a real GLB")
    return {"asset_key": "exact.asset", "mesh_uri": str(mesh),
            "mesh_sha256": hashlib.sha256(mesh.read_bytes()).hexdigest(),
            "bbox_size_local": [0.7, 1.3, 1.1], "bbox_center_local": [0.31, -0.27, 0.44],
            "native_scale": [1, 1, 1], "physical_dimensions": [0.7, 1.3, 1.1],
            "canonical_front": [0, -1, 0]}


def test_binding_is_exact_and_defensively_copied(factory_module, binding):
    original = deepcopy(binding)
    result = factory_module.validate_binding("slot_a", binding, tolerance=1e-4)
    assert binding == original
    assert result["asset_key"] == "exact.asset"
    assert result["canonical_front"] == [0, -1, 0]
    result["native_scale"][0] = 99
    assert binding == original


@pytest.mark.parametrize("field,value,message", [
    ("asset_key", None, "requires asset_key"),
    ("mesh_sha256", "0" * 64, "hash mismatch"),
    ("mesh_uri", "/missing/exact.glb", "existing exact GLB"),
    ("physical_dimensions", [9, 9, 9], "conflict"),
    ("native_scale", [-1, 1, 1], "finite and positive"),
    ("canonical_front", [1, 1, 0], "non-cardinal"),
    ("canonical_front", [float("nan"), 0, 0], "finite"),
    ("canonical_front", [0, 0, 1], "horizontal"),
    ("bbox_center_local", [1, 2], "three-vector"),
])
def test_invalid_frozen_input_fails_before_blender_import(factory_module, binding, field, value, message):
    binding[field] = value
    with pytest.raises((ValueError, RuntimeError), match=message):
        factory_module.load_exact_glb("slot_a", binding)


def test_source_change_is_rejected(factory_module, binding):
    factory_module.validate_binding("slot_a", binding, tolerance=1e-4)
    Path(binding["mesh_uri"]).write_bytes(b"replacement bytes")
    with pytest.raises(ValueError, match="hash mismatch"):
        factory_module.validate_binding("slot_a", binding, tolerance=1e-4)


def test_declared_physical_scale_is_applied_once(factory_module, binding):
    binding["native_scale"] = [2, 3, 4]
    binding["physical_dimensions"] = [1.4, 3.9, 4.4]
    value = factory_module.validate_binding("slot_a", binding, tolerance=1e-4)
    assert value["bbox_size_local"] == [0.7, 1.3, 1.1]
    anchor = factory_module._anchor_basis(value)
    assert anchor["canonical_bottom_center_local"] == pytest.approx([0.62, -0.81, -0.44])


def test_missing_front_is_not_invented(factory_module, binding):
    del binding["canonical_front"]
    value = factory_module.validate_binding("slot_a", binding, tolerance=1e-4)
    assert "canonical_front" not in value
    assert factory_module._orientation_basis(value)["applied"] is False


def test_factory_only_overrides_native_creation_hooks(factory_module, binding, monkeypatch):
    class NativeBaseShape:
        def __init__(self, factory_seed, coarse=False):
            self.factory_seed = factory_seed
            self.coarse = coarse

        def spawn_asset(self, **params):
            return self.create_asset(**params)

    calls = []

    def loader(slot, asset, **kwargs):
        calls.append((slot, asset["asset_key"]))
        return SimpleNamespace(name="created mesh"), {"slot_id": slot, "observed": True}

    monkeypatch.setattr(factory_module, "load_exact_glb", loader)
    one = factory_module.make_frozen_factory(NativeBaseShape, "slot_a", binding)
    two = factory_module.make_frozen_factory(NativeBaseShape, "slot_b", binding)
    assert one is factory_module.make_frozen_factory(NativeBaseShape, "slot_a", binding)
    assert one is not two
    assert one.spawn_asset is NativeBaseShape.spawn_asset
    obj = one(123, coarse=False)
    binding["asset_key"] = "changed caller binding"
    assert obj.spawn_asset(i=10).name == "created mesh"
    assert calls == [("slot_a", "exact.asset")]
    assert obj.factory_seed == 123
    assert obj.frozen_input_observations == [{"slot_id": "slot_a", "observed": True}]


def test_bounds_do_not_hide_bad_geometry(factory_module):
    with pytest.raises(ValueError, match="no mesh vertices"):
        factory_module._bounds([])
    with pytest.raises(ValueError, match="non-finite"):
        factory_module._bounds([(0, 0, float("nan"))])
