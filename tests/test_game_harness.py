from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from benchmark.game_scene.harness import (
    HARNESS_SCRIPT_PATH,
    THREE_SHIM_PATH,
    THREE_SOURCE_SPECIFIER,
    THREE_VENDOR_PATH,
    GameHarnessError,
    GameSourceServer,
    build_harness_script,
    build_three_shim,
    rewrite_entry_html,
)


_SCRIPT_TAG_GAME = """<!DOCTYPE html>
<html>
<head>
  <title>script tag game</title>
  <script src="three.min.js"></script>
</head>
<body><canvas id="c"></canvas><script src="game.js"></script></body>
</html>
"""

_IMPORT_MAP_GAME = """<!DOCTYPE html>
<html>
<head>
  <script type="importmap">
  {
    "imports": {
      "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
      "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
    }
  }
  </script>
</head>
<body><script type="module" src="game.js"></script></body>
</html>
"""

_CDN_SCRIPT_GAME = """<!DOCTYPE html>
<html><head>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head><body></body></html>
"""


def _import_map(html: str) -> dict:
    match = re.search(
        r'<script type="importmap">\s*(?P<body>\{.*?\})\s*</script>', html, re.DOTALL
    )
    assert match is not None, "rewritten document lost its import map"
    return json.loads(match.group("body"))


def _game_dir(tmp_path: Path, html: str, *, name: str = "index.html") -> Path:
    root = tmp_path / "game"
    root.mkdir(exist_ok=True)
    (root / name).write_text(html, encoding="utf-8")
    (root / "game.js").write_text("// game source\n", encoding="utf-8")
    (root / "three.min.js").write_text("// vendored three\n", encoding="utf-8")
    return root


def _get(url: str) -> tuple[int, str, str]:
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.status, response.read().decode("utf-8"), response.headers["Content-Type"]


def test_harness_script_precedes_every_page_script() -> None:
    html, report = rewrite_entry_html(_SCRIPT_TAG_GAME)

    assert report.harness_injected is True
    # The trap must be installed before three.js assigns window.THREE, so the
    # injected tag has to sit ahead of every other script in the document.
    positions = [match.start() for match in re.finditer(r"<script", html)]
    harness_position = html.index(HARNESS_SCRIPT_PATH)
    assert positions[0] < harness_position < positions[1]


def test_import_map_repoints_three_at_the_shim_and_keeps_the_original() -> None:
    html, report = rewrite_entry_html(_IMPORT_MAP_GAME)

    imports = _import_map(html)["imports"]
    assert imports["three"] == THREE_SHIM_PATH
    # The game must still run against the exact build it shipped with.
    assert imports[THREE_SOURCE_SPECIFIER] == "https://unpkg.com/three@0.160.0/build/three.module.js"
    assert imports["three/addons/"] == "https://unpkg.com/three@0.160.0/examples/jsm/"
    assert report.import_map_rewritten is True
    assert report.instrumentation == "import_map"


def test_shim_re_exports_the_original_module_and_shadows_scene() -> None:
    shim = build_three_shim()

    assert f'import * as THREE from "{THREE_SOURCE_SPECIFIER}"' in shim
    assert f'export * from "{THREE_SOURCE_SPECIFIER}"' in shim
    assert "export { Scene, WebGLRenderer }" in shim
    # Subclassing is what keeps isScene, instanceof, and the prototype chain
    # intact for addons that branch on them.
    assert "class Scene extends THREE.Scene" in shim
    assert "class WebGLRenderer extends THREE.WebGLRenderer" in shim
    assert "__benchmarkRegisterRender" in shim


def test_instrumentation_traps_the_namespace_not_just_the_assignment() -> None:
    script = build_harness_script(seed=7, step_ms=16.0)

    # three's UMD wrapper assigns an empty object and populates it afterwards,
    # so a plain window.THREE setter would never observe Scene.
    assert "__benchmarkTrapThreeNamespace" in script
    assert "__BENCHMARK_SCENES__" in script
    assert "__BENCHMARK_RENDERS__" in script
    assert "wrapRendererConstructor" in script
    assert "ambiguous_multiple_webgl_renderers" in (
        Path("src/benchmark/game_scene/render_camera_view.js").read_text(
            encoding="utf-8"
        )
    )
    assert "__benchmarkStep" in script
    assert '"seed": 7' in script or '"seed":7' in script


def test_external_three_is_flagged_when_no_local_replacement_exists() -> None:
    _, import_map_report = rewrite_entry_html(_IMPORT_MAP_GAME)
    _, script_report = rewrite_entry_html(_CDN_SCRIPT_GAME)

    assert import_map_report.network_required is True
    assert script_report.network_required is True
    assert script_report.script_tag_three is True
    assert "cdnjs.cloudflare.com" in script_report.external_three_urls[0]


def test_local_replacement_removes_the_network_dependency() -> None:
    html, report = rewrite_entry_html(_CDN_SCRIPT_GAME, three_replacement_url=THREE_VENDOR_PATH)

    assert report.network_required is False
    assert report.localized_three_urls == [
        "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"
    ]
    assert THREE_VENDOR_PATH in html
    assert "cdnjs.cloudflare.com" not in html


def test_vendored_three_script_tag_is_left_alone() -> None:
    html, report = rewrite_entry_html(_SCRIPT_TAG_GAME)

    assert report.script_tag_three is True
    assert report.network_required is False
    assert '<script src="three.min.js"></script>' in html


def test_document_without_head_still_gets_the_harness_first() -> None:
    html, report = rewrite_entry_html("<body><script src='game.js'></script></body>")

    assert report.harness_injected is True
    assert html.index(HARNESS_SCRIPT_PATH) < html.index("game.js")


def test_malformed_import_map_is_rejected_rather_than_ignored() -> None:
    broken = '<html><head><script type="importmap">{not json}</script></head></html>'

    with pytest.raises(GameHarnessError, match="malformed import map"):
        rewrite_entry_html(broken)


def test_server_serves_the_rewritten_entry_over_http(tmp_path: Path) -> None:
    root = _game_dir(tmp_path, _IMPORT_MAP_GAME)

    with GameSourceServer(root, entry="index.html") as server:
        assert server.entry_url.startswith("http://127.0.0.1:")
        status, body, content_type = _get(server.entry_url)

    assert status == 200
    assert content_type.startswith("text/html")
    # A file:// origin cannot honour an import map at all, which is why the
    # game is served rather than opened.
    assert _import_map(body)["imports"]["three"] == THREE_SHIM_PATH
    assert HARNESS_SCRIPT_PATH in body


def test_server_publishes_the_harness_and_shim_assets(tmp_path: Path) -> None:
    root = _game_dir(tmp_path, _IMPORT_MAP_GAME)

    with GameSourceServer(root, entry="index.html", seed=99) as server:
        harness_status, harness_body, harness_type = _get(server.base_url + HARNESS_SCRIPT_PATH)
        shim_status, shim_body, shim_type = _get(server.base_url + THREE_SHIM_PATH)

    assert (harness_status, shim_status) == (200, 200)
    assert harness_type.startswith("text/javascript")
    assert shim_type.startswith("text/javascript")
    assert "__benchmarkStep" in harness_body
    assert str(99) in harness_body
    assert THREE_SOURCE_SPECIFIER in shim_body


def test_server_streams_ordinary_game_files_untouched(tmp_path: Path) -> None:
    root = _game_dir(tmp_path, _SCRIPT_TAG_GAME)

    with GameSourceServer(root, entry="index.html") as server:
        status, body, _ = _get(f"{server.base_url}/game.js")

    assert status == 200
    assert body == "// game source\n"


def test_server_refuses_paths_that_escape_the_game_root(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("private", encoding="utf-8")
    root = _game_dir(tmp_path, _SCRIPT_TAG_GAME)

    with GameSourceServer(root, entry="index.html") as server:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            _get(f"{server.base_url}/../secret.txt")

    assert excinfo.value.code == 404


def test_server_can_substitute_a_local_three_for_a_cdn_url(tmp_path: Path) -> None:
    root = _game_dir(tmp_path, _CDN_SCRIPT_GAME)

    with GameSourceServer(
        root, entry="index.html", three_replacement=root / "three.min.js"
    ) as server:
        _, body, _ = _get(server.entry_url)
        vendor_status, vendor_body, _ = _get(server.base_url + THREE_VENDOR_PATH)
        report = server.rewrite_report

    assert "cdnjs.cloudflare.com" not in body
    assert vendor_status == 200
    assert vendor_body == "// vendored three\n"
    assert report is not None and report.network_required is False


def test_server_rejects_an_entry_outside_the_served_root(tmp_path: Path) -> None:
    root = _game_dir(tmp_path, _SCRIPT_TAG_GAME)
    outside = tmp_path / "elsewhere.html"
    outside.write_text("<html></html>", encoding="utf-8")

    with pytest.raises(GameHarnessError, match="inside the game root"):
        GameSourceServer(root, entry=outside)


def test_server_rejects_a_missing_entry(tmp_path: Path) -> None:
    root = _game_dir(tmp_path, _SCRIPT_TAG_GAME)

    with pytest.raises(GameHarnessError, match="entry HTML does not exist"):
        GameSourceServer(root, entry="nope.html")


def test_probe_reads_the_instrumentation_registry() -> None:
    probe = Path("src/benchmark/game_scene/probe.js").read_text(encoding="utf-8")

    # Scanning window finds nothing on real games; the registry is the only
    # reliable source, and the probe must prefer it.
    assert "__BENCHMARK_SCENES__" in probe
    assert "window.__BENCHMARK_THREE__ || window.THREE" in probe
