"""Serve a browser game under benchmark control and instrument it at load time.

Two problems make a game page unusable as an evidence source, and both are
solved before the page ever executes rather than after.

The first is the origin. A ``file://`` page cannot load ES modules or honour an
import map, which is how most of the observed games pull in three.js, so the
game is served from a loopback HTTP origin instead.

The second is reachability. A probe that searches ``window`` at runtime finds
nothing: a classic script's top-level ``const`` never becomes a window property,
and a scene held as a class field lives inside a closure. The scene has to be
captured at construction time, which means wrapping the ``Scene`` constructor
before any game code runs. Owning the HTTP origin makes that a server-side HTML
rewrite, so the mechanism is independent of any particular browser driver and is
testable without one.

The rewrite is deliberately conservative. It injects one classic script ahead of
all page scripts, and it repoints the ``three`` import-map specifier at a shim
that re-exports the real module. Nothing else about the page is altered, and the
original three.js source is reachable unchanged under a private specifier so the
game still runs against the exact build it shipped with.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

BENCHMARK_PREFIX = "/__benchmark__/"
HARNESS_SCRIPT_PATH = BENCHMARK_PREFIX + "harness.js"
THREE_SHIM_PATH = BENCHMARK_PREFIX + "three-shim.mjs"
THREE_VENDOR_PATH = BENCHMARK_PREFIX + "three-vendor.js"

# The shim imports the game's own three build under this specifier, so the
# rewritten import map keeps pointing at whatever the page originally shipped.
THREE_SOURCE_SPECIFIER = "__benchmark_three_source__"

_ASSET_DIR = Path(__file__).resolve().parent

_IMPORT_MAP_RE = re.compile(
    r"<script\b[^>]*\btype\s*=\s*[\"']importmap[\"'][^>]*>(?P<body>.*?)</script>",
    re.IGNORECASE | re.DOTALL,
)
_HEAD_OPEN_RE = re.compile(r"<head\b[^>]*>", re.IGNORECASE)
_HTML_OPEN_RE = re.compile(r"<html\b[^>]*>", re.IGNORECASE)
_THREE_SCRIPT_SRC_RE = re.compile(
    r"(<script\b[^>]*\bsrc\s*=\s*[\"'])(?P<url>[^\"']*three[^\"']*\.js)([\"'][^>]*>)",
    re.IGNORECASE,
)
_BARE_THREE_IMPORT_RE = re.compile(
    r"""\bfrom\s*["'](three)["']""",
    re.IGNORECASE,
)

_CONTENT_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".gif": "image/gif",
    ".glb": "model/gltf-binary",
    ".gltf": "model/gltf+json",
    ".html": "text/html; charset=utf-8",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".wasm": "application/wasm",
    ".wav": "audio/wav",
}


class GameHarnessError(RuntimeError):
    """Raised when a game page cannot be prepared for instrumented capture."""


@dataclass
class EntryRewriteReport:
    """What the rewriter did to one entry document, and what it could not do."""

    harness_injected: bool = False
    import_map_found: bool = False
    import_map_rewritten: bool = False
    bare_three_import: bool = False
    script_tag_three: bool = False
    external_three_urls: list[str] = field(default_factory=list)
    localized_three_urls: list[str] = field(default_factory=list)

    @property
    def network_required(self) -> bool:
        """True when the page still points at an origin the harness does not serve."""

        return bool(self.external_three_urls)

    @property
    def instrumentation(self) -> str:
        """Which capture mechanism the rewritten page will actually use."""

        if self.import_map_rewritten and self.script_tag_three:
            return "import_map_and_script_global"
        if self.import_map_rewritten:
            return "import_map"
        if self.script_tag_three:
            return "script_global"
        return "script_global_only"

    def to_dict(self) -> dict[str, Any]:
        return {
            "harness_injected": self.harness_injected,
            "instrumentation": self.instrumentation,
            "import_map_found": self.import_map_found,
            "import_map_rewritten": self.import_map_rewritten,
            "bare_three_import": self.bare_three_import,
            "script_tag_three": self.script_tag_three,
            "external_three_urls": list(self.external_three_urls),
            "localized_three_urls": list(self.localized_three_urls),
            "network_required": self.network_required,
        }


def read_asset(name: str) -> str:
    return (_ASSET_DIR / name).read_text(encoding="utf-8")


def build_harness_script(*, seed: int, step_ms: float) -> str:
    """Concatenate the determinism and instrumentation harnesses into one script.

    Both must be installed before the first page script observes ``Math.random``
    or assigns ``window.THREE``, so they ship as a single classic script that the
    rewriter places ahead of everything else.
    """

    config = json.dumps({"seed": int(seed), "stepMs": float(step_ms)})
    return "\n".join(
        [
            "/* benchmark harness: deterministic replay + scene instrumentation */",
            f"({read_asset('determinism.js')})({config});",
            f"({read_asset('instrument.js')})({config});",
            "",
        ]
    )


def build_three_shim(source_specifier: str = THREE_SOURCE_SPECIFIER) -> str:
    return read_asset("three_shim.mjs").replace("__THREE_SOURCE__", source_specifier)


def rewrite_entry_html(html: str, *, three_replacement_url: str | None = None) -> tuple[str, EntryRewriteReport]:
    """Return the instrumented entry document plus a report of what changed."""

    report = EntryRewriteReport()
    html, report = _rewrite_import_map(html, report, three_replacement_url=three_replacement_url)
    html, report = _rewrite_three_script_tags(html, report, three_replacement_url=three_replacement_url)
    if _BARE_THREE_IMPORT_RE.search(html):
        report.bare_three_import = True
    return _inject_harness_script(html, report)


def _rewrite_import_map(
    html: str,
    report: EntryRewriteReport,
    *,
    three_replacement_url: str | None,
) -> tuple[str, EntryRewriteReport]:
    match = _IMPORT_MAP_RE.search(html)
    if match is None:
        return html, report
    report.import_map_found = True
    try:
        import_map = json.loads(match.group("body"))
    except json.JSONDecodeError as exc:
        raise GameHarnessError(f"entry document has a malformed import map: {exc}") from exc
    imports = import_map.get("imports")
    if not isinstance(imports, dict) or "three" not in imports:
        return html, report

    original = str(imports["three"])
    if _is_external(original):
        if three_replacement_url is not None:
            report.localized_three_urls.append(original)
            original = three_replacement_url
        else:
            report.external_three_urls.append(original)
    imports[THREE_SOURCE_SPECIFIER] = original
    imports["three"] = THREE_SHIM_PATH
    report.import_map_rewritten = True

    body = json.dumps(import_map, indent=2, sort_keys=True)
    replacement = f'<script type="importmap">\n{body}\n</script>'
    return html[: match.start()] + replacement + html[match.end() :], report


def _rewrite_three_script_tags(
    html: str,
    report: EntryRewriteReport,
    *,
    three_replacement_url: str | None,
) -> tuple[str, EntryRewriteReport]:
    def replace(match: re.Match[str]) -> str:
        url = match.group("url")
        report.script_tag_three = True
        if not _is_external(url):
            return match.group(0)
        if three_replacement_url is None:
            report.external_three_urls.append(url)
            return match.group(0)
        report.localized_three_urls.append(url)
        return match.group(1) + three_replacement_url + match.group(3)

    return _THREE_SCRIPT_SRC_RE.sub(replace, html), report


def _inject_harness_script(html: str, report: EntryRewriteReport) -> tuple[str, EntryRewriteReport]:
    tag = f'<script src="{HARNESS_SCRIPT_PATH}"></script>'
    for pattern in (_HEAD_OPEN_RE, _HTML_OPEN_RE):
        match = pattern.search(html)
        if match is not None:
            report.harness_injected = True
            return html[: match.end()] + "\n" + tag + html[match.end() :], report
    report.harness_injected = True
    return tag + "\n" + html, report


def _is_external(url: str) -> bool:
    return url.startswith(("http://", "https://", "//"))


class GameSourceServer:
    """Serve one game directory from loopback with the entry document rewritten.

    Used as a context manager. The bound port is chosen by the OS, so concurrent
    captures never contend, and the server refuses any path that escapes the game
    root.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        entry: str | Path,
        seed: int = 20260727,
        step_ms: float = 1000.0 / 60.0,
        three_replacement: str | Path | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise GameHarnessError(f"game root does not exist: {self.root}")
        entry_path = Path(entry).expanduser()
        self.entry = (entry_path if entry_path.is_absolute() else self.root / entry_path).resolve()
        if not self.entry.is_file():
            raise GameHarnessError(f"game entry HTML does not exist: {self.entry}")
        if not self._within_root(self.entry):
            raise GameHarnessError("game entry HTML must live inside the game root")
        self.seed = int(seed)
        self.step_ms = float(step_ms)
        self.three_replacement = (
            Path(three_replacement).expanduser().resolve() if three_replacement is not None else None
        )
        if self.three_replacement is not None and not self.three_replacement.is_file():
            raise GameHarnessError(f"three replacement does not exist: {self.three_replacement}")
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._report: EntryRewriteReport | None = None

    def __enter__(self) -> "GameSourceServer":
        handler = _make_handler(self)
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    @property
    def port(self) -> int:
        if self._server is None:
            raise GameHarnessError("game source server is not running")
        return int(self._server.server_address[1])

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def entry_url(self) -> str:
        relative = self.entry.relative_to(self.root).as_posix()
        return f"{self.base_url}/{relative}"

    @property
    def rewrite_report(self) -> EntryRewriteReport | None:
        """The report from the most recent entry-document fetch, if any."""

        return self._report

    def render_entry(self) -> str:
        """Rewrite the entry document without going through HTTP."""

        html, report = rewrite_entry_html(
            self.entry.read_text(encoding="utf-8", errors="ignore"),
            three_replacement_url=THREE_VENDOR_PATH if self.three_replacement is not None else None,
        )
        self._report = report
        return html

    def _within_root(self, path: Path) -> bool:
        try:
            path.relative_to(self.root)
        except ValueError:
            return False
        return True

    def _resolve(self, url_path: str) -> Path | None:
        relative = url_path.lstrip("/")
        if not relative:
            return self.entry
        candidate = (self.root / relative).resolve()
        if not self._within_root(candidate) or not candidate.is_file():
            return None
        return candidate


def _make_handler(server: GameSourceServer) -> type[BaseHTTPRequestHandler]:
    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
            path = self.path.split("?", 1)[0].split("#", 1)[0]
            if path.startswith(BENCHMARK_PREFIX):
                self._serve_benchmark_asset(path)
                return
            resolved = server._resolve(path)
            if resolved is None:
                self._send(404, b"not found", "text/plain; charset=utf-8")
                return
            if resolved == server.entry:
                self._send(200, server.render_entry().encode("utf-8"), _CONTENT_TYPES[".html"])
                return
            self._send(200, resolved.read_bytes(), _content_type(resolved))

        def do_HEAD(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
            self.do_GET()

        def log_message(self, *args: object) -> None:
            """Silence the default stderr access log."""

        def _serve_benchmark_asset(self, path: str) -> None:
            if path == HARNESS_SCRIPT_PATH:
                payload = build_harness_script(seed=server.seed, step_ms=server.step_ms)
                self._send(200, payload.encode("utf-8"), _CONTENT_TYPES[".js"])
                return
            if path == THREE_SHIM_PATH:
                self._send(200, build_three_shim().encode("utf-8"), _CONTENT_TYPES[".mjs"])
                return
            if path == THREE_VENDOR_PATH and server.three_replacement is not None:
                self._send(200, server.three_replacement.read_bytes(), _CONTENT_TYPES[".js"])
                return
            self._send(404, b"not found", "text/plain; charset=utf-8")

        def _send(self, status: int, payload: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            # Every capture must observe the same bytes the harness just produced.
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(payload)

    return _Handler


def _content_type(path: Path) -> str:
    return _CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")
