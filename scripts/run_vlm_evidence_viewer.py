#!/usr/bin/env python3
"""Build and serve the read-only VLM evidence viewer on localhost."""

from __future__ import annotations

import argparse
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
from pathlib import Path
import sys
import threading
from typing import Any
from urllib.parse import quote
import webbrowser


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_vlm_evidence_viewer import build_viewer  # noqa: E402


REPORT_SUFFIXES = {".json", ".jsonl"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_root",
        nargs="?",
        type=Path,
        help="Scene-level run output root.",
    )
    parser.add_argument(
        "--run-root",
        dest="run_root_option",
        type=Path,
        help="Scene-level run output root (named form).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Local port. The default 0 asks the OS for a free port.",
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=None,
        help="Viewer bundle path; defaults to RUN_ROOT/viewer_bundle.",
    )
    parser.add_argument(
        "--scene",
        default=None,
        help="Scene ID to select when the browser opens.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Print the URL without opening the default browser.",
    )
    parser.add_argument(
        "--no-watch",
        action="store_true",
        help="Do not rebuild from newer report files when the page refreshes.",
    )
    args = parser.parse_args()

    run_root = _resolve_run_root(parser, args)
    validate_loopback_host(parser, args.host)
    if args.port < 0 or args.port > 65535:
        parser.error("--port must be from 0 to 65535")
    if not run_root.is_dir():
        parser.error(f"run root does not exist: {run_root}")
    if not (run_root / "run_manifest.json").is_file():
        parser.error(f"run manifest does not exist: {run_root}")

    bundle_dir = (
        args.bundle_dir.expanduser().resolve()
        if args.bundle_dir is not None
        else run_root / "viewer_bundle"
    )
    runtime = ViewerRuntime(
        run_root=run_root,
        bundle_dir=bundle_dir,
        watch=not args.no_watch,
    )
    runtime.rebuild(force=True)

    handler = partial(
        ViewerHandler,
        directory=str(bundle_dir),
        runtime=runtime,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    port = int(server.server_address[1])
    scene = args.scene or first_scene_id(run_root)
    url = viewer_url(args.host, port, scene=scene)

    print(f"VLM evidence viewer: {url}", flush=True)
    print(f"Run output: {run_root}", flush=True)
    print(f"Read-only viewer bundle: {bundle_dir}", flush=True)
    if runtime.watch:
        print(
            "Refresh the page to rebuild from newly persisted run reports.",
            flush=True,
        )
    print("Press Ctrl-C to stop the local server.", flush=True)

    if not args.no_open:
        timer = threading.Timer(0.2, webbrowser.open, args=(url,))
        timer.daemon = True
        timer.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _resolve_run_root(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> Path:
    positional = args.run_root
    named = args.run_root_option
    if positional is None and named is None:
        parser.error("provide RUN_ROOT or --run-root RUN_ROOT")
    if positional is not None and named is not None:
        if positional.expanduser().resolve() != named.expanduser().resolve():
            parser.error("positional RUN_ROOT and --run-root disagree")
    return (named or positional).expanduser().resolve()


def validate_loopback_host(
    parser: argparse.ArgumentParser,
    host: str,
) -> None:
    if host == "localhost":
        return
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        parser.error("--host must be localhost or a loopback IP address")
    if not address.is_loopback:
        parser.error("viewer server must bind to a loopback address")


def viewer_url(host: str, port: int, *, scene: str | None) -> str:
    rendered_host = f"[{host}]" if ":" in host else host
    fragment = f"#{quote(scene)}" if scene else ""
    return f"http://{rendered_host}:{port}/index.html{fragment}"


def first_scene_id(run_root: Path) -> str | None:
    cases_root = run_root / "cases"
    if not cases_root.is_dir():
        return None
    case_ids = sorted(path.name for path in cases_root.iterdir() if path.is_dir())
    return case_ids[0] if case_ids else None


def source_revision(run_root: Path, *, bundle_dir: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(
        path
        for path in run_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in REPORT_SUFFIXES
        and not _is_relative_to(path, bundle_dir)
    )
    for path in paths:
        stat = path.stat()
        digest.update(str(path.relative_to(run_root)).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(str(stat.st_size).encode("ascii"))
    return digest.hexdigest()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


class ViewerRuntime:
    def __init__(
        self,
        *,
        run_root: Path,
        bundle_dir: Path,
        watch: bool,
    ) -> None:
        self.run_root = run_root.expanduser().resolve()
        self.bundle_dir = bundle_dir.expanduser().resolve()
        self.watch = watch
        self._revision: str | None = None
        self._lock = threading.Lock()

    def rebuild(self, *, force: bool = False) -> bool:
        if not force and not self.watch:
            return False
        revision = source_revision(
            self.run_root,
            bundle_dir=self.bundle_dir,
        )
        if not force and revision == self._revision:
            return False
        with self._lock:
            revision = source_revision(
                self.run_root,
                bundle_dir=self.bundle_dir,
            )
            if not force and revision == self._revision:
                return False
            build_viewer(
                self.run_root,
                serve_root=PROJECT_ROOT,
                bundle_dir=self.bundle_dir,
            )
            self._revision = revision
        return True


class ViewerHandler(SimpleHTTPRequestHandler):
    def __init__(
        self,
        *args: Any,
        runtime: ViewerRuntime,
        **kwargs: Any,
    ) -> None:
        self.runtime = runtime
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        self._refresh_viewer()
        super().do_GET()

    def do_HEAD(self) -> None:
        self._refresh_viewer()
        super().do_HEAD()

    def _refresh_viewer(self) -> None:
        try:
            rebuilt = self.runtime.rebuild()
        except (OSError, RuntimeError, ValueError) as exc:
            print(
                f"Viewer rebuild failed; serving the last good bundle: {exc}",
                file=sys.stderr,
                flush=True,
            )
            return
        if rebuilt:
            print("Viewer rebuilt from updated reports.", flush=True)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        super().end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write(
            "%s - - [%s] %s\n"
            % (
                self.address_string(),
                self.log_date_time_string(),
                format % args,
            )
        )


if __name__ == "__main__":
    main()
