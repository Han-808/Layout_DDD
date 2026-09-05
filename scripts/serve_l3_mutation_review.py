#!/usr/bin/env python3
"""Serve the L3 mutation review UI and open trusted cases in Blender."""

from __future__ import annotations

import argparse
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_l3_mutation_dataset import DEFAULT_CONFIG, load_config
from scripts.build_l3_mutation_review import REVIEW_SCHEMA_VERSION


HUMAN_REVIEW_SCHEMA_VERSION = "l3_mutation_human_reviews_v1"
MAX_REQUEST_BYTES = 4 * 1024 * 1024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config, output_override=args.output_root)
    output_root = Path(config["_output_root"])
    review_data_path = output_root / "review" / "review_data.json"
    if not review_data_path.is_file():
        raise SystemExit(
            "review UI is not built; run build_l3_mutation_review.py"
        )
    review_data = _read_json(review_data_path)
    if review_data.get("schema_version") != REVIEW_SCHEMA_VERSION:
        raise SystemExit("review data schema is incompatible")
    host = str(args.host or config["review"]["host"])
    address = ipaddress.ip_address(host)
    if not address.is_loopback:
        parser.error("review server must bind to a loopback address")
    port = int(args.port or config["review"]["port"])
    if port <= 0 or port > 65535:
        parser.error("port must be in 1..65535")
    review_path = output_root / str(config["review"]["persist_file"])
    blender_bin = _path(
        config["render"]["blender_bin"],
        repo_root=PROJECT_ROOT,
    )
    handler = partial(
        ReviewHandler,
        directory=str(output_root),
        review_data=review_data,
        review_path=review_path,
        blender_bin=blender_bin,
        output_root=output_root,
    )
    server = ThreadingHTTPServer((host, port), handler)
    print(
        f"L3 mutation review: http://{host}:{port}/review/index.html",
        flush=True,
    )
    print(f"Review records: {review_path}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


class ReviewHandler(SimpleHTTPRequestHandler):
    def __init__(
        self,
        *args: Any,
        review_data: dict[str, Any],
        review_path: Path,
        blender_bin: Path,
        output_root: Path,
        **kwargs: Any,
    ) -> None:
        self.review_data = review_data
        self.review_path = review_path
        self.blender_bin = blender_bin
        self.output_root = output_root.resolve()
        self.case_by_id = {
            str(item["review_id"]): item
            for item in review_data["cases"]
        }
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        if urlsplit(self.path).path == "/api/reviews":
            payload = (
                _read_json(self.review_path)
                if self.review_path.is_file()
                else {
                    "schema_version": HUMAN_REVIEW_SCHEMA_VERSION,
                    "experiment_id": self.review_data["experiment_id"],
                    "dataset_fingerprint": self.review_data[
                        "dataset_fingerprint"
                    ],
                    "answers": {},
                }
            )
            self._send_json(HTTPStatus.OK, payload)
            return
        super().do_GET()

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        try:
            payload = self._read_payload()
            if path == "/api/reviews":
                validated = validate_review_payload(
                    payload,
                    review_data=self.review_data,
                )
                _write_json(self.review_path, validated)
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "ok": True,
                        "answer_count": len(validated["answers"]),
                        "saved_path": str(self.review_path),
                    },
                )
                return
            if path == "/api/open-blender":
                result = self._open_blender(payload)
                self._send_json(HTTPStatus.OK, result)
                return
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"error": "unknown API route"},
            )
        except (TypeError, ValueError, OSError) as exc:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": str(exc)[:1000]},
            )

    def _read_payload(self) -> Any:
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length <= 0 or length > MAX_REQUEST_BYTES:
            raise ValueError("request body size is invalid")
        try:
            return json.loads(self.rfile.read(length))
        except json.JSONDecodeError as exc:
            raise ValueError("request body is not valid JSON") from exc

    def _open_blender(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("open-blender payload must be an object")
        if set(payload) != {"review_id", "which"}:
            raise ValueError("open-blender payload fields are invalid")
        review_id = str(payload["review_id"])
        which = str(payload["which"])
        if which not in {"source", "variant"}:
            raise ValueError("which must be source or variant")
        case = self.case_by_id.get(review_id)
        if case is None:
            raise ValueError("unknown review_id")
        web_path = str(case[which].get("blend") or "")
        if not web_path.startswith("/"):
            raise ValueError("requested blend file is unavailable")
        blend_path = (self.output_root / web_path.lstrip("/")).resolve()
        try:
            blend_path.relative_to(self.output_root)
        except ValueError as exc:
            raise ValueError("blend path escaped output root") from exc
        if not blend_path.is_file() or blend_path.suffix != ".blend":
            raise ValueError("trusted blend file is missing")
        subprocess.Popen(
            [str(self.blender_bin), str(blend_path)],
            cwd=str(blend_path.parent),
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return {
            "ok": True,
            "review_id": review_id,
            "which": which,
            "blend_file": str(blend_path),
        }

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

    def _send_json(
        self,
        status: HTTPStatus,
        value: dict[str, Any],
    ) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header(
            "Content-Type", "application/json; charset=utf-8"
        )
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def validate_review_payload(
    value: Any,
    *,
    review_data: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("review payload must be an object")
    expected = {
        "schema_version",
        "experiment_id",
        "dataset_fingerprint",
        "answers",
    }
    if set(value) != expected:
        raise ValueError("review payload fields are invalid")
    if value["schema_version"] != HUMAN_REVIEW_SCHEMA_VERSION:
        raise ValueError("review schema_version is invalid")
    if value["experiment_id"] != review_data["experiment_id"]:
        raise ValueError("review experiment_id is invalid")
    if value["dataset_fingerprint"] != review_data["dataset_fingerprint"]:
        raise ValueError("review dataset_fingerprint is invalid")
    answers = value.get("answers")
    if not isinstance(answers, dict):
        raise ValueError("review answers must be an object")
    known = {
        str(item["review_id"]) for item in review_data["cases"]
    }
    if set(answers) - known:
        raise ValueError("review payload references unknown cases")
    contract = review_data["annotation_contract"]
    normalized = {
        review_id: _validate_answer(
            answer,
            contract=contract,
            label=f"answers.{review_id}",
        )
        for review_id, answer in answers.items()
    }
    return {
        "schema_version": HUMAN_REVIEW_SCHEMA_VERSION,
        "experiment_id": review_data["experiment_id"],
        "dataset_fingerprint": review_data["dataset_fingerprint"],
        "answers": normalized,
    }


def _validate_answer(
    value: Any,
    *,
    contract: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    expected = {
        "overall_label",
        "severity",
        "issues",
        "evidence_sufficiency",
        "render_integrity",
        "notes",
        "reviewed",
    }
    if set(value) != expected:
        raise ValueError(f"{label} fields are invalid")
    for field, allowed_key in (
        ("overall_label", "overall_labels"),
        ("severity", "severity_labels"),
        ("evidence_sufficiency", "evidence_labels"),
        ("render_integrity", "render_labels"),
    ):
        if value[field] not in contract[allowed_key]:
            raise ValueError(f"{label}.{field} is invalid")
    issues = value["issues"]
    if (
        not isinstance(issues, list)
        or any(item not in contract["issue_labels"] for item in issues)
        or len(issues) != len(set(issues))
    ):
        raise ValueError(f"{label}.issues is invalid")
    notes = value["notes"]
    if not isinstance(notes, str) or len(notes) > 5000:
        raise ValueError(f"{label}.notes is invalid")
    if not isinstance(value["reviewed"], bool):
        raise ValueError(f"{label}.reviewed must be boolean")
    return {
        "overall_label": value["overall_label"],
        "severity": value["severity"],
        "issues": list(issues),
        "evidence_sufficiency": value["evidence_sufficiency"],
        "render_integrity": value["render_integrity"],
        "notes": notes,
        "reviewed": value["reviewed"],
    }


def _path(value: Any, *, repo_root: Path) -> Path:
    path = Path(str(value)).expanduser()
    resolved = path if path.is_absolute() else repo_root / path
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved.resolve()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()
