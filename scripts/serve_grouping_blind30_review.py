#!/usr/bin/env python3
"""Serve the blind grouping UI and persist human reviews on localhost."""

from __future__ import annotations

import argparse
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
from pathlib import Path
import sys
from typing import Any
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.grouping_blind30_contracts import (
    atomic_write_json,
    read_json,
)


HUMAN_REVIEW_SCHEMA_VERSION = "grouping_blind30_human_reviews_v1"
QUALITY_VALUES = {
    "",
    "correct",
    "partially_correct",
    "incorrect",
    "unclear",
}
BEST_VALUES = {"", "A", "B", "C", "tie", "unclear"}
BLIND_LABELS = {"A", "B", "C"}
MAX_REQUEST_BYTES = 2 * 1024 * 1024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    address = ipaddress.ip_address(args.host)
    if not address.is_loopback:
        parser.error("review server host must be a loopback address")
    if args.port <= 0 or args.port > 65535:
        parser.error("--port must be from 1 to 65535")
    output_root = args.output_root.expanduser().resolve()
    review_root = output_root / "blind_review"
    review_data_path = review_root / "review_data.json"
    if not (review_root / "index.html").is_file():
        raise SystemExit(
            f"blind review is not built: {review_root / 'index.html'}"
        )
    review_data = read_json(review_data_path)
    review_path = output_root / "human_reviews" / "blind_reviews.json"
    handler = partial(
        ReviewHandler,
        directory=str(review_root),
        review_data=review_data,
        review_path=review_path,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(
        f"Blind grouping review: http://{args.host}:{args.port}/index.html",
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
        **kwargs: Any,
    ) -> None:
        self.review_data = review_data
        self.review_path = review_path
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        if urlsplit(self.path).path == "/api/reviews":
            value = (
                read_json(self.review_path)
                if self.review_path.is_file()
                else {
                    "schema_version": HUMAN_REVIEW_SCHEMA_VERSION,
                    "experiment_id": self.review_data["experiment_id"],
                    "answers": {},
                }
            )
            self._send_json(HTTPStatus.OK, value)
            return
        super().do_GET()

    def do_POST(self) -> None:
        if urlsplit(self.path).path != "/api/reviews":
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"error": "unknown API route"},
            )
            return
        raw_length = self.headers.get("Content-Length")
        try:
            length = int(raw_length or "0")
        except ValueError:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "invalid Content-Length"},
            )
            return
        if length <= 0 or length > MAX_REQUEST_BYTES:
            self._send_json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"error": "review payload size is invalid"},
            )
            return
        try:
            payload = json.loads(self.rfile.read(length))
            validated = validate_review_payload(
                payload,
                review_data=self.review_data,
            )
            atomic_write_json(self.review_path, validated)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": str(exc)[:500]},
            )
            return
        self._send_json(
            HTTPStatus.OK,
            {
                "ok": True,
                "saved_path": str(self.review_path),
                "answer_count": len(validated["answers"]),
            },
        )

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
        self.send_header("Content-Type", "application/json; charset=utf-8")
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
    if set(value) != {"schema_version", "experiment_id", "answers"}:
        raise ValueError("review payload contains unsupported fields")
    if value.get("schema_version") != HUMAN_REVIEW_SCHEMA_VERSION:
        raise ValueError("review payload schema_version is invalid")
    if value.get("experiment_id") != review_data.get("experiment_id"):
        raise ValueError("review payload experiment_id is invalid")
    answers = value.get("answers")
    if not isinstance(answers, dict):
        raise ValueError("review payload answers must be an object")
    known_cases = {
        str(item["case_id"]) for item in review_data.get("cases", [])
    }
    unknown_cases = sorted(set(answers) - known_cases)
    if unknown_cases:
        raise ValueError(
            f"review payload references unknown cases {unknown_cases}"
        )
    normalized: dict[str, Any] = {}
    for case_id, answer in answers.items():
        normalized[case_id] = _validate_case_answer(
            answer,
            label=f"answers.{case_id}",
        )
    return {
        "schema_version": HUMAN_REVIEW_SCHEMA_VERSION,
        "experiment_id": review_data["experiment_id"],
        "answers": normalized,
    }


def _validate_case_answer(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    expected = {"reviewed", "best_result", "notes", "variants"}
    if set(value) != expected:
        raise ValueError(f"{label} contains unsupported fields")
    if not isinstance(value["reviewed"], bool):
        raise ValueError(f"{label}.reviewed must be boolean")
    best = value["best_result"]
    if best not in BEST_VALUES:
        raise ValueError(f"{label}.best_result is invalid")
    notes = _review_text(value["notes"], f"{label}.notes")
    variants = value["variants"]
    if not isinstance(variants, dict) or set(variants) != BLIND_LABELS:
        raise ValueError(f"{label}.variants must contain A, B, and C")
    normalized_variants: dict[str, Any] = {}
    for blind_label in sorted(BLIND_LABELS):
        variant = variants[blind_label]
        variant_label = f"{label}.variants.{blind_label}"
        if not isinstance(variant, dict) or set(variant) != {
            "quality",
            "notes",
        }:
            raise ValueError(
                f"{variant_label} contains unsupported fields"
            )
        quality = variant["quality"]
        if quality not in QUALITY_VALUES:
            raise ValueError(f"{variant_label}.quality is invalid")
        normalized_variants[blind_label] = {
            "quality": quality,
            "notes": _review_text(
                variant["notes"],
                f"{variant_label}.notes",
            ),
        }
    return {
        "reviewed": value["reviewed"],
        "best_result": best,
        "notes": notes,
        "variants": normalized_variants,
    }


def _review_text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    if len(value) > 20_000:
        raise ValueError(f"{label} is too long")
    return value


if __name__ == "__main__":
    main()
