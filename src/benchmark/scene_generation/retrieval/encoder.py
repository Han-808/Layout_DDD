"""Encoder descriptor and content-addressed local snapshot validation v2."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
from pathlib import Path
from typing import Any, Mapping

from ._common import (
    RetrievalContractError,
    array_value,
    canonical_json_bytes,
    exact_keys,
    identifier,
    object_value,
    positive_int,
    safe_relative_path,
    sha256_bytes,
    sha256_file,
    sha256_value,
    string_value,
)


@dataclass(frozen=True, slots=True)
class RequiredSnapshotFile:
    path: str
    bytes: int
    sha256: str

    @classmethod
    def parse(cls, value: Any, *, label: str) -> "RequiredSnapshotFile":
        raw = object_value(value, label=label)
        exact_keys(raw, label=label, required=("path", "bytes", "sha256"))
        return cls(
            path=safe_relative_path(raw["path"], label=f"{label}.path"),
            bytes=positive_int(raw["bytes"], label=f"{label}.bytes"),
            sha256=sha256_value(raw["sha256"], label=f"{label}.sha256"),
        )


@dataclass(frozen=True, slots=True)
class EncoderDescriptor:
    encoder_id: str
    implementation: str
    model_resource_id: str
    upstream_model_id: str
    revision: str
    expected_dimension: int
    whitespace_normalization: str
    prompt_name: str
    required_files: tuple[RequiredSnapshotFile, ...]
    snapshot_manifest_sha256: str
    package_expectations: Mapping[str, str]

    @classmethod
    def parse(cls, value: Any, *, label: str) -> "EncoderDescriptor":
        raw = object_value(value, label=label)
        exact_keys(
            raw,
            label=label,
            required=(
                "encoder_id",
                "implementation",
                "model_resource_id",
                "upstream_model_id",
                "revision",
                "expected_dimension",
                "query",
                "required_files",
                "snapshot_manifest_sha256",
                "package_expectations",
            ),
        )
        implementation = identifier(
            raw["implementation"], label=f"{label}.implementation"
        )
        if implementation != "sentence_transformers_v2":
            raise RetrievalContractError(
                "unsupported encoder implementation contract; model aliases are unrestricted, "
                "but new executable encoder contracts require reviewed code"
            )
        query = object_value(raw["query"], label=f"{label}.query")
        exact_keys(
            query,
            label=f"{label}.query",
            required=("whitespace_normalization", "prompt_name"),
        )
        normalization = string_value(
            query["whitespace_normalization"],
            label=f"{label}.query.whitespace_normalization",
        )
        if normalization != "collapse_ascii_whitespace_v2":
            raise RetrievalContractError("unsupported query whitespace normalization")
        file_values = array_value(raw["required_files"], label=f"{label}.required_files")
        if not file_values:
            raise RetrievalContractError(f"{label}.required_files must not be empty")
        files = tuple(
            RequiredSnapshotFile.parse(item, label=f"{label}.required_files[{index}]")
            for index, item in enumerate(file_values)
        )
        if len({item.path for item in files}) != len(files):
            raise RetrievalContractError(f"{label}.required_files contains duplicate paths")
        expected_manifest = sha256_bytes(
            canonical_json_bytes(
                [
                    {"path": item.path, "bytes": item.bytes, "sha256": item.sha256}
                    for item in files
                ]
            )
        )
        declared_manifest = sha256_value(
            raw["snapshot_manifest_sha256"],
            label=f"{label}.snapshot_manifest_sha256",
        )
        if expected_manifest != declared_manifest:
            raise RetrievalContractError(
                f"{label}.snapshot_manifest_sha256 does not match required_files"
            )
        packages = object_value(
            raw["package_expectations"], label=f"{label}.package_expectations"
        )
        normalized_packages = {
            string_value(name, label=f"{label}.package name"): string_value(
                version, label=f"{label}.package_expectations.{name}"
            )
            for name, version in packages.items()
        }
        return cls(
            encoder_id=identifier(raw["encoder_id"], label=f"{label}.encoder_id"),
            implementation=implementation,
            model_resource_id=identifier(
                raw["model_resource_id"], label=f"{label}.model_resource_id"
            ),
            upstream_model_id=string_value(
                raw["upstream_model_id"], label=f"{label}.upstream_model_id"
            ),
            revision=identifier(raw["revision"], label=f"{label}.revision"),
            expected_dimension=positive_int(
                raw["expected_dimension"], label=f"{label}.expected_dimension"
            ),
            whitespace_normalization=normalization,
            prompt_name=string_value(
                query["prompt_name"], label=f"{label}.query.prompt_name"
            ),
            required_files=files,
            snapshot_manifest_sha256=declared_manifest,
            package_expectations=normalized_packages,
        )

    def validate_snapshot(self, root: Path) -> dict[str, Any]:
        if not root.is_dir():
            raise RetrievalContractError(
                f"encoder resource {self.model_resource_id!r} is unavailable"
            )
        declared = {item.path: item for item in self.required_files}
        actual = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        }
        if actual != set(declared):
            raise RetrievalContractError(
                f"encoder snapshot inventory differs: missing={sorted(set(declared)-actual)}, "
                f"extra={sorted(actual-set(declared))}"
            )
        for relative, item in declared.items():
            candidate = root / relative
            # Hugging Face snapshots are symlink forests into the blob store.
            # Following a symlink is allowed; trust comes from bytes+hash, not
            # from where the binding points.
            if not candidate.is_file():
                raise RetrievalContractError(
                    f"encoder snapshot file is unavailable: {relative}"
                )
            if candidate.stat().st_size != item.bytes:
                raise RetrievalContractError(
                    f"encoder snapshot byte size differs: {relative}"
                )
            if sha256_file(candidate) != item.sha256:
                raise RetrievalContractError(
                    f"encoder snapshot hash differs: {relative}"
                )
        return {
            "resource_id": self.model_resource_id,
            "revision": self.revision,
            "snapshot_manifest_sha256": self.snapshot_manifest_sha256,
            "file_count": len(self.required_files),
        }

    def package_drift(self) -> list[dict[str, str | None]]:
        drift: list[dict[str, str | None]] = []
        for distribution, expected in self.package_expectations.items():
            try:
                actual: str | None = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                actual = None
            if actual != expected:
                drift.append(
                    {
                        "distribution": distribution,
                        "expected": expected,
                        "actual": actual,
                    }
                )
        return drift
