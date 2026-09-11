"""Content-addressed dense index descriptor v2."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from ._common import (
    RetrievalContractError,
    exact_keys,
    identifier,
    object_value,
    positive_int,
    sha256_file,
    sha256_value,
    string_value,
)
from .dataset import DatasetDescriptor


@dataclass(frozen=True, slots=True)
class BoundFileDescriptor:
    resource_id: str
    bytes: int
    sha256: str

    @classmethod
    def parse(cls, value: Any, *, label: str) -> "BoundFileDescriptor":
        raw = object_value(value, label=label)
        exact_keys(raw, label=label, required=("resource_id", "bytes", "sha256"))
        return cls(
            resource_id=identifier(raw["resource_id"], label=f"{label}.resource_id"),
            bytes=positive_int(raw["bytes"], label=f"{label}.bytes"),
            sha256=sha256_value(raw["sha256"], label=f"{label}.sha256"),
        )

    def verify(self, path: Path) -> None:
        if not path.is_file():
            raise RetrievalContractError(f"resource {self.resource_id!r} is unavailable")
        if path.stat().st_size != self.bytes:
            raise RetrievalContractError(f"resource {self.resource_id!r} byte size differs")
        if sha256_file(path) != self.sha256:
            raise RetrievalContractError(f"resource {self.resource_id!r} hash differs")


@dataclass(frozen=True, slots=True)
class IndexDescriptor:
    index_id: str
    dataset_id: str
    encoder_id: str
    implementation: str
    metadata_file: BoundFileDescriptor
    matrix_file: BoundFileDescriptor
    expected_rows: int
    expected_dimension: int
    dtype: str
    order_semantics: str

    @classmethod
    def parse(cls, value: Any, *, label: str) -> "IndexDescriptor":
        raw = object_value(value, label=label)
        exact_keys(
            raw,
            label=label,
            required=(
                "index_id",
                "dataset_id",
                "encoder_id",
                "implementation",
                "metadata_file",
                "matrix_file",
                "expected_rows",
                "expected_dimension",
                "dtype",
                "order_semantics",
            ),
        )
        implementation = identifier(
            raw["implementation"], label=f"{label}.implementation"
        )
        if implementation != "dense_numpy_cosine_v2":
            raise RetrievalContractError("unsupported index implementation")
        order = identifier(raw["order_semantics"], label=f"{label}.order_semantics")
        if order != "declared_stable_order_v2":
            raise RetrievalContractError("unsupported index order semantics")
        dtype = string_value(raw["dtype"], label=f"{label}.dtype")
        try:
            np.dtype(dtype)
        except TypeError as exc:
            raise RetrievalContractError(f"unsupported NumPy dtype: {dtype}") from exc
        return cls(
            index_id=identifier(raw["index_id"], label=f"{label}.index_id"),
            dataset_id=identifier(raw["dataset_id"], label=f"{label}.dataset_id"),
            encoder_id=identifier(raw["encoder_id"], label=f"{label}.encoder_id"),
            implementation=implementation,
            metadata_file=BoundFileDescriptor.parse(
                raw["metadata_file"], label=f"{label}.metadata_file"
            ),
            matrix_file=BoundFileDescriptor.parse(
                raw["matrix_file"], label=f"{label}.matrix_file"
            ),
            expected_rows=positive_int(
                raw["expected_rows"], label=f"{label}.expected_rows"
            ),
            expected_dimension=positive_int(
                raw["expected_dimension"], label=f"{label}.expected_dimension"
            ),
            dtype=np.dtype(dtype).name,
            order_semantics=order,
        )

    def load_and_validate(
        self,
        *,
        dataset: DatasetDescriptor,
        metadata_path: Path,
        matrix_path: Path,
    ) -> tuple[dict[str, dict[str, Any]], list[str], np.ndarray]:
        self.metadata_file.verify(metadata_path)
        self.matrix_file.verify(matrix_path)
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise RetrievalContractError(f"invalid index metadata JSON: {exc}") from exc
        if not isinstance(metadata, dict):
            raise RetrievalContractError("index metadata root must be an object")
        assets, order = dataset.validate_and_normalize_assets(metadata)
        try:
            matrix = np.load(matrix_path, allow_pickle=False)
        except Exception as exc:
            raise RetrievalContractError(
                f"cannot load index matrix: {type(exc).__name__}: {exc}"
            ) from exc
        if matrix.shape != (self.expected_rows, self.expected_dimension):
            raise RetrievalContractError(
                f"index matrix shape differs: expected={(self.expected_rows, self.expected_dimension)}, "
                f"actual={matrix.shape}"
            )
        if matrix.dtype.name != self.dtype:
            raise RetrievalContractError(
                f"index matrix dtype differs: expected={self.dtype}, actual={matrix.dtype.name}"
            )
        if len(assets) != self.expected_rows or len(order) != self.expected_rows:
            raise RetrievalContractError("dataset asset count and index rows differ")
        if not np.isfinite(matrix).all():
            raise RetrievalContractError("index matrix contains non-finite values")
        return assets, order, matrix
