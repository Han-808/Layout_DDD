"""Retrieval-profile composition and cross-validation v2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._common import (
    RetrievalContractError,
    exact_keys,
    finite_float,
    identifier,
    object_value,
    positive_int,
    safe_relative_path,
    sha256_value,
)
from .dataset import DatasetDescriptor
from .encoder import EncoderDescriptor
from .index import IndexDescriptor


@dataclass(frozen=True, slots=True)
class GoldenSuiteDescriptor:
    path: str
    sha256: str
    score_tolerance: float

    @classmethod
    def parse(cls, value: Any, *, label: str) -> "GoldenSuiteDescriptor":
        raw = object_value(value, label=label)
        exact_keys(raw, label=label, required=("path", "sha256", "score_tolerance"))
        tolerance = finite_float(
            raw["score_tolerance"], label=f"{label}.score_tolerance"
        )
        if tolerance < 0:
            raise RetrievalContractError(f"{label}.score_tolerance must be non-negative")
        return cls(
            path=safe_relative_path(raw["path"], label=f"{label}.path"),
            sha256=sha256_value(raw["sha256"], label=f"{label}.sha256"),
            score_tolerance=tolerance,
        )


@dataclass(frozen=True, slots=True)
class RetrievalPolicy:
    algorithm: str
    category_argument: str | None
    size_tolerance: float
    top_k: int
    min_score: float
    tie_order: str

    @classmethod
    def parse(cls, value: Any, *, label: str) -> "RetrievalPolicy":
        raw = object_value(value, label=label)
        exact_keys(
            raw,
            label=label,
            required=(
                "algorithm",
                "category_argument",
                "size_tolerance",
                "top_k",
                "min_score",
                "tie_order",
            ),
        )
        algorithm = identifier(raw["algorithm"], label=f"{label}.algorithm")
        if algorithm != "cosine_log_size_stable_top1_v2":
            raise RetrievalContractError("unsupported retrieval algorithm")
        category = raw["category_argument"]
        if category is not None:
            raise RetrievalContractError(
                "the current stable Top-1 contract requires category_argument=null"
            )
        tolerance = finite_float(raw["size_tolerance"], label=f"{label}.size_tolerance")
        if tolerance != 0.5:
            raise RetrievalContractError(
                "the current stable Top-1 contract requires size_tolerance=0.5"
            )
        top_k = positive_int(raw["top_k"], label=f"{label}.top_k")
        if top_k != 1:
            raise RetrievalContractError("stable Top-1 runtime requires top_k=1")
        tie_order = identifier(raw["tie_order"], label=f"{label}.tie_order")
        if tie_order != "declared_index_order_v2":
            raise RetrievalContractError("unsupported tie order")
        min_score = finite_float(raw["min_score"], label=f"{label}.min_score")
        if min_score != 0.3:
            raise RetrievalContractError(
                "the current stable Top-1 contract requires min_score=0.3"
            )
        return cls(
            algorithm=algorithm,
            category_argument=category,
            size_tolerance=tolerance,
            top_k=top_k,
            min_score=min_score,
            tie_order=tie_order,
        )


@dataclass(frozen=True, slots=True)
class RetrievalProfile:
    retrieval_profile_id: str
    dataset_id: str
    encoder_id: str
    index_id: str
    official_strict: bool
    policy: RetrievalPolicy
    golden_suite: GoldenSuiteDescriptor

    @classmethod
    def parse(cls, value: Any, *, label: str) -> "RetrievalProfile":
        raw = object_value(value, label=label)
        exact_keys(
            raw,
            label=label,
            required=(
                "retrieval_profile_id",
                "dataset_id",
                "encoder_id",
                "index_id",
                "official_strict",
                "policy",
                "golden_suite",
            ),
        )
        strict = raw["official_strict"]
        if not isinstance(strict, bool):
            raise RetrievalContractError(f"{label}.official_strict must be boolean")
        return cls(
            retrieval_profile_id=identifier(
                raw["retrieval_profile_id"], label=f"{label}.retrieval_profile_id"
            ),
            dataset_id=identifier(raw["dataset_id"], label=f"{label}.dataset_id"),
            encoder_id=identifier(raw["encoder_id"], label=f"{label}.encoder_id"),
            index_id=identifier(raw["index_id"], label=f"{label}.index_id"),
            official_strict=strict,
            policy=RetrievalPolicy.parse(raw["policy"], label=f"{label}.policy"),
            golden_suite=GoldenSuiteDescriptor.parse(
                raw["golden_suite"], label=f"{label}.golden_suite"
            ),
        )


@dataclass(frozen=True, slots=True)
class ComposedRetrievalProfile:
    profile: RetrievalProfile
    dataset: DatasetDescriptor
    encoder: EncoderDescriptor
    index: IndexDescriptor
    catalog_path: Path
    catalog_sha256: str
    profile_sha256: str

    def validate_composition(self) -> None:
        if self.profile.dataset_id != self.dataset.dataset_id:
            raise RetrievalContractError("retrieval profile dataset reference differs")
        if self.profile.encoder_id != self.encoder.encoder_id:
            raise RetrievalContractError("retrieval profile encoder reference differs")
        if self.profile.index_id != self.index.index_id:
            raise RetrievalContractError("retrieval profile index reference differs")
        if self.index.dataset_id != self.dataset.dataset_id:
            raise RetrievalContractError("index dataset reference differs")
        if self.index.encoder_id != self.encoder.encoder_id:
            raise RetrievalContractError("index encoder reference differs")
        if self.index.expected_dimension != self.encoder.expected_dimension:
            raise RetrievalContractError("index and encoder dimensions differ")
