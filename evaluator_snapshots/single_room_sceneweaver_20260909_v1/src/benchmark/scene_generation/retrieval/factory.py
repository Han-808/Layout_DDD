"""Factory for shared retrieval runtimes and local binding selection v2."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from ._common import RetrievalContractError
from .bindings import LocalResourceBindings, select_binding_path
from .profiles import RetrievalCatalog
from .runtime import SharedRetrieverRuntime


DEFAULT_PROFILE_ID = "imaginarium-qwen3-embedding-0.6b-stable-top1-v2"


def build_runtime(
    *,
    catalog_path: str | Path,
    retrieval_profile_id: str,
    resource_bindings_path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    encoder: Callable[[str], np.ndarray] | None = None,
) -> SharedRetrieverRuntime:
    catalog = RetrievalCatalog.load(catalog_path)
    binding_path = select_binding_path(
        catalog_path=catalog.path,
        explicit_path=resource_bindings_path,
        environ=os.environ if environ is None else environ,
    )
    bindings = LocalResourceBindings.load(binding_path)
    composed = catalog.compose(retrieval_profile_id)
    required_resource_ids = {
        composed.index.metadata_file.resource_id,
        composed.index.matrix_file.resource_id,
        composed.encoder.model_resource_id,
    }
    actual_resource_ids = set(bindings.paths)
    missing_resource_ids = required_resource_ids - actual_resource_ids
    if missing_resource_ids:
        raise RetrievalContractError(
            "resource bindings do not cover the selected retrieval profile: "
            f"missing={sorted(missing_resource_ids)}"
        )
    return SharedRetrieverRuntime(
        composed,
        bindings,
        encoder=encoder,
    )
