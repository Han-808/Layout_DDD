"""Local logical resource bindings for portable retrieval profiles v2."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping

from ._common import (
    RetrievalContractError,
    exact_keys,
    identifier,
    object_value,
    strict_json_object,
    string_value,
)


BINDINGS_SCHEMA_VERSION = "generation_resource_bindings_v2"
BINDINGS_ENV = "LAYOUT_DDD_RETRIEVAL_BINDINGS"
DEFAULT_LOCAL_RELATIVE = Path(".runtime/retrieval_bindings.local.json")


@dataclass(frozen=True, slots=True)
class LocalResourceBindings:
    source: str
    paths: Mapping[str, Path]

    @classmethod
    def load(cls, path: str | Path) -> "LocalResourceBindings":
        selected = Path(path).expanduser().absolute()
        if selected.is_symlink():
            raise RetrievalContractError("resource binding must not be a symlink")
        binding_path = selected.resolve()
        if not binding_path.is_file():
            raise RetrievalContractError("resource binding must be a regular JSON file")
        raw = strict_json_object(binding_path, maximum_bytes=200_000)
        exact_keys(raw, label="resource bindings", required=("schema_version", "bindings"))
        if raw["schema_version"] != BINDINGS_SCHEMA_VERSION:
            raise RetrievalContractError(
                f"resource binding schema must be {BINDINGS_SCHEMA_VERSION!r}"
            )
        values = object_value(raw["bindings"], label="bindings")
        paths: dict[str, Path] = {}
        for resource_id_value, binding_value in values.items():
            resource_id = identifier(resource_id_value, label="binding resource ID")
            binding = object_value(binding_value, label=f"bindings.{resource_id}")
            exact_keys(binding, label=f"bindings.{resource_id}", required=("path",))
            text = string_value(binding["path"], label=f"bindings.{resource_id}.path")
            candidate = Path(text).expanduser()
            if not candidate.is_absolute():
                candidate = binding_path.parent / candidate
            paths[resource_id] = candidate.resolve()
        if not paths:
            raise RetrievalContractError("resource bindings must not be empty")
        return cls(source="local_binding_v2", paths=paths)

    def require(self, resource_id: str) -> Path:
        try:
            return self.paths[resource_id]
        except KeyError as exc:
            raise RetrievalContractError(
                f"logical resource has no local binding: {resource_id}"
            ) from exc

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": BINDINGS_SCHEMA_VERSION,
            "source": self.source,
            "bound_resource_ids": sorted(self.paths),
        }


def select_binding_path(
    *,
    catalog_path: Path,
    explicit_path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve explicit > env-selected > ignored repo-local binding.

    No HOME, Hugging Face cache, mount point, or other ambient location is
    scanned. The chosen file is local state and is never serialized publicly.
    """

    if explicit_path is not None:
        return Path(explicit_path).expanduser().absolute()
    environment = os.environ if environ is None else environ
    selected = environment.get(BINDINGS_ENV)
    if selected:
        return Path(selected).expanduser().absolute()
    catalog_root = catalog_path.parent
    repo_root = (
        catalog_root.parent.parent
        if catalog_root.parent.name == "configs"
        else catalog_root
    )
    return (repo_root / DEFAULT_LOCAL_RELATIVE).absolute()
