"""Loader-or-converter routing for scene-generator outputs.

This module owns the boundary between a harness-native artifact and the
canonical scene consumed by evaluation.  A route has exactly one behavior:
load an already supported representation, or convert a foreign one.  The
evaluator stays downstream of both choices and never dispatches on a harness.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal


OUTPUT_LOADER = "loader"
OUTPUT_CONVERTER = "converter"
OUTPUT_INGESTION_KINDS = {OUTPUT_LOADER, OUTPUT_CONVERTER}

OutputIngestionKind = Literal["loader", "converter"]
SceneOutputHandler = Callable[[Path, dict, Path, dict | None], str | Path]


@dataclass(frozen=True)
class SceneOutputRoute:
    """One selected path from a native artifact to a canonical scene.

    ``handler`` is intentionally a plain callable.  Concrete harness support
    can therefore live in an adapter class or in a small converter function
    without changing this dispatcher.
    """

    kind: OutputIngestionKind
    handler: SceneOutputHandler

    def __post_init__(self) -> None:
        if self.kind not in OUTPUT_INGESTION_KINDS:
            raise ValueError(
                f"output ingestion kind must be one of {sorted(OUTPUT_INGESTION_KINDS)}, "
                f"got {self.kind!r}"
            )
        if not callable(self.handler):
            raise TypeError("scene output route handler must be callable")

    @classmethod
    def existing_loader(cls, handler: SceneOutputHandler) -> "SceneOutputRoute":
        """Route an already supported artifact through an existing loader."""

        return cls(kind=OUTPUT_LOADER, handler=handler)

    @classmethod
    def converter(cls, handler: SceneOutputHandler) -> "SceneOutputRoute":
        """Route a foreign harness artifact through its converter."""

        return cls(kind=OUTPUT_CONVERTER, handler=handler)

    def materialize(
        self,
        source_path: Path,
        generation_input: dict,
        out_dir: Path,
        config: dict | None = None,
    ) -> Path:
        """Return the canonical scene path produced by the selected handler."""

        return Path(
            self.handler(
                Path(source_path),
                generation_input,
                Path(out_dir),
                config,
            )
        )


__all__ = [
    "OUTPUT_CONVERTER",
    "OUTPUT_INGESTION_KINDS",
    "OUTPUT_LOADER",
    "OutputIngestionKind",
    "SceneOutputHandler",
    "SceneOutputRoute",
]
