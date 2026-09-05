"""Write-once artifact paths for the frozen two-stage generator.

See ``docs/generation_transport_compatibility.md`` for the compatibility
contract.  The filenames here preserve the generation/evaluation file boundary;
this module does not import or invoke the evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


JsonWriter = Callable[[Path, Mapping[str, Any]], None]


@dataclass(frozen=True, slots=True)
class ArtifactLayout:
    """Canonical write-once paths owned by one generation run."""

    output_root: Path
    run_manifest_name: str = "run_manifest.json"
    execution_policy_name: str = "execution_policy.json"
    summary_name: str = "summary.json"

    def __post_init__(self) -> None:
        output_root = Path(self.output_root).expanduser()
        object.__setattr__(self, "output_root", output_root)
        names = (
            self.run_manifest_name,
            self.execution_policy_name,
            self.summary_name,
        )
        for name in names:
            self._validate_filename(name)
        if len(set(names)) != len(names):
            raise ValueError("artifact filenames must be unique")

    @staticmethod
    def _validate_filename(name: str) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("artifact filename must be a non-empty string")
        path = Path(name)
        if path.name != name or name in {".", ".."}:
            raise ValueError(f"artifact filename must be a basename: {name!r}")
        if path.suffix != ".json":
            raise ValueError(f"artifact filename must end in .json: {name!r}")

    @property
    def run_manifest_path(self) -> Path:
        return self.output_root / self.run_manifest_name

    @property
    def execution_policy_path(self) -> Path:
        return self.output_root / self.execution_policy_name

    @property
    def summary_path(self) -> Path:
        return self.output_root / self.summary_name

    def case_dir(self, brief_id: str) -> Path:
        """Return the canonical case directory without permitting traversal."""

        if (
            not isinstance(brief_id, str)
            or not brief_id
            or Path(brief_id).name != brief_id
            or brief_id in {".", ".."}
        ):
            raise ValueError(f"invalid brief_id for artifact path: {brief_id!r}")
        return self.output_root / brief_id

    def require_fresh_output(self) -> None:
        """Refuse any pre-existing root; generation never overwrites or resumes."""

        if self.output_root.exists():
            raise FileExistsError(
                f"refusing to overwrite existing output: {self.output_root}"
            )

    def verify_initialized(self) -> None:
        """Verify that the frozen core initialized only the expected run root."""

        if not self.output_root.is_dir():
            raise RuntimeError(
                f"frozen core did not create output root: {self.output_root}"
            )
        if not self.run_manifest_path.is_file():
            raise RuntimeError(
                f"frozen core did not write run manifest: {self.run_manifest_path}"
            )
        if self.execution_policy_path.exists():
            raise FileExistsError(
                "execution policy already exists after run initialization: "
                f"{self.execution_policy_path}"
            )
        if self.summary_path.exists():
            raise FileExistsError(
                f"summary already exists after run initialization: {self.summary_path}"
            )

    def write_execution_policy(
        self,
        writer: JsonWriter,
        execution_policy: Mapping[str, Any],
    ) -> None:
        """Write the caller's policy exactly once without adding fields."""

        writer(self.execution_policy_path, execution_policy)

    def write_summary(
        self,
        writer: JsonWriter,
        summary: Mapping[str, Any],
    ) -> None:
        """Write the terminal run summary exactly once."""

        writer(self.summary_path, summary)

    def to_public_dict(self) -> dict[str, str]:
        """Return the stable run-level path contract for public provenance."""

        return {
            "output_root": str(self.output_root),
            "run_manifest": self.run_manifest_name,
            "execution_policy": self.execution_policy_name,
            "summary": self.summary_name,
        }
