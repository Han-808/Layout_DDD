"""Hash-verified trust inventory for configurable frozen generation inputs.

The trust boundary is defined in
``docs/generation_transport_compatibility.md``.  Static checks consume this
inventory before any configurable Python module is imported or executed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


TRUST_SCHEMA_VERSION = "active_generation_bundles_v1"
_DEFAULT_MANIFEST_RELATIVE = Path(
    "configs/runners/active_generation_bundles_v1.json"
)
_IGNORED_NAMES = frozenset({".DS_Store"})
_IGNORED_SUFFIXES = frozenset({".pyc"})
_IGNORED_PARTS = frozenset({"__pycache__"})
_MAX_MANIFEST_BYTES = 5_000_000


class TrustError(RuntimeError):
    """Raised when a configured generation input is not hash-trusted."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise TrustError(f"duplicate trust-manifest key: {key}")
        value[key] = child
    return value


def _reject_constant(value: str) -> None:
    raise TrustError(f"non-finite trust-manifest number is forbidden: {value}")


def _load_manifest(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if len(data) > _MAX_MANIFEST_BYTES:
        raise TrustError(f"trust manifest exceeds {_MAX_MANIFEST_BYTES} bytes")
    try:
        value = json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise TrustError(f"invalid trust manifest: {exc}") from exc
    if not isinstance(value, dict):
        raise TrustError("trust manifest must be a JSON object")
    if value.get("schema_version") != TRUST_SCHEMA_VERSION:
        raise TrustError(
            f"trust schema must be {TRUST_SCHEMA_VERSION!r}"
        )
    return value


def _default_manifest() -> Path:
    repo_candidate = Path(__file__).resolve().parents[4] / _DEFAULT_MANIFEST_RELATIVE
    if repo_candidate.is_file():
        return repo_candidate
    raise TrustError(
        "no default generation trust manifest is available outside a source "
        "checkout; pass trust_manifest explicitly"
    )


def _manifest_repo_root(path: Path) -> Path:
    if path.parent.name != "runners" or path.parent.parent.name != "configs":
        raise TrustError(
            "trust manifest must be located at <repo>/configs/runners/<name>.json"
        )
    return path.parents[2].resolve()


def _safe_relative(value: Any, *, field_name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise TrustError(f"{field_name} must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise TrustError(f"{field_name} is not a safe relative path: {value!r}")
    return path


def _discover_files(root: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.name not in _IGNORED_NAMES
            and path.suffix not in _IGNORED_SUFFIXES
            and not (set(path.parts) & _IGNORED_PARTS)
        )
    )


@dataclass(frozen=True, slots=True)
class TrustedFile:
    relative_path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class TrustedBundle:
    bundle_id: str
    role: str
    root_text: str
    root: Path
    files: tuple[TrustedFile, ...]
    declaration_sha256: str

    def file(self, path: Path) -> TrustedFile | None:
        try:
            relative = path.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return None
        return next(
            (item for item in self.files if item.relative_path == relative),
            None,
        )

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "role": self.role,
            "root": self.root_text,
            "declaration_sha256": self.declaration_sha256,
            "file_count": len(self.files),
        }


@dataclass(frozen=True, slots=True)
class TrustReport:
    manifest_path: Path
    manifest_sha256: str
    core_bundle: TrustedBundle
    models_bundle: TrustedBundle
    models_file: TrustedFile
    briefs_bundle: TrustedBundle
    briefs_file: TrustedFile
    retrieval_runtime_bundle: TrustedBundle
    retrieval_runtime_source_sha256: str
    retrieval_catalog_bundle: TrustedBundle
    retrieval_catalog_file: TrustedFile
    run_config_bundle: TrustedBundle
    run_config_file: TrustedFile

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "frozen_two_stage_trust_report_v3",
            "trust_manifest": {
                "path": str(self.manifest_path),
                "sha256": self.manifest_sha256,
            },
            "core_bundle": self.core_bundle.to_public_dict(),
            "models": {
                **self.models_bundle.to_public_dict(),
                "file": self.models_file.relative_path,
                "file_sha256": self.models_file.sha256,
            },
            "briefs": {
                **self.briefs_bundle.to_public_dict(),
                "file": self.briefs_file.relative_path,
                "file_sha256": self.briefs_file.sha256,
            },
            "retrieval_runtime": {
                **self.retrieval_runtime_bundle.to_public_dict(),
                "source_manifest_sha256": self.retrieval_runtime_source_sha256,
            },
            "retrieval_catalog": {
                **self.retrieval_catalog_bundle.to_public_dict(),
                "file": self.retrieval_catalog_file.relative_path,
                "file_sha256": self.retrieval_catalog_file.sha256,
            },
            "run_config": {
                **self.run_config_bundle.to_public_dict(),
                "file": self.run_config_file.relative_path,
                "file_sha256": self.run_config_file.sha256,
            },
        }


@dataclass(frozen=True, slots=True)
class TrustInventory:
    """Immutable bundle inventory loaded from an explicit hash manifest."""

    manifest_path: Path
    repo_root: Path
    manifest_sha256: str
    bundles: tuple[TrustedBundle, ...]

    @classmethod
    def load(cls, manifest_path: str | Path | None = None) -> "TrustInventory":
        path = (
            _default_manifest()
            if manifest_path is None
            else Path(manifest_path).expanduser().resolve()
        )
        if not path.is_file() or path.is_symlink():
            raise TrustError(f"trust manifest is not a regular file: {path}")
        repo_root = _manifest_repo_root(path)
        value = _load_manifest(path)
        bundles_value = value.get("bundles")
        if not isinstance(bundles_value, list) or not bundles_value:
            raise TrustError("trust manifest bundles must be a non-empty array")
        bundles: list[TrustedBundle] = []
        seen_ids: set[str] = set()
        seen_roots: set[Path] = set()
        for index, raw_bundle in enumerate(bundles_value):
            if not isinstance(raw_bundle, dict):
                raise TrustError(f"bundles[{index}] must be an object")
            bundle_id = raw_bundle.get("bundle_id")
            role = raw_bundle.get("role")
            if not isinstance(bundle_id, str) or not bundle_id:
                raise TrustError(f"bundles[{index}].bundle_id is invalid")
            if bundle_id in seen_ids:
                raise TrustError(f"duplicate trusted bundle ID: {bundle_id}")
            seen_ids.add(bundle_id)
            if not isinstance(role, str) or not role:
                raise TrustError(f"bundle role is invalid: {bundle_id}")
            root_relative = _safe_relative(
                raw_bundle.get("root"), field_name=f"{bundle_id}.root"
            )
            root = (repo_root / root_relative).resolve()
            try:
                root.relative_to(repo_root)
            except ValueError as exc:
                raise TrustError(f"bundle root escapes repository: {bundle_id}") from exc
            if root in seen_roots:
                raise TrustError(f"duplicate trusted bundle root: {root_relative}")
            seen_roots.add(root)
            files_value = raw_bundle.get("files")
            if not isinstance(files_value, list) or not files_value:
                raise TrustError(f"trusted bundle has no files: {bundle_id}")
            files: list[TrustedFile] = []
            seen_files: set[str] = set()
            for file_index, raw_file in enumerate(files_value):
                if not isinstance(raw_file, dict):
                    raise TrustError(
                        f"{bundle_id}.files[{file_index}] must be an object"
                    )
                relative = _safe_relative(
                    raw_file.get("path"),
                    field_name=f"{bundle_id}.files[{file_index}].path",
                ).as_posix()
                digest = raw_file.get("sha256")
                if relative in seen_files:
                    raise TrustError(f"duplicate trusted file: {bundle_id}/{relative}")
                seen_files.add(relative)
                if not isinstance(digest, str) or len(digest) != 64:
                    raise TrustError(f"invalid trusted hash: {bundle_id}/{relative}")
                try:
                    int(digest, 16)
                except ValueError as exc:
                    raise TrustError(
                        f"non-hex trusted hash: {bundle_id}/{relative}"
                    ) from exc
                files.append(TrustedFile(relative_path=relative, sha256=digest))
            bundles.append(
                TrustedBundle(
                    bundle_id=bundle_id,
                    role=role,
                    root_text=root_relative.as_posix(),
                    root=root,
                    files=tuple(files),
                    declaration_sha256=hashlib.sha256(
                        _canonical_json_bytes(raw_bundle)
                    ).hexdigest(),
                )
            )
        return cls(
            manifest_path=path,
            repo_root=repo_root,
            manifest_sha256=_sha256(path),
            bundles=tuple(bundles),
        )

    def _bundle_for_root(self, root: Path, *, purpose: str) -> TrustedBundle:
        resolved = root.resolve()
        matches = [bundle for bundle in self.bundles if bundle.root == resolved]
        if len(matches) != 1:
            raise TrustError(
                f"{purpose} must equal exactly one trusted bundle root: {resolved}"
            )
        return matches[0]

    def _bundle_for_file(
        self, path: Path, *, purpose: str
    ) -> tuple[TrustedBundle, TrustedFile]:
        resolved = path.resolve()
        matches: list[tuple[TrustedBundle, TrustedFile]] = []
        for bundle in self.bundles:
            item = bundle.file(resolved)
            if item is not None:
                matches.append((bundle, item))
        if len(matches) != 1:
            raise TrustError(
                f"{purpose} must be one file in exactly one trusted bundle: {resolved}"
            )
        return matches[0]

    @staticmethod
    def _verify_bundle(bundle: TrustedBundle) -> None:
        root = bundle.root
        if not root.is_dir() or root.is_symlink():
            raise TrustError(f"trusted bundle root is unavailable: {bundle.root_text}")
        actual = {
            path.relative_to(root).as_posix(): path for path in _discover_files(root)
        }
        declared = {item.relative_path: item for item in bundle.files}
        if actual.keys() != declared.keys():
            raise TrustError(
                f"trusted bundle inventory mismatch for {bundle.bundle_id}: "
                f"missing={sorted(declared.keys() - actual.keys())}, "
                f"extra={sorted(actual.keys() - declared.keys())}"
            )
        for relative, item in declared.items():
            path = actual[relative]
            if path.is_symlink():
                raise TrustError(
                    f"trusted bundle file may not be a symlink: "
                    f"{bundle.bundle_id}/{relative}"
                )
            actual_hash = _sha256(path)
            if actual_hash != item.sha256:
                raise TrustError(
                    f"trusted hash mismatch for {bundle.bundle_id}/{relative}: "
                    f"expected={item.sha256}, actual={actual_hash}"
                )

    @staticmethod
    def _retrieval_source_manifest_sha256(bundle: TrustedBundle) -> str:
        """Compute the shared runtime identity from trusted declarations.

        The runtime publishes the same path/byte/hash payload after import.
        Hashes come from the already-verified declaration, rather than a
        second untrusted read, so a post-gate source mutation cannot produce a
        matching static identity.
        """

        files = []
        for item in sorted(bundle.files, key=lambda child: child.relative_path):
            if Path(item.relative_path).suffix != ".py":
                continue
            path = bundle.root / item.relative_path
            files.append(
                {
                    "path": item.relative_path,
                    "bytes": path.stat().st_size,
                    "sha256": item.sha256,
                }
            )
        payload = {
            "schema_version": "generation_retrieval_source_manifest_v2",
            "logical_root": "benchmark.scene_generation.retrieval",
            "files": files,
        }
        return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()

    def verify_run_inputs(
        self,
        *,
        core_root: str | Path,
        models_path: str | Path,
        briefs_path: str | Path,
        retrieval_runtime_root: str | Path,
        retrieval_catalog_path: str | Path,
        run_config_path: str | Path,
    ) -> TrustReport:
        """Hash-verify every executable/data bundle used by a configured run."""

        core_bundle = self._bundle_for_root(Path(core_root), purpose="core_root")
        if core_bundle.role != "frozen_generation_core":
            raise TrustError("core_root is not declared as a frozen_generation_core")
        models_bundle, models_file = self._bundle_for_file(
            Path(models_path), purpose="models_path"
        )
        briefs_bundle, briefs_file = self._bundle_for_file(
            Path(briefs_path), purpose="briefs_path"
        )
        retrieval_runtime_bundle = self._bundle_for_root(
            Path(retrieval_runtime_root), purpose="retrieval_runtime_root"
        )
        if retrieval_runtime_bundle.role != "shared_generation_retrieval_runtime":
            raise TrustError(
                "retrieval_runtime_root is not declared as a shared retrieval runtime"
            )
        retrieval_catalog_bundle, retrieval_catalog_file = self._bundle_for_file(
            Path(retrieval_catalog_path), purpose="retrieval_catalog_path"
        )
        if retrieval_catalog_bundle.role != "generation_retrieval_profiles":
            raise TrustError(
                "retrieval_catalog_path is not declared in a retrieval-profile bundle"
            )
        run_config_bundle, run_config_file = self._bundle_for_file(
            Path(run_config_path), purpose="run_config_path"
        )
        if run_config_bundle.role != "config_only_generation_routes":
            raise TrustError(
                "run_config_path is not declared in a config-only generation bundle"
            )
        unique_bundles = {
            bundle.bundle_id: bundle
            for bundle in (
                core_bundle,
                models_bundle,
                briefs_bundle,
                retrieval_runtime_bundle,
                retrieval_catalog_bundle,
                run_config_bundle,
            )
        }
        for bundle in unique_bundles.values():
            self._verify_bundle(bundle)
        retrieval_runtime_source_sha256 = (
            self._retrieval_source_manifest_sha256(retrieval_runtime_bundle)
        )
        return TrustReport(
            manifest_path=self.manifest_path,
            manifest_sha256=self.manifest_sha256,
            core_bundle=core_bundle,
            models_bundle=models_bundle,
            models_file=models_file,
            briefs_bundle=briefs_bundle,
            briefs_file=briefs_file,
            retrieval_runtime_bundle=retrieval_runtime_bundle,
            retrieval_runtime_source_sha256=retrieval_runtime_source_sha256,
            retrieval_catalog_bundle=retrieval_catalog_bundle,
            retrieval_catalog_file=retrieval_catalog_file,
            run_config_bundle=run_config_bundle,
            run_config_file=run_config_file,
        )

    def verify_campaign_inputs(
        self,
        *,
        core_root: str | Path,
        campaign_runtime_root: str | Path,
        campaign_profile_path: str | Path,
        retrieval_runtime_root: str | Path,
        retrieval_catalog_path: str | Path,
    ) -> dict[str, Any]:
        """Hash-verify the complete executable/config surface of Campaign v2.

        The returned projection is path-free and safe to embed in generation
        provenance.  Every bundle is checked as an exact inventory, not merely
        by hashing one entrypoint.
        """

        core_bundle = self._bundle_for_root(Path(core_root), purpose="core_root")
        if core_bundle.role != "frozen_generation_core":
            raise TrustError("core_root is not declared as a frozen generation core")
        campaign_runtime_bundle = self._bundle_for_root(
            Path(campaign_runtime_root), purpose="campaign_runtime_root"
        )
        if campaign_runtime_bundle.role != "generation_campaign_runtime":
            raise TrustError(
                "campaign_runtime_root is not declared as a generation campaign runtime"
            )
        campaign_profile_bundle, campaign_profile_file = self._bundle_for_file(
            Path(campaign_profile_path), purpose="campaign_profile_path"
        )
        if campaign_profile_bundle.role != "config_only_generation_routes":
            raise TrustError(
                "campaign profile is not declared in the generation config bundle"
            )
        retrieval_runtime_bundle = self._bundle_for_root(
            Path(retrieval_runtime_root), purpose="retrieval_runtime_root"
        )
        if retrieval_runtime_bundle.role != "shared_generation_retrieval_runtime":
            raise TrustError(
                "retrieval_runtime_root is not declared as a shared retrieval runtime"
            )
        retrieval_catalog_bundle, retrieval_catalog_file = self._bundle_for_file(
            Path(retrieval_catalog_path), purpose="retrieval_catalog_path"
        )
        if retrieval_catalog_bundle.role != "generation_retrieval_profiles":
            raise TrustError(
                "retrieval catalog is not declared in a retrieval-profile bundle"
            )
        unique_bundles = {
            bundle.bundle_id: bundle
            for bundle in (
                core_bundle,
                campaign_runtime_bundle,
                campaign_profile_bundle,
                retrieval_runtime_bundle,
                retrieval_catalog_bundle,
            )
        }
        for bundle in unique_bundles.values():
            self._verify_bundle(bundle)
        return {
            "schema_version": "generation_campaign_trust_report_v1",
            "trust_manifest_sha256": self.manifest_sha256,
            "core_bundle": core_bundle.to_public_dict(),
            "campaign_runtime_bundle": campaign_runtime_bundle.to_public_dict(),
            "campaign_profile_bundle": campaign_profile_bundle.to_public_dict(),
            "campaign_profile_file": {
                "path": campaign_profile_file.relative_path,
                "sha256": campaign_profile_file.sha256,
            },
            "retrieval_runtime_bundle": retrieval_runtime_bundle.to_public_dict(),
            "retrieval_catalog_bundle": retrieval_catalog_bundle.to_public_dict(),
            "retrieval_catalog_file": {
                "path": retrieval_catalog_file.relative_path,
                "sha256": retrieval_catalog_file.sha256,
            },
        }
