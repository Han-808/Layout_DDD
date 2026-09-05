"""Cross-process ownership lock for one credential-bearing API-family run.

The in-memory cooldown gate coordinates models inside one experiment process.
This advisory lock closes the separate-terminal gap: at most one official
invocation for a given API family may own a credential and issue requests at a
time.  Different API families use different lock files and may coexist.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from datetime import datetime, timezone
import errno
import fcntl
import json
import os
from pathlib import Path
import re
import stat
import threading
from typing import Any


PORTABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_PROCESS_GUARD = threading.Lock()
_PROCESS_OWNED_FAMILIES: set[str] = set()


class ApiFamilyLockError(RuntimeError):
    """Raised when exclusive API-family ownership cannot be proven."""


class ApiFamilyInvocationLock(AbstractContextManager["ApiFamilyInvocationLock"]):
    """Hold an OS advisory lock for a full credential-bearing invocation."""

    def __init__(
        self,
        *,
        lock_root: str | Path,
        api_family_id: str,
        experiment_id: str,
        invocation_nonce: str,
    ) -> None:
        for value, label in (
            (api_family_id, "api_family_id"),
            (experiment_id, "experiment_id"),
            (invocation_nonce, "invocation_nonce"),
        ):
            if not isinstance(value, str) or PORTABLE_ID.fullmatch(value) is None:
                raise ApiFamilyLockError(f"invalid {label}")
        self.lock_root = Path(lock_root).expanduser().absolute()
        self.api_family_id = api_family_id
        self.experiment_id = experiment_id
        self.invocation_nonce = invocation_nonce
        self.path = self.lock_root / f"{api_family_id}.lock"
        self._descriptor: int | None = None
        self._entered = False

    def __enter__(self) -> "ApiFamilyInvocationLock":
        if self._entered:
            raise ApiFamilyLockError("API-family lock object cannot be reused")
        self._entered = True
        self._prepare_root()
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_CLOEXEC", 0)
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except OSError as exc:
            raise ApiFamilyLockError("API-family lock file is unavailable") from exc
        self._descriptor = descriptor
        process_claimed = False
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                raise ApiFamilyLockError("API-family lock file is not trusted")
            with _PROCESS_GUARD:
                if self.api_family_id in _PROCESS_OWNED_FAMILIES:
                    raise ApiFamilyLockError(
                        "another invocation already owns this API family"
                    )
                _PROCESS_OWNED_FAMILIES.add(self.api_family_id)
                process_claimed = True
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in {errno.EACCES, errno.EAGAIN}:
                    raise ApiFamilyLockError(
                        "another invocation already owns this API family"
                    ) from exc
                raise ApiFamilyLockError("API-family lock acquisition failed") from exc
            os.fchmod(descriptor, 0o600)
            self._write_owner_record(descriptor)
            return self
        except BaseException:
            if process_claimed:
                with _PROCESS_GUARD:
                    _PROCESS_OWNED_FAMILIES.discard(self.api_family_id)
            try:
                os.close(descriptor)
            finally:
                self._descriptor = None
            raise

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        descriptor = self._descriptor
        self._descriptor = None
        if descriptor is not None:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
        with _PROCESS_GUARD:
            _PROCESS_OWNED_FAMILIES.discard(self.api_family_id)

    def _prepare_root(self) -> None:
        parent = self.lock_root.parent
        if not parent.is_dir() or parent.is_symlink():
            raise ApiFamilyLockError("API-family lock parent is invalid")
        try:
            self.lock_root.mkdir(mode=0o700, exist_ok=True)
        except OSError as exc:
            raise ApiFamilyLockError("API-family lock directory is unavailable") from exc
        if not self.lock_root.is_dir() or self.lock_root.is_symlink():
            raise ApiFamilyLockError("API-family lock directory is invalid")
        info = self.lock_root.stat()
        if info.st_uid != os.getuid():
            raise ApiFamilyLockError("API-family lock directory has another owner")
        self.lock_root.chmod(0o700)

    def _write_owner_record(self, descriptor: int) -> None:
        record = {
            "schema_version": "sieve_api_family_invocation_lock_v1",
            "api_family_id": self.api_family_id,
            "experiment_id": self.experiment_id,
            "invocation_nonce": self.invocation_nonce,
            "owner_pid": os.getpid(),
            "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
            "credential_endpoint_or_request_recorded": False,
        }
        encoded = (
            json.dumps(
                record,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.ftruncate(descriptor, 0)
        written = 0
        while written < len(encoded):
            count = os.write(descriptor, encoded[written:])
            if count <= 0:
                raise ApiFamilyLockError("API-family lock record write failed")
            written += count
        os.fsync(descriptor)


__all__ = ["ApiFamilyInvocationLock", "ApiFamilyLockError"]
