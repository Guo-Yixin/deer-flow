"""Regression coverage for the Windows lock follow-up from PR #6888.

Several independent Gateway and harness paths used the blocking ``LK_LOCK``
mode even after the shared retry helper existed.  The blocking mode reports
contention as ``PermissionError`` after about ten seconds, so these tests run
each remaining path against a deterministic msvcrt double and verify that
only contention is retried.
"""

from __future__ import annotations

import errno
from pathlib import Path
from types import SimpleNamespace

import pytest


class _FakeMsvcrt:
    LK_UNLCK = 0
    LK_LOCK = 1
    LK_NBLCK = 2

    def __init__(self, *, failures: int = 0, error_number: int = errno.EACCES) -> None:
        self.failures = failures
        self.error_number = error_number
        self.calls: list[tuple[int, int, int]] = []

    def locking(self, fd: int, mode: int, size: int) -> None:
        self.calls.append((fd, mode, size))
        if mode == self.LK_UNLCK:
            return
        attempts = sum(call[1] != self.LK_UNLCK for call in self.calls)
        if attempts <= self.failures:
            raise OSError(self.error_number, "Lock unavailable")


def _patch_windows_lock(module, monkeypatch: pytest.MonkeyPatch, fake: _FakeMsvcrt) -> None:
    monkeypatch.setattr(module, "fcntl", None)
    monkeypatch.setattr(module, "msvcrt", fake, raising=False)
    real_retry = module.retry_until_locked

    def retry(acquire, **kwargs):
        return real_retry(acquire, sleep=lambda _: None, **kwargs)

    monkeypatch.setattr(module, "retry_until_locked", retry)


def _exercise_lock(module_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake: _FakeMsvcrt) -> Path:
    """Acquire one of the four remaining locks and return its sidecar path."""
    import importlib

    module = importlib.import_module(module_name)
    _patch_windows_lock(module, monkeypatch, fake)

    if module_name.endswith("extensions_config"):
        target = tmp_path / "extensions_config.json"
        with module.extensions_config_file_lock(target):
            pass
        return target.parent / f".{target.name}.lock"

    if module_name.endswith("skills.projection"):
        root = tmp_path / "projection"
        with module._projection_lock(root):
            pass
        return root.parent / f".{root.name}.projection.lock"

    if module_name.endswith("routers.skills"):
        custom_root = tmp_path / "custom"
        storage = SimpleNamespace(get_user_custom_root=lambda: custom_root)
        with module._custom_skill_mutation(storage):
            pass
        return custom_root.parent / f".{custom_root.name}.mutation.lock"

    if module_name.endswith("aio_sandbox_provider"):
        lock_path = tmp_path / "sandbox.lock"
        with lock_path.open("a", encoding="utf-8") as lock_file:
            module._lock_file_exclusive(lock_file)
            module._unlock_file(lock_file)
        return lock_path

    raise AssertionError(f"unhandled lock module: {module_name}")


LOCK_MODULES = (
    "deerflow.config.extensions_config",
    "deerflow.skills.projection",
    "app.gateway.routers.skills",
    "deerflow.community.aio_sandbox.aio_sandbox_provider",
)


@pytest.mark.parametrize("module_name", LOCK_MODULES)
def test_remaining_windows_locks_retry_contention(
    module_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeMsvcrt(failures=3)

    sidecar = _exercise_lock(module_name, tmp_path, monkeypatch, fake)

    assert [mode for _, mode, _ in fake.calls] == [fake.LK_NBLCK] * 4 + [fake.LK_UNLCK]
    assert [size for _, _, size in fake.calls] == [1] * 5
    assert sidecar.read_bytes()[:1] == b"\0"


@pytest.mark.parametrize("module_name", LOCK_MODULES)
def test_remaining_windows_locks_propagate_permanent_errors(
    module_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    module = importlib.import_module(module_name)
    fake = _FakeMsvcrt(failures=1, error_number=errno.EBADF)
    _patch_windows_lock(module, monkeypatch, fake)

    def acquire_once() -> None:
        if module_name.endswith("extensions_config"):
            with module.extensions_config_file_lock(tmp_path / "extensions_config.json"):
                pytest.fail("permanent lock errors must not enter the critical section")
        elif module_name.endswith("skills.projection"):
            with module._projection_lock(tmp_path / "projection"):
                pytest.fail("permanent lock errors must not enter the critical section")
        elif module_name.endswith("routers.skills"):
            root = tmp_path / "custom"
            storage = SimpleNamespace(get_user_custom_root=lambda: root)
            with module._custom_skill_mutation(storage):
                pytest.fail("permanent lock errors must not enter the critical section")
        else:
            lock_path = tmp_path / "sandbox.lock"
            with lock_path.open("a", encoding="utf-8") as lock_file:
                module._lock_file_exclusive(lock_file)

    with pytest.raises(OSError) as caught:
        acquire_once()

    assert caught.value.errno == errno.EBADF
    assert [mode for _, mode, _ in fake.calls] == [fake.LK_NBLCK]


@pytest.mark.parametrize("module_name", LOCK_MODULES)
def test_remaining_windows_locks_never_call_blocking_mode(
    module_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeMsvcrt()
    _exercise_lock(module_name, tmp_path, monkeypatch, fake)

    assert all(mode != fake.LK_LOCK for _, mode, _ in fake.calls)
