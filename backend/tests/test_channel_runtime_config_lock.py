"""Cross-process lock behavior for the channel runtime-config store.

Windows' blocking ``msvcrt.LK_LOCK`` gives up after roughly ten seconds and
surfaces contention as ``PermissionError``. A Gateway worker can legitimately
hold this lock while an atomic JSON write is in progress, so the store must
retry the non-blocking acquisition just as the extension and Lark installers
do.
"""

from __future__ import annotations

import errno
import threading
import time
from types import SimpleNamespace

import pytest

import app.channels.runtime_config_store as runtime_config_store


class _FakeMsvcrt:
    """Small locking double that makes Windows contention deterministic."""

    LK_UNLCK = 0
    LK_LOCK = 1
    LK_NBLCK = 2

    def __init__(self, *, failures: int = 0, error_number: int = errno.EACCES) -> None:
        self.failures = failures
        self.error_number = error_number
        self.calls: list[tuple[int, int, int]] = []
        self.unlocks = 0

    def locking(self, fd: int, mode: int, size: int) -> None:
        self.calls.append((fd, mode, size))
        if mode == self.LK_UNLCK:
            self.unlocks += 1
            return
        attempts = len([call for call in self.calls if call[1] != self.LK_UNLCK])
        if attempts <= self.failures:
            raise OSError(self.error_number, "Lock unavailable")


@pytest.fixture
def windows_lock(monkeypatch):
    """Run the platform branch without changing the host's real ``os.name``."""

    monkeypatch.setattr(runtime_config_store, "fcntl", None)
    fake = _FakeMsvcrt()
    monkeypatch.setattr(runtime_config_store, "msvcrt", fake, raising=False)
    return fake


def test_windows_lock_retries_nonblocking_contention_and_releases(tmp_path, monkeypatch):
    fake = _FakeMsvcrt(failures=3)
    monkeypatch.setattr(runtime_config_store, "fcntl", None)
    monkeypatch.setattr(runtime_config_store, "msvcrt", fake, raising=False)
    delays: list[float] = []

    real_retry = runtime_config_store.retry_until_locked

    def retry(acquire, **kwargs):
        return real_retry(acquire, sleep=delays.append, **kwargs)

    monkeypatch.setattr(runtime_config_store, "retry_until_locked", retry)
    path = tmp_path / "runtime-config.json"

    with runtime_config_store._runtime_config_file_lock(path):
        pass

    assert [mode for _, mode, _ in fake.calls] == [fake.LK_NBLCK] * 4 + [fake.LK_UNLCK]
    assert [size for _, _, size in fake.calls] == [1] * 5
    assert fake.unlocks == 1
    assert delays == [0.2] * 3
    assert (path.parent / f".{path.name}.lock").read_bytes() == b"\0"


@pytest.mark.parametrize("error_number", [errno.EACCES, errno.EAGAIN, errno.EDEADLK])
def test_windows_lock_retries_all_documented_contention_errors(tmp_path, monkeypatch, error_number):
    fake = _FakeMsvcrt(failures=1, error_number=error_number)
    monkeypatch.setattr(runtime_config_store, "fcntl", None)
    monkeypatch.setattr(runtime_config_store, "msvcrt", fake, raising=False)
    calls: list[int] = []
    real_retry = runtime_config_store.retry_until_locked

    def retry(acquire, **kwargs):
        calls.append(1)
        return real_retry(acquire, sleep=lambda _: None, **kwargs)

    monkeypatch.setattr(runtime_config_store, "retry_until_locked", retry)
    with runtime_config_store._runtime_config_file_lock(tmp_path / "runtime-config.json"):
        pass

    assert calls == [1]
    assert [mode for _, mode, _ in fake.calls] == [fake.LK_NBLCK] * 2 + [fake.LK_UNLCK]


@pytest.mark.parametrize("error_number", [errno.EBADF, errno.EINVAL, errno.EIO, errno.EPERM])
def test_windows_lock_propagates_noncontention_errors(tmp_path, monkeypatch, error_number):
    fake = _FakeMsvcrt(failures=1, error_number=error_number)
    monkeypatch.setattr(runtime_config_store, "fcntl", None)
    monkeypatch.setattr(runtime_config_store, "msvcrt", fake, raising=False)
    slept: list[float] = []
    real_retry = runtime_config_store.retry_until_locked

    def retry(acquire, **kwargs):
        return real_retry(acquire, sleep=slept.append, **kwargs)

    monkeypatch.setattr(runtime_config_store, "retry_until_locked", retry)
    with pytest.raises(OSError) as caught:
        with runtime_config_store._runtime_config_file_lock(tmp_path / "runtime-config.json"):
            pytest.fail("the lock must not enter after a permanent error")

    assert caught.value.errno == error_number
    assert fake.unlocks == 0
    assert slept == []


def test_windows_lock_initializes_a_zero_byte_region(tmp_path, windows_lock):
    path = tmp_path / "runtime-config.json"

    with runtime_config_store._runtime_config_file_lock(path):
        pass

    lock_path = path.parent / f".{path.name}.lock"
    assert lock_path.read_bytes() == b"\0"
    assert windows_lock.calls[-1][1] == windows_lock.LK_UNLCK


def test_windows_lock_keeps_an_existing_region_byte(tmp_path, monkeypatch):
    fake = _FakeMsvcrt()
    monkeypatch.setattr(runtime_config_store, "fcntl", None)
    monkeypatch.setattr(runtime_config_store, "msvcrt", fake, raising=False)
    path = tmp_path / "runtime-config.json"
    lock_path = path.parent / f".{path.name}.lock"
    lock_path.write_bytes(b"X")

    with runtime_config_store._runtime_config_file_lock(path):
        pass

    assert lock_path.read_bytes() == b"X"
    assert [mode for _, mode, _ in fake.calls] == [fake.LK_NBLCK, fake.LK_UNLCK]


def test_posix_lock_still_uses_flock_and_unlocks(tmp_path, monkeypatch):
    class _FakeFcntl:
        LOCK_EX = 2
        LOCK_UN = 8

        def __init__(self):
            self.calls: list[tuple[int, int]] = []

        def flock(self, fd: int, operation: int) -> None:
            self.calls.append((fd, operation))

    fake = _FakeFcntl()
    monkeypatch.setattr(runtime_config_store, "fcntl", fake)
    monkeypatch.setattr(runtime_config_store, "msvcrt", SimpleNamespace(), raising=False)

    with runtime_config_store._runtime_config_file_lock(tmp_path / "runtime-config.json"):
        pass

    assert [operation for _, operation in fake.calls] == [fake.LOCK_EX, fake.LOCK_UN]


def test_store_mutations_remain_atomic_after_windows_lock_retry(tmp_path, monkeypatch):
    fake = _FakeMsvcrt(failures=2)
    monkeypatch.setattr(runtime_config_store, "fcntl", None)
    monkeypatch.setattr(runtime_config_store, "msvcrt", fake, raising=False)
    real_retry = runtime_config_store.retry_until_locked

    def retry(acquire, **kwargs):
        return real_retry(acquire, sleep=lambda _: None, **kwargs)

    monkeypatch.setattr(runtime_config_store, "retry_until_locked", retry)
    store = runtime_config_store.ChannelRuntimeConfigStore(tmp_path / "runtime-config.json")
    store.set_provider_config("slack", {"enabled": True, "bot_token": "xoxb-test"})

    assert store.load_all() == {"slack": {"enabled": True, "bot_token": "xoxb-test"}}
    assert fake.unlocks == 1


@pytest.mark.skipif(runtime_config_store.fcntl is not None, reason="exercises native Windows locking")
def test_real_windows_lock_can_wait_past_blocking_lock_limit(tmp_path):
    """A second worker acquires after a holder keeps the region for 12 seconds."""
    import msvcrt

    path = tmp_path / "runtime-config.json"
    lock_path = path.parent / f".{path.name}.lock"
    lock_path.write_bytes(b"\0")
    holder = lock_path.open("a+b")
    holder.seek(0)
    msvcrt.locking(holder.fileno(), msvcrt.LK_NBLCK, 1)
    acquired = threading.Event()
    errors: list[BaseException] = []

    def contender() -> None:
        try:
            with runtime_config_store._runtime_config_file_lock(path):
                acquired.set()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=contender, daemon=True)
    thread.start()
    try:
        time.sleep(12)
        holder.seek(0)
        msvcrt.locking(holder.fileno(), msvcrt.LK_UNLCK, 1)
        holder.close()
        assert acquired.wait(timeout=5), errors
    finally:
        thread.join(timeout=10)

    assert not errors
