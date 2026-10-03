"""Shared fixtures for the test suite."""

import os
import sys
import tempfile
from pathlib import Path

# Set valid URLs before any module-level Settings() instantiation (#200).
os.environ.setdefault("EMBEDDING_SVC_URL", "http://localhost:9092")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")

# scripts/ is not a package; add it once here so any test module can import
# from it directly, instead of each one repeating this sys.path.insert (#272).
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import pytest

# Settings reads ".env" relative to the cwd (#505). Import the package from an
# empty cwd so a developer's local .env cannot leak into the module-level
# `settings` singleton, then disable env_file for any later Settings() calls.
_orig_cwd = os.getcwd()
with tempfile.TemporaryDirectory() as _empty_dir:
    os.chdir(_empty_dir)
    try:
        from academic_paper import config as _config
        from academic_paper.config import Settings, get_settings

        Settings.model_config["env_file"] = None
        from academic_paper.db import init_db
        from academic_paper.jobs import job_store
    finally:
        os.chdir(_orig_cwd)


@pytest.fixture
def temp_db():
    """Initialized temporary database, removed after the test (#151).

    WAL mode creates -wal/-shm sidecar files, so all three are unlinked —
    previously each test module defined its own fixture and most leaked
    the files into /tmp on self-hosted runners.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = f.name
    init_db(db_path)
    yield db_path
    for suffix in ("", "-wal", "-shm"):
        Path(db_path + suffix).unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def _reset_job_store():
    """Reset the global JobStore singleton around every test (#151).

    Without this, module execution order leaks jobs/db-path between test
    modules, and xdist parallelisation would race on shared state.
    """
    job_store._jobs.clear()
    job_store._db_path = None
    yield
    job_store._jobs.clear()
    job_store._db_path = None


@pytest.fixture(autouse=True)
def _restore_settings_cache(monkeypatch):
    """Keep ``get_settings() is settings`` true after tests that cache_clear() (#604).

    After ``get_settings.cache_clear()`` the next call would build a fresh
    Settings that differs from the module-level ``settings`` alias, so re-prime
    the cache with the original instance on teardown.
    """
    original = _config.settings
    yield
    if get_settings() is not original:
        get_settings.cache_clear()
        with monkeypatch.context() as m:
            m.setattr(_config, "Settings", lambda: original)
            get_settings()


@pytest.fixture
def override_settings(monkeypatch):
    """Temporarily override Settings fields; restored automatically.

    Usage: ``override_settings(max_upload_mb=1)``. Patches the shared instance
    returned by get_settings() (== the ``settings`` alias), so both access
    styles see the override and no test assigns ``settings.x = ...`` directly.
    """

    def _override(**values):
        target = get_settings()
        for name, value in values.items():
            monkeypatch.setattr(target, name, value)
        return target

    return _override
