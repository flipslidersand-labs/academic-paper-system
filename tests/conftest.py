"""Shared fixtures for the test suite."""

import os
import tempfile
from pathlib import Path

# Set valid URLs before any module-level Settings() instantiation (#200).
os.environ.setdefault("EMBEDDING_SVC_URL", "http://localhost:9092")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")

import pytest

from academic_paper.db import init_db
from academic_paper.jobs import job_store


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
