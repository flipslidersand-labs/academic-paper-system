"""Shared pytest fixtures for the full test suite.

job_store is a module-level singleton in academic_paper.jobs.  Tests that
exercise the jobs API must reset it between runs to avoid state leakage.
Placing reset_job_store here (autouse, session-scoped per module) makes it
effective across *all* test files, not just test_jobs.py.
"""

import os
import tempfile

import pytest

from academic_paper.db import init_db
from academic_paper.jobs import job_store


@pytest.fixture
def temp_db():
    """Create an isolated SQLite DB for one test; delete it on teardown."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = f.name
    init_db(db_path)
    yield db_path
    os.unlink(db_path)


@pytest.fixture(autouse=True)
def reset_job_store(temp_db):
    """Reset the global job_store before and after every test."""
    job_store._jobs.clear()
    job_store._db_path = temp_db
    yield
    job_store._jobs.clear()
    job_store._db_path = None
