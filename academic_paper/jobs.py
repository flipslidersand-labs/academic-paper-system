"""Background job tracking for bulk operations."""

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

JobStatus = Literal["pending", "running", "done", "failed"]


@dataclass
class Job:
    id: str
    status: JobStatus
    kind: str = ""
    total: int = 0
    processed: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    result: dict | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "kind": self.kind,
            "total": self.total,
            "processed": self.processed,
            "failed": self.failed,
            "errors": self.errors[:10],
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result": self.result,
        }


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._db_path: str | None = None
        self._lock = threading.Lock()
        # Jobs whose _persist() ran before init() set _db_path (#302) — flushed
        # to SQLite once init() completes, so they aren't lost on restart.
        self._pending_writes: list[Job] = []

    async def init(self, db_path: str) -> None:
        """Load existing jobs from SQLite into memory on startup.

        Jobs that were 'pending' or 'running' at last shutdown are converted to
        'failed' — their BackgroundTasks did not survive the restart, and a
        stale in-flight status would block has_running() forever.

        The SQLite reads/writes are synchronous (see busy_timeout in db.py); run
        them in a worker thread via asyncio.to_thread so a lock-contended load
        can't block the event loop and delay unrelated asyncio.wait_for timeouts
        (#277).
        """
        self._db_path = db_path
        with self._lock:
            pending = self._pending_writes
            self._pending_writes = []
        if pending:
            await asyncio.to_thread(self._flush_pending, pending)
        jobs = await asyncio.to_thread(self._load_and_reconcile_jobs, db_path)
        with self._lock:
            for job in jobs:
                self._jobs[job.id] = job

    def _flush_pending(self, jobs: list[Job]) -> None:
        """Synchronous helper for init(): write jobs buffered before init() completed."""
        for job in jobs:
            self._persist(job)

    def _load_and_reconcile_jobs(self, db_path: str) -> list[Job]:
        """Synchronous helper for init(): read jobs and reconcile stale in-flight rows."""
        from academic_paper.db import db_connection, load_all_jobs, upsert_job

        jobs: list[Job] = []
        with db_connection(db_path) as conn:
            for row in load_all_jobs(conn):
                status = row["status"]
                errors = list(row["errors"])
                if status in ("pending", "running"):
                    status = "failed"
                    errors.append("Server restarted while job was in flight")
                    upsert_job(
                        conn,
                        row["id"],
                        status,
                        row["total"],
                        row["processed"],
                        row["failed"],
                        errors,
                        row["started_at"],
                        row["finished_at"],
                        kind=row.get("kind", ""),
                    )
                jobs.append(
                    Job(
                        id=row["id"],
                        status=status,
                        kind=row.get("kind", ""),
                        total=row["total"],
                        processed=row["processed"],
                        failed=row["failed"],
                        errors=errors,
                        started_at=row["started_at"],
                        finished_at=row["finished_at"],
                    )
                )
        return jobs

    def _persist(self, job: Job) -> None:
        """Write current job state to SQLite (synchronous; call via asyncio.to_thread).

        If init() hasn't set _db_path yet (job created before startup finished
        loading the DB, see #302), buffer the job instead of silently dropping
        it — init() flushes the buffer once _db_path is set.
        """
        if not self._db_path:
            logger.warning("JobStore._persist called before init() completed; buffering job %s", job.id)
            with self._lock:
                self._pending_writes.append(job)
            return
        from academic_paper.db import db_connection, upsert_job

        with db_connection(self._db_path) as conn:
            upsert_job(
                conn,
                job.id,
                job.status,
                job.total,
                job.processed,
                job.failed,
                job.errors,
                job.started_at,
                job.finished_at,
                kind=job.kind,
            )

    async def create(self, kind: str = "") -> Job:
        job = Job(id=str(uuid.uuid4()), status="pending", kind=kind)
        with self._lock:
            self._jobs[job.id] = job
        await asyncio.to_thread(self._persist, job)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_all(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def has_running(self, kind: str | None = None) -> bool:
        """True if any job (optionally filtered by kind) is pending or running.

        Counting "pending" closes the TOCTOU window between create() and the
        background task flipping the status to "running"; filtering by kind
        keeps unrelated job types (e.g. per-paper ingest) from blocking each
        other.
        """
        with self._lock:
            return any(
                j.status in ("pending", "running") and (kind is None or j.kind == kind) for j in self._jobs.values()
            )

    async def create_if_not_running(self, kind: str = "") -> Job | None:
        """Atomically check for a running/pending job of `kind` and create one if none exists.

        has_running() and create() each take self._lock independently, so a caller doing
        "check then create" (e.g. the summarize-all endpoint) still races: two callers can
        both see has_running() == False before either has created a job (see #235). This
        method performs the check and the create under a single lock acquisition, closing
        that window. Returns None if a job of this kind is already pending/running.
        """
        with self._lock:
            if any(j.status in ("pending", "running") and j.kind == kind for j in self._jobs.values()):
                return None
            job = Job(id=str(uuid.uuid4()), status="pending", kind=kind)
            self._jobs[job.id] = job
        await asyncio.to_thread(self._persist, job)
        return job

    async def persist(self, job: Job) -> None:
        """Persist job state to SQLite (call on status transitions).

        Runs the synchronous sqlite3 write in a worker thread (asyncio.to_thread) so
        that lock contention (busy_timeout up to 5s, see db.py) can't block the event
        loop and delay unrelated asyncio.wait_for timeouts elsewhere in the process (#277).
        """
        await asyncio.to_thread(self._persist, job)


job_store = JobStore()
