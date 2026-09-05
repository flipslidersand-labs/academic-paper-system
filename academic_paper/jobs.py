"""Background job tracking for bulk operations."""

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal

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

    def init(self, db_path: str) -> None:
        """Load existing jobs from SQLite into memory on startup.

        Jobs that were 'pending' or 'running' at last shutdown are converted to
        'failed' — their BackgroundTasks did not survive the restart, and a
        stale in-flight status would block has_running() forever.
        """
        from academic_paper.db import db_connection, load_all_jobs, upsert_job

        self._db_path = db_path
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
                job = Job(
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
                with self._lock:
                    self._jobs[job.id] = job

    def _persist(self, job: Job) -> None:
        """Write current job state to SQLite."""
        if not self._db_path:
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

    def create(self, kind: str = "") -> Job:
        job = Job(id=str(uuid.uuid4()), status="pending", kind=kind)
        with self._lock:
            self._jobs[job.id] = job
        self._persist(job)
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
                j.status in ("pending", "running") and (kind is None or j.kind == kind)
                for j in self._jobs.values()
            )

    def persist(self, job: Job) -> None:
        """Persist job state to SQLite (call on status transitions)."""
        self._persist(job)


job_store = JobStore()
