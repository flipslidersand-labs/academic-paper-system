"""Background job store with SQLite persistence."""

import uuid
from datetime import datetime

from academic_paper import db


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, dict] = {}
        self._db_path: str | None = None

    def init(self, db_path: str) -> None:
        """Load persisted jobs from SQLite on startup.

        Jobs that were `running` at last shutdown are converted to `failed`
        to avoid phantom running state.
        """
        self._db_path = db_path
        conn = db.get_connection(db_path)
        try:
            for job in db.load_all_jobs(conn):
                if job["status"] == "running":
                    job["status"] = "failed"
                    job["error"] = "Server restarted while job was running"
                    db.upsert_job(conn, job)
                self._jobs[job["id"]] = job
        finally:
            conn.close()

    def create(self, job_type: str) -> dict:
        """Create a new pending job, persist it, and return it."""
        job: dict = {
            "id": str(uuid.uuid4()),
            "type": job_type,
            "status": "pending",
            "created_at": datetime.utcnow().isoformat(),
            "started_at": None,
            "finished_at": None,
            "result": None,
            "error": None,
        }
        self._jobs[job["id"]] = job
        self.persist(job)
        return job

    def persist(self, job: dict) -> None:
        """Upsert a job to SQLite. No-op if db not initialized."""
        if self._db_path is None:
            return
        conn = db.get_connection(self._db_path)
        try:
            db.upsert_job(conn, job)
        finally:
            conn.close()

    def get(self, job_id: str) -> dict | None:
        return self._jobs.get(job_id)

    def all(self) -> list[dict]:
        return sorted(self._jobs.values(), key=lambda j: j["created_at"], reverse=True)
