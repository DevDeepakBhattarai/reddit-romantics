from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


JobHandler = Callable[[dict[str, Any], Path], int | None]


class VideoJobQueue:
    """Durable single-worker FIFO queue for expensive video pipeline runs."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.queue_root = self.root / ".work" / "video-queue"
        self.pending_dir = self.queue_root / "pending"
        self.running_dir = self.queue_root / "running"
        self.completed_dir = self.queue_root / "completed"
        self.failed_dir = self.queue_root / "failed"
        self.logs_dir = self.queue_root / "logs"
        self.lock_path = self.queue_root / "worker.lock"
        self.worker_log = self.queue_root / "worker.log"
        for path in (
            self.pending_dir,
            self.running_dir,
            self.completed_dir,
            self.failed_dir,
            self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, path)

    def _append_worker_log(self, message: str) -> None:
        self.queue_root.mkdir(parents=True, exist_ok=True)
        with self.worker_log.open("a", encoding="utf-8") as handle:
            handle.write(f"[{self._utc_now()}] {message}\n")

    def _active_job_for_run(self, run_dir: Path) -> dict[str, Any] | None:
        wanted = os.path.normcase(str(run_dir.resolve()))
        for folder in (self.running_dir, self.pending_dir):
            for path in sorted(folder.glob("*.json")):
                try:
                    job = self._read_json(path)
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
                existing = job.get("run_dir")
                if existing and os.path.normcase(str(Path(existing).resolve())) == wanted:
                    return job
        return None

    def enqueue(self, run_dir: str | Path, run_args: list[str]) -> tuple[dict[str, Any], bool]:
        run_path = Path(run_dir)
        if not run_path.is_absolute():
            run_path = self.root / run_path
        run_path = run_path.resolve()
        story_path = run_path / "story.md"
        if not run_path.is_dir():
            raise FileNotFoundError(f"Run folder not found: {run_path}")
        if not story_path.exists() or not story_path.read_text(encoding="utf-8-sig").strip():
            raise ValueError(f"Run is not ready for video generation; non-empty story.md is required: {run_path}")

        active = self._active_job_for_run(run_path)
        if active is not None:
            return active, False

        job_id = f"{time.time_ns():020d}-{uuid.uuid4().hex[:8]}"
        job = {
            "id": job_id,
            "state": "pending",
            "created_at": self._utc_now(),
            "run_dir": str(run_path),
            "run_args": list(run_args),
        }
        self._atomic_write_json(self.pending_dir / f"{job_id}.json", job)
        return job, True

    def start_detached_worker(self, main_script: str | Path) -> int:
        """Launch a worker detached from the caller/session and return its PID."""
        command = [sys.executable, str(Path(main_script).resolve()), "queue-worker"]
        kwargs: dict[str, Any] = {
            "cwd": str(self.root),
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "close_fds": True,
        }
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        process = subprocess.Popen(command, **kwargs)
        return int(process.pid)

    def _acquire_worker_lock(self, *, wait_seconds: float = 5.0) -> bool:
        self.queue_root.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + max(0.0, wait_seconds)
        while True:
            try:
                fd = os.open(self.lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            except FileExistsError:
                try:
                    raw = self.lock_path.read_text(encoding="utf-8").strip()
                    pid = int(raw)
                except (OSError, ValueError):
                    pid = -1
                if not self._pid_alive(pid):
                    try:
                        self.lock_path.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                if time.monotonic() >= deadline:
                    return False
                # A newly spawned candidate waits briefly instead of exiting immediately.
                # This closes the race where the active worker is releasing its lock just
                # as a new job is enqueued. Long-running workers keep ownership, so this
                # candidate naturally gives up after the short handoff window.
                time.sleep(0.2)
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(str(os.getpid()))
            return True

    def _release_worker_lock(self) -> None:
        try:
            owner = int(self.lock_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            owner = -1
        if owner == os.getpid():
            try:
                self.lock_path.unlink()
            except FileNotFoundError:
                pass

    def _recover_orphaned_jobs(self) -> None:
        for path in sorted(self.running_dir.glob("*.json")):
            try:
                job = self._read_json(path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self._append_worker_log(f"Could not recover {path.name}: {exc}")
                continue
            job["state"] = "pending"
            job["recovered_at"] = self._utc_now()
            job.pop("started_at", None)
            job.pop("worker_pid", None)
            destination = self.pending_dir / path.name
            self._atomic_write_json(path, job)
            os.replace(path, destination)
            self._append_worker_log(f"Recovered orphaned job {job.get('id', path.stem)}")

    def _next_pending(self) -> Path | None:
        jobs = sorted(self.pending_dir.glob("*.json"), key=lambda path: path.name)
        return jobs[0] if jobs else None

    def run_worker(self, handler: JobHandler, *, idle_grace_seconds: float = 3.0) -> int:
        """Drain the queue sequentially, then exit when it remains empty."""
        if not self._acquire_worker_lock():
            return 0

        self._append_worker_log(f"Worker {os.getpid()} started")
        try:
            self._recover_orphaned_jobs()
            while True:
                pending = self._next_pending()
                if pending is None:
                    time.sleep(max(0.0, idle_grace_seconds))
                    pending = self._next_pending()
                    if pending is None:
                        break

                running = self.running_dir / pending.name
                try:
                    os.replace(pending, running)
                except FileNotFoundError:
                    continue

                try:
                    job = self._read_json(running)
                except Exception as exc:
                    bad = {
                        "id": running.stem,
                        "state": "failed",
                        "failed_at": self._utc_now(),
                        "error": f"Invalid queue job: {exc}",
                    }
                    self._atomic_write_json(running, bad)
                    os.replace(running, self.failed_dir / running.name)
                    continue

                job["state"] = "running"
                job["started_at"] = self._utc_now()
                job["worker_pid"] = os.getpid()
                self._atomic_write_json(running, job)
                log_path = self.logs_dir / f"{job['id']}.log"
                self._append_worker_log(f"Starting job {job['id']} for {job['run_dir']}")

                exit_code = 0
                error: str | None = None
                try:
                    result = handler(job, log_path)
                    exit_code = int(result or 0)
                    if exit_code != 0:
                        error = f"Pipeline exited with code {exit_code}"
                except Exception as exc:
                    exit_code = 1
                    error = f"{type(exc).__name__}: {exc}"
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(f"\nQUEUE ERROR: {error}\n")

                job["exit_code"] = exit_code
                job["finished_at"] = self._utc_now()
                if exit_code == 0:
                    job["state"] = "completed"
                    destination = self.completed_dir / running.name
                    self._append_worker_log(f"Completed job {job['id']}")
                else:
                    job["state"] = "failed"
                    job["error"] = error
                    destination = self.failed_dir / running.name
                    self._append_worker_log(f"Failed job {job['id']}: {error}")
                self._atomic_write_json(running, job)
                os.replace(running, destination)
        finally:
            self._append_worker_log(f"Worker {os.getpid()} stopped")
            self._release_worker_lock()
        return 0

    def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        result: dict[str, list[dict[str, Any]]] = {}
        for name, folder in (
            ("pending", self.pending_dir),
            ("running", self.running_dir),
            ("completed", self.completed_dir),
            ("failed", self.failed_dir),
        ):
            jobs: list[dict[str, Any]] = []
            for path in sorted(folder.glob("*.json")):
                try:
                    jobs.append(self._read_json(path))
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
            result[name] = jobs
        return result
