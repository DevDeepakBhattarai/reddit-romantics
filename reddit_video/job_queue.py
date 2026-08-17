from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from datetime import datetime, timezone
from multiprocessing.connection import Client, Listener
from pathlib import Path
from typing import Any


JobHandler = Callable[[dict[str, Any], Path], int | None]


class VideoJobQueue:
    """Single-worker, in-memory FIFO queue for expensive video pipeline runs.

    The detached worker owns the only authoritative queue. CLI `enqueue` processes send
    jobs to that worker over a localhost authenticated IPC socket. Nothing pending or
    running is persisted, so if the worker dies its queue dies with it.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.queue_root = self.root / ".work" / "video-queue"
        self.logs_dir = self.queue_root / "logs"
        self.lock_path = self.queue_root / "worker.lock"
        self.state_path = self.queue_root / "worker.json"
        self.worker_log = self.queue_root / "worker.log"

        # Legacy durable-queue paths. New code never writes queue jobs here; they are
        # cleared whenever a fresh worker starts so old versions cannot leak backlog.
        self.pending_dir = self.queue_root / "pending"
        self.running_dir = self.queue_root / "running"
        self.completed_dir = self.queue_root / "completed"
        self.failed_dir = self.queue_root / "failed"

        self.logs_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False

        if os.name == "nt":
            # On Windows, os.kill(pid, 0) is not a harmless existence probe.
            import ctypes
            from ctypes import wintypes

            process_query_limited_information = 0x1000
            still_active = 259
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
            kernel32.GetExitCodeProcess.restype = wintypes.BOOL
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel32.CloseHandle.restype = wintypes.BOOL

            handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
            if not handle:
                return False
            try:
                exit_code = wintypes.DWORD()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                    return False
                return exit_code.value == still_active
            finally:
                kernel32.CloseHandle(handle)

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
        line = f"[{self._utc_now()}] {message}"
        with self.worker_log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)

    def _clear_legacy_queue_state(self) -> None:
        """Delete all disk-backed job state left by older queue versions."""
        for folder in (self.pending_dir, self.running_dir, self.completed_dir, self.failed_dir):
            if folder.exists():
                shutil.rmtree(folder, ignore_errors=True)

    def _raw_worker_state(self) -> dict[str, Any] | None:
        try:
            state = self._read_json(self.state_path)
            pid = int(state["pid"])
            port = int(state["port"])
            authkey = str(state["authkey"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None
        if not self._pid_alive(pid):
            self._discard_stale_worker_state(expected_pid=pid)
            return None
        if not (1 <= port <= 65535) or not authkey:
            return None
        return state

    def _discard_stale_worker_state(self, *, expected_pid: int | None = None) -> None:
        try:
            if expected_pid is not None and self.state_path.exists():
                current = self._read_json(self.state_path)
                if int(current.get("pid", -1)) != expected_pid:
                    return
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
        try:
            self.state_path.unlink()
        except FileNotFoundError:
            pass

        try:
            owner = int(self.lock_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            owner = -1
        if owner <= 0 or not self._pid_alive(owner):
            try:
                self.lock_path.unlink()
            except FileNotFoundError:
                pass

        # A dead worker means the old queue is dead too. This also removes leftovers
        # produced by the previous disk-backed implementation.
        self._clear_legacy_queue_state()

    def _request_state(self, state: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        address = (str(state.get("host", "127.0.0.1")), int(state["port"]))
        authkey = bytes.fromhex(str(state["authkey"]))
        connection = Client(address, family="AF_INET", authkey=authkey)
        try:
            connection.send(payload)
            response = connection.recv()
        finally:
            connection.close()
        if not isinstance(response, dict):
            raise RuntimeError("Queue worker returned an invalid IPC response")
        if response.get("ok") is False:
            raise RuntimeError(str(response.get("error", "Queue worker request failed")))
        return response

    def _live_worker_state(self) -> dict[str, Any] | None:
        state = self._raw_worker_state()
        if state is None:
            return None
        try:
            response = self._request_state(state, {"op": "ping"})
            if response.get("pid") != int(state["pid"]):
                raise RuntimeError("Queue worker PID mismatch")
            return state
        except (OSError, EOFError, ValueError, RuntimeError):
            self._discard_stale_worker_state(expected_pid=int(state["pid"]))
            return None

    def _wait_for_worker_ready(self, *, timeout_seconds: float = 5.0) -> dict[str, Any]:
        deadline = time.monotonic() + max(0.1, timeout_seconds)
        while time.monotonic() < deadline:
            state = self._live_worker_state()
            if state is not None:
                return state
            time.sleep(0.05)
        raise RuntimeError(
            "Detached queue worker did not become ready within "
            f"{timeout_seconds:.1f}s"
        )

    def start_detached_worker(self, main_script: str | Path) -> int:
        """Ensure one detached in-memory queue worker is alive and return its PID."""
        state = self._live_worker_state()
        if state is not None:
            return int(state["pid"])

        # Starting a fresh worker defines a fresh queue lifetime.
        self._discard_stale_worker_state()
        command = [sys.executable, str(Path(main_script).resolve()), "queue-worker"]
        kwargs: dict[str, Any] = {
            "cwd": str(self.root),
            "close_fds": True,
        }
        if os.name == "nt":
            kwargs["creationflags"] = (
                subprocess.CREATE_NEW_CONSOLE
                | subprocess.CREATE_NEW_PROCESS_GROUP
                | subprocess.CREATE_BREAKAWAY_FROM_JOB
            )
        else:
            kwargs["start_new_session"] = True
        subprocess.Popen(command, **kwargs)
        return int(self._wait_for_worker_ready()["pid"])

    def enqueue(
        self,
        run_dir: str | Path,
        run_args: list[str],
        *,
        main_script: str | Path | None = None,
    ) -> tuple[dict[str, Any], bool, int]:
        run_path = Path(run_dir)
        if not run_path.is_absolute():
            run_path = self.root / run_path
        run_path = run_path.resolve()
        story_path = run_path / "story.md"
        if not run_path.is_dir():
            raise FileNotFoundError(f"Run folder not found: {run_path}")
        if not story_path.exists() or not story_path.read_text(encoding="utf-8-sig").strip():
            raise ValueError(f"Run is not ready for video generation; non-empty story.md is required: {run_path}")

        state = self._live_worker_state()
        if state is None:
            if main_script is None:
                raise RuntimeError("No live video queue worker")
            self.start_detached_worker(main_script)
            state = self._live_worker_state()
            if state is None:
                raise RuntimeError("Video queue worker started but is not reachable")

        request = {
            "op": "enqueue",
            "run_dir": str(run_path),
            "run_args": list(run_args),
        }
        try:
            response = self._request_state(state, request)
        except (OSError, EOFError, ValueError) as exc:
            # The queue belonged to that worker. If it died during submission, never
            # resurrect its old jobs; start a new empty worker and submit only this job.
            self._discard_stale_worker_state(expected_pid=int(state["pid"]))
            if main_script is None:
                raise RuntimeError("Video queue worker died while enqueueing") from exc
            self.start_detached_worker(main_script)
            state = self._live_worker_state()
            if state is None:
                raise RuntimeError("Replacement video queue worker is not reachable") from exc
            response = self._request_state(state, request)

        job = response.get("job")
        if not isinstance(job, dict):
            raise RuntimeError("Queue worker did not return the queued job")
        return job, bool(response.get("created")), int(state["pid"])

    def _acquire_worker_lock(self, *, wait_seconds: float = 5.0) -> bool:
        self.queue_root.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + max(0.0, wait_seconds)
        while True:
            try:
                fd = os.open(self.lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            except FileExistsError:
                try:
                    pid = int(self.lock_path.read_text(encoding="utf-8").strip())
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

    def _release_worker_state(self) -> None:
        try:
            state = self._read_json(self.state_path)
            owner = int(state.get("pid", -1))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            owner = -1
        if owner == os.getpid():
            try:
                self.state_path.unlink()
            except FileNotFoundError:
                pass

    def run_worker(self, handler: JobHandler, *, idle_grace_seconds: float = 3.0) -> int:
        """Own an in-memory FIFO queue, drain it sequentially, then exit when idle."""
        if not self._acquire_worker_lock():
            return 0

        self._clear_legacy_queue_state()
        authkey = secrets.token_bytes(32)
        listener = Listener(("127.0.0.1", 0), family="AF_INET", authkey=authkey)
        host, port = listener.address
        self._atomic_write_json(
            self.state_path,
            {
                "pid": os.getpid(),
                "host": host,
                "port": port,
                "authkey": authkey.hex(),
                "started_at": self._utc_now(),
            },
        )

        pending: deque[dict[str, Any]] = deque()
        running: dict[str, Any] | None = None
        completed: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        condition = threading.Condition()
        stopping = threading.Event()

        def same_run(left: str, right: str) -> bool:
            return os.path.normcase(str(Path(left).resolve())) == os.path.normcase(str(Path(right).resolve()))

        def ipc_server() -> None:
            nonlocal running
            while not stopping.is_set():
                try:
                    connection = listener.accept()
                except (OSError, EOFError):
                    break
                try:
                    request = connection.recv()
                    op = request.get("op") if isinstance(request, dict) else None
                    if op == "ping":
                        connection.send({"ok": True, "pid": os.getpid()})
                        continue
                    if op == "snapshot":
                        with condition:
                            connection.send(
                                {
                                    "ok": True,
                                    "snapshot": {
                                        "running": [dict(running)] if running is not None else [],
                                        "pending": [dict(job) for job in pending],
                                        "completed": [dict(job) for job in completed],
                                        "failed": [dict(job) for job in failed],
                                    },
                                }
                            )
                        continue
                    if op != "enqueue":
                        connection.send({"ok": False, "error": f"Unknown queue operation: {op}"})
                        continue

                    run_dir = str(request.get("run_dir", ""))
                    run_args = list(request.get("run_args", []))
                    with condition:
                        existing = None
                        if running is not None and same_run(str(running.get("run_dir", "")), run_dir):
                            existing = running
                        if existing is None:
                            existing = next(
                                (job for job in pending if same_run(str(job.get("run_dir", "")), run_dir)),
                                None,
                            )
                        if existing is not None:
                            connection.send({"ok": True, "job": dict(existing), "created": False})
                            continue

                        job_id = f"{time.time_ns():020d}-{uuid.uuid4().hex[:8]}"
                        job = {
                            "id": job_id,
                            "state": "pending",
                            "created_at": self._utc_now(),
                            "run_dir": run_dir,
                            "run_args": run_args,
                        }
                        pending.append(job)
                        condition.notify_all()
                    connection.send({"ok": True, "job": dict(job), "created": True})
                except (OSError, EOFError, TypeError, ValueError) as exc:
                    try:
                        connection.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
                    except (OSError, EOFError):
                        pass
                finally:
                    connection.close()

        server_thread = threading.Thread(target=ipc_server, name="video-queue-ipc", daemon=True)
        server_thread.start()
        self._append_worker_log(f"Worker {os.getpid()} started with an empty in-memory queue")

        try:
            while True:
                with condition:
                    if not pending:
                        condition.wait(timeout=max(0.0, idle_grace_seconds))
                    if not pending:
                        break
                    job = pending.popleft()
                    job = dict(job)
                    job["state"] = "running"
                    job["started_at"] = self._utc_now()
                    job["worker_pid"] = os.getpid()
                    running = job

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
                with condition:
                    running = None
                    if exit_code == 0:
                        job["state"] = "completed"
                        completed.append(job)
                        self._append_worker_log(f"Completed job {job['id']}")
                    else:
                        job["state"] = "failed"
                        job["error"] = error
                        failed.append(job)
                        self._append_worker_log(f"Failed job {job['id']}: {error}")
        finally:
            stopping.set()
            try:
                listener.close()
            except OSError:
                pass
            self._append_worker_log(f"Worker {os.getpid()} stopped; in-memory queue discarded")
            self._release_worker_state()
            self._release_worker_lock()
        return 0

    def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        state = self._live_worker_state()
        if state is None:
            # No worker means no queue. Clean old on-disk active jobs opportunistically.
            self._clear_legacy_queue_state()
            return {"running": [], "pending": [], "completed": [], "failed": []}
        try:
            response = self._request_state(state, {"op": "snapshot"})
        except (OSError, EOFError, ValueError, RuntimeError):
            self._discard_stale_worker_state(expected_pid=int(state["pid"]))
            return {"running": [], "pending": [], "completed": [], "failed": []}
        snapshot = response.get("snapshot")
        if not isinstance(snapshot, dict):
            return {"running": [], "pending": [], "completed": [], "failed": []}
        return {
            name: list(snapshot.get(name, []))
            for name in ("running", "pending", "completed", "failed")
        }
