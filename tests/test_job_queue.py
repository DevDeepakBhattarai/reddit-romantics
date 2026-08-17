from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from reddit_video.job_queue import VideoJobQueue


def _ready_run(root: Path, name: str) -> Path:
    run = root / "runs" / name
    run.mkdir(parents=True)
    (run / "story.md").write_text("Speaker 0: Ready.\n", encoding="utf-8")
    return run


def _start_worker_thread(
    queue: VideoJobQueue,
    handler,
    *,
    idle_grace_seconds: float = 0.4,
) -> threading.Thread:
    thread = threading.Thread(
        target=lambda: queue.run_worker(handler, idle_grace_seconds=idle_grace_seconds),
        daemon=True,
    )
    thread.start()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if queue._live_worker_state() is not None:
            return thread
        time.sleep(0.01)
    raise AssertionError("worker did not become ready")


def test_pid_alive_probe_does_not_terminate_process(tmp_path: Path):
    queue = VideoJobQueue(tmp_path)
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert queue._pid_alive(sleeper.pid) is True
        assert sleeper.poll() is None
    finally:
        sleeper.terminate()
        sleeper.wait(timeout=5)


def test_enqueue_deduplicates_within_live_worker_memory(tmp_path: Path):
    queue = VideoJobQueue(tmp_path)
    run = _ready_run(tmp_path, "2026-08-17_00-00-01_first")
    release = threading.Event()

    def handler(_job: dict, _log_path: Path) -> int:
        release.wait(timeout=2)
        return 0

    worker = _start_worker_thread(queue, handler)
    first, created_first, _ = queue.enqueue(run, ["--run-dir", str(run)])
    second, created_second, _ = queue.enqueue(run, ["--run-dir", str(run)])

    assert created_first is True
    assert created_second is False
    assert second["id"] == first["id"]
    assert len(queue.snapshot()["running"]) + len(queue.snapshot()["pending"]) == 1

    release.set()
    worker.join(timeout=3)
    assert not worker.is_alive()


def test_live_worker_processes_jobs_fifo(tmp_path: Path):
    queue = VideoJobQueue(tmp_path)
    first_run = _ready_run(tmp_path, "2026-08-17_00-00-01_first")
    second_run = _ready_run(tmp_path, "2026-08-17_00-00-02_second")
    first_started = threading.Event()
    release_first = threading.Event()
    order: list[str] = []

    def handler(job: dict, _log_path: Path) -> int:
        order.append(Path(job["run_dir"]).name)
        if len(order) == 1:
            first_started.set()
            release_first.wait(timeout=2)
        return 0

    worker = _start_worker_thread(queue, handler)
    queue.enqueue(first_run, ["--run-dir", str(first_run)])
    assert first_started.wait(timeout=2)
    queue.enqueue(second_run, ["--run-dir", str(second_run)])
    release_first.set()
    worker.join(timeout=3)

    assert order == [first_run.name, second_run.name]
    assert not worker.is_alive()


def test_queue_disappears_when_worker_exits(tmp_path: Path):
    queue = VideoJobQueue(tmp_path)
    run = _ready_run(tmp_path, "2026-08-17_00-00-01_one-shot")
    processed: list[str] = []

    def handler(job: dict, _log_path: Path) -> int:
        processed.append(job["id"])
        return 0

    worker = _start_worker_thread(queue, handler, idle_grace_seconds=0.1)
    job, _, _ = queue.enqueue(run, ["--run-dir", str(run)])
    worker.join(timeout=3)

    assert processed == [job["id"]]
    assert not worker.is_alive()
    assert queue.snapshot() == {"running": [], "pending": [], "completed": [], "failed": []}
    assert not queue.state_path.exists()
    assert not queue.lock_path.exists()


def test_fresh_worker_clears_legacy_disk_backlog_instead_of_recovering_it(tmp_path: Path):
    queue = VideoJobQueue(tmp_path)
    for folder in (queue.pending_dir, queue.running_dir, queue.completed_dir, queue.failed_dir):
        folder.mkdir(parents=True)
        (folder / "old.json").write_text('{"run_dir":"old"}', encoding="utf-8")

    worker = _start_worker_thread(queue, lambda _job, _log: 0, idle_grace_seconds=0.1)
    worker.join(timeout=3)

    assert not queue.pending_dir.exists()
    assert not queue.running_dir.exists()
    assert not queue.completed_dir.exists()
    assert not queue.failed_dir.exists()


def test_worker_marks_failure_and_continues_to_next_in_memory_job(tmp_path: Path):
    queue = VideoJobQueue(tmp_path)
    bad_run = _ready_run(tmp_path, "2026-08-17_00-00-01_bad")
    good_run = _ready_run(tmp_path, "2026-08-17_00-00-02_good")
    first_started = threading.Event()
    release_first = threading.Event()
    processed: list[str] = []

    def handler(job: dict, _log_path: Path) -> int:
        name = Path(job["run_dir"]).name
        processed.append(name)
        if name == bad_run.name:
            first_started.set()
            release_first.wait(timeout=2)
            raise RuntimeError("render exploded")
        return 0

    worker = _start_worker_thread(queue, handler)
    queue.enqueue(bad_run, ["--run-dir", str(bad_run)])
    assert first_started.wait(timeout=2)
    queue.enqueue(good_run, ["--run-dir", str(good_run)])
    release_first.set()
    worker.join(timeout=3)

    assert processed == [bad_run.name, good_run.name]
    assert not worker.is_alive()


def test_enqueue_rejects_empty_story_before_starting_worker(tmp_path: Path):
    run = tmp_path / "runs" / "2026-08-17_00-00-01_empty"
    run.mkdir(parents=True)
    (run / "story.md").write_text("", encoding="utf-8")
    queue = VideoJobQueue(tmp_path)

    try:
        queue.enqueue(run, ["--run-dir", str(run)], main_script=tmp_path / "main.py")
    except ValueError as exc:
        assert "non-empty story.md" in str(exc)
    else:
        raise AssertionError("empty story must not be queued")

    assert queue._live_worker_state() is None


def test_detached_worker_reuses_live_worker(tmp_path: Path, monkeypatch):
    queue = VideoJobQueue(tmp_path)
    monkeypatch.setattr(queue, "_live_worker_state", lambda: {"pid": 31337})

    def fail_popen(*_args, **_kwargs):
        raise AssertionError("must not spawn a second worker")

    monkeypatch.setattr(subprocess, "Popen", fail_popen)
    assert queue.start_detached_worker(tmp_path / "main.py") == 31337


def test_detached_worker_uses_independent_visible_console_on_windows(tmp_path: Path, monkeypatch):
    queue = VideoJobQueue(tmp_path)
    captured: dict = {}

    class FakeProcess:
        pid = 4242

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return FakeProcess()

    states = iter([None, {"pid": 5151}])
    monkeypatch.setattr(queue, "_live_worker_state", lambda: next(states))
    monkeypatch.setattr(queue, "_discard_stale_worker_state", lambda **_kwargs: None)
    monkeypatch.setattr(queue, "_wait_for_worker_ready", lambda: {"pid": 5151})
    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    pid = queue.start_detached_worker(tmp_path / "main.py")
    assert pid == 5151
    if os.name == "nt":
        flags = captured["kwargs"]["creationflags"]
        assert flags & subprocess.CREATE_NEW_CONSOLE
        assert flags & subprocess.CREATE_NEW_PROCESS_GROUP
        assert flags & subprocess.CREATE_BREAKAWAY_FROM_JOB
        assert not flags & subprocess.DETACHED_PROCESS
        assert "stdin" not in captured["kwargs"]
        assert "stdout" not in captured["kwargs"]
        assert "stderr" not in captured["kwargs"]
    else:
        assert captured["kwargs"]["start_new_session"] is True
