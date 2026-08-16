from __future__ import annotations

from pathlib import Path

from reddit_video.job_queue import VideoJobQueue


def _ready_run(root: Path, name: str) -> Path:
    run = root / "runs" / name
    run.mkdir(parents=True)
    (run / "story.md").write_text("Speaker 0: Ready.\n", encoding="utf-8")
    return run


def test_enqueue_is_durable_and_deduplicates_active_run(tmp_path: Path):
    run = _ready_run(tmp_path, "2026-08-17_00-00-01_first")
    queue = VideoJobQueue(tmp_path)

    first, created_first = queue.enqueue(run, ["--run-dir", str(run)])
    second, created_second = queue.enqueue(run, ["--run-dir", str(run)])

    assert created_first is True
    assert created_second is False
    assert second["id"] == first["id"]
    assert len(queue.snapshot()["pending"]) == 1


def test_worker_processes_all_jobs_in_fifo_order_then_exits(tmp_path: Path):
    first_run = _ready_run(tmp_path, "2026-08-17_00-00-01_first")
    second_run = _ready_run(tmp_path, "2026-08-17_00-00-02_second")
    queue = VideoJobQueue(tmp_path)
    first, _ = queue.enqueue(first_run, ["--run-dir", str(first_run)])
    second, _ = queue.enqueue(second_run, ["--run-dir", str(second_run)])
    order: list[str] = []

    def handler(job: dict, log_path: Path) -> int:
        order.append(Path(job["run_dir"]).name)
        log_path.write_text(f"processed {job['id']}", encoding="utf-8")
        return 0

    assert queue.run_worker(handler, idle_grace_seconds=0) == 0

    snapshot = queue.snapshot()
    assert order == [first_run.name, second_run.name]
    assert snapshot["pending"] == []
    assert snapshot["running"] == []
    assert [job["id"] for job in snapshot["completed"]] == [first["id"], second["id"]]
    assert snapshot["failed"] == []
    assert not queue.lock_path.exists()


def test_worker_marks_failure_and_continues_to_next_job(tmp_path: Path):
    bad_run = _ready_run(tmp_path, "2026-08-17_00-00-01_bad")
    good_run = _ready_run(tmp_path, "2026-08-17_00-00-02_good")
    queue = VideoJobQueue(tmp_path)
    bad, _ = queue.enqueue(bad_run, ["--run-dir", str(bad_run)])
    good, _ = queue.enqueue(good_run, ["--run-dir", str(good_run)])
    processed: list[str] = []

    def handler(job: dict, _log_path: Path) -> int:
        processed.append(job["id"])
        if job["id"] == bad["id"]:
            raise RuntimeError("render exploded")
        return 0

    queue.run_worker(handler, idle_grace_seconds=0)
    snapshot = queue.snapshot()

    assert processed == [bad["id"], good["id"]]
    assert [job["id"] for job in snapshot["failed"]] == [bad["id"]]
    assert snapshot["failed"][0]["error"] == "RuntimeError: render exploded"
    assert [job["id"] for job in snapshot["completed"]] == [good["id"]]


def test_enqueue_rejects_empty_story(tmp_path: Path):
    run = tmp_path / "runs" / "2026-08-17_00-00-01_empty"
    run.mkdir(parents=True)
    (run / "story.md").write_text("", encoding="utf-8")

    queue = VideoJobQueue(tmp_path)

    try:
        queue.enqueue(run, ["--run-dir", str(run)])
    except ValueError as exc:
        assert "non-empty story.md" in str(exc)
    else:
        raise AssertionError("empty story must not be queued")
