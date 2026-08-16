from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


RUNS_DIR_NAME = "runs"


def slugify(value: str, fallback: str = "story") -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "-", value.strip().lower())
    value = value.strip("-")
    return value[:60] or fallback


def title_from_story(text: str) -> str:
    for raw_line in text.splitlines():
        line = re.sub(r"^Speaker\s+\d+\s*:\s*", "", raw_line.strip(), flags=re.IGNORECASE)
        line = re.sub(r"\[[^\]]+\]", "", line).strip()
        if line:
            words = re.findall(r"[A-Za-z0-9']+", line)
            if words:
                return " ".join(words[:8])
    return "story"


@dataclass(frozen=True)
class StoryRun:
    path: Path

    @property
    def story(self) -> Path:
        return self.path / "story.md"

    @property
    def narration(self) -> Path:
        return self.path / "narration.wav"

    @property
    def transcript(self) -> Path:
        return self.path / "transcript.json"

    @property
    def captions(self) -> Path:
        return self.path / "captions.ass"

    @property
    def full_video(self) -> Path:
        return self.path / "full.mp4"

    @property
    def short_audio(self) -> Path:
        return self.path / "short.wav"

    @property
    def short_transcript(self) -> Path:
        return self.path / "short-transcript.json"

    @property
    def short_captions(self) -> Path:
        return self.path / "short-captions.ass"

    @property
    def short_video(self) -> Path:
        return self.path / "short.mp4"

    @property
    def thumbnail(self) -> Path:
        return self.path / "thumbnail.png"


def create_story_run(root: Path, title: str, now: datetime | None = None) -> StoryRun:
    root = root.resolve()
    timestamp = (now or datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")
    base = f"{timestamp}_{slugify(title)}"
    runs_root = root / RUNS_DIR_NAME
    runs_root.mkdir(parents=True, exist_ok=True)

    candidate = runs_root / base
    suffix = 2
    while candidate.exists():
        candidate = runs_root / f"{base}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return StoryRun(candidate)


def resolve_story_run(root: Path, value: str | Path) -> StoryRun:
    root = root.resolve()
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if path.is_file():
        path = path.parent
    if not path.exists():
        raise FileNotFoundError(f"Story run not found: {path}")
    return StoryRun(path)


def list_story_runs(root: Path) -> list[str]:
    root = root.resolve()
    runs_root = root / RUNS_DIR_NAME
    if not runs_root.exists():
        return []
    paths = [path for path in runs_root.iterdir() if path.is_dir() and (path / "story.md").exists()]
    return [str(path.relative_to(root)).replace("\\", "/") for path in sorted(paths, reverse=True)]
