from __future__ import annotations

from pathlib import Path

import pytest

from reddit_video.pipeline import (
    PipelineOptions,
    RedditVideoPipeline,
    list_background_videos,
    list_input_stories,
    slugify,
)


def test_slugify_produces_safe_output_name():
    assert slugify(" My dramatic story?! ") == "My_dramatic_story"
    assert slugify("***") == "story"


def test_repo_discovers_inputs_and_backgrounds():
    stories = list_input_stories()
    backgrounds = list_background_videos()
    assert any(item.endswith("input/test.txt") for item in stories)
    assert any(item.endswith("videos/minecraft/minecraft.mp4") for item in backgrounds)
    assert any("videos/asmr/" in item for item in backgrounds)


def test_story_text_is_written_for_reusable_pipeline(tmp_path: Path):
    (tmp_path / "input").mkdir()
    pipeline = RedditVideoPipeline(root=tmp_path, log=lambda _message: None)
    options = PipelineOptions(story_text="A complete story.", output_name="my output")

    path, text, base = pipeline._prepare_story(options)

    assert base == "my_output"
    assert text == "A complete story."
    assert path.read_text(encoding="utf-8") == "A complete story."


def test_local_whisperx_environment_is_preferred(tmp_path: Path):
    executable = tmp_path / ".whisperx-venv" / "Scripts" / "whisperx.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"")
    pipeline = RedditVideoPipeline(root=tmp_path, log=lambda _message: None)

    command, shell_prefix = pipeline._whisper_command(PipelineOptions())

    assert command == [str(executable)]
    assert shell_prefix is None


def test_narration_guard_rejects_catastrophic_truncation(tmp_path: Path):
    pipeline = RedditVideoPipeline(root=tmp_path, log=lambda _message: None)
    story = " ".join(["word"] * 100)

    with pytest.raises(RuntimeError, match="implausibly short"):
        pipeline._validate_narration_duration(story, 5.0)


def test_narration_guard_allows_plausible_audio(tmp_path: Path):
    pipeline = RedditVideoPipeline(root=tmp_path, log=lambda _message: None)
    story = " ".join(["word"] * 100)
    pipeline._validate_narration_duration(story, 30.0)
