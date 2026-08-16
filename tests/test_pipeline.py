from __future__ import annotations

from pathlib import Path

import pytest

from reddit_video import pipeline as pipeline_module
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



def test_gemini_rejects_story_with_more_than_two_speakers(tmp_path: Path):
    pipeline = RedditVideoPipeline(root=tmp_path, log=lambda _message: None)
    story = "Speaker 0: A.\nSpeaker 1: B.\nSpeaker 2: C."

    with pytest.raises(ValueError, match="at most 2"):
        pipeline._generate_narration(
            PipelineOptions(story_text=story, tts_engine="gemini"),
            tmp_path / "story.txt",
            story,
            tmp_path / "out.wav",
        )


def test_vibevoice_pipeline_strips_decorators_and_maps_per_speaker_voices(tmp_path: Path, monkeypatch):
    captured: dict[str, object] = {}

    def fake_generate(_root, text, output, **kwargs):
        captured["text"] = text
        captured.update(kwargs)
        return output

    monkeypatch.setattr(pipeline_module, "generate_vibevoice", fake_generate)
    pipeline = RedditVideoPipeline(root=tmp_path, log=lambda _message: None)
    story = "Speaker 3: Hello. [laugh]\nSpeaker 8: Hi. <pause>"
    options = PipelineOptions(
        story_text=story,
        tts_engine="vibevoice",
        vibevoice_speaker_voices={3: "Alice", 8: "Frank"},
    )

    pipeline._generate_narration(options, tmp_path / "story.txt", story, tmp_path / "out.wav")

    assert captured["text"] == "Speaker 0: Hello.\nSpeaker 1: Hi."
    assert captured["speaker_names"] == {0: "Alice", 1: "Frank"}


def test_fish_pipeline_uses_native_speaker_tokens_and_remaps_references(tmp_path: Path, monkeypatch):
    captured: dict[str, object] = {}
    ref_a = tmp_path / "a.wav"
    ref_b = tmp_path / "b.wav"
    ref_a.write_bytes(b"a")
    ref_b.write_bytes(b"b")

    def fake_generate(_root, text, output, **kwargs):
        captured["text"] = text
        captured.update(kwargs)
        return output

    monkeypatch.setattr(pipeline_module, "generate_fish_s2", fake_generate)
    pipeline = RedditVideoPipeline(root=tmp_path, log=lambda _message: None)
    story = "Speaker 4: First. [giggle]\nSpeaker 9: Second. <laught>"
    options = PipelineOptions(
        story_text=story,
        tts_engine="fish",
        fish_speaker_references={4: (ref_a, "Reference A"), 9: (ref_b, "Reference B")},
    )

    pipeline._generate_narration(options, tmp_path / "story.txt", story, tmp_path / "out.wav")

    assert captured["text"] == "<|speaker:0|>First. [chuckle]\n<|speaker:1|>Second. [laugh]"
    assert captured["speaker_references"] == {
        0: (ref_a, "Reference A"),
        1: (ref_b, "Reference B"),
    }


def test_gemini_requires_voice_for_every_detected_speaker(tmp_path: Path):
    pipeline = RedditVideoPipeline(root=tmp_path, log=lambda _message: None)
    story = "Speaker 0: Hello.\nSpeaker 1: Hi."
    options = PipelineOptions(
        story_text=story,
        tts_engine="gemini",
        gemini_speaker_voices={0: "Kore"},
    )

    with pytest.raises(ValueError, match="explicit voice preset.*Speaker 1"):
        pipeline._generate_narration(options, tmp_path / "story.txt", story, tmp_path / "out.wav")


def test_vibevoice_requires_preset_for_every_detected_speaker(tmp_path: Path):
    pipeline = RedditVideoPipeline(root=tmp_path, log=lambda _message: None)
    story = "Speaker 0: Hello.\nSpeaker 1: Hi."
    options = PipelineOptions(
        story_text=story,
        tts_engine="vibevoice",
        vibevoice_speaker_voices={0: "Alice"},
    )

    with pytest.raises(ValueError, match="explicit preset.*Speaker 1"):
        pipeline._generate_narration(options, tmp_path / "story.txt", story, tmp_path / "out.wav")


def test_fish_never_allows_random_voice_for_detected_speaker(tmp_path: Path):
    pipeline = RedditVideoPipeline(root=tmp_path, log=lambda _message: None)
    story = "Speaker 0: Hello.\nSpeaker 1: Hi."
    ref = tmp_path / "speaker0.wav"
    ref.write_bytes(b"reference")
    options = PipelineOptions(
        story_text=story,
        tts_engine="fish",
        fish_speaker_references={0: (ref, "Reference transcript")},
    )

    with pytest.raises(ValueError, match="random model-selected voices are disabled.*Speaker 1"):
        pipeline._generate_narration(options, tmp_path / "story.txt", story, tmp_path / "out.wav")
