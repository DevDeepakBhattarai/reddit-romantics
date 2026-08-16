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
    assert slugify(" My dramatic story?! ") == "my-dramatic-story"
    assert slugify("***") == "story"


def test_repo_discovers_story_runs_and_backgrounds(tmp_path: Path):
    story = tmp_path / "runs" / "2026-08-16_16-30_test-story" / "story.md"
    minecraft = tmp_path / "videos" / "minecraft" / "minecraft.mp4"
    asmr = tmp_path / "videos" / "asmr" / "clip.mp4"
    for path in (story, minecraft, asmr):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test")

    assert list_input_stories(tmp_path) == ["runs/2026-08-16_16-30_test-story/story.md"]
    assert list_background_videos(tmp_path) == [
        "videos/asmr/clip.mp4",
        "videos/minecraft/minecraft.mp4",
    ]


def test_story_text_is_written_into_one_dated_run_folder(tmp_path: Path):
    pipeline = RedditVideoPipeline(root=tmp_path, log=lambda _message: None)
    options = PipelineOptions(story_text="A complete story.", output_name="my output")

    run, path, text, base = pipeline._prepare_story(options)

    assert base.endswith("_my-output")
    assert run.path.parent == tmp_path / "runs"
    assert path == run.path / "story.md"
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


def test_fish_requires_gender_metadata_when_no_override_exists(tmp_path: Path):
    pipeline = RedditVideoPipeline(root=tmp_path, log=lambda _message: None)
    story = "Speaker 0: Hello.\nSpeaker 1: Hi."
    ref = tmp_path / "speaker0.wav"
    ref.write_bytes(b"reference")
    options = PipelineOptions(
        story_text=story,
        tts_engine="fish",
        fish_speaker_references={0: (ref, "Reference transcript")},
    )

    with pytest.raises(ValueError, match="Speaker 1 has no gender metadata"):
        pipeline._generate_narration(options, tmp_path / "story.txt", story, tmp_path / "out.wav")


def test_pipeline_defaults_to_fish_audio():
    assert PipelineOptions().tts_engine == "fish"


def test_fish_auto_casts_by_character_gender_not_speaker_number(tmp_path: Path, monkeypatch):
    captured: dict[str, object] = {}
    for preset in ("Ethan", "Sarah"):
        folder = tmp_path / "voice_presets" / "fish" / preset
        folder.mkdir(parents=True)
        (folder / "sample.wav").write_bytes(preset.encode())
        (folder / "sample.lab").write_text(f"{preset} reference transcript", encoding="utf-8")

    def fake_generate(_root, text, output, **kwargs):
        captured["text"] = text
        captured.update(kwargs)
        return output

    monkeypatch.setattr(pipeline_module, "generate_fish_s2", fake_generate)
    logs: list[str] = []
    pipeline = RedditVideoPipeline(root=tmp_path, log=logs.append)
    story = (
        "Speaker 0 - gender=female; narrator; Maya\n"
        "Speaker 1 - gender=male; main counterpart; Adrian\n"
        "Speaker 0: I opened the door.\n"
        "Speaker 1: I was already waiting."
    )

    pipeline._generate_narration(
        PipelineOptions(story_text=story),
        tmp_path / "story.md",
        story,
        tmp_path / "out.wav",
    )

    refs = captured["speaker_references"]
    assert refs[0][0] == tmp_path / "voice_presets" / "fish" / "Sarah" / "sample.wav"
    assert refs[1][0] == tmp_path / "voice_presets" / "fish" / "Ethan" / "sample.wav"
    assert "Speaker 0=female->Sarah" in "\n".join(logs)
    assert "Speaker 1=male->Ethan" in "\n".join(logs)
    assert "gender=" not in captured["text"]





def test_short_reuses_existing_audio_and_transcript_without_tts_or_whisper(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "runs" / "2026-08-16_16-30_test-story"
    run_dir.mkdir(parents=True)
    (run_dir / "story.md").write_text(
        "Speaker 0: one two three four five six.\n"
        "[[SHORTS_CLIFFHANGER]]\n"
        "Speaker 0: seven eight nine.",
        encoding="utf-8",
    )
    (run_dir / "narration.wav").write_bytes(b"audio")
    (run_dir / "transcript.json").write_text(
        '{"word_segments":['
        '{"word":"one","start":0.0,"end":0.5},'
        '{"word":"two","start":0.5,"end":1.0},'
        '{"word":"three","start":1.0,"end":1.5},'
        '{"word":"four","start":1.5,"end":2.0},'
        '{"word":"five","start":2.0,"end":2.5},'
        '{"word":"six","start":2.5,"end":3.0},'
        '{"word":"seven","start":3.0,"end":3.5}]}' ,
        encoding="utf-8",
    )
    background = tmp_path / "videos" / "minecraft" / "minecraft.mp4"
    background.parent.mkdir(parents=True)
    background.write_bytes(b"video")

    pipeline = RedditVideoPipeline(root=tmp_path, log=lambda _message: None)
    monkeypatch.setattr(pipeline_module.shutil, "which", lambda _name: "tool.exe")
    monkeypatch.setattr(pipeline, "_audio_duration", lambda _path: 180.0)
    monkeypatch.setattr(pipeline, "_resolve_background", lambda _value: background)
    monkeypatch.setattr(
        pipeline,
        "_generate_narration",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("short must not run TTS")),
    )
    monkeypatch.setattr(
        pipeline,
        "_transcribe_and_style",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("short must not run WhisperX")),
    )

    def fake_extract(_source, output, _end):
        output.write_bytes(b"short-audio")
        return output

    def fake_render(_background, _audio, _caption, output, _options, _duration):
        output.write_bytes(b"short-video")

    monkeypatch.setattr(pipeline, "_extract_audio", fake_extract)
    monkeypatch.setattr(pipeline, "_render", fake_render)

    result = pipeline.render_short(run_dir, PipelineOptions(captions=False, background=background))

    assert result.video_path == run_dir / "short.mp4"
    assert result.audio_path == run_dir / "short.wav"
    assert result.whisper_json_path == run_dir / "short-transcript.json"
    assert result.short_end_seconds == 3.0
    assert result.video_path.read_bytes() == b"short-video"


def test_full_pipeline_strips_marker_and_automatically_renders_short(tmp_path: Path, monkeypatch):
    background = tmp_path / "videos" / "minecraft" / "minecraft.mp4"
    background.parent.mkdir(parents=True)
    background.write_bytes(b"video")
    pipeline = RedditVideoPipeline(root=tmp_path, log=lambda _message: None)
    story = (
        "Speaker 0: one two three four five six.\n"
        "[[SHORTS_CLIFFHANGER]]\n"
        "Speaker 0: seven eight nine ten."
    )
    captured: dict[str, str] = {}

    monkeypatch.setattr(pipeline_module.shutil, "which", lambda _name: "tool.exe")
    monkeypatch.setattr(pipeline, "_resolve_background", lambda _value: background)
    monkeypatch.setattr(pipeline, "_probe_dimensions", lambda _path: (1920, 1080))
    monkeypatch.setattr(pipeline, "_audio_duration", lambda _path: 10.0)

    def fake_tts(_options, _story_path, story_text, output):
        captured["tts_text"] = story_text
        output.write_bytes(b"audio")
        return output

    def fake_transcribe(_audio, _work, transcript, _caption, _options, _resolution):
        transcript.write_text(
            '{"word_segments":['
            '{"word":"one","start":0.0,"end":0.5},'
            '{"word":"two","start":0.5,"end":1.0},'
            '{"word":"three","start":1.0,"end":1.5},'
            '{"word":"four","start":1.5,"end":2.0},'
            '{"word":"five","start":2.0,"end":2.5},'
            '{"word":"six","start":2.5,"end":3.0},'
            '{"word":"seven","start":3.0,"end":3.5},'
            '{"word":"eight","start":3.5,"end":4.0}]}' ,
            encoding="utf-8",
        )
        return transcript

    def fake_render(_background, _audio, _caption, output, _options, _duration):
        output.write_bytes(b"video")

    def fake_extract(_source, output, _end):
        output.write_bytes(b"short-audio")
        return output

    monkeypatch.setattr(pipeline, "_generate_narration", fake_tts)
    monkeypatch.setattr(pipeline, "_transcribe_and_style", fake_transcribe)
    monkeypatch.setattr(pipeline, "_render", fake_render)
    monkeypatch.setattr(pipeline, "_extract_audio", fake_extract)

    result = pipeline.run(PipelineOptions(story_text=story, output_name="automatic short", captions=False))

    assert "[[SHORTS_CLIFFHANGER]]" not in captured["tts_text"]
    assert captured["tts_text"] == (
        "Speaker 0: one two three four five six.\nSpeaker 0: seven eight nine ten."
    )
    assert result.short_video_path == result.run_dir / "short.mp4"
    assert result.short_end_seconds == 3.0
    assert result.short_video_path.exists()
