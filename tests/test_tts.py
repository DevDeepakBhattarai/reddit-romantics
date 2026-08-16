from __future__ import annotations

from pathlib import Path

import pytest

from reddit_video import tts
from reddit_video.tts import (
    _format_vibevoice_script,
    get_gemini_voice_preview,
    get_vibevoice_voice_preview,
)


def test_vibevoice_story_is_one_valid_speaker_line():
    story = "First paragraph.\n\nSecond paragraph.\nThird line with: a colon."

    script = _format_vibevoice_script(story)

    assert script == (
        "Speaker 1: First paragraph.\n"
        "Speaker 1: Second paragraph.\n"
        "Speaker 1: Third line with: a colon."
    )
    assert all(line.startswith("Speaker 1: ") for line in script.splitlines())


def test_vibevoice_story_normalizes_smart_quotes():
    script = _format_vibevoice_script('“Hello,” she said. It’s fine.')
    assert script == 'Speaker 1: "Hello," she said. It\'s fine.'


def test_vibevoice_story_rejects_blank_input():
    with pytest.raises(ValueError, match="empty"):
        _format_vibevoice_script("  \n \t")


def test_vibevoice_preview_is_the_exact_reference_wav(tmp_path):
    voice_dir = tmp_path / "vendor" / "VibeVoice" / "demo" / "voices"
    voice_dir.mkdir(parents=True)
    expected = voice_dir / "en-Alice_woman.wav"
    expected.write_bytes(b"RIFF" + b"\x00" * 64)

    preview = get_vibevoice_voice_preview(tmp_path, "Alice")

    assert preview == expected


def test_gemini_preview_reuses_existing_cache_without_generation(tmp_path, monkeypatch):
    script = tmp_path / "gemini-tts" / "gemini_tts.py"
    script.parent.mkdir(parents=True)
    script.write_text("# placeholder", encoding="utf-8")
    preview = (
        tmp_path
        / ".work"
        / "voice_previews"
        / "gemini"
        / "gemini-3.1-flash-tts-preview__Kore.wav"
    )
    preview.parent.mkdir(parents=True)
    preview.write_bytes(b"RIFF" + b"\x00" * 64)

    def fail_run(*_args, **_kwargs):
        raise AssertionError("cached previews must not call Gemini")

    monkeypatch.setattr(tts, "_run", fail_run)

    result = get_gemini_voice_preview(
        tmp_path,
        "Kore",
        "gemini-3.1-flash-tts-preview",
        log=lambda _message: None,
    )

    assert result == preview


def test_gemini_preview_generates_once_and_caches(tmp_path, monkeypatch):
    script = tmp_path / "gemini-tts" / "gemini_tts.py"
    script.parent.mkdir(parents=True)
    script.write_text("# placeholder", encoding="utf-8")
    calls = []

    def fake_run(command, cwd, log):
        calls.append(command)
        text_file = command[command.index("--text_file") + 1]
        generated = tmp_path / "gemini-tts" / "output" / f"{Path(text_file).stem}.wav"
        generated.parent.mkdir(parents=True, exist_ok=True)
        generated.write_bytes(b"RIFF" + b"\x00" * 64)

    monkeypatch.setattr(tts, "_run", fake_run)

    first = get_gemini_voice_preview(
        tmp_path,
        "Aoede",
        "gemini-3.1-flash-tts-preview",
        log=lambda _message: None,
    )
    second = get_gemini_voice_preview(
        tmp_path,
        "Aoede",
        "gemini-3.1-flash-tts-preview",
        log=lambda _message: None,
    )

    assert first == second
    assert first.exists()
    assert len(calls) == 1
    assert "--voice" in calls[0]
    assert calls[0][calls[0].index("--voice") + 1] == "Aoede"
