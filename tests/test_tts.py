from __future__ import annotations

import pytest

from reddit_video.tts import _format_vibevoice_script


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
