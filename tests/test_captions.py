from __future__ import annotations

import json

import pytest

from reddit_video.captions import (
    CAPTION_THEMES,
    convert_whisperx_json_to_ass,
    format_time,
    trim_whisperx_json,
)


@pytest.fixture
def whisperx_json(tmp_path):
    path = tmp_path / "words.json"
    path.write_text(
        json.dumps(
            {
                "word_segments": [
                    {"word": "This", "start": 0.0, "end": 0.2},
                    {"word": "is", "start": 0.21, "end": 0.32},
                    {"word": "a", "start": 0.33, "end": 0.38},
                    {"word": "story", "start": 0.39, "end": 0.7},
                    {"word": "ending", "start": 1.4, "end": 1.8},
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("theme_name", sorted(CAPTION_THEMES))
def test_every_caption_theme_generates_valid_ass(tmp_path, whisperx_json, theme_name):
    output = tmp_path / f"{theme_name}.ass"
    convert_whisperx_json_to_ass(whisperx_json, output, theme_name=theme_name)

    text = output.read_text(encoding="utf-8")
    theme = CAPTION_THEMES[theme_name]
    assert "[V4+ Styles]" in text
    assert "[Events]" in text
    assert f"Style: Default,{theme.font},{theme.font_size}" in text
    assert theme.active in text
    assert text.count("Dialogue: 0,") == 5


def test_pause_creates_a_new_word_group(tmp_path, whisperx_json):
    output = tmp_path / "captions.ass"
    convert_whisperx_json_to_ass(
        whisperx_json,
        output,
        theme_name="classic_yellow",
        max_words=8,
        pause_threshold=0.5,
    )
    text = output.read_text(encoding="utf-8")

    first_word_event = next(line for line in text.splitlines() if line.startswith("Dialogue:"))
    # The active word is wrapped in ASS color tags, so assert on the rest of the group.
    assert "is a story" in first_word_event
    assert "ending" not in first_word_event


def test_long_caption_timestamp_rolls_over_cleanly():
    assert format_time(3599.999) == "1:00:00.00"
    assert format_time(5400.0) == "1:30:00.00"



def test_trim_whisperx_json_keeps_only_words_before_cliffhanger(tmp_path):
    source = tmp_path / "transcript.json"
    target = tmp_path / "short-transcript.json"
    source.write_text(
        json.dumps({
            "word_segments": [
                {"word": "one", "start": 1.0, "end": 1.3},
                {"word": "two", "start": 59.0, "end": 60.5},
                {"word": "three", "start": 61.0, "end": 61.2},
            ],
            "segments": [
                {"start": 0.0, "end": 62.0, "text": "one two three", "words": [
                    {"word": "one", "start": 1.0, "end": 1.3},
                    {"word": "two", "start": 59.0, "end": 60.5},
                    {"word": "three", "start": 61.0, "end": 61.2},
                ]}
            ],
        }),
        encoding="utf-8",
    )

    trim_whisperx_json(source, target, 60.0)
    data = json.loads(target.read_text(encoding="utf-8"))

    assert [word["word"] for word in data["word_segments"]] == ["one", "two"]
    assert data["word_segments"][-1]["end"] == 60.0
    assert [word["word"] for word in data["segments"][0]["words"]] == ["one", "two"]
    assert data["segments"][0]["end"] == 60.0
