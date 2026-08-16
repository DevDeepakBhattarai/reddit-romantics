from __future__ import annotations

import json

import pytest

from reddit_video.shorts import SHORTS_CLIFFHANGER, resolve_cliffhanger_time, split_story_at_cliffhanger


def test_split_story_marker_is_non_spoken_metadata():
    story = (
        "Speaker 0: The door opened.\n"
        f"{SHORTS_CLIFFHANGER}\n"
        "Speaker 1: And then I saw him."
    )
    spoken, prefix = split_story_at_cliffhanger(story)

    assert SHORTS_CLIFFHANGER not in spoken
    assert SHORTS_CLIFFHANGER not in prefix
    assert prefix == "Speaker 0: The door opened."
    assert "Speaker 1: And then I saw him." in spoken


def test_resolve_cliffhanger_time_tolerates_small_transcript_differences(tmp_path):
    story = (
        "Speaker 0: I opened the locked office and found my manager standing beside the safe.\n"
        f"{SHORTS_CLIFFHANGER}\n"
        "Speaker 0: He finally told me why."
    )
    transcript = tmp_path / "transcript.json"
    words = [
        ("I", 0.4), ("opened", 0.8), ("the", 1.0), ("locked", 1.4),
        ("office", 1.8), ("and", 2.0), ("found", 2.3), ("manager", 2.8),
        ("standing", 3.2), ("beside", 3.6), ("the", 3.8), ("safe", 4.2),
        ("he", 4.6), ("finally", 5.0),
    ]
    transcript.write_text(
        json.dumps({"word_segments": [
            {"word": word, "start": max(0, end - 0.2), "end": end} for word, end in words
        ]}),
        encoding="utf-8",
    )

    assert resolve_cliffhanger_time(story, transcript) == 4.2


def test_resolve_cliffhanger_rejects_marker_at_two_minutes(tmp_path):
    story = (
        "Speaker 0: one two three four five six.\n"
        f"{SHORTS_CLIFFHANGER}\n"
        "Speaker 0: after."
    )
    transcript = tmp_path / "transcript.json"
    transcript.write_text(
        json.dumps({"word_segments": [
            {"word": word, "start": index * 20.0, "end": (index + 1) * 20.0}
            for index, word in enumerate(["one", "two", "three", "four", "five", "six"])
        ]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="under 120s"):
        resolve_cliffhanger_time(story, transcript)
