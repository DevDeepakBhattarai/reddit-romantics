from __future__ import annotations

import pytest

from reddit_video.tts_text import (
    detect_speakers,
    normalize_decorators,
    prepare_text_for_provider,
    validate_speaker_count,
)

STORY = '''""" Speaker 0 - Maya, 28, narrator. Warm and quick-witted.
Speaker 1 - Adrian, 31. Deep voice.
Speaker 2 - Lena, 28. Energetic.

Speaker 0: I tried not to laugh. [giggle]
Speaker 1: Maya. <pause>
Speaker 2: Both of you are impossible. <laught>
'''


def test_detect_speakers_uses_metadata_descriptions_and_turns():
    speakers = detect_speakers(STORY)
    assert [speaker.speaker_id for speaker in speakers] == [0, 1, 2]
    assert speakers[0].description.startswith("Maya")
    assert speakers[1].description.startswith("Adrian")


def test_gemini_normalizes_common_decorator_aliases():
    text = normalize_decorators("Hi. [laugh] <giggle> <laught> [pause]", "gemini")
    assert text == "Hi. [laughs] [giggles] [laughs] [short pause]"


def test_vibevoice_removes_decorator_markup_instead_of_speaking_it():
    text = normalize_decorators("Hello [laugh] there <pause> friend.", "vibevoice")
    assert text == "Hello there friend."


def test_fish_converts_speaker_lines_to_native_tokens_and_keeps_tags():
    text, mapping = prepare_text_for_provider(STORY, "fish")
    assert mapping == {0: 0, 1: 1, 2: 2}
    assert "Speaker 0 - Maya" not in text
    assert "<|speaker:0|>I tried not to laugh. [chuckle]" in text
    assert "<|speaker:1|>Maya. [pause]" in text
    assert "<|speaker:2|>Both of you are impossible. [laugh]" in text


def test_vibevoice_remaps_sparse_speaker_ids_to_voice_sample_order():
    text, mapping = prepare_text_for_provider(
        "Speaker 4: Four. [laugh]\nSpeaker 9: Nine. <pause>",
        "vibevoice",
    )
    assert mapping == {4: 0, 9: 1}
    assert text == "Speaker 0: Four.\nSpeaker 1: Nine."


def test_gemini_rejects_three_speaker_story_but_vibevoice_accepts_it():
    with pytest.raises(ValueError, match="at most 2"):
        validate_speaker_count(STORY, "gemini")
    assert len(validate_speaker_count(STORY, "vibevoice")) == 3
