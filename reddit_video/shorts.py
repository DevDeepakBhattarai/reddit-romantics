from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .tts_text import strip_speaker_metadata

SHORTS_CLIFFHANGER = "[[SHORTS_CLIFFHANGER]]"
_SPEAKER_TURN_RE = re.compile(r"^\s*Speaker\s+\d+\s*:\s*(.*)$", re.IGNORECASE)
_CUE_RE = re.compile(r"\[[^\]]+\]|<[^>]+>")
_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def split_story_at_cliffhanger(story_text: str) -> tuple[str, str | None]:
    """Return TTS-safe full story and the spoken prefix selected for the Short."""
    count = story_text.count(SHORTS_CLIFFHANGER)
    if count > 1:
        raise ValueError(f"Story must contain at most one {SHORTS_CLIFFHANGER} marker; found {count}.")
    if count == 0:
        return story_text.strip(), None

    before, after = story_text.split(SHORTS_CLIFFHANGER, 1)
    full = f"{before.rstrip()}\n{after.lstrip()}".strip()
    if not before.strip():
        raise ValueError(f"{SHORTS_CLIFFHANGER} cannot appear before any spoken story text.")
    if not after.strip():
        raise ValueError(f"{SHORTS_CLIFFHANGER} must leave story content after the cliffhanger.")
    return full, before.strip()


def _spoken_words(text: str) -> list[str]:
    text = strip_speaker_metadata(text)
    spoken_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        turn = _SPEAKER_TURN_RE.match(line)
        if turn:
            line = turn.group(1)
        line = _CUE_RE.sub(" ", line)
        spoken_lines.append(line)
    normalized = " ".join(spoken_lines).lower().replace("’", "'").replace("‘", "'")
    return _WORD_RE.findall(normalized)


def _transcript_words(data: dict[str, Any]) -> list[tuple[str, float]]:
    words = data.get("word_segments") or []
    if not words:
        for segment in data.get("segments", []) or []:
            words.extend(segment.get("words", []) or [])

    result: list[tuple[str, float]] = []
    for item in words:
        raw = str(item.get("word", "")).lower().replace("’", "'").replace("‘", "'")
        tokens = _WORD_RE.findall(raw)
        end = item.get("end")
        if end is None:
            continue
        for token in tokens:
            result.append((token, float(end)))
    return result


def _exact_tail_match(expected: list[str], actual: list[str]) -> int | None:
    max_anchor = min(18, len(expected))
    expected_end = len(expected) - 1
    for size in range(max_anchor, 3, -1):
        anchor = expected[-size:]
        matches = [
            start + size - 1
            for start in range(0, len(actual) - size + 1)
            if actual[start:start + size] == anchor
        ]
        if matches:
            return min(matches, key=lambda index: abs(index - expected_end))
    return None


def _fuzzy_tail_match(expected: list[str], actual: list[str]) -> tuple[int, float]:
    anchor = expected[-min(24, len(expected)):]
    expected_position = max(0, len(expected) - 1)
    best_index = -1
    best_score = -1.0
    min_size = max(4, len(anchor) - 5)
    max_size = min(len(actual), len(anchor) + 5)

    for size in range(min_size, max_size + 1):
        for start in range(0, len(actual) - size + 1):
            window = actual[start:start + size]
            similarity = SequenceMatcher(None, anchor, window, autojunk=False).ratio()
            end_index = start + size - 1
            distance = abs(end_index - expected_position)
            positional = max(0.0, 1.0 - distance / max(20.0, len(expected) * 0.35))
            score = similarity * 0.9 + positional * 0.1
            if score > best_score:
                best_score = score
                best_index = end_index
    return best_index, best_score


def resolve_cliffhanger_time(story_text: str, transcript_file: str | Path) -> float | None:
    """Map the story marker to the corresponding end timestamp in WhisperX output."""
    _full_story, prefix = split_story_at_cliffhanger(story_text)
    if prefix is None:
        return None

    expected = _spoken_words(prefix)
    if len(expected) < 4:
        raise ValueError("The Shorts marker needs at least four spoken words before it.")

    data = json.loads(Path(transcript_file).read_text(encoding="utf-8"))
    timed_words = _transcript_words(data)
    if not timed_words:
        raise ValueError(f"Transcript contains no word timestamps: {transcript_file}")
    actual = [word for word, _end in timed_words]

    index = _exact_tail_match(expected, actual)
    if index is None:
        index, score = _fuzzy_tail_match(expected, actual)
        if index < 0 or score < 0.72:
            raise RuntimeError(
                "Could not reliably align the Shorts cliffhanger marker with the transcript. "
                f"Best trailing-word match score was {score:.2f}."
            )

    cutoff = timed_words[index][1]
    if cutoff >= 120.0:
        raise ValueError(
            f"The Shorts cliffhanger resolves to {cutoff:.2f}s. Move {SHORTS_CLIFFHANGER} earlier so it is under 120s."
        )
    return cutoff
