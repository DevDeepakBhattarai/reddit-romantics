from __future__ import annotations

import base64
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "gemini-tts" / "gemini_tts.py"
SPEC = importlib.util.spec_from_file_location("gemini_tts_module", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
gemini_tts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gemini_tts)


def test_latest_gemini_tts_model_is_the_default():
    assert gemini_tts.DEFAULT_MODEL == "gemini-3.1-flash-tts-preview"


def test_semantic_chunker_uses_sentence_boundaries_for_long_paragraphs():
    text = " ".join(
        f"This is complete sentence number {index}." for index in range(350)
    )

    chunks = gemini_tts.split_text_semantically(
        text, target_seconds=180, estimated_wpm=140
    )

    assert len(chunks) > 1
    assert all(chunk.endswith(".") for chunk in chunks)
    assert " ".join(chunks) == text
    assert all(gemini_tts._word_count(chunk) <= 420 for chunk in chunks)


def test_semantic_chunker_prefers_paragraph_boundaries():
    paragraph_one = "First paragraph stays coherent. " * 30
    paragraph_two = "Second paragraph also stays coherent. " * 30
    paragraph_three = "Third paragraph remains intact. " * 30
    text = f"{paragraph_one.strip()}\n\n{paragraph_two.strip()}\n\n{paragraph_three.strip()}"

    chunks = gemini_tts.split_text_semantically(
        text, target_seconds=90, estimated_wpm=140
    )

    assert len(chunks) >= 2
    # If a paragraph fits individually, the splitter never cuts through one just to fill a chunk.
    for paragraph in (
        paragraph_one.strip(),
        paragraph_two.strip(),
        paragraph_three.strip(),
    ):
        assert any(paragraph in chunk for chunk in chunks)


def test_explicit_separator_is_a_hard_boundary_but_can_be_ignored():
    text = "First section is short.\n\nStill first.\n-------------\nSecond section is short."

    hard_chunks = gemini_tts.split_text_semantically(
        text, respect_explicit_separator=True
    )
    automatic_only = gemini_tts.split_text_semantically(
        text, respect_explicit_separator=False
    )

    assert hard_chunks == [
        "First section is short.\n\nStill first.",
        "Second section is short.",
    ]
    assert len(automatic_only) == 1
    assert "-------------" not in automatic_only[0]
    assert "First section is short." in automatic_only[0]
    assert "Second section is short." in automatic_only[0]


def test_semantic_chunker_never_cuts_a_single_huge_token_mid_word():
    huge_token = "x" * 2000

    chunks = gemini_tts.split_text_semantically(
        huge_token, target_seconds=1, estimated_wpm=140
    )

    assert chunks == [huge_token]


def test_preprocess_preserves_paragraph_boundaries():
    text = "First paragraph: hello; world.\n\nSecond paragraph: still here."

    cleaned = gemini_tts.preprocess_text(text)

    assert "\n\n" in cleaned
    assert cleaned.startswith("First paragraph, hello, world.")
    assert cleaned.endswith("Second paragraph, still here.")


def test_generate_tts_audio_uses_interactions_api_and_decodes_audio():
    pcm = b"\x01\x02\x03\x04"
    calls: list[dict] = []

    class FakeInteractions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                output_audio=SimpleNamespace(data=base64.b64encode(pcm).decode("ascii"))
            )

    client = SimpleNamespace(interactions=FakeInteractions())

    result = gemini_tts.generate_tts_audio(
        client, "Hello from the story.", voice_name="Kore"
    )

    assert result == pcm
    assert calls[0]["model"] == "gemini-3.1-flash-tts-preview"
    assert calls[0]["response_format"] == {"type": "audio"}
    assert calls[0]["generation_config"] == {"speech_config": [{"voice": "Kore"}]}
    assert "Speak only the text under TRANSCRIPT" in calls[0]["input"]
    assert calls[0]["input"].endswith("Hello from the story.")


def test_generate_tts_audio_retries_transient_500(monkeypatch: pytest.MonkeyPatch):
    pcm = b"\x00\x00"

    class FakeInteractions:
        def __init__(self):
            self.attempts = 0

        def create(self, **_kwargs):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("500 INTERNAL server error")
            return SimpleNamespace(
                output_audio=SimpleNamespace(data=base64.b64encode(pcm).decode("ascii"))
            )

    interactions = FakeInteractions()
    client = SimpleNamespace(interactions=interactions)
    monkeypatch.setattr(gemini_tts.time, "sleep", lambda _seconds: None)

    assert gemini_tts.generate_tts_audio(client, "Retry me.", max_retries=3) == pcm
    assert interactions.attempts == 2
