from __future__ import annotations

import wave
from pathlib import Path

from reddit_video import fish


def _write_silent_wav(path: Path, seconds: float = 0.1, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * frames)


def _create_hybrid_runtime(root: Path) -> tuple[Path, Path, Path]:
    s2_root = root / "vendor" / "s2.cpp"
    binary = s2_root / "build-cuda" / "bin" / "Release" / "s2.exe"
    tokenizer = s2_root / "tokenizer.json"
    model = root / ".work" / "fish-native" / "s2-pro-f16.gguf"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"s2")
    tokenizer.write_text("{}", encoding="utf-8")
    model.parent.mkdir(parents=True)
    model.write_bytes(b"full-f16-model")
    return binary, tokenizer, model


def test_fish_hybrid_uses_unquantized_f16_partial_cuda_offload(tmp_path, monkeypatch):
    _binary, _tokenizer, model = _create_hybrid_runtime(tmp_path)
    output = tmp_path / "fish-hybrid.wav"
    seen: list[str] = []
    logs: list[str] = []
    monkeypatch.setenv("TTS_FISH_CPU_THREADS", "16")

    def fake_run(command, _cwd, _log):
        seen.extend(command)
        _write_silent_wav(output)

    monkeypatch.setattr(fish, "_run", fake_run)
    result = fish.generate_fish_s2(
        tmp_path,
        "A short full-precision hybrid Fish test.",
        output,
        gpu_layers=28,
        temperature=0.9,
        log=logs.append,
    )

    assert result == output
    assert str(model) in seen
    assert seen[seen.index("--cuda") + 1] == "0"
    assert seen[seen.index("--gpu-layers") + 1] == "28"
    assert "--codec-follow-backend" in seen
    assert seen[seen.index("--threads") + 1] == "16"
    assert not any("q6" in arg.lower() or "q8" in arg.lower() or "q4" in arg.lower() for arg in seen)
    assert any("unquantized F16" in line for line in logs)


def test_fish_long_form_defaults_to_500_character_semantic_budget(monkeypatch):
    monkeypatch.delenv("TTS_FISH_LONG_FORM_CHARS", raising=False)
    monkeypatch.delenv("TTS_FISH_SEGMENT_MAX_CHARS", raising=False)
    assert fish._fish_long_form_threshold() == 500
    assert fish._fish_segment_max_chars() == 500


def test_fish_long_form_repeats_active_speaker_for_every_segment():
    text = (
        "<|speaker:0|>First sentence. Second sentence!\n"
        "<|speaker:1|>Third sentence? Fourth sentence."
    )
    assert fish._fish_retag_segmented_text(text).splitlines() == [
        "<|speaker:0|>First sentence.",
        "<|speaker:0|>Second sentence!",
        "<|speaker:1|>Third sentence?",
        "<|speaker:1|>Fourth sentence.",
    ]


def test_fish_long_form_patch_packs_complete_sentences_by_character_budget():
    patch = Path(__file__).parents[1] / "patches" / "s2cpp-batched-segments.patch"
    contents = patch.read_text(encoding="utf-8")
    assert "segment_group_sentences" in contents
    assert "sentences_per_segment == 0" in contents
    assert "semantic_text_length" in contents
    assert "current_semantic_chars + separator_chars + sentence_semantic_chars > max_chars_per_segment" in contents
    assert "current += sentence" in contents


def test_fish_hybrid_long_form_uses_segmented_server_once(tmp_path, monkeypatch):
    binary, _tokenizer, model = _create_hybrid_runtime(tmp_path)
    output = tmp_path / "fish-long.wav"
    reference = tmp_path / "voice.wav"
    _write_silent_wav(reference)
    captured: dict[str, object] = {}
    monkeypatch.setenv("TTS_FISH_LONG_FORM_CHARS", "100")
    monkeypatch.setattr(
        fish,
        "_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("long form must not use one-shot CLI")),
    )

    def fake_segmented_server(**kwargs):
        captured.update(kwargs)
        _write_silent_wav(output)
        return output

    monkeypatch.setattr(fish, "_run_fish_s2_segmented_server", fake_segmented_server)
    result = fish.generate_fish_s2(
        tmp_path,
        "<|speaker:0|>" + ("This is a long Fish narration sentence. " * 8),
        output,
        gpu_layers=28,
        temperature=0.9,
        reference_audio=reference,
        reference_text="Reference transcript.",
        log=lambda _message: None,
    )

    assert result == output
    assert captured["binary"] == binary
    assert captured["model"] == model
    assert captured["gpu_layers"] == 28
    assert captured["reference_audio"] == reference
    assert captured["reference_text"] == "Reference transcript."


def test_fish_runtime_status_has_no_download_side_effects(tmp_path):
    status = fish.fish_runtime_status(tmp_path)
    assert "not built" in status.lower()
    assert not (tmp_path / "vendor").exists()
    assert not (tmp_path / ".work").exists()


def test_fish_saved_reference_presets_are_listed_and_resolved(tmp_path: Path):
    preset = tmp_path / "voice_presets" / "fish" / "deep-male"
    preset.mkdir(parents=True)
    audio = preset / "sample.wav"
    audio.write_bytes(b"wav")
    (preset / "sample.lab").write_text("This is the exact reference transcript.", encoding="utf-8")

    assert fish.list_fish_reference_presets(tmp_path) == ["deep-male"]
    resolved_audio, resolved_text = fish.resolve_fish_reference_preset(tmp_path, "deep-male")
    assert resolved_audio == audio
    assert resolved_text == "This is the exact reference transcript."


def test_fish_uploaded_reference_is_cached_for_future_sessions(tmp_path: Path):
    source = tmp_path / "girl voice.wav"
    source.write_bytes(b"voice-bytes")
    preset_id, cached_audio, cached_text = fish.cache_fish_reference_preset(
        tmp_path, source, "Hello from the reference clip.", "e-girl"
    )
    assert preset_id == "e-girl"
    assert cached_audio == tmp_path / "voice_presets" / "fish" / "e-girl" / "sample.wav"
    assert cached_audio.read_bytes() == b"voice-bytes"
    assert cached_text == "Hello from the reference clip."
    assert fish.list_fish_reference_presets(tmp_path) == ["e-girl"]
    assert fish.resolve_fish_reference_preset(tmp_path, "e-girl") == (cached_audio, cached_text)


def test_fish_cache_deduplicates_identical_unnamed_uploaded_voice(tmp_path: Path):
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    first.write_bytes(b"same-voice")
    second.write_bytes(b"same-voice")
    first_id, first_audio, _ = fish.cache_fish_reference_preset(tmp_path, first, "Same transcript")
    second_id, second_audio, _ = fish.cache_fish_reference_preset(tmp_path, second, "Same transcript")
    assert first_id == second_id == "first"
    assert first_audio == second_audio
    assert fish.list_fish_reference_presets(tmp_path) == ["first"]


def test_fish_explicit_preset_name_is_always_honored(tmp_path: Path):
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    first.write_bytes(b"same-voice")
    second.write_bytes(b"same-voice")
    fish.cache_fish_reference_preset(tmp_path, first, "Same transcript", "old-name")
    preset_id, _audio, _text = fish.cache_fish_reference_preset(
        tmp_path, second, "Same transcript", "e-girl"
    )
    assert preset_id == "e-girl"
    assert fish.list_fish_reference_presets(tmp_path) == ["e-girl", "old-name"]
