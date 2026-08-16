from __future__ import annotations

import wave
from pathlib import Path

from reddit_video import tts_models
from reddit_video.pipeline import PipelineOptions, RedditVideoPipeline


def _write_silent_wav(path: Path, seconds: float = 0.1, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * frames)


def test_audio_only_pipeline_never_needs_a_background(tmp_path, monkeypatch):
    pipeline = RedditVideoPipeline(root=tmp_path, log=lambda _message: None)

    def fake_generate(_options, _story_path, _story_text, audio_path):
        _write_silent_wav(audio_path)
        return audio_path

    monkeypatch.setattr(pipeline, "_generate_narration", fake_generate)

    result = pipeline.run_audio(PipelineOptions(story_text="A short audio-only preview.", tts_engine="magpie"))

    assert result.audio_path.exists()
    assert result.audio_path.suffix == ".wav"
    assert not (tmp_path / "output").exists()


def test_fish_cpu_uses_multicore_thread_environment(tmp_path, monkeypatch):
    root = tmp_path / "vendor" / "fish-speech"
    script = root / "fish_speech" / "models" / "text2semantic" / "inference.py"
    checkpoint = root / "checkpoints" / "s2-pro"
    runtime = root / ".venv" / "bin" / "python"
    pyvenv = root / ".venv" / "pyvenv.cfg"
    script.parent.mkdir(parents=True)
    script.write_text("# runner", encoding="utf-8")
    checkpoint.mkdir(parents=True)
    runtime.parent.mkdir(parents=True)
    runtime.write_text("", encoding="utf-8")
    pyvenv.write_text("home = /usr/bin", encoding="utf-8")
    output = tmp_path / "fish.wav"
    seen: list[str] = []
    logs: list[str] = []

    monkeypatch.setenv("TTS_FISH_CPU_THREADS", "16")
    monkeypatch.setattr(tts_models.shutil, "which", lambda name: "wsl.exe" if name == "wsl.exe" else None)
    monkeypatch.setattr(tts_models, "_wsl_path", lambda path: "/wsl/" + path.name)

    def fake_run(command, _cwd, _log):
        seen.extend(command)
        _write_silent_wav(output)

    monkeypatch.setattr(tts_models, "_run", fake_run)

    result = tts_models.generate_fish_s2(
        tmp_path,
        "A short Fish CPU test.",
        output,
        device="cpu",
        log=logs.append,
    )

    assert result == output
    assert "env" in seen
    assert "OMP_NUM_THREADS=16" in seen
    assert "MKL_NUM_THREADS=16" in seen
    assert "OPENBLAS_NUM_THREADS=16" in seen
    assert "NUMEXPR_NUM_THREADS=16" in seen
    assert any("16 compute threads" in line for line in logs)


def test_fish_prefers_native_wsl_runtime_and_checkpoint(tmp_path, monkeypatch):
    root = tmp_path / "vendor" / "fish-speech"
    script = root / "fish_speech" / "models" / "text2semantic" / "inference.py"
    script.parent.mkdir(parents=True)
    script.write_text("# runner", encoding="utf-8")
    marker = tmp_path / ".work" / "fish-wsl-native-ready"
    marker.parent.mkdir(parents=True)
    marker.write_text("wsl-native-v1", encoding="utf-8")
    output = tmp_path / "fish-native.wav"
    seen: list[str] = []

    monkeypatch.setattr(tts_models, "_wsl_home", lambda: "/home/test")
    monkeypatch.setattr(tts_models.shutil, "which", lambda name: "wsl.exe" if name == "wsl.exe" else None)
    monkeypatch.setattr(tts_models, "_wsl_path", lambda path: "/mnt/project/" + path.name)

    def fake_run(command, _cwd, _log):
        seen.extend(command)
        _write_silent_wav(output)

    monkeypatch.setattr(tts_models, "_run", fake_run)

    result = tts_models.generate_fish_s2(
        tmp_path,
        "Fast native WSL test.",
        output,
        device="cpu",
        log=lambda _message: None,
    )

    assert result == output
    assert "/home/test/.cache/reddit-romantics/fish/.venv/bin/python" in seen
    assert "/home/test/.cache/reddit-romantics/fish/checkpoints/s2-pro" in seen
    assert not any("vendor/fish-speech/.venv" in part for part in seen)


def test_fish_hybrid_uses_unquantized_f16_partial_cuda_offload(tmp_path, monkeypatch):
    s2_root = tmp_path / "vendor" / "s2.cpp"
    binary = s2_root / "build-cuda" / "bin" / "Release" / "s2.exe"
    tokenizer = s2_root / "tokenizer.json"
    model = tmp_path / ".work" / "fish-native" / "s2-pro-f16.gguf"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"s2")
    tokenizer.write_text("{}", encoding="utf-8")
    model.parent.mkdir(parents=True)
    model.write_bytes(b"full-f16-model")
    output = tmp_path / "fish-hybrid.wav"
    seen: list[str] = []
    logs: list[str] = []

    monkeypatch.setenv("TTS_FISH_CPU_THREADS", "16")

    def fake_run(command, _cwd, _log):
        seen.extend(command)
        _write_silent_wav(output)

    monkeypatch.setattr(tts_models, "_run", fake_run)

    result = tts_models.generate_fish_s2(
        tmp_path,
        "A short full-precision hybrid Fish test.",
        output,
        device="hybrid",
        gpu_layers=20,
        temperature=0.9,
        log=logs.append,
    )

    assert result == output
    assert str(model) in seen
    assert "--cuda" in seen and seen[seen.index("--cuda") + 1] == "0"
    assert "--gpu-layers" in seen and seen[seen.index("--gpu-layers") + 1] == "20"
    assert "--codec-follow-backend" in seen
    assert "--threads" in seen and seen[seen.index("--threads") + 1] == "16"
    assert not any("q6" in arg.lower() or "q8" in arg.lower() or "q4" in arg.lower() for arg in seen)
    assert any("unquantized F16" in line for line in logs)


def test_step_prefers_awq_and_memory_saving_flags(tmp_path, monkeypatch):
    root = tmp_path / "vendor" / "Step-Audio-EditX"
    (root / "models" / "Step-Audio-EditX-AWQ-4bit").mkdir(parents=True)
    (root / "models" / "Step-Audio-Tokenizer").mkdir(parents=True)
    (root / "tts_infer.py").write_text("# runner", encoding="utf-8")
    runtime = tmp_path / ".tts-venvs" / "step" / "Scripts" / "python.exe"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("", encoding="utf-8")
    reference = tmp_path / "reference.wav"
    _write_silent_wav(reference)
    output = tmp_path / "preview.wav"
    seen: list[str] = []

    def fake_run(command, _cwd, _log):
        seen.extend(command)
        output_dir = Path(command[command.index("--output-dir") + 1])
        _write_silent_wav(output_dir / "step_output.wav")

    monkeypatch.setattr(tts_models, "_run", fake_run)

    result = tts_models.generate_step_editx(
        tmp_path,
        "I couldn't believe it [laugh], but there it was.",
        output,
        reference_audio=reference,
        reference_text="This is the reference voice.",
        mode="paralinguistic",
        log=lambda _message: None,
    )

    assert result == output
    assert "--quantization" in seen
    assert seen[seen.index("--quantization") + 1] == "awq"
    assert "--enforce-eager" in seen
    assert "--no-cosyvoice-cuda-graph" in seen
    assert seen[seen.index("--edit-type") + 1] == "paralinguistic"


def test_runtime_status_does_not_import_or_download_models(tmp_path):
    statuses = tts_models.backend_runtime_status(tmp_path)

    assert set(statuses) == {"fish", "step", "magpie", "chatterbox", "higgs"}
    assert all(
        "not installed" in value.lower() or "incomplete" in value.lower()
        for value in statuses.values()
    )


def test_fish_saved_reference_presets_are_listed_and_resolved(tmp_path: Path):
    preset = tmp_path / "vendor" / "fish-speech" / "references" / "deep-male"
    preset.mkdir(parents=True)
    audio = preset / "sample.wav"
    audio.write_bytes(b"wav")
    (preset / "sample.lab").write_text("This is the exact reference transcript.", encoding="utf-8")

    assert tts_models.list_fish_reference_presets(tmp_path) == ["deep-male"]
    resolved_audio, resolved_text = tts_models.resolve_fish_reference_preset(tmp_path, "deep-male")
    assert resolved_audio == audio
    assert resolved_text == "This is the exact reference transcript."


def test_fish_uploaded_reference_is_cached_for_future_sessions(tmp_path: Path):
    source = tmp_path / "girl voice.wav"
    source.write_bytes(b"voice-bytes")

    preset_id, cached_audio, cached_text = tts_models.cache_fish_reference_preset(
        tmp_path, source, "Hello from the reference clip.", "e-girl"
    )

    assert preset_id == "e-girl"
    assert cached_audio == tmp_path / "voice_presets" / "fish" / "e-girl" / "sample.wav"
    assert cached_audio.read_bytes() == b"voice-bytes"
    assert cached_text == "Hello from the reference clip."
    assert tts_models.list_fish_reference_presets(tmp_path) == ["e-girl"]
    resolved_audio, resolved_text = tts_models.resolve_fish_reference_preset(tmp_path, "e-girl")
    assert resolved_audio == cached_audio
    assert resolved_text == cached_text


def test_fish_cache_deduplicates_identical_unnamed_uploaded_voice(tmp_path: Path):
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    first.write_bytes(b"same-voice")
    second.write_bytes(b"same-voice")

    first_id, first_audio, _ = tts_models.cache_fish_reference_preset(
        tmp_path, first, "Same transcript"
    )
    second_id, second_audio, _ = tts_models.cache_fish_reference_preset(
        tmp_path, second, "Same transcript"
    )

    assert first_id == second_id == "first"
    assert first_audio == second_audio
    assert tts_models.list_fish_reference_presets(tmp_path) == ["first"]


def test_fish_explicit_preset_name_is_always_honored(tmp_path: Path):
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    first.write_bytes(b"same-voice")
    second.write_bytes(b"same-voice")

    tts_models.cache_fish_reference_preset(tmp_path, first, "Same transcript", "old-name")
    preset_id, _audio, _text = tts_models.cache_fish_reference_preset(
        tmp_path, second, "Same transcript", "e-girl"
    )

    assert preset_id == "e-girl"
    assert tts_models.list_fish_reference_presets(tmp_path) == ["e-girl", "old-name"]


def test_user_fish_preset_takes_precedence_over_native_reference(tmp_path: Path):
    user = tmp_path / "voice_presets" / "fish" / "same-name"
    native = tmp_path / "vendor" / "fish-speech" / "references" / "same-name"
    user.mkdir(parents=True)
    native.mkdir(parents=True)
    (user / "sample.wav").write_bytes(b"user")
    (user / "sample.lab").write_text("user transcript", encoding="utf-8")
    (native / "sample.wav").write_bytes(b"native")
    (native / "sample.lab").write_text("native transcript", encoding="utf-8")

    audio, text = tts_models.resolve_fish_reference_preset(tmp_path, "same-name")
    assert audio.read_bytes() == b"user"
    assert text == "user transcript"
