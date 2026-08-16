from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import warnings
from collections.abc import Callable
from pathlib import Path

LogFn = Callable[[str], None]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".work" / "hf-cache"))

_VIBEVOICE_CACHE: dict[tuple[str, str, str], tuple[object, object]] = {}
_GEMINI_PREVIEW_TEXT = (
    "Some stories start quietly, right before everything changes. "
    "This is where ours begins."
)


def _run(command: list[str], cwd: Path, log: LogFn) -> None:
    log("$ " + subprocess.list2cmdline(command))
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        stripped = line.rstrip()
        if stripped:
            log(stripped)
    code = process.wait()
    if code != 0:
        raise RuntimeError(f"Command failed with exit code {code}: {subprocess.list2cmdline(command)}")


def _safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return safe.strip("._") or "default"


def get_gemini_voice_preview(
    project_root: Path,
    voice: str,
    model: str,
    log: LogFn,
) -> Path:
    """Return a cached short Gemini voice preview, generating it once when needed."""
    voice = voice.strip()
    model = model.strip()
    if not voice:
        raise ValueError("Select a Gemini voice before loading a preview.")
    if not model:
        raise ValueError("Enter a Gemini TTS model before loading a preview.")

    cache_dir = project_root / ".work" / "voice_previews" / "gemini"
    cache_dir.mkdir(parents=True, exist_ok=True)
    preview_path = cache_dir / f"{_safe_filename(model)}__{_safe_filename(voice)}.wav"
    if preview_path.exists() and preview_path.stat().st_size > 44:
        return preview_path

    script = project_root / "gemini-tts" / "gemini_tts.py"
    if not script.exists():
        raise FileNotFoundError(f"Gemini TTS script is missing: {script}")

    gemini_input = project_root / "gemini-tts" / "input"
    gemini_output = project_root / "gemini-tts" / "output"
    gemini_input.mkdir(parents=True, exist_ok=True)
    gemini_output.mkdir(parents=True, exist_ok=True)

    stem = f"__voice_preview_{_safe_filename(model)}_{_safe_filename(voice)}"
    staged_input = gemini_input / f"{stem}.txt"
    generated = gemini_output / f"{stem}.wav"
    staged_input.write_text(_GEMINI_PREVIEW_TEXT, encoding="utf-8")
    try:
        try:
            generated.unlink()
        except FileNotFoundError:
            pass
        log(f"Generating one-time Gemini preview for {voice}; later previews use the local cache.")
        _run(
            [
                sys.executable,
                str(script),
                "--text_file",
                staged_input.name,
                "--voice",
                voice,
                "--model",
                model,
                "--chunk-seconds",
                "60",
                "--max-retries",
                "2",
                "--high_quality",
            ],
            project_root,
            log,
        )
        if not generated.exists() or generated.stat().st_size <= 44:
            raise RuntimeError(f"Gemini preview generation did not create valid audio: {generated}")
        shutil.copyfile(generated, preview_path)
    finally:
        for temporary_path in (staged_input, generated):
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass

    return preview_path


def generate_gemini(
    project_root: Path,
    story_path: Path,
    output_path: Path,
    voice: str,
    model: str,
    preprocess: bool,
    split_on_separator: bool,
    chunk_seconds: int,
    log: LogFn,
    *,
    text: str | None = None,
    speaker_voices: dict[int, str] | None = None,
) -> Path:
    script = project_root / "gemini-tts" / "gemini_tts.py"
    if not script.exists():
        raise FileNotFoundError(f"Gemini TTS script is missing: {script}")

    gemini_input = project_root / "gemini-tts" / "input"
    gemini_output = project_root / "gemini-tts" / "output"
    gemini_input.mkdir(parents=True, exist_ok=True)
    gemini_output.mkdir(parents=True, exist_ok=True)

    staged_input = gemini_input / story_path.name
    if text is None:
        shutil.copyfile(story_path, staged_input)
    else:
        staged_input.write_text(text.strip(), encoding="utf-8")
    generated = gemini_output / f"{story_path.stem}.wav"

    command = [
        sys.executable, str(script), "--text_file", story_path.name, "--voice", voice, "--model", model,
        "--chunk-seconds", str(int(chunk_seconds)), "--high_quality",
    ]
    if preprocess:
        command.append("--preprocess")
    if not split_on_separator:
        command.append("--no_split")
    for speaker_id, speaker_voice in sorted((speaker_voices or {}).items()):
        if speaker_voice and speaker_voice.strip():
            command.extend(["--speaker-voice", f"{speaker_id}={speaker_voice.strip()}"])

    try:
        _run(command, project_root, log)
        if not generated.exists():
            raise RuntimeError(f"Gemini TTS completed without creating {generated}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(generated, output_path)
    finally:
        try:
            staged_input.unlink()
        except OSError:
            pass

    return output_path


def _voice_presets(vibevoice_root: Path) -> dict[str, Path]:
    voices_dir = vibevoice_root / "demo" / "voices"
    presets: dict[str, Path] = {}
    if not voices_dir.exists():
        return presets
    for wav in sorted(voices_dir.glob("*.wav")):
        name = wav.stem
        presets[name.lower()] = wav
        simplified = name.split("_")[0].split("-")[-1]
        presets.setdefault(simplified.lower(), wav)
    return presets


def list_vibevoice_presets(vibevoice_root: Path) -> list[str]:
    presets = _voice_presets(vibevoice_root)
    unique: dict[str, str] = {}
    for path in presets.values():
        unique[path.stem.lower()] = path.stem
    return sorted(unique.values(), key=str.lower)


def _resolve_voice(vibevoice_root: Path, speaker_name: str) -> Path:
    presets = _voice_presets(vibevoice_root)
    if not presets:
        raise RuntimeError(
            f"No VibeVoice WAV presets found in {vibevoice_root / 'demo' / 'voices'}. "
            "Run setup.ps1 to install the pinned VibeVoice community runtime."
        )

    requested = speaker_name.strip().lower()
    if requested in presets:
        return presets[requested]
    for name, path in presets.items():
        if requested in name or name in requested:
            return path
    available = ", ".join(list_vibevoice_presets(vibevoice_root))
    raise ValueError(f"Unknown VibeVoice speaker '{speaker_name}'. Available presets: {available}")


def get_vibevoice_voice_preview(project_root: Path, speaker_name: str) -> Path:
    """Return the exact WAV reference sample used by the selected VibeVoice preset."""
    if not speaker_name or not speaker_name.strip():
        raise ValueError("Select a VibeVoice speaker before loading a preview.")
    return _resolve_voice(project_root / "vendor" / "VibeVoice", speaker_name)


def _format_vibevoice_script(story_text: str) -> str:
    """Normalize VibeVoice script while preserving explicit `Speaker N:` turns."""
    normalized = (
        story_text.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '\"')
        .replace("\u201d", '\"')
    )
    turns: list[str] = []
    current_speaker = 0
    for raw_line in normalized.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        match = re.match(r"^Speaker\s+(\d+)\s*:\s*(.*)$", line, flags=re.IGNORECASE)
        if match:
            current_speaker = int(match.group(1))
            body = match.group(2).strip()
            if body:
                turns.append(f"Speaker {current_speaker}: {body}")
        else:
            turns.append(f"Speaker {current_speaker}: {line}")
    if not turns:
        raise ValueError("VibeVoice story text is empty.")
    return "\n".join(turns)

def _load_vibevoice(model_id: str, device: str, dtype_name: str, log: LogFn):
    cache_key = (model_id, device, dtype_name)
    if cache_key in _VIBEVOICE_CACHE:
        log("Reusing the loaded VibeVoice model.")
        return _VIBEVOICE_CACHE[cache_key]

    try:
        import torch
        from transformers.utils import logging as transformers_logging

        # Optional accelerators are not installed here. The native/SDPA path is intentional.
        transformers_logging.set_verbosity_error()
        transformers_logging.disable_progress_bar()
        logging.getLogger("vibevoice.modular.modular_vibevoice_tokenizer").setLevel(logging.ERROR)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="audio_utils not available.*")
            from vibevoice.modular.modeling_vibevoice_inference import (
                VibeVoiceForConditionalGenerationInference,
            )
            from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
    except ImportError as exc:
        raise RuntimeError(
            "VibeVoice runtime is not installed. Run .\\setup.ps1 first; it installs the pinned community runtime "
            "used to execute microsoft/VibeVoice-1.5B."
        ) from exc

    resolved_device = device
    if resolved_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    if resolved_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("VibeVoice was asked to use CUDA, but PyTorch cannot see a CUDA GPU.")

    if dtype_name == "auto":
        dtype = torch.bfloat16 if resolved_device == "cuda" else torch.float32
    else:
        dtype = getattr(torch, dtype_name)

    log(f"Loading VibeVoice model {model_id} on {resolved_device} ({dtype}, SDPA).")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The tokenizer class you load from this checkpoint.*")
        processor = VibeVoiceProcessor.from_pretrained(model_id)
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=resolved_device,
        attn_implementation="sdpa",
    )

    model.eval()
    _VIBEVOICE_CACHE[cache_key] = (processor, model)
    return processor, model


def generate_vibevoice(
    project_root: Path,
    story_text: str,
    output_path: Path,
    model_id: str,
    speaker_name: str,
    cfg_scale: float,
    diffusion_steps: int,
    seed: int,
    device: str,
    dtype_name: str,
    log: LogFn,
    *,
    speaker_names: dict[int, str] | None = None,
) -> Path:
    """Generate the complete story in a single VibeVoice pass (no text chunking)."""
    vibevoice_root = project_root / "vendor" / "VibeVoice"
    if not vibevoice_root.exists():
        raise RuntimeError(
            f"VibeVoice runtime not found at {vibevoice_root}. Run .\\setup.ps1 before using VibeVoice."
        )

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed. Run .\\setup.ps1 first.") from exc

    selected_speakers = dict(sorted((speaker_names or {0: speaker_name}).items()))
    if not selected_speakers:
        selected_speakers = {0: speaker_name}
    expected_ids = list(range(len(selected_speakers)))
    if list(selected_speakers) != expected_ids:
        raise ValueError(
            f"VibeVoice speaker IDs must be contiguous from 0; got {list(selected_speakers)}."
        )
    voice_paths = [_resolve_voice(vibevoice_root, selected_speakers[i]) for i in expected_ids]
    processor, model = _load_vibevoice(model_id, device, dtype_name, log)
    resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    full_script = _format_vibevoice_script(story_text)
    log(
        f"Generating the full story with VibeVoice in one pass: "
        f"speakers={[path.stem for path in voice_paths]}, cfg={cfg_scale}, "
        f"diffusion_steps={diffusion_steps}, seed={seed}."
    )
    model.set_ddpm_inference_steps(num_steps=int(diffusion_steps))
    inputs = processor(
        text=[full_script],
        voice_samples=[[str(path) for path in voice_paths]],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    for key, value in inputs.items():
        if torch.is_tensor(value):
            inputs[key] = value.to(resolved_device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=float(cfg_scale),
            tokenizer=processor.tokenizer,
            generation_config={"do_sample": False},
            verbose=False,
            show_progress_bar=False,
            is_prefill=True,
        )

    if not outputs.speech_outputs or outputs.speech_outputs[0] is None:
        raise RuntimeError("VibeVoice returned no speech output.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processor.save_audio(outputs.speech_outputs[0], output_path=str(output_path))
    if not output_path.exists():
        raise RuntimeError(f"VibeVoice did not create the expected audio file: {output_path}")
    return output_path

