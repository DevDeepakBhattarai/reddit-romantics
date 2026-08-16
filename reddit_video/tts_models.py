from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

LogFn = Callable[[str], None]

MAGPIE_SPEAKERS = ["Aria", "Jason", "John", "Leo", "Sofia"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".work" / "hf-cache"))


@lru_cache(maxsize=1)
def _wsl_home() -> str:
    if shutil.which("wsl.exe") is None:
        raise RuntimeError("WSL is required for this TTS backend on Windows, but wsl.exe was not found.")
    result = subprocess.run(
        ["wsl.exe", "bash", "-lc", 'printf %s "$HOME"'],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    home = result.stdout.strip()
    if result.returncode != 0 or not home:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"Could not resolve the WSL home directory: {detail}")
    return home


def _wsl_native_root(backend: str) -> str:
    return f"{_wsl_home()}/.cache/reddit-romantics/{backend}"


def _wsl_native_ready(project_root: Path, backend: str) -> bool:
    return os.name == "nt" and (project_root / ".work" / f"{backend}-wsl-native-ready").exists()


def _runtime_python(project_root: Path, backend: str) -> Path:
    env_name = f"TTS_{backend.upper()}_PYTHON"
    override = os.getenv(env_name, "").strip()
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override))
    base = project_root / ".tts-venvs" / backend
    candidates.extend([base / "Scripts" / "python.exe", base / "bin" / "python"])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise RuntimeError(
        f"{backend} TTS runtime is not installed. Run .\\setup_tts_models.ps1 -Backend {backend}, "
        f"or set {env_name} to that backend's Python executable."
    )


def _wsl_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name == "nt" and len(resolved) >= 3 and resolved[1] == ":" and resolved[2] in {"\\", "/"}:
        drive = resolved[0].lower()
        rest = resolved[3:].replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    if shutil.which("wsl.exe") is None:
        raise RuntimeError("WSL is required for this TTS backend on Windows, but wsl.exe was not found.")
    escaped = resolved.replace("'", "'\"'\"'")
    result = subprocess.run(
        ["wsl.exe", "bash", "-lc", f"wslpath -a -u '{escaped}'"],
        capture_output=True, text=True, check=True, encoding="utf-8", errors="replace",
    )
    return result.stdout.strip()


def _runtime_prefix(project_root: Path, backend: str, repo_root: Path | None = None) -> tuple[list[str], bool]:
    if _wsl_native_ready(project_root, backend):
        return [
            "wsl.exe",
            "env",
            "PYTHONUNBUFFERED=1",
            f"{_wsl_native_root(backend)}/.venv/bin/python",
        ], True
    try:
        return [str(_runtime_python(project_root, backend))], False
    except RuntimeError:
        if repo_root is not None and (repo_root / ".venv" / "pyvenv.cfg").exists() and shutil.which("wsl.exe"):
            return [
                "wsl.exe",
                "env",
                "PYTHONUNBUFFERED=1",
                _wsl_path(repo_root / ".venv" / "bin" / "python"),
            ], True
        raise


def _for_runtime(path: Path, use_wsl: bool) -> str:
    return _wsl_path(path) if use_wsl else str(path)



def _magpie_wsl_prefix(project_root: Path) -> list[str]:
    if shutil.which("wsl.exe") is None:
        raise RuntimeError("NVIDIA Magpie requires WSL2 on Windows, but wsl.exe was not found.")
    marker = project_root / ".work" / "magpie-runtime-ready"
    if not marker.exists():
        raise RuntimeError(
            "Magpie TTS runtime is incomplete. Run .\\setup_tts_models.ps1 -Backend magpie."
        )
    runtime_python = f"{_wsl_native_root('magpie')}/.venv/bin/python"
    return [
        "wsl.exe",
        "env",
        f"HF_HOME={_wsl_home()}/.cache/huggingface",
        "HF_HUB_DISABLE_XET=0",
        "PYTHONUNBUFFERED=1",
        runtime_python,
    ]


def _repo_root(project_root: Path, backend: str, default_folder: str) -> Path:
    env_name = f"TTS_{backend.upper()}_ROOT"
    override = os.getenv(env_name, "").strip()
    root = Path(override) if override else project_root / "vendor" / default_folder
    if not root.exists():
        raise RuntimeError(
            f"{backend} source runtime was not found at {root}. Run .\\setup_tts_models.ps1 -Backend {backend}, "
            f"or set {env_name}."
        )
    return root


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
        line = line.rstrip()
        if line:
            log(line)
    code = process.wait()
    if code != 0:
        raise RuntimeError(f"Command failed with exit code {code}: {subprocess.list2cmdline(command)}")


def _runner(project_root: Path, name: str) -> Path:
    path = project_root / "reddit_video" / "tts_runners" / f"{name}.py"
    if not path.exists():
        raise FileNotFoundError(f"TTS runner is missing: {path}")
    return path


def _write_text(work_dir: Path, text: str) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / "tts_input.txt"
    path.write_text(text.strip(), encoding="utf-8")
    return path


def _validate_output(path: Path, engine: str) -> Path:
    if not path.exists() or path.stat().st_size <= 44:
        raise RuntimeError(f"{engine} finished without creating valid WAV audio: {path}")
    return path



def _fish_s2_native_binary(project_root: Path) -> Path:
    override = os.getenv("TTS_FISH_S2_BINARY", "").strip()
    candidates = [Path(override)] if override else []
    root = project_root / "vendor" / "s2.cpp"
    candidates.extend([
        root / "build-cuda" / "bin" / "Release" / "s2.exe",
        root / "build-cuda" / "Release" / "s2.exe",
        root / "build-cuda" / "s2.exe",
        root / "build" / "s2.exe",
        root / "build" / "s2",
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise RuntimeError(
        "Fish hybrid CUDA runtime is not built. Run .\\setup_tts_models.ps1 -Backend fish "
        "to build the native s2.cpp CUDA backend."
    )


def _fish_s2_native_assets(project_root: Path) -> tuple[Path, Path]:
    model_override = os.getenv("TTS_FISH_GGUF", "").strip()
    tokenizer_override = os.getenv("TTS_FISH_TOKENIZER", "").strip()
    model = Path(model_override) if model_override else project_root / ".work" / "fish-native" / "s2-pro-f16.gguf"
    tokenizer = (
        Path(tokenizer_override)
        if tokenizer_override
        else project_root / "vendor" / "s2.cpp" / "tokenizer.json"
    )
    if not model.exists():
        raise RuntimeError(
            f"Fish hybrid requires the full unquantized F16 S2 Pro GGUF at {model}. "
            "Run .\\setup_tts_models.ps1 -Backend fish once to export it from the existing Fish checkpoint."
        )
    if not tokenizer.exists():
        raise RuntimeError(f"Fish hybrid tokenizer is missing: {tokenizer}")
    return model, tokenizer


_FISH_REFERENCE_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def _fish_user_preset_root(project_root: Path) -> Path:
    return project_root / "voice_presets" / "fish"


def _fish_native_reference_root(project_root: Path) -> Path:
    return project_root / "vendor" / "fish-speech" / "references"


def _safe_voice_preset_id(value: str, fallback: str = "voice") -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip(".-_")
    return value[:80] or fallback


def _valid_fish_preset_folders(reference_root: Path) -> list[Path]:
    if not reference_root.exists():
        return []
    folders: list[Path] = []
    for folder in sorted(reference_root.iterdir(), key=lambda item: item.name.lower()):
        if not folder.is_dir():
            continue
        if any(
            audio.is_file()
            and audio.suffix.lower() in _FISH_REFERENCE_AUDIO_EXTENSIONS
            and audio.with_suffix(".lab").is_file()
            for audio in folder.iterdir()
        ):
            folders.append(folder)
    return folders


def list_fish_reference_presets(project_root: Path) -> list[str]:
    """List persistent user presets plus native Fish reference IDs."""
    names = {folder.name for folder in _valid_fish_preset_folders(_fish_user_preset_root(project_root))}
    names.update(folder.name for folder in _valid_fish_preset_folders(_fish_native_reference_root(project_root)))
    return sorted(names, key=str.lower)


def _resolve_fish_preset_folder(project_root: Path, preset_id: str) -> Path:
    preset_id = preset_id.strip()
    if not preset_id:
        raise ValueError("Choose a Fish voice preset or upload a reference clip for this speaker.")
    if any(part in preset_id for part in ("/", "\\", "..")):
        raise ValueError(f"Invalid Fish voice preset ID: {preset_id!r}")
    for root in (_fish_user_preset_root(project_root), _fish_native_reference_root(project_root)):
        folder = root / preset_id
        if folder.is_dir():
            return folder
    raise FileNotFoundError(f"Fish voice preset not found: {preset_id}")


def resolve_fish_reference_preset(project_root: Path, preset_id: str) -> tuple[Path, str]:
    """Resolve one persistent/native Fish preset to an audio file and exact transcript."""
    folder = _resolve_fish_preset_folder(project_root, preset_id)
    for audio in sorted(folder.iterdir(), key=lambda item: item.name.lower()):
        if not audio.is_file() or audio.suffix.lower() not in _FISH_REFERENCE_AUDIO_EXTENSIONS:
            continue
        transcript_path = audio.with_suffix(".lab")
        if not transcript_path.is_file():
            continue
        transcript = transcript_path.read_text(encoding="utf-8-sig").strip()
        if transcript:
            return audio, transcript
    raise ValueError(
        f"Fish voice preset '{preset_id}' has no valid audio + matching .lab transcript pair."
    )


def cache_fish_reference_preset(
    project_root: Path,
    audio_path: str | Path,
    transcript: str,
    preset_name: str = "",
) -> tuple[str, Path, str]:
    """Persist an uploaded Fish reference so it can be reused in later sessions.

    A supplied name is stable and replaces that named preset. If the name is blank,
    the upload filename is used. Identical content is deduplicated by SHA-256.
    """
    source = Path(audio_path)
    if not source.is_file():
        raise FileNotFoundError(f"Fish reference audio not found: {source}")
    if source.suffix.lower() not in _FISH_REFERENCE_AUDIO_EXTENSIONS:
        raise ValueError(f"Unsupported Fish reference audio format: {source.suffix}")
    transcript = transcript.strip()
    if not transcript:
        raise ValueError("Fish voice preset requires the exact transcript of its reference audio.")

    audio_bytes = source.read_bytes()
    digest = hashlib.sha256(audio_bytes + b"\0" + transcript.encode("utf-8")).hexdigest()
    root = _fish_user_preset_root(project_root)
    root.mkdir(parents=True, exist_ok=True)

    requested_name = preset_name.strip()
    if not requested_name:
        # For unnamed uploads, avoid storing the exact same reference more than once.
        for folder in _valid_fish_preset_folders(root):
            metadata_path = folder / "preset.json"
            if metadata_path.is_file():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    metadata = {}
                if metadata.get("sha256") == digest:
                    audio, saved_transcript = resolve_fish_reference_preset(project_root, folder.name)
                    return folder.name, audio, saved_transcript

    requested = requested_name or source.stem
    preset_id = _safe_voice_preset_id(requested, fallback=f"voice-{digest[:8]}")
    folder = root / preset_id
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=False)

    destination = folder / f"sample{source.suffix.lower()}"
    shutil.copy2(source, destination)
    destination.with_suffix(".lab").write_text(transcript, encoding="utf-8")
    (folder / "preset.json").write_text(
        json.dumps(
            {
                "name": preset_id,
                "sha256": digest,
                "source_filename": source.name,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return preset_id, destination, transcript


def _merge_fish_reference_audio(references: list[Path], output_path: Path, log: LogFn) -> Path:
    """Concatenate per-speaker reference clips into the single reference stream s2.cpp expects."""
    if len(references) == 1:
        return references[0]
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to combine multiple Fish speaker reference clips for hybrid mode.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for reference in references:
        command.extend(["-i", str(reference)])
    normalized = []
    for index in range(len(references)):
        normalized.append(
            f"[{index}:a]aresample=44100,aformat=sample_fmts=s16:channel_layouts=mono[a{index}]"
        )
    inputs = "".join(f"[a{index}]" for index in range(len(references)))
    filter_graph = ";".join(normalized + [f"{inputs}concat=n={len(references)}:v=0:a=1[out]"])
    command.extend([
        "-filter_complex",
        filter_graph,
        "-map", "[out]",
        "-ar", "44100",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(output_path),
    ])
    log(f"Fish hybrid: combining {len(references)} per-speaker reference clips into one tagged reference stream.")
    _run(command, output_path.parent, log)
    return output_path


def _validated_fish_speaker_references(
    speaker_references: dict[int, tuple[str | Path, str]] | None,
) -> list[tuple[int, Path, str]]:
    validated: list[tuple[int, Path, str]] = []
    for speaker_id, (audio, transcript) in sorted((speaker_references or {}).items()):
        path = Path(audio)
        if not path.exists():
            raise FileNotFoundError(f"Fish reference audio for Speaker {speaker_id} not found: {path}")
        if not transcript.strip():
            raise ValueError(f"Fish reference transcript is required for Speaker {speaker_id}.")
        validated.append((speaker_id, path, transcript.strip()))
    if validated and [speaker_id for speaker_id, _, _ in validated] != list(range(len(validated))):
        raise ValueError("Fish speaker references must use contiguous speaker IDs starting at 0.")
    return validated


def _generate_fish_s2_hybrid(
    project_root: Path,
    text: str,
    output_path: Path,
    *,
    gpu_layers: int,
    temperature: float,
    reference_audio: str | Path | None,
    reference_text: str,
    log: LogFn,
    speaker_references: dict[int, tuple[str | Path, str]] | None = None,
) -> Path:
    binary = _fish_s2_native_binary(project_root)
    model, tokenizer = _fish_s2_native_assets(project_root)
    layers = max(1, min(36, int(gpu_layers)))
    default_threads = min(16, os.cpu_count() or 16)
    cpu_threads = max(1, int(os.getenv("TTS_FISH_CPU_THREADS", str(default_threads))))
    cuda_device = max(0, int(os.getenv("TTS_FISH_CUDA_DEVICE", "0")))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log(
        "Fish hybrid: native Windows s2.cpp, full unquantized F16 weights; "
        f"{layers}/36 transformer layers on CUDA:{cuda_device}, remaining transformer layers on CPU."
    )
    log(
        "Fish hybrid: Fast-AR, KV cache, and codec use CUDA; the codec can fall back to CPU if GPU allocation fails; "
        f"CPU worker threads={cpu_threads}."
    )
    command = [
        str(binary),
        "--model", str(model),
        "--tokenizer", str(tokenizer),
        "--text", text.strip(),
        "--output", str(output_path),
        "--cuda", str(cuda_device),
        "--gpu-layers", str(layers),
        "--codec-follow-backend",
        "--threads", str(cpu_threads),
        "--temperature", str(float(temperature)),
        "--max-tokens", os.getenv("TTS_FISH_MAX_TOKENS", "4096"),
    ]
    refs = _validated_fish_speaker_references(speaker_references)
    if refs:
        merged_audio = _merge_fish_reference_audio(
            [ref for _, ref, _ in refs],
            output_path.parent / f".{output_path.stem}_fish" / "multi_speaker_reference.wav",
            log,
        )
        tagged_text = "\n".join(
            f"<|speaker:{speaker_id}|>{transcript}" for speaker_id, _, transcript in refs
        )
        command.extend(["--prompt-audio", str(merged_audio), "--prompt-text", tagged_text])
    elif reference_audio:
        ref = Path(reference_audio)
        if not ref.exists():
            raise FileNotFoundError(f"Fish reference audio not found: {ref}")
        if not reference_text.strip():
            raise ValueError("Fish voice cloning requires the transcript of the reference audio.")
        command.extend(["--prompt-audio", str(ref), "--prompt-text", reference_text.strip()])

    _run(command, binary.parent, log)
    return _validate_output(output_path, "Fish Audio S2 Pro (hybrid CUDA)")

def generate_fish_s2(
    project_root: Path,
    text: str,
    output_path: Path,
    *,
    device: str = "hybrid",
    gpu_layers: int = 20,
    half: bool = False,
    temperature: float = 1.0,
    seed: int = 42,
    reference_audio: str | Path | None = None,
    reference_text: str = "",
    speaker_references: dict[int, tuple[str | Path, str]] | None = None,
    log: LogFn,
) -> Path:
    if device == "hybrid":
        return _generate_fish_s2_hybrid(
            project_root, text, output_path, gpu_layers=gpu_layers, temperature=temperature,
            reference_audio=reference_audio, reference_text=reference_text, log=log,
            speaker_references=speaker_references,
        )

    root = _repo_root(project_root, "fish", "fish-speech")
    python_prefix, use_wsl = _runtime_prefix(project_root, "fish", root)
    script = root / "fish_speech" / "models" / "text2semantic" / "inference.py"
    checkpoint = root / "checkpoints" / "s2-pro"
    native_runtime = _wsl_native_ready(project_root, "fish")
    if not script.exists():
        raise FileNotFoundError(f"Fish S2 inference script is missing: {script}")
    if not native_runtime and not checkpoint.exists():
        raise RuntimeError(
            f"Fish S2 Pro weights are missing at {checkpoint}. Run .\\setup_tts_models.ps1 -Backend fish."
        )
    checkpoint_runtime = (
        f"{_wsl_native_root('fish')}/checkpoints/s2-pro"
        if native_runtime
        else _for_runtime(checkpoint, use_wsl)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = output_path.parent / f".{output_path.stem}_fish"
    work_dir.mkdir(parents=True, exist_ok=True)
    # Fish CPU inference is dominated by PyTorch/oneDNN matrix operations.
    # Give those libraries a real multi-core worker pool instead of relying on
    # whatever conservative thread count the parent process/runtime inherited.
    # Pick a high but sane default automatically and keep it overridable.
    # This workstation resolves to 16; smaller hosts will not be oversubscribed.
    default_fish_threads = min(16, os.cpu_count() or 16)
    fish_cpu_threads = max(
        1, int(os.getenv("TTS_FISH_CPU_THREADS", str(default_fish_threads)))
    )
    if device == "cpu":
        log(f"Fish CPU inference: using up to {fish_cpu_threads} compute threads.")
        thread_env = [
            f"OMP_NUM_THREADS={fish_cpu_threads}",
            f"MKL_NUM_THREADS={fish_cpu_threads}",
            f"OPENBLAS_NUM_THREADS={fish_cpu_threads}",
            f"NUMEXPR_NUM_THREADS={fish_cpu_threads}",
        ]
        if use_wsl:
            # `env` after wsl.exe guarantees these Linux-side variables reach
            # PyTorch; plain Windows environment variables are not sufficient.
            runtime_python = python_prefix[-1]
            python_prefix = [
                "wsl.exe", "env", *thread_env, "PYTHONUNBUFFERED=1", runtime_python
            ]

    if native_runtime:
        log("Fish: starting native-WSL runtime; framework imports and weights are on Linux ext4.")
    elif use_wsl:
        log("Fish: WARNING - using the legacy /mnt Windows-backed WSL runtime; startup will be slower.")

    command = [
        *python_prefix,
        _for_runtime(script, use_wsl),
        "--text",
        text.strip(),
        "--checkpoint-path",
        checkpoint_runtime,
        "--device",
        device,
        "--temperature",
        str(float(temperature)),
        "--seed",
        str(int(seed)),
        "--output",
        _for_runtime(output_path, use_wsl),
        "--output-dir",
        _for_runtime(work_dir, use_wsl),
        "--no-compile",
    ]
    if half and device != "cpu":
        command.append("--half")
    refs = _validated_fish_speaker_references(speaker_references)
    if refs:
        for speaker_id, ref, transcript in refs:
            command.extend([
                "--prompt-audio", _for_runtime(ref, use_wsl),
                "--prompt-text", f"<|speaker:{speaker_id}|>{transcript}",
            ])
    elif reference_audio:
        ref = Path(reference_audio)
        if not ref.exists():
            raise FileNotFoundError(f"Fish reference audio not found: {ref}")
        if not reference_text.strip():
            raise ValueError("Fish voice cloning requires the transcript of the reference audio.")
        command.extend(["--prompt-audio", _for_runtime(ref, use_wsl), "--prompt-text", reference_text.strip()])

    _run(command, root, log)
    return _validate_output(output_path, "Fish Audio S2 Pro")


def generate_step_editx(
    project_root: Path,
    text: str,
    output_path: Path,
    *,
    reference_audio: str | Path | None,
    reference_text: str,
    mode: str = "clone",
    log: LogFn,
) -> Path:
    root = _repo_root(project_root, "step", "Step-Audio-EditX")
    if not reference_audio:
        reference_audio = root / "examples" / "zero_shot_en_prompt.wav"
        reference_text = "His political stance was conservative, and he was particularly close to margaret thatcher."
    elif not reference_text.strip():
        raise ValueError("A custom Step Audio reference clip requires its exact transcript.")

    python_prefix, use_wsl = _runtime_prefix(project_root, "step", root)
    script = root / "tts_infer.py"
    model_root_override = os.getenv("STEP_AUDIO_MODEL_ROOT", "").strip()
    model_root = Path(model_root_override) if model_root_override else root / "models"
    native_runtime = _wsl_native_ready(project_root, "step")
    if not script.exists():
        raise FileNotFoundError(f"Step Audio EditX inference script is missing: {script}")
    if not native_runtime and not model_root.exists():
        raise RuntimeError(
            f"Step Audio model files are missing at {model_root}. Run .\\setup_tts_models.ps1 -Backend step."
        )
    model_root_runtime = (
        f"{_wsl_native_root('step')}/models"
        if native_runtime
        else _for_runtime(model_root, use_wsl)
    )

    ref = Path(reference_audio)
    if not ref.exists():
        raise FileNotFoundError(f"Step Audio reference audio not found: {ref}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_dir = output_path.parent / f".{output_path.stem}_step_{int(time.time() * 1000)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    awq_model = model_root / "Step-Audio-EditX-AWQ-4bit"
    standard_model = model_root / "Step-Audio-EditX"
    if native_runtime:
        selected_model_runtime = f"{model_root_runtime}/Step-Audio-EditX-AWQ-4bit"
        selected_is_awq = True
        log("Step Audio: starting native-WSL runtime; dependencies and model weights are on Linux ext4.")
    else:
        selected_model = awq_model if awq_model.exists() else (standard_model if standard_model.exists() else model_root)
        selected_model_runtime = _for_runtime(selected_model, use_wsl)
        selected_is_awq = selected_model == awq_model
        if use_wsl:
            log("Step Audio: WARNING - using the legacy /mnt Windows-backed WSL runtime; startup will be slower.")
    command = [
        *python_prefix,
        _for_runtime(script, use_wsl),
        "--model-path",
        selected_model_runtime,
        "--tokenizer-path",
        f"{model_root_runtime}/Step-Audio-Tokenizer" if native_runtime else _for_runtime(model_root / "Step-Audio-Tokenizer", use_wsl),
        "--prompt-text",
        reference_text.strip(),
        "--prompt-audio",
        _for_runtime(ref, use_wsl),
        "--generated-text",
        text.strip(),
        "--edit-type",
        "paralinguistic" if mode == "paralinguistic" else "clone",
        "--output-dir",
        _for_runtime(run_dir, use_wsl),
        "--gpu-memory-utilization", "0.1",
        "--enforce-eager",
        "--max-num-seqs", "1",
        "--cosyvoice-dtype", "bfloat16",
        "--no-cosyvoice-cuda-graph",
    ]
    if selected_is_awq:
        command.extend(["--quantization", "awq"])
    _run(command, root, log)
    candidates = sorted(run_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError(f"Step Audio EditX completed but produced no WAV file in {run_dir}")
    shutil.copyfile(candidates[0], output_path)
    return _validate_output(output_path, "Step Audio EditX")


def generate_magpie(
    project_root: Path,
    text: str,
    output_path: Path,
    *,
    model_id: str = "nvidia/magpie_tts_multilingual_357m",
    speaker: str = "John",
    language: str = "en",
    device: str = "auto",
    use_cfg: bool = True,
    cfg_scale: float = 2.5,
    log: LogFn,
) -> Path:
    if os.name == "nt":
        use_wsl = True
        python_prefix = _magpie_wsl_prefix(project_root)
    else:
        use_wsl = False
        python_prefix = [str(_runtime_python(project_root, "magpie"))]
    text_file = _write_text(output_path.parent / f".{output_path.stem}_magpie", text)
    if use_wsl:
        log("Magpie: starting native-WSL runtime; importing NeMo and loading the checkpoint from Linux ext4.")
    command = [
        *python_prefix,
        _for_runtime(_runner(project_root, "magpie"), use_wsl),
        "--text-file",
        _for_runtime(text_file, use_wsl),
        "--output",
        _for_runtime(output_path, use_wsl),
        "--model",
        model_id,
        "--speaker",
        speaker,
        "--language",
        language,
        "--device",
        device,
        "--cfg-scale",
        str(float(cfg_scale)),
    ]
    if use_cfg:
        command.append("--use-cfg")
    _run(command, project_root, log)
    return _validate_output(output_path, "NVIDIA Magpie")


def generate_chatterbox(
    project_root: Path,
    text: str,
    output_path: Path,
    *,
    variant: str = "turbo",
    device: str = "auto",
    reference_audio: str | Path | None = None,
    log: LogFn,
) -> Path:
    python = _runtime_python(project_root, "chatterbox")
    text_file = _write_text(output_path.parent / f".{output_path.stem}_chatterbox", text)
    command = [
        str(python),
        str(_runner(project_root, "chatterbox_runner")),
        "--text-file",
        str(text_file),
        "--output",
        str(output_path),
        "--variant",
        variant,
        "--device",
        device,
    ]
    if not reference_audio and variant in {"turbo", "nano"}:
        default_refs = sorted((project_root / "vendor" / "VibeVoice" / "demo" / "voices").glob("*.wav"))
        preferred = project_root / "vendor" / "VibeVoice" / "demo" / "voices" / "en-Alice_woman.wav"
        reference_audio = preferred if preferred.exists() else (default_refs[0] if default_refs else None)
        if reference_audio:
            log(f"Using bundled reference voice for Chatterbox {variant}: {Path(reference_audio).stem}")
    if reference_audio:
        ref = Path(reference_audio)
        if not ref.exists():
            raise FileNotFoundError(f"Chatterbox reference audio not found: {ref}")
        command.extend(["--reference-audio", str(ref)])
    _run(command, project_root, log)
    return _validate_output(output_path, "Chatterbox")


_HIGGS_DOCKER_IMAGE = "reddit-romantics-sglang-omni:local"
_HIGGS_CONTAINER = "reddit-romantics-higgs-tts"
_HIGGS_PORT = 18080


def _docker_command(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    if shutil.which("docker") is None:
        raise RuntimeError("Docker Desktop is required for Higgs TTS 3, but docker.exe was not found on PATH.")
    return subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=check,
    )


def _higgs_health() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{_HIGGS_PORT}/health", timeout=2) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def _higgs_container_state() -> tuple[bool, bool, str]:
    result = _docker_command(
        [
            "inspect",
            "--format",
            '{{.State.Running}}|{{index .Config.Labels "reddit-romantics.higgs-model"}}',
            _HIGGS_CONTAINER,
        ],
        check=False,
    )
    if result.returncode != 0:
        return False, False, ""
    running, _, model = result.stdout.strip().partition("|")
    return True, running.lower() == "true", model.strip()


def _ensure_higgs_server(project_root: Path, model_id: str, device: str, log: LogFn) -> None:
    image = _docker_command(["image", "inspect", _HIGGS_DOCKER_IMAGE], check=False)
    if image.returncode != 0:
        raise RuntimeError(
            "The Higgs SGLang-Omni image is not installed. Run .\\setup_tts_models.ps1 -Backend higgs once."
        )

    exists, running, loaded_model = _higgs_container_state()
    if exists and loaded_model != model_id:
        log(f"Recreating Higgs server because the selected model changed to {model_id}.")
        _docker_command(["rm", "-f", _HIGGS_CONTAINER], check=False)
        exists = running = False

    if exists and not running:
        log("Starting the existing Higgs TTS container.")
        _docker_command(["start", _HIGGS_CONTAINER])
    elif not exists:
        cache_dir = (project_root / ".work" / "hf-cache").resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Higgs stages are CUDA-oriented, but SGLang can keep a substantial part
        # of the AR backbone in host RAM. On this 8 GB card, use aggressive
        # offload and tiny serving/KV budgets; `cpu` means "prefer host RAM",
        # not a CUDA-free Higgs pipeline (the codec/vocoder still use CUDA).
        offload_gb = "12" if device == "cpu" else "8"
        log(
            f"Launching Higgs TTS via SGLang-Omni (CPU offload={offload_gb} GB, "
            "single-request low-memory mode)."
        )
        command = [
            "run",
            "-d",
            "--name",
            _HIGGS_CONTAINER,
            "--label",
            f"reddit-romantics.higgs-model={model_id}",
            "--gpus",
            "all",
            "--shm-size",
            "16g",
            "--ipc",
            "host",
            "-p",
            f"127.0.0.1:{_HIGGS_PORT}:8000",
            "-v",
            f"{cache_dir}:/root/.cache/huggingface",
            _HIGGS_DOCKER_IMAGE,
            "serve",
            "--model-path",
            model_id,
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--cpu-offload-gb",
            offload_gb,
            "--talker-mem-fraction-static",
            "0.55",
            "--max-running-requests",
            "1",
            "--max-total-tokens",
            "2048",
            "--talker-cuda-graph",
            "off",
            "--decode-mode",
            "sync",
            "--log-level",
            "warning",
        ]
        result = _docker_command(command, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Could not launch Higgs TTS container: {result.stderr.strip() or result.stdout.strip()}")

    if _higgs_health():
        return

    log("Waiting for the Higgs model to finish loading (first launch only).")
    for attempt in range(300):
        if _higgs_health():
            log("Higgs TTS server is ready.")
            return
        exists, running, _ = _higgs_container_state()
        if not exists or not running:
            logs = _docker_command(["logs", "--tail", "120", _HIGGS_CONTAINER], check=False)
            detail = (logs.stdout + "\n" + logs.stderr).strip()
            raise RuntimeError(f"Higgs TTS container stopped while loading.\n{detail}")
        if attempt and attempt % 15 == 0:
            log("Higgs is still loading/offloading model weights...")
        time.sleep(2)
    logs = _docker_command(["logs", "--tail", "120", _HIGGS_CONTAINER], check=False)
    detail = (logs.stdout + "\n" + logs.stderr).strip()
    raise RuntimeError(f"Higgs TTS did not become healthy after startup.\n{detail}")


def generate_higgs_tts3(
    project_root: Path,
    text: str,
    output_path: Path,
    *,
    model_id: str = "bosonai/higgs-audio-v3-tts-4b",
    device: str = "auto",
    log: LogFn,
) -> Path:
    _ensure_higgs_server(project_root, model_id, device, log)
    request = urllib.request.Request(
        f"http://127.0.0.1:{_HIGGS_PORT}/v1/audio/speech",
        data=json.dumps(
            {
                "model": model_id,
                "voice": "default",
                "input": text.strip(),
                "response_format": "wav",
                "stream": False,
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    log("Generating Higgs TTS audio through the local SGLang-Omni server.")
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            audio = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Higgs TTS request failed with HTTP {exc.code}: {detail}") from exc
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(f"Higgs TTS request failed: {exc}") from exc
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(audio)
    return _validate_output(output_path, "Higgs TTS 3")


def backend_runtime_status(project_root: Path) -> dict[str, str]:
    statuses: dict[str, str] = {}

    fish_root = project_root / "vendor" / "fish-speech"
    fish_native = _wsl_native_ready(project_root, "fish")
    fish_ready = fish_native or ((
        (fish_root / ".venv" / "pyvenv.cfg").exists()
        and (fish_root / "checkpoints" / "s2-pro" / "codec.pth").stat().st_size > 1_000_000_000
        and (fish_root / "checkpoints" / "s2-pro" / "model-00001-of-00002.safetensors").exists()
    ) if (fish_root / "checkpoints" / "s2-pro" / "codec.pth").exists() else False)
    fish_s2_root = project_root / "vendor" / "s2.cpp"
    fish_s2_binaries = [
        fish_s2_root / "build-cuda" / "Release" / "s2.exe",
        fish_s2_root / "build-cuda" / "bin" / "Release" / "s2.exe",
        fish_s2_root / "build-cuda" / "s2.exe",
    ]
    fish_s2_model = project_root / ".work" / "fish-native" / "s2-pro-f16.gguf"
    fish_hybrid_ready = (
        any(binary.exists() for binary in fish_s2_binaries)
        and fish_s2_model.exists()
        and fish_s2_model.stat().st_size > 9_000_000_000
    )
    statuses["fish"] = (
        "ready: native Windows hybrid CUDA + CPU, unquantized F16 S2 Pro" if fish_hybrid_ready
        else "official Fish ready, but hybrid CUDA assets are incomplete; rerun setup_tts_models.ps1 -Backend fish" if fish_ready
        else "Fish S2 Pro runtime/weights are incomplete; run setup_tts_models.ps1 -Backend fish"
    )

    step_root = project_root / "vendor" / "Step-Audio-EditX"
    step_model = step_root / "models" / "Step-Audio-EditX-AWQ-4bit" / "model.safetensors"
    step_tokenizer = step_root / "models" / "Step-Audio-Tokenizer"
    step_native = _wsl_native_ready(project_root, "step")
    step_ready = step_native or (
        (step_root / ".venv" / "pyvenv.cfg").exists()
        and step_model.exists()
        and step_model.stat().st_size > 2_000_000_000
        and step_tokenizer.exists()
        and (step_root / "examples" / "zero_shot_en_prompt.wav").exists()
    )
    statuses["step"] = (
        "ready via native WSL filesystem (fast startup)" if step_native
        else "ready via legacy /mnt WSL runtime (slow startup; rerun setup_tts_models.ps1 -Backend step)" if step_ready
        else "Step Audio EditX runtime/weights are incomplete; run setup_tts_models.ps1 -Backend step"
    )

    magpie_marker = project_root / ".work" / "magpie-runtime-ready"
    magpie_cache = project_root / ".work" / "hf-cache" / "hub" / "models--nvidia--magpie_tts_multilingual_357m"
    statuses["magpie"] = (
        "ready via WSL native runtime (NeMo + Magpie checkpoint installed)"
        if magpie_marker.exists() and magpie_cache.exists()
        else "Magpie runtime is incomplete; run setup_tts_models.ps1 -Backend magpie"
    )

    chatterbox_python = project_root / ".tts-venvs" / "chatterbox" / "Scripts" / "python.exe"
    chatterbox_pkg = project_root / ".tts-venvs" / "chatterbox" / "Lib" / "site-packages" / "chatterbox"
    statuses["chatterbox"] = (
        "ready (Turbo/Nano/Original installed)"
        if chatterbox_python.exists() and chatterbox_pkg.exists()
        else "Chatterbox runtime is incomplete; run setup_tts_models.ps1 -Backend chatterbox"
    )

    higgs_model = project_root / ".work" / "hf-cache" / "hub" / "models--bosonai--higgs-audio-v3-tts-4b"
    try:
        image_ready = _docker_command(["image", "inspect", _HIGGS_DOCKER_IMAGE], check=False).returncode == 0
    except RuntimeError:
        image_ready = False
    statuses["higgs"] = (
        "ready (SGLang-Omni Docker image + Higgs v3 checkpoint installed)"
        if image_ready and higgs_model.exists()
        else "Higgs Docker runtime/model is incomplete; run setup_tts_models.ps1 -Backend higgs"
    )
    return statuses
