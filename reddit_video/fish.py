from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path

LogFn = Callable[[str], None]


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
        raise RuntimeError(
            f"Command failed with exit code {code}: {subprocess.list2cmdline(command)}"
        )


def _validate_output(path: Path, engine: str) -> Path:
    if not path.exists() or path.stat().st_size <= 44:
        raise RuntimeError(f"{engine} finished without creating valid WAV audio: {path}")
    return path


_FISH_SPEAKER_TAG_RE = re.compile(r"<\|speaker:(\d+)\|>")


def _fish_sentence_units(text: str) -> list[str]:
    """Split text the same way s2.cpp's segmented server does, without losing punctuation."""
    segments: list[str] = []
    current: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        current.append(char)
        if char not in ".!?\n":
            index += 1
            continue

        if char in ".!?":
            while index + 1 < len(text) and text[index + 1] in ".!?\"') ]":
                next_char = text[index + 1]
                if next_char == " ":
                    break
                current.append(next_char)
                index += 1

        segment = "".join(current).strip()
        if segment:
            segments.append(segment)
        current = []
        index += 1

    tail = "".join(current).strip()
    if tail:
        segments.append(tail)
    return segments


def _fish_retag_segmented_text(text: str) -> str:
    """Repeat the active speaker tag for every sentence/line segment.

    s2.cpp first splits long-form text into sentence units and then semantically
    packs as many whole sentences as fit the character budget. Repeating the tag
    keeps speaker identity explicit when a packed chunk contains speaker changes.
    """
    parts = re.split(r"(<\|speaker:\d+\|>)", text.strip())
    active_speaker = 0
    tagged_segments: list[str] = []
    for part in parts:
        if not part:
            continue
        match = _FISH_SPEAKER_TAG_RE.fullmatch(part.strip())
        if match:
            active_speaker = int(match.group(1))
            continue
        for segment in _fish_sentence_units(part):
            tagged_segments.append(f"<|speaker:{active_speaker}|>{segment}")
    return "\n".join(tagged_segments)


def _fish_long_form_threshold() -> int:
    return max(100, int(os.getenv("TTS_FISH_LONG_FORM_CHARS", "500")))


def _fish_segment_max_chars() -> int:
    return max(120, int(os.getenv("TTS_FISH_SEGMENT_MAX_CHARS", "500")))


def _fish_needs_long_form(text: str) -> bool:
    spoken_text = _FISH_SPEAKER_TAG_RE.sub("", text).strip()
    return len(spoken_text) > _fish_long_form_threshold()


def _reserve_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _multipart_body(
    fields: dict[str, str],
    reference_audio: Path | None,
) -> tuple[bytes, str]:
    boundary = f"----RedditRomanticsFish{uuid.uuid4().hex}"
    body = bytearray()

    def add_line(value: str = "") -> None:
        body.extend(value.encode("utf-8"))
        body.extend(b"\r\n")

    for name, value in fields.items():
        add_line(f"--{boundary}")
        add_line(f'Content-Disposition: form-data; name="{name}"')
        add_line()
        add_line(value)

    if reference_audio is not None:
        safe_name = reference_audio.name.replace('"', "_")
        add_line(f"--{boundary}")
        add_line(f'Content-Disposition: form-data; name="reference"; filename="{safe_name}"')
        add_line("Content-Type: application/octet-stream")
        add_line()
        body.extend(reference_audio.read_bytes())
        body.extend(b"\r\n")

    add_line(f"--{boundary}--")
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def _run_fish_s2_segmented_server(
    *,
    binary: Path,
    model: Path,
    tokenizer: Path,
    text: str,
    output_path: Path,
    cuda_device: int,
    gpu_layers: int,
    cpu_threads: int,
    temperature: float,
    reference_audio: Path | None,
    reference_text: str,
    log: LogFn,
) -> Path:
    """Generate long Fish audio with one model load and sentence-aware segmentation."""
    port = _reserve_local_port()
    max_tokens = max(128, min(4096, int(os.getenv("TTS_FISH_SEGMENT_MAX_TOKENS", "1024"))))
    max_chars = _fish_segment_max_chars()
    pause_ms = max(0, int(os.getenv("TTS_FISH_SENTENCE_PAUSE_MS", "180")))
    startup_timeout = max(30.0, float(os.getenv("TTS_FISH_SERVER_START_TIMEOUT", "900")))

    command = [
        str(binary),
        "--model", str(model),
        "--tokenizer", str(tokenizer),
        "--cuda", str(cuda_device),
        "--gpu-layers", str(gpu_layers),
        "--codec-follow-backend",
        "--threads", str(cpu_threads),
        "--temperature", str(float(temperature)),
        "--max-tokens", str(max_tokens),
        "--server",
        "--host", "127.0.0.1",
        "--port", str(port),
    ]
    log(
        "Fish long-form: loading S2 Pro once, then packing as many complete sentences "
        f"as fit within a {max_chars}-character target "
        f"(single longer sentences stay intact; <= {max_tokens} audio tokens each)."
    )
    log("$ " + subprocess.list2cmdline(command))
    process = subprocess.Popen(
        command,
        cwd=str(binary.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    def pump_logs() -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            stripped = line.rstrip()
            if stripped:
                log(stripped)

    log_thread = threading.Thread(target=pump_logs, name="fish-s2-server-log", daemon=True)
    log_thread.start()
    try:
        deadline = time.monotonic() + startup_timeout
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"Fish S2 long-form server exited during startup with code {process.returncode}.")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    break
            except OSError:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"Fish S2 long-form server did not become ready within {startup_timeout:.0f}s."
                    )
                time.sleep(0.25)

        segmented_text = _fish_retag_segmented_text(text)
        sentence_count = len([line for line in segmented_text.splitlines() if line.strip()])
        log(
            f"Fish long-form: {sentence_count} sentence/line units will be packed by semantic "
            f"sentence boundaries under the {max_chars}-character target; voice reference is encoded once and reused."
        )
        params = json.dumps(
            {
                "max_new_tokens": max_tokens,
                "temperature": float(temperature),
                "n_threads": cpu_threads,
                "codec_follow_backend": True,
                "segment_sentences": True,
                "segment_max_chars": max_chars,
                "segment_group_sentences": 0,
                "sentence_pause_ms": pause_ms,
                "output_format": "wav",
            },
            separators=(",", ":"),
        )
        fields = {"text": segmented_text, "params": params}
        if reference_audio is not None:
            fields["reference_text"] = reference_text
        body, content_type = _multipart_body(fields, reference_audio)

        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=None)
        try:
            connection.request(
                "POST",
                "/generate",
                body=body,
                headers={"Content-Type": content_type, "Content-Length": str(len(body))},
            )
            response = connection.getresponse()
            audio = response.read()
            if response.status != 200:
                detail = audio.decode("utf-8", errors="replace")
                raise RuntimeError(f"Fish S2 long-form request failed with HTTP {response.status}: {detail}")
        finally:
            connection.close()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(audio)
        return _validate_output(output_path, "Fish Audio S2 Pro (hybrid CUDA long-form)")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        log_thread.join(timeout=2)



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
        "Fish hybrid CUDA runtime is not built. Run .\\setup.ps1 "
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
            "Run .\\setup.ps1 once to export it from the existing Fish checkpoint."
        )
    if not tokenizer.exists():
        raise RuntimeError(f"Fish hybrid tokenizer is missing: {tokenizer}")
    return model, tokenizer


_FISH_REFERENCE_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def _fish_user_preset_root(project_root: Path) -> Path:
    return project_root / "voice_presets" / "fish"


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
    """List persistent Fish voice presets saved by this application."""
    return [
        folder.name
        for folder in _valid_fish_preset_folders(_fish_user_preset_root(project_root))
    ]


def _resolve_fish_preset_folder(project_root: Path, preset_id: str) -> Path:
    preset_id = preset_id.strip()
    if not preset_id:
        raise ValueError("Choose a Fish voice preset or upload a reference clip for this speaker.")
    if any(part in preset_id for part in ("/", "\\", "..")):
        raise ValueError(f"Invalid Fish voice preset ID: {preset_id!r}")
    folder = _fish_user_preset_root(project_root) / preset_id
    if folder.is_dir():
        return folder
    raise FileNotFoundError(f"Fish voice preset not found: {preset_id}")


def resolve_fish_reference_preset(project_root: Path, preset_id: str) -> tuple[Path, str]:
    """Resolve one saved Fish preset to an audio file and exact transcript."""
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

    refs = _validated_fish_speaker_references(speaker_references)
    prompt_audio: Path | None = None
    prompt_text = ""
    if refs:
        prompt_audio = _merge_fish_reference_audio(
            [ref for _, ref, _ in refs],
            output_path.parent / f".{output_path.stem}_fish" / "multi_speaker_reference.wav",
            log,
        )
        prompt_text = "\n".join(
            f"<|speaker:{speaker_id}|>{transcript}" for speaker_id, _, transcript in refs
        )
    elif reference_audio:
        prompt_audio = Path(reference_audio)
        if not prompt_audio.exists():
            raise FileNotFoundError(f"Fish reference audio not found: {prompt_audio}")
        if not reference_text.strip():
            raise ValueError("Fish voice cloning requires the transcript of the reference audio.")
        prompt_text = reference_text.strip()

    if _fish_needs_long_form(text):
        return _run_fish_s2_segmented_server(
            binary=binary, model=model, tokenizer=tokenizer, text=text, output_path=output_path,
            cuda_device=cuda_device, gpu_layers=layers, cpu_threads=cpu_threads,
            temperature=temperature, reference_audio=prompt_audio, reference_text=prompt_text, log=log,
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
    if prompt_audio is not None:
        command.extend(["--prompt-audio", str(prompt_audio), "--prompt-text", prompt_text])

    _run(command, binary.parent, log)
    return _validate_output(output_path, "Fish Audio S2 Pro (hybrid CUDA)")

def generate_fish_s2(
    project_root: Path,
    text: str,
    output_path: Path,
    *,
    gpu_layers: int = 20,
    temperature: float = 1.0,
    reference_audio: str | Path | None = None,
    reference_text: str = "",
    speaker_references: dict[int, tuple[str | Path, str]] | None = None,
    log: LogFn,
) -> Path:
    """Generate Fish Audio S2 Pro with the single maintained hybrid F16 runtime."""
    return _generate_fish_s2_hybrid(
        project_root,
        text,
        output_path,
        gpu_layers=gpu_layers,
        temperature=temperature,
        reference_audio=reference_audio,
        reference_text=reference_text,
        speaker_references=speaker_references,
        log=log,
    )


def fish_runtime_status(project_root: Path) -> str:
    try:
        binary = _fish_s2_native_binary(project_root)
        model, tokenizer = _fish_s2_native_assets(project_root)
    except RuntimeError as exc:
        return str(exc)
    if model.stat().st_size <= 9_000_000_000:
        return f"Fish F16 model looks incomplete: {model}"
    return (
        "ready: native Windows s2.cpp hybrid, unquantized F16 S2 Pro "
        f"({binary.name}, {tokenizer.name})"
    )
