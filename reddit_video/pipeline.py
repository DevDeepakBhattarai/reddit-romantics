from __future__ import annotations

import os
import random
import re
import shlex
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from .captions import CAPTION_THEMES, convert_whisperx_json_to_ass
from .tts import generate_gemini, generate_vibevoice
from .tts_models import generate_fish_s2
from .tts_text import prepare_text_for_provider, validate_speaker_count

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

LogFn = Callable[[str], None]
ProgressFn = Callable[[float, str], None]


@dataclass
class PipelineOptions:
    story_file: str | Path | None = None
    story_text: str = ""
    output_name: str = ""

    tts_engine: str = "gemini"
    gemini_voice: str = "Kore"
    gemini_model: str = "gemini-3.1-flash-tts-preview"
    gemini_preprocess: bool = True
    gemini_split_on_separator: bool = True
    gemini_chunk_seconds: int = 180
    gemini_speaker_voices: dict[int, str] = field(default_factory=dict)

    vibevoice_model: str = "microsoft/VibeVoice-1.5B"
    vibevoice_speaker: str = "Alice"
    vibevoice_cfg_scale: float = 1.3
    vibevoice_diffusion_steps: int = 10
    vibevoice_seed: int = 42
    vibevoice_device: str = "auto"
    vibevoice_dtype: str = "auto"
    vibevoice_speaker_voices: dict[int, str] = field(default_factory=dict)

    fish_device: str = "hybrid"
    fish_gpu_layers: int = 20
    fish_half: bool = False
    fish_temperature: float = 1.0
    fish_seed: int = 42
    fish_reference_audio: str | Path | None = None
    fish_reference_text: str = ""
    fish_speaker_references: dict[int, tuple[str | Path, str]] = field(default_factory=dict)

    background: str | Path = "videos/minecraft/minecraft.mp4"
    output_format: str = "shorts"
    randomize_background_start: bool = True
    end_padding_seconds: float = 1.0

    captions: bool = True
    caption_theme: str = "classic_yellow"
    caption_max_words: int = 0
    caption_pause_threshold: float = 0.5

    whisper_model: str = "large-v2"
    whisper_language: str = "en"
    whisper_align_model: str = "WAV2VEC2_ASR_LARGE_LV60K_960H"
    whisper_compute_type: str = "float16"
    whisperx_command: str = ""

    encoder: str = "auto"
    video_quality: int = 20
    audio_bitrate: str = "128k"


@dataclass(frozen=True)
class PipelineResult:
    video_path: Path
    audio_path: Path
    caption_path: Path | None
    whisper_json_path: Path | None
    elapsed_seconds: float


@dataclass(frozen=True)
class AudioPipelineResult:
    audio_path: Path
    elapsed_seconds: float
    duration_seconds: float


def slugify(value: str, fallback: str = "story") -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    value = value.strip("._-")
    return value[:100] or fallback


def list_input_stories(root: Path = PROJECT_ROOT) -> list[str]:
    folder = root / "input"
    folder.mkdir(parents=True, exist_ok=True)
    return [str(path.relative_to(root)).replace("\\", "/") for path in sorted(folder.glob("*.txt"))]


def list_background_videos(root: Path = PROJECT_ROOT) -> list[str]:
    folder = root / "videos"
    if not folder.exists():
        return []
    return [str(path.relative_to(root)).replace("\\", "/") for path in sorted(folder.rglob("*.mp4"))]


class RedditVideoPipeline:
    def __init__(self, root: Path = PROJECT_ROOT, log: LogFn | None = None, progress: ProgressFn | None = None):
        self.root = root.resolve()
        self.log = log or print
        self.progress = progress or (lambda _value, _message: None)

    def _stage(self, value: float, message: str) -> None:
        self.log(f"[{int(value * 100):02d}%] {message}")
        self.progress(value, message)

    def _resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.root / path)

    def _prepare_story(self, options: PipelineOptions) -> tuple[Path, str, str]:
        text = options.story_text.strip()
        if text:
            base = slugify(options.output_name or f"story_{time.strftime('%Y%m%d_%H%M%S')}")
            story_path = self.root / "input" / f"{base}.txt"
            story_path.parent.mkdir(parents=True, exist_ok=True)
            story_path.write_text(text, encoding="utf-8")
            return story_path, text, base

        if not options.story_file:
            raise ValueError("Provide story text or choose a .txt story file.")
        story_path = self._resolve_path(options.story_file)
        if not story_path.exists():
            raise FileNotFoundError(f"Story file not found: {story_path}")
        text = story_path.read_text(encoding="utf-8-sig").strip()
        if not text:
            raise ValueError(f"Story file is empty: {story_path}")
        base = slugify(options.output_name or story_path.stem)
        return story_path, text, base

    def _probe_duration(self, path: Path) -> float:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())

    def _audio_duration(self, path: Path) -> float:
        try:
            import soundfile as sf

            return float(sf.info(str(path)).duration)
        except (OSError, RuntimeError):
            return self._probe_duration(path)

    def _generate_narration(
        self,
        options: PipelineOptions,
        story_path: Path,
        story_text: str,
        audio_path: Path,
    ) -> Path:
        engine = options.tts_engine
        if engine not in {"gemini", "vibevoice", "fish"}:
            raise ValueError(
                f"Unsupported TTS engine '{engine}'. This pipeline intentionally exposes only "
                "Gemini, VibeVoice, and Fish Audio because multi-speaker generation is required."
            )

        speakers = validate_speaker_count(story_text, engine)
        prepared_text, id_map = prepare_text_for_provider(story_text, engine)
        explicit_speakers = bool(speakers)
        original_ids = [speaker.speaker_id for speaker in speakers] or [0]
        if not id_map:
            id_map = {0: 0}

        if engine == "gemini":
            if explicit_speakers:
                missing = [
                    speaker_id
                    for speaker_id in original_ids
                    if not options.gemini_speaker_voices.get(speaker_id, "").strip()
                ]
                if missing:
                    raise ValueError(
                        "Gemini requires an explicit voice preset for every detected speaker. Missing: "
                        + ", ".join(f"Speaker {speaker_id}" for speaker_id in missing)
                    )
            selected = {
                speaker_id: options.gemini_speaker_voices.get(speaker_id, options.gemini_voice).strip()
                for speaker_id in original_ids
            }
            if len(selected) == 1:
                options.gemini_voice = next(iter(selected.values()))
            self._stage(
                0.10,
                f"Generating Gemini TTS ({len(selected) or 1} speaker{'s' if len(selected) != 1 else ''})",
            )
            return generate_gemini(
                self.root,
                story_path,
                audio_path,
                voice=options.gemini_voice,
                model=options.gemini_model,
                preprocess=options.gemini_preprocess,
                split_on_separator=options.gemini_split_on_separator,
                chunk_seconds=options.gemini_chunk_seconds,
                log=self.log,
                text=prepared_text,
                speaker_voices=selected,
            )

        if engine == "vibevoice":
            if explicit_speakers:
                missing = [
                    speaker_id
                    for speaker_id in original_ids
                    if not options.vibevoice_speaker_voices.get(speaker_id, "").strip()
                ]
                if missing:
                    raise ValueError(
                        "VibeVoice requires an explicit preset for every detected speaker. Missing: "
                        + ", ".join(f"Speaker {speaker_id}" for speaker_id in missing)
                    )
            selected_original = {
                speaker_id: options.vibevoice_speaker_voices.get(speaker_id, options.vibevoice_speaker).strip()
                for speaker_id in original_ids
            }
            selected = {id_map[speaker_id]: voice for speaker_id, voice in selected_original.items()}
            self._stage(0.10, f"Generating VibeVoice ({len(selected)} speaker(s)) in one pass")
            return generate_vibevoice(
                self.root,
                prepared_text,
                audio_path,
                model_id=options.vibevoice_model,
                speaker_name=options.vibevoice_speaker,
                speaker_names=selected,
                cfg_scale=options.vibevoice_cfg_scale,
                diffusion_steps=options.vibevoice_diffusion_steps,
                seed=options.vibevoice_seed,
                device=options.vibevoice_device,
                dtype_name=options.vibevoice_dtype,
                log=self.log,
            )

        selected_refs = {
            id_map[speaker_id]: reference
            for speaker_id, reference in options.fish_speaker_references.items()
            if speaker_id in id_map
        }
        if not explicit_speakers and not selected_refs and options.fish_reference_audio:
            selected_refs[0] = (options.fish_reference_audio, options.fish_reference_text)
        missing = [
            speaker_id
            for speaker_id in original_ids
            if id_map[speaker_id] not in selected_refs
        ]
        if missing:
            raise ValueError(
                "Fish Audio requires an explicit saved preset or uploaded reference voice for every speaker; "
                "random model-selected voices are disabled. Missing: "
                + ", ".join(f"Speaker {speaker_id}" for speaker_id in missing)
            )
        self._stage(0.10, f"Fish Audio: starting S2 Pro for {len(speakers) or 1} speaker(s)")
        return generate_fish_s2(
            self.root,
            prepared_text,
            audio_path,
            device=options.fish_device,
            gpu_layers=options.fish_gpu_layers,
            half=options.fish_half,
            temperature=options.fish_temperature,
            seed=options.fish_seed,
            reference_audio=options.fish_reference_audio,
            reference_text=options.fish_reference_text,
            speaker_references=selected_refs,
            log=self.log,
        )

    def _validate_narration_duration(self, story_text: str, audio_duration: float) -> None:
        words = re.findall(r"\b[\w'-]+\b", story_text, flags=re.UNICODE)
        if len(words) < 40:
            return
        minimum_plausible_seconds = len(words) / 5.0  # 300 WPM: intentionally very permissive.
        if audio_duration < minimum_plausible_seconds:
            raise RuntimeError(
                "Narration is implausibly short and was probably truncated: "
                f"{len(words)} story words produced only {audio_duration:.1f}s of audio. "
                "Rendering was stopped so a broken video is not produced."
            )

    def _probe_dimensions(self, path: Path) -> tuple[int, int]:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                "-of", "csv=p=0:s=x", str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        width, height = result.stdout.strip().split("x", 1)
        return int(width), int(height)

    def _find_windows_shared_ffmpeg(self) -> Path | None:
        if os.name != "nt":
            return None
        override = os.getenv("WHISPERX_FFMPEG_DIR", "").strip()
        candidates: list[Path] = [Path(override)] if override else []
        try:
            found = subprocess.run(
                ["where.exe", "ffmpeg"], capture_output=True, text=True, check=False
            )
            candidates.extend(Path(line.strip()).parent for line in found.stdout.splitlines() if line.strip())
        except OSError:
            pass
        winget = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
        if winget.exists():
            candidates.extend(path.parent for path in winget.rglob("ffmpeg.exe"))
        seen: set[Path] = set()
        for directory in candidates:
            try:
                directory = directory.resolve()
            except OSError:
                continue
            if directory in seen or not directory.exists():
                continue
            seen.add(directory)
            # TorchCodec 0.7 / torch 2.8 supports FFmpeg 4-7. Shared Windows
            # distributions expose these avcodec DLL majors alongside ffmpeg.exe.
            if any((directory / f"avcodec-{major}.dll").exists() for major in (58, 59, 60, 61)):
                return directory
        return None

    def _whisper_command(self, options: PipelineOptions) -> tuple[list[str] | None, str | None]:
        configured = options.whisperx_command.strip() or os.getenv("WHISPERX_COMMAND", "").strip()
        local_python = self.root / ".whisperx-venv" / "Scripts" / "python.exe"
        local_whisperx = self.root / ".whisperx-venv" / "Scripts" / "whisperx.exe"
        if local_whisperx.exists() and not configured:
            ffmpeg_dir = self._find_windows_shared_ffmpeg()
            if ffmpeg_dir is not None and local_python.exists():
                bootstrap = (
                    "import os, warnings; "
                    "warnings.filterwarnings(\"ignore\", message=\"TensorFloat-32.*\"); "
                    "os.environ.setdefault(\"HF_HUB_DISABLE_XET\", \"1\"); "
                    f"os.add_dll_directory({str(ffmpeg_dir)!r}); "
                    "from whisperx.__main__ import cli; cli()"
                )
                return [str(local_python), "-c", bootstrap], None
            return [str(local_whisperx)], None
        direct = shutil.which("whisperx")
        if direct and not configured:
            return [direct], None
        if configured:
            if os.name == "nt":
                return None, configured
            return shlex.split(configured), None
        conda = shutil.which("conda")
        if conda:
            env_name = os.getenv("WHISPERX_CONDA_ENV", "whisperx")
            return [conda, "run", "-n", env_name, "whisperx"], None
        raise RuntimeError(
            "WhisperX was not found. Install it so `whisperx` is on PATH, or set WHISPERX_COMMAND in .env "
            "(for example: conda run -n whisperx whisperx)."
        )

    def _run_streamed(self, command: list[str] | str, cwd: Path, shell: bool = False) -> None:
        display = command if isinstance(command, str) else subprocess.list2cmdline(command)
        self.log("$ " + display)
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            shell=shell,
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
            if not line:
                continue
            if "Lightning automatically upgraded your loaded checkpoint" in line:
                continue
            self.log(line)
        code = process.wait()
        if code != 0:
            raise RuntimeError(f"Command failed with exit code {code}: {display}")

    def _generate_captions(
        self,
        audio_path: Path,
        work_dir: Path,
        caption_path: Path,
        options: PipelineOptions,
        resolution: tuple[int, int],
    ) -> Path:
        whisper_dir = work_dir / "whisperx"
        whisper_dir.mkdir(parents=True, exist_ok=True)
        base_command, shell_prefix = self._whisper_command(options)
        args = [
            str(audio_path),
            "--model", options.whisper_model,
            "--output_format", "json",
            "--highlight_words", "True",
            "--compute_type", options.whisper_compute_type,
            "--output_dir", str(whisper_dir),
            "--log-level", "error",
        ]
        if options.whisper_language.strip().lower() not in {"", "auto"}:
            args.extend(["--language", options.whisper_language.strip()])
        if options.whisper_align_model.strip():
            args.extend(["--align_model", options.whisper_align_model.strip()])

        if shell_prefix is not None:
            command = shell_prefix + " " + subprocess.list2cmdline(args)
            self._run_streamed(command, self.root, shell=True)
        else:
            assert base_command is not None
            self._run_streamed(base_command + args, self.root)

        whisper_json = whisper_dir / f"{audio_path.stem}.json"
        if not whisper_json.exists():
            candidates = list(whisper_dir.glob("*.json"))
            if len(candidates) == 1:
                whisper_json = candidates[0]
            else:
                raise RuntimeError(f"WhisperX finished but no JSON output was found in {whisper_dir}")

        width, height = resolution
        convert_whisperx_json_to_ass(
            whisper_json,
            caption_path,
            theme_name=options.caption_theme,
            max_words=(options.caption_max_words or None),
            pause_threshold=options.caption_pause_threshold,
            width=width,
            height=height,
        )
        final_json = self.root / "output" / "captions" / f"{caption_path.stem}.json"
        shutil.copyfile(whisper_json, final_json)
        return final_json

    def _resolve_encoder(self, requested: str) -> str:
        if requested == "cpu":
            return "libx264"
        if requested == "nvenc":
            return "h264_nvenc"
        result = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, check=True)
        return "h264_nvenc" if "h264_nvenc" in result.stdout else "libx264"

    def _render(
        self,
        background: Path,
        audio: Path,
        caption: Path | None,
        output: Path,
        options: PipelineOptions,
        duration: float,
    ) -> None:
        background_duration = self._probe_duration(background)
        start_offset = 0.0
        if options.randomize_background_start and background_duration > 2:
            start_offset = random.uniform(0, max(0.0, background_duration - 1.0))

        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-stream_loop", "-1"]
        if start_offset > 0:
            command.extend(["-ss", f"{start_offset:.3f}"])
        command.extend(["-i", str(background), "-i", str(audio)])

        filters: list[str] = []
        if options.output_format == "shorts":
            filters.extend([
                "scale=1080:1920:force_original_aspect_ratio=increase",
                "crop=1080:1920",
                "setsar=1",
            ])
        elif options.output_format != "source":
            raise ValueError(f"Unknown output format: {options.output_format}")

        if caption is not None:
            relative = caption.relative_to(self.root).as_posix().replace("'", "\\'")
            filters.append(f"subtitles=filename='{relative}'")
        if filters:
            command.extend(["-vf", ",".join(filters)])

        encoder = self._resolve_encoder(options.encoder)
        if encoder == "h264_nvenc":
            command.extend(["-c:v", encoder, "-preset", "p6", "-rc", "vbr", "-cq", str(options.video_quality), "-b:v", "0"])
        else:
            command.extend(["-c:v", encoder, "-preset", "slow", "-crf", str(options.video_quality)])

        padded_duration = duration + max(0.0, float(options.end_padding_seconds))
        if options.end_padding_seconds > 0:
            command.extend(["-af", f"apad=pad_dur={float(options.end_padding_seconds):.3f}"])
        command.extend([
            "-map", "0:v:0", "-map", "1:a:0",
            "-t", f"{padded_duration:.3f}",
            "-c:a", "aac", "-b:a", options.audio_bitrate,
            "-movflags", "+faststart",
            str(output),
        ])
        self._run_streamed(command, self.root)

    def run_audio(self, options: PipelineOptions) -> AudioPipelineResult:
        started = time.perf_counter()
        story_path, story_text, base = self._prepare_story(options)
        work_dir = self.root / ".work" / "audio_tests"
        work_dir.mkdir(parents=True, exist_ok=True)
        audio_path = work_dir / f"{base}_{slugify(options.tts_engine, 'tts')}.wav"
        self._stage(0.05, f"Story loaded: {story_path.name}")
        self._generate_narration(options, story_path, story_text, audio_path)
        duration = self._audio_duration(audio_path)
        self._validate_narration_duration(story_text, duration)
        elapsed = time.perf_counter() - started
        self._stage(1.0, f"Audio ready: {audio_path}")
        return AudioPipelineResult(audio_path=audio_path, elapsed_seconds=elapsed, duration_seconds=duration)

    def run(self, options: PipelineOptions) -> PipelineResult:
        started = time.perf_counter()
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            raise RuntimeError("ffmpeg and ffprobe must be available on PATH.")
        if options.caption_theme not in CAPTION_THEMES:
            raise ValueError(f"Unknown caption theme: {options.caption_theme}")

        story_path, story_text, base = self._prepare_story(options)
        background_value = str(options.background)
        if background_value.lower() in {"asmr", "minecraft"}:
            matches = sorted((self.root / "videos" / background_value.lower()).glob("*.mp4"))
            if not matches:
                raise FileNotFoundError(f"No background videos found for category: {background_value}")
            background = matches[0]
        else:
            background = self._resolve_path(options.background)
        if not background.exists():
            raise FileNotFoundError(f"Background video not found: {background}")

        work_dir = self.root / ".work" / base
        work_dir.mkdir(parents=True, exist_ok=True)
        audio_path = work_dir / f"{base}.wav"
        output_path = self.root / "output" / f"{base}_final.mp4"
        caption_path = self.root / "output" / "captions" / f"{base}_{options.caption_theme}.ass" if options.captions else None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        (self.root / "output" / "captions").mkdir(parents=True, exist_ok=True)

        self._stage(0.05, f"Story loaded: {story_path.name}")
        self._generate_narration(options, story_path, story_text, audio_path)

        audio_duration = self._audio_duration(audio_path)
        self._validate_narration_duration(story_text, audio_duration)
        self.log(f"Narration duration: {audio_duration:.2f}s")

        whisper_json: Path | None = None
        if options.output_format == "shorts":
            resolution = (1080, 1920)
        else:
            resolution = self._probe_dimensions(background)

        if caption_path is not None:
            self._stage(0.58, f"Transcribing and styling captions: {CAPTION_THEMES[options.caption_theme].label}")
            whisper_json = self._generate_captions(audio_path, work_dir, caption_path, options, resolution)

        self._stage(0.82, f"Rendering final {options.output_format} video")
        self._render(background, audio_path, caption_path, output_path, options, audio_duration)

        elapsed = time.perf_counter() - started
        self._stage(1.0, f"Finished: {output_path}")
        return PipelineResult(output_path, audio_path, caption_path, whisper_json, elapsed)
