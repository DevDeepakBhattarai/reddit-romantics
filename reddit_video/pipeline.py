from __future__ import annotations

import os
import random
import re
import shlex
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

from dotenv import load_dotenv

from .captions import CAPTION_THEMES, convert_whisperx_json_to_ass, trim_whisperx_json
from .tts import generate_gemini, generate_vibevoice
from .fish import generate_fish_s2, resolve_fish_reference_preset
from .tts_text import infer_speaker_gender, prepare_text_for_provider, validate_speaker_count
from .runs import StoryRun, create_story_run, list_story_runs, resolve_story_run, slugify, title_from_story
from .shorts import resolve_cliffhanger_time, split_story_at_cliffhanger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

LogFn = Callable[[str], None]
ProgressFn = Callable[[float, str], None]


@dataclass
class PipelineOptions:
    run_dir: str | Path | None = None
    story_file: str | Path | None = None
    story_text: str = ""
    output_name: str = ""

    tts_engine: str = "fish"
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

    fish_gpu_layers: int = 20
    fish_temperature: float = 1.0
    fish_reference_audio: str | Path | None = None
    fish_reference_text: str = ""
    fish_speaker_references: dict[int, tuple[str | Path, str]] = field(default_factory=dict)

    background: str | Path = "videos/minecraft/minecraft.mp4"
    output_format: str = "source"
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
    run_dir: Path
    video_path: Path
    audio_path: Path
    caption_path: Path | None
    whisper_json_path: Path | None
    elapsed_seconds: float
    short_video_path: Path | None = None
    short_end_seconds: float | None = None



def list_input_stories(root: Path = PROJECT_ROOT) -> list[str]:
    return [f"{run}/story.md" for run in list_story_runs(root)]



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

    def _prepare_story(self, options: PipelineOptions) -> tuple[StoryRun, Path, str, str]:
        text = options.story_text.strip()
        source_path: Path | None = None
        if options.story_file:
            source_path = self._resolve_path(options.story_file)
            if not source_path.exists():
                raise FileNotFoundError(f"Story file not found: {source_path}")
            if not text:
                text = source_path.read_text(encoding="utf-8-sig").strip()

        if options.run_dir:
            run_path = self._resolve_path(options.run_dir)
            run_path.mkdir(parents=True, exist_ok=True)
            run = StoryRun(run_path)
        elif source_path is not None and source_path.name.lower() in {"story.md", "story.txt"}:
            runs_root = (self.root / "runs").resolve()
            try:
                source_path.parent.resolve().relative_to(runs_root)
                run = StoryRun(source_path.parent.resolve())
            except ValueError:
                run = create_story_run(self.root, options.output_name or source_path.stem)
        else:
            run = create_story_run(self.root, options.output_name or title_from_story(text))

        if not text and run.story.exists():
            text = run.story.read_text(encoding="utf-8-sig").strip()
        if not text:
            raise ValueError("Story is empty. Put the reviewed story in story.md before generating video.")

        run.story.write_text(text, encoding="utf-8")
        return run, run.story, text, run.path.name

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

    def _resolve_default_fish_voice(self, gender: str) -> tuple[str, tuple[Path, str]]:
        candidates = ("Ethan",) if gender == "male" else ("Sarah",)
        errors: list[str] = []
        for preset in candidates:
            try:
                return preset, resolve_fish_reference_preset(self.root, preset)
            except (FileNotFoundError, ValueError) as exc:
                errors.append(str(exc))
        preferred = candidates[0]
        raise FileNotFoundError(
            f"Default Fish {gender} preset '{preferred}' is unavailable. "
            + " | ".join(errors)
        )

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

        auto_cast: list[str] = []
        for speaker in speakers:
            mapped_id = id_map[speaker.speaker_id]
            if mapped_id in selected_refs:
                continue
            gender = infer_speaker_gender(speaker)
            if gender is None:
                raise ValueError(
                    f"Speaker {speaker.speaker_id} has no gender metadata for automatic Fish casting. "
                    "Production stories must include a non-spoken line such as "
                    f"'Speaker {speaker.speaker_id} - gender=male; narrator' or "
                    f"'Speaker {speaker.speaker_id} - gender=female; main counterpart'."
                )
            preset, reference = self._resolve_default_fish_voice(gender)
            selected_refs[mapped_id] = reference
            auto_cast.append(f"Speaker {speaker.speaker_id}={gender}->{preset}")

        missing = [
            speaker_id
            for speaker_id in original_ids
            if id_map[speaker_id] not in selected_refs
        ]
        if missing:
            raise ValueError(
                "Fish Audio could not resolve a voice for: "
                + ", ".join(f"Speaker {speaker_id}" for speaker_id in missing)
            )
        if auto_cast:
            self.log("Fish automatic casting: " + "; ".join(auto_cast))
        self._stage(0.10, f"Fish Audio S2 Pro: starting {len(speakers) or 1} speaker(s)")
        return generate_fish_s2(
            self.root,
            prepared_text,
            audio_path,
            gpu_layers=options.fish_gpu_layers,
            temperature=options.fish_temperature,
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

    def _transcribe_and_style(
        self,
        audio_path: Path,
        work_dir: Path,
        transcript_path: Path,
        caption_path: Path | None,
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

        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(whisper_json, transcript_path)
        if caption_path is not None:
            width, height = resolution
            convert_whisperx_json_to_ass(
                transcript_path,
                caption_path,
                theme_name=options.caption_theme,
                max_words=(options.caption_max_words or None),
                pause_threshold=options.caption_pause_threshold,
                width=width,
                height=height,
            )
        return transcript_path

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

    def _resolve_background(self, value: str | Path) -> Path:
        background_value = str(value)
        if background_value.lower() in {"asmr", "minecraft"}:
            matches = sorted((self.root / "videos" / background_value.lower()).glob("*.mp4"))
            if not matches:
                raise FileNotFoundError(f"No background videos found for category: {background_value}")
            return matches[0]
        background = self._resolve_path(value)
        if not background.exists():
            raise FileNotFoundError(f"Background video not found: {background}")
        return background

    def _extract_audio(self, source: Path, output: Path, end_time: float) -> Path:
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source), "-t", f"{end_time:.3f}",
            "-c:a", "pcm_s16le", str(output),
        ]
        self._run_streamed(command, self.root)
        return output

    def _render_short_from_cutoff(
        self,
        run: StoryRun,
        end_time: float,
        options: PipelineOptions,
        background: Path,
    ) -> None:
        short_options = replace(
            options,
            output_format="shorts",
            end_padding_seconds=min(float(options.end_padding_seconds), 0.25),
        )
        self._extract_audio(run.narration, run.short_audio, end_time)
        trim_whisperx_json(run.transcript, run.short_transcript, end_time)

        caption_path: Path | None = None
        if short_options.captions:
            caption_path = run.short_captions
            convert_whisperx_json_to_ass(
                run.short_transcript,
                caption_path,
                theme_name=short_options.caption_theme,
                max_words=(short_options.caption_max_words or None),
                pause_threshold=short_options.caption_pause_threshold,
                width=1080,
                height=1920,
            )

        self._render(background, run.short_audio, caption_path, run.short_video, short_options, end_time)

    def run(self, options: PipelineOptions) -> PipelineResult:
        started = time.perf_counter()
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            raise RuntimeError("ffmpeg and ffprobe must be available on PATH.")
        if options.caption_theme not in CAPTION_THEMES:
            raise ValueError(f"Unknown caption theme: {options.caption_theme}")

        run, story_path, story_text, base = self._prepare_story(options)
        spoken_story_text, short_prefix = split_story_at_cliffhanger(story_text)
        background = self._resolve_background(options.background)
        work_dir = self.root / ".work" / base
        work_dir.mkdir(parents=True, exist_ok=True)
        caption_path = run.captions if options.captions else None

        self._stage(0.05, f"Story loaded: {story_path}")
        self._generate_narration(options, story_path, spoken_story_text, run.narration)

        audio_duration = self._audio_duration(run.narration)
        self._validate_narration_duration(spoken_story_text, audio_duration)
        self.log(f"Narration duration: {audio_duration:.2f}s")

        resolution = (1080, 1920) if options.output_format == "shorts" else self._probe_dimensions(background)
        self._stage(0.58, "Transcribing narration for captions and automatic Short alignment")
        transcript = self._transcribe_and_style(
            run.narration,
            work_dir,
            run.transcript,
            caption_path,
            options,
            resolution,
        )

        self._stage(0.78, f"Rendering full {options.output_format} video")
        self._render(background, run.narration, caption_path, run.full_video, options, audio_duration)

        short_video: Path | None = None
        short_end: float | None = None
        if short_prefix is not None:
            short_end = resolve_cliffhanger_time(story_text, run.transcript)
            assert short_end is not None
            self._stage(0.90, f"Rendering automatic cliffhanger Short through {short_end:.2f}s")
            self._render_short_from_cutoff(run, short_end, options, background)
            short_video = run.short_video
        else:
            self.log("No Shorts cliffhanger marker found; full video completed and Short was skipped.")

        elapsed = time.perf_counter() - started
        self._stage(1.0, f"Finished: {run.full_video}")
        return PipelineResult(
            run.path,
            run.full_video,
            run.narration,
            caption_path,
            transcript,
            elapsed,
            short_video,
            short_end,
        )

    def render_short(
        self,
        run_dir: str | Path,
        options: PipelineOptions | None = None,
    ) -> PipelineResult:
        """Programmatically rebuild a Short from a run's story marker and existing transcript."""
        started = time.perf_counter()
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            raise RuntimeError("ffmpeg and ffprobe must be available on PATH.")

        options = options or PipelineOptions()
        if options.caption_theme not in CAPTION_THEMES:
            raise ValueError(f"Unknown caption theme: {options.caption_theme}")
        run = resolve_story_run(self.root, run_dir)
        if not run.story.exists():
            raise FileNotFoundError(f"Story is missing: {run.story}")
        if not run.narration.exists():
            raise FileNotFoundError(f"Full narration is missing: {run.narration}")
        if not run.transcript.exists():
            raise FileNotFoundError(f"Transcript is missing: {run.transcript}")

        story_text = run.story.read_text(encoding="utf-8-sig")
        end_time = resolve_cliffhanger_time(story_text, run.transcript)
        if end_time is None:
            raise ValueError("Story has no Shorts cliffhanger marker, so there is no programmatic cutoff to render.")

        narration_duration = self._audio_duration(run.narration)
        if end_time > narration_duration:
            raise ValueError(
                f"Short end time {end_time:.2f}s exceeds narration duration {narration_duration:.2f}s."
            )

        background = self._resolve_background(options.background)
        self._stage(0.15, f"Reusing narration/transcript; marker resolves to {end_time:.2f}s")
        self._render_short_from_cutoff(run, end_time, options, background)
        elapsed = time.perf_counter() - started
        self._stage(1.0, f"Finished: {run.short_video}")
        return PipelineResult(
            run.path,
            run.short_video,
            run.short_audio,
            run.short_captions if options.captions else None,
            run.short_transcript,
            elapsed,
            run.short_video,
            end_time,
        )
