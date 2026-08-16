from __future__ import annotations

import argparse
import sys
from pathlib import Path

from reddit_video.captions import CAPTION_THEMES
from reddit_video.fish import list_fish_reference_presets, resolve_fish_reference_preset
from reddit_video.pipeline import PipelineOptions, RedditVideoPipeline, list_background_videos, list_input_stories
from reddit_video.runs import create_story_run, list_story_runs
from reddit_video.tts import list_vibevoice_presets

ROOT = Path(__file__).resolve().parent


def _speaker_map(values: list[str]) -> dict[int, str]:
    result: dict[int, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid speaker assignment '{value}'. Use SPEAKER_ID=PRESET.")
        speaker, preset = value.split("=", 1)
        result[int(speaker.strip())] = preset.strip()
    return result


def _add_render_args(parser: argparse.ArgumentParser, *, full: bool) -> None:
    parser.add_argument("--background", default="videos/minecraft/minecraft.mp4")
    if full:
        parser.add_argument("--format", dest="output_format", choices=["source", "shorts"], default="source")
    parser.add_argument("--no-random-background-start", action="store_true")
    parser.add_argument("--end-padding", type=float, default=1.0 if full else 0.25)
    parser.add_argument("--no-captions", action="store_true")
    parser.add_argument("--caption-theme", choices=list(CAPTION_THEMES), default="classic_yellow")
    parser.add_argument("--caption-max-words", type=int, default=0)
    parser.add_argument("--caption-pause", type=float, default=0.5)
    parser.add_argument("--encoder", choices=["auto", "nvenc", "cpu"], default="auto")
    parser.add_argument("--video-quality", type=int, default=20)
    parser.add_argument("--audio-bitrate", default="128k")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reddit Romantics video automation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    new_run = subparsers.add_parser("new-run", help="Create one dated story workspace under runs/")
    new_run.add_argument("--title", required=True, help="Short human-readable story title")

    run = subparsers.add_parser("run", help="Generate the full video and transcript for one story run")
    run.add_argument("--run-dir", help="Existing runs/<timestamp>_<title> folder. Reads story.md from it.")
    run.add_argument("--story-file", help="Story file. Legacy/external files are copied into a new run folder.")
    run.add_argument("--story-text", default="", help="Story text directly on the command line")
    run.add_argument("--output-name", default="", help="Title used only when a new run folder must be created")

    run.add_argument("--tts", choices=["gemini", "vibevoice", "fish"], default="fish")
    run.add_argument(
        "--speaker-preset",
        action="append",
        default=[],
        metavar="ID=PRESET",
        help="Optional voice override. Fish auto-casts from story gender metadata when omitted; repeat for manual overrides.",
    )
    run.add_argument("--gemini-voice", default="Kore")
    run.add_argument("--gemini-model", default="gemini-3.1-flash-tts-preview")
    run.add_argument("--no-gemini-preprocess", action="store_true")
    run.add_argument("--no-gemini-split", action="store_true")
    run.add_argument("--gemini-chunk-seconds", type=int, default=180)
    run.add_argument("--vibevoice-model", default="microsoft/VibeVoice-1.5B")
    run.add_argument("--vibevoice-speaker", default="Alice")
    run.add_argument("--vibevoice-cfg-scale", type=float, default=1.3)
    run.add_argument("--vibevoice-diffusion-steps", type=int, default=10)
    run.add_argument("--vibevoice-seed", type=int, default=42)
    run.add_argument("--vibevoice-device", choices=["auto", "cuda", "cpu"], default="auto")
    run.add_argument("--vibevoice-dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    run.add_argument("--fish-gpu-layers", type=int, default=28)
    run.add_argument("--fish-temperature", type=float, default=1.0)
    run.add_argument("--fish-reference-audio")
    run.add_argument("--fish-reference-text", default="")

    _add_render_args(run, full=True)
    run.add_argument("--whisper-model", default="large-v2")
    run.add_argument("--whisper-language", default="en")
    run.add_argument("--whisper-align-model", default="WAV2VEC2_ASR_LARGE_LV60K_960H")
    run.add_argument("--whisper-compute-type", default="float16")
    run.add_argument("--whisperx-command", default="")


    ui = subparsers.add_parser("ui", help="Launch the Gradio browser UI")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=7860)
    ui.add_argument("--share", action="store_true")

    subparsers.add_parser("list", help="List story runs, backgrounds, caption themes, and VibeVoice presets")
    return parser


def _render_options(args: argparse.Namespace) -> dict:
    return dict(
        background=args.background,
        randomize_background_start=not args.no_random_background_start,
        end_padding_seconds=args.end_padding,
        captions=not args.no_captions,
        caption_theme=args.caption_theme,
        caption_max_words=args.caption_max_words,
        caption_pause_threshold=args.caption_pause,
        encoder=args.encoder,
        video_quality=args.video_quality,
        audio_bitrate=args.audio_bitrate,
    )


def run_pipeline(args: argparse.Namespace) -> int:
    assignments = _speaker_map(args.speaker_preset)
    fish_references = {}
    if args.tts == "fish":
        fish_references = {
            speaker_id: resolve_fish_reference_preset(ROOT, preset)
            for speaker_id, preset in assignments.items()
        }

    options = PipelineOptions(
        run_dir=args.run_dir,
        story_file=args.story_file,
        story_text=args.story_text,
        output_name=args.output_name,
        tts_engine=args.tts,
        gemini_voice=args.gemini_voice,
        gemini_model=args.gemini_model,
        gemini_preprocess=not args.no_gemini_preprocess,
        gemini_split_on_separator=not args.no_gemini_split,
        gemini_chunk_seconds=args.gemini_chunk_seconds,
        gemini_speaker_voices=assignments if args.tts == "gemini" else {},
        vibevoice_model=args.vibevoice_model,
        vibevoice_speaker=args.vibevoice_speaker,
        vibevoice_cfg_scale=args.vibevoice_cfg_scale,
        vibevoice_diffusion_steps=args.vibevoice_diffusion_steps,
        vibevoice_seed=args.vibevoice_seed,
        vibevoice_device=args.vibevoice_device,
        vibevoice_dtype=args.vibevoice_dtype,
        vibevoice_speaker_voices=assignments if args.tts == "vibevoice" else {},
        fish_gpu_layers=args.fish_gpu_layers,
        fish_temperature=args.fish_temperature,
        fish_reference_audio=args.fish_reference_audio,
        fish_reference_text=args.fish_reference_text,
        fish_speaker_references=fish_references,
        output_format=args.output_format,
        whisper_model=args.whisper_model,
        whisper_language=args.whisper_language,
        whisper_align_model=args.whisper_align_model,
        whisper_compute_type=args.whisper_compute_type,
        whisperx_command=args.whisperx_command,
        **_render_options(args),
    )
    result = RedditVideoPipeline().run(options)
    print(f"\nRun: {result.run_dir}")
    print(f"Video: {result.video_path}")
    print(f"Audio: {result.audio_path}")
    print(f"Transcript: {result.whisper_json_path}")
    if result.caption_path:
        print(f"Captions: {result.caption_path}")
    if result.short_video_path:
        print(f"Short: {result.short_video_path}")
        print(f"Short cutoff: {result.short_end_seconds:.2f}s")
    return 0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args()
    if args.command == "new-run":
        run = create_story_run(ROOT, args.title)
        run.story.touch()
        print(run.path)
        return 0
    if args.command == "run":
        return run_pipeline(args)
    if args.command == "ui":
        from reddit_video.ui import build_ui

        build_ui().queue(default_concurrency_limit=1).launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            show_error=True,
        )
        return 0
    if args.command == "list":
        print("Story runs:")
        for item in list_story_runs(ROOT):
            print(f"  {item}")
        print("\nStory files:")
        for item in list_input_stories():
            print(f"  {item}")
        print("\nBackgrounds:")
        for item in list_background_videos():
            print(f"  {item}")
        print("\nCaption themes:")
        for key, theme in CAPTION_THEMES.items():
            print(f"  {key}: {theme.label}")
        print("\nVibeVoice presets:")
        presets = list_vibevoice_presets(ROOT / "vendor" / "VibeVoice")
        if not presets:
            print("  (runtime not installed; run setup.ps1)")
        else:
            for item in presets:
                print(f"  {item}")
        print("\nFish presets:")
        fish_presets = list_fish_reference_presets(ROOT)
        if not fish_presets:
            print("  (no saved presets)")
        else:
            for item in fish_presets:
                print(f"  {item}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
