from __future__ import annotations

import argparse
from pathlib import Path

from reddit_video.captions import CAPTION_THEMES
from reddit_video.pipeline import (
    PipelineOptions,
    RedditVideoPipeline,
    list_background_videos,
    list_input_stories,
)
from reddit_video.tts import list_vibevoice_presets

ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reddit Romantics video automation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Generate a video from a story")
    run.add_argument("--story-file", help="Story text file. Relative paths are resolved from the repo root.")
    run.add_argument("--story-text", default="", help="Story text directly on the command line")
    run.add_argument("--output-name", default="")

    run.add_argument("--tts", choices=["gemini", "vibevoice"], default="gemini")
    run.add_argument("--gemini-voice", default="Kore")
    run.add_argument("--gemini-model", default="gemini-3.1-flash-tts-preview")
    run.add_argument("--no-gemini-preprocess", action="store_true")
    run.add_argument("--no-gemini-split", action="store_true", help="Ignore explicit ------------- boundaries; automatic semantic chunking remains enabled")
    run.add_argument("--gemini-chunk-seconds", type=int, default=180, help="Target Gemini narration duration per semantic chunk")

    run.add_argument("--vibevoice-model", default="microsoft/VibeVoice-1.5B")
    run.add_argument("--vibevoice-speaker", default="Alice")
    run.add_argument("--vibevoice-cfg-scale", type=float, default=1.3)
    run.add_argument("--vibevoice-diffusion-steps", type=int, default=10)
    run.add_argument("--vibevoice-seed", type=int, default=42)
    run.add_argument("--vibevoice-device", choices=["auto", "cuda", "cpu"], default="auto")
    run.add_argument("--vibevoice-dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")

    run.add_argument("--background", default="videos/minecraft/minecraft.mp4")
    run.add_argument("--format", dest="output_format", choices=["shorts", "source"], default="shorts")
    run.add_argument("--no-random-background-start", action="store_true")
    run.add_argument("--end-padding", type=float, default=1.0)

    run.add_argument("--no-captions", action="store_true")
    run.add_argument("--caption-theme", choices=list(CAPTION_THEMES), default="classic_yellow")
    run.add_argument("--caption-max-words", type=int, default=0)
    run.add_argument("--caption-pause", type=float, default=0.5)
    run.add_argument("--whisper-model", default="large-v2")
    run.add_argument("--whisper-language", default="en", help="Language code, or auto")
    run.add_argument("--whisper-align-model", default="WAV2VEC2_ASR_LARGE_LV60K_960H")
    run.add_argument("--whisper-compute-type", default="float16")
    run.add_argument("--whisperx-command", default="")

    run.add_argument("--encoder", choices=["auto", "nvenc", "cpu"], default="auto")
    run.add_argument("--video-quality", type=int, default=20)
    run.add_argument("--audio-bitrate", default="128k")

    ui = subparsers.add_parser("ui", help="Launch the Gradio browser UI")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=7860)
    ui.add_argument("--share", action="store_true")

    subparsers.add_parser("list", help="List stories, backgrounds, caption themes, and VibeVoice presets")
    return parser


def run_pipeline(args: argparse.Namespace) -> int:
    options = PipelineOptions(
        story_file=args.story_file,
        story_text=args.story_text,
        output_name=args.output_name,
        tts_engine=args.tts,
        gemini_voice=args.gemini_voice,
        gemini_model=args.gemini_model,
        gemini_preprocess=not args.no_gemini_preprocess,
        gemini_split_on_separator=not args.no_gemini_split,
        gemini_chunk_seconds=args.gemini_chunk_seconds,
        vibevoice_model=args.vibevoice_model,
        vibevoice_speaker=args.vibevoice_speaker,
        vibevoice_cfg_scale=args.vibevoice_cfg_scale,
        vibevoice_diffusion_steps=args.vibevoice_diffusion_steps,
        vibevoice_seed=args.vibevoice_seed,
        vibevoice_device=args.vibevoice_device,
        vibevoice_dtype=args.vibevoice_dtype,
        background=args.background,
        output_format=args.output_format,
        randomize_background_start=not args.no_random_background_start,
        end_padding_seconds=args.end_padding,
        captions=not args.no_captions,
        caption_theme=args.caption_theme,
        caption_max_words=args.caption_max_words,
        caption_pause_threshold=args.caption_pause,
        whisper_model=args.whisper_model,
        whisper_language=args.whisper_language,
        whisper_align_model=args.whisper_align_model,
        whisper_compute_type=args.whisper_compute_type,
        whisperx_command=args.whisperx_command,
        encoder=args.encoder,
        video_quality=args.video_quality,
        audio_bitrate=args.audio_bitrate,
    )
    result = RedditVideoPipeline().run(options)
    print(f"\nVideo: {result.video_path}")
    print(f"Audio: {result.audio_path}")
    if result.caption_path:
        print(f"Captions: {result.caption_path}")
    return 0


def main() -> int:
    args = build_parser().parse_args()
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
        print("Stories:")
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
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
