from __future__ import annotations

import traceback

import gradio as gr

from .captions import theme_choices
from .pipeline import (
    PROJECT_ROOT,
    PipelineOptions,
    RedditVideoPipeline,
    list_background_videos,
    list_input_stories,
)
from .tts import list_vibevoice_presets

GEMINI_VOICES = [
    "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede", "Callirrhoe", "Autonoe",
    "Enceladus", "Iapetus", "Umbriel", "Algieba", "Despina", "Erinome", "Algenib", "Rasalgethi",
    "Laomedeia", "Achernar", "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi",
    "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat",
]


def _vibe_choices() -> list[str]:
    installed = list_vibevoice_presets(PROJECT_ROOT / "vendor" / "VibeVoice")
    return installed or ["Alice", "Frank"]


def _first_or_none(values: list[str], preferred: str | None = None) -> str | None:
    if preferred and preferred in values:
        return preferred
    return values[0] if values else None


def build_ui() -> gr.Blocks:
    stories = list_input_stories()
    backgrounds = list_background_videos()
    vibe_voices = _vibe_choices()

    with gr.Blocks(title="Reddit Romantics Automation") as demo:
        gr.Markdown(
            "# Reddit Romantics Video Automation\n"
            "Paste a story or choose an input file, pick the narration engine, background, caption theme, and render settings. "
            "The same pipeline is also available from `main.py run` for scripting."
        )

        with gr.Row():
            with gr.Column(scale=3):
                story_file = gr.Dropdown(
                    label="Story file (optional when pasting text)",
                    choices=stories,
                    value=_first_or_none(stories),
                    allow_custom_value=True,
                )
                story_upload = gr.File(label="Upload story .txt", file_types=[".txt"], type="filepath")
                story_text = gr.Textbox(
                    label="Story text (takes priority over file)",
                    lines=14,
                    placeholder="Paste the complete Reddit story here...",
                )
                output_name = gr.Textbox(label="Output name (optional)", placeholder="my_story")

            with gr.Column(scale=2):
                tts_engine = gr.Radio(
                    choices=[("Gemini TTS", "gemini"), ("Microsoft VibeVoice 1.5B", "vibevoice")],
                    value="gemini",
                    label="Narration engine",
                )
                gemini_voice = gr.Dropdown(GEMINI_VOICES, value="Kore", label="Gemini voice")
                gemini_model = gr.Textbox(value="gemini-3.1-flash-tts-preview", label="Gemini TTS model")
                gemini_preprocess = gr.Checkbox(value=True, label="Gemini: preprocess text")
                gemini_split = gr.Checkbox(value=True, label="Gemini: treat ------------- as a hard chunk boundary")
                gemini_chunk_seconds = gr.Slider(60, 240, value=180, step=15, label="Gemini semantic chunk target (seconds)")

                gr.Markdown("**VibeVoice settings** — VibeVoice generates the whole story in one pass; there is no Gemini-style chunking.")
                vibe_speaker = gr.Dropdown(
                    choices=vibe_voices,
                    value=_first_or_none(vibe_voices, "Alice"),
                    allow_custom_value=True,
                    label="VibeVoice speaker preset",
                )
                vibe_model = gr.Textbox(value="microsoft/VibeVoice-1.5B", label="VibeVoice model")
                with gr.Row():
                    vibe_cfg = gr.Slider(1.0, 2.0, value=1.3, step=0.05, label="CFG scale")
                    vibe_steps = gr.Slider(4, 30, value=10, step=1, label="Diffusion steps")
                with gr.Row():
                    vibe_seed = gr.Number(value=42, precision=0, label="Seed")
                    vibe_device = gr.Dropdown(["auto", "cuda", "cpu"], value="auto", label="Device")
                    vibe_dtype = gr.Dropdown(["auto", "bfloat16", "float16", "float32"], value="auto", label="Dtype")

        gr.Markdown("## Video and captions")
        with gr.Row():
            background = gr.Dropdown(
                choices=backgrounds,
                value=_first_or_none(backgrounds, "videos/minecraft/minecraft.mp4"),
                allow_custom_value=True,
                label="Background video",
            )
            background_upload = gr.File(label="Upload custom background", file_types=["video"], type="filepath")
            output_format = gr.Radio(
                choices=[("YouTube Shorts 1080×1920", "shorts"), ("Keep source dimensions", "source")],
                value="shorts",
                label="Output format",
            )
            random_start = gr.Checkbox(value=True, label="Randomize background start")

        with gr.Row():
            captions = gr.Checkbox(value=True, label="Render captions")
            caption_theme = gr.Dropdown(theme_choices(), value="classic_yellow", label="Caption theme")
            caption_max_words = gr.Slider(0, 8, value=0, step=1, label="Max words (0 = theme default)")
            caption_pause = gr.Slider(0.1, 1.5, value=0.5, step=0.05, label="Pause split (seconds)")

        with gr.Accordion("Advanced render / WhisperX", open=False):
            with gr.Row():
                whisper_model = gr.Dropdown(["large-v2", "large-v3", "medium", "small"], value="large-v2", allow_custom_value=True, label="WhisperX model")
                whisper_language = gr.Dropdown(["en", "auto"], value="en", allow_custom_value=True, label="WhisperX language")
                whisper_compute = gr.Dropdown(["float16", "int8_float16", "int8", "float32"], value="float16", label="WhisperX compute type")
                whisper_align = gr.Textbox(value="WAV2VEC2_ASR_LARGE_LV60K_960H", label="WhisperX alignment model")
            with gr.Row():
                encoder = gr.Dropdown(["auto", "nvenc", "cpu"], value="auto", label="Video encoder")
                quality = gr.Slider(14, 30, value=20, step=1, label="Video quality (lower = higher quality)")
                end_padding = gr.Slider(0, 5, value=1, step=0.25, label="Silence/padding at end (seconds)")

        with gr.Row():
            refresh = gr.Button("Refresh files / voices")
            run = gr.Button("Generate video", variant="primary")

        with gr.Row():
            video_output = gr.Video(label="Final video")
            audio_output = gr.Audio(label="Narration", type="filepath")
        status = gr.Markdown()

        def refresh_choices(current_story: str | None, current_background: str | None, current_voice: str | None):
            new_stories = list_input_stories()
            new_backgrounds = list_background_videos()
            new_voices = _vibe_choices()
            return (
                gr.Dropdown(choices=new_stories, value=_first_or_none(new_stories, current_story)),
                gr.Dropdown(choices=new_backgrounds, value=_first_or_none(new_backgrounds, current_background)),
                gr.Dropdown(choices=new_voices, value=_first_or_none(new_voices, current_voice)),
            )

        refresh.click(
            refresh_choices,
            inputs=[story_file, background, vibe_speaker],
            outputs=[story_file, background, vibe_speaker],
        )

        def generate(
            story_file_value,
            story_upload_value,
            story_text_value,
            output_name_value,
            tts_engine_value,
            gemini_voice_value,
            gemini_model_value,
            gemini_preprocess_value,
            gemini_split_value,
            gemini_chunk_seconds_value,
            vibe_speaker_value,
            vibe_model_value,
            vibe_cfg_value,
            vibe_steps_value,
            vibe_seed_value,
            vibe_device_value,
            vibe_dtype_value,
            background_value,
            background_upload_value,
            output_format_value,
            random_start_value,
            captions_value,
            caption_theme_value,
            caption_max_words_value,
            caption_pause_value,
            whisper_model_value,
            whisper_language_value,
            whisper_compute_value,
            whisper_align_value,
            encoder_value,
            quality_value,
            end_padding_value,
            progress=gr.Progress(),  # noqa: B008 - Gradio injects this special dependency
        ):
            logs: list[str] = []

            def log(message: str) -> None:
                logs.append(message)
                print(message, flush=True)

            def report(value: float, message: str) -> None:
                progress(value, desc=message)

            try:
                options = PipelineOptions(
                    story_file=story_upload_value or story_file_value or None,
                    story_text=story_text_value or "",
                    output_name=output_name_value or "",
                    tts_engine=tts_engine_value,
                    gemini_voice=gemini_voice_value,
                    gemini_model=gemini_model_value,
                    gemini_preprocess=bool(gemini_preprocess_value),
                    gemini_split_on_separator=bool(gemini_split_value),
                    gemini_chunk_seconds=int(gemini_chunk_seconds_value),
                    vibevoice_model=vibe_model_value or "microsoft/VibeVoice-1.5B",
                    vibevoice_speaker=vibe_speaker_value or "Alice",
                    vibevoice_cfg_scale=float(vibe_cfg_value),
                    vibevoice_diffusion_steps=int(vibe_steps_value),
                    vibevoice_seed=int(vibe_seed_value),
                    vibevoice_device=vibe_device_value,
                    vibevoice_dtype=vibe_dtype_value,
                    background=background_upload_value or background_value,
                    output_format=output_format_value,
                    randomize_background_start=bool(random_start_value),
                    captions=bool(captions_value),
                    caption_theme=caption_theme_value,
                    caption_max_words=int(caption_max_words_value),
                    caption_pause_threshold=float(caption_pause_value),
                    whisper_model=whisper_model_value,
                    whisper_language=whisper_language_value,
                    whisper_compute_type=whisper_compute_value,
                    whisper_align_model=whisper_align_value or "",
                    encoder=encoder_value,
                    video_quality=int(quality_value),
                    end_padding_seconds=float(end_padding_value),
                )
                result = RedditVideoPipeline(log=log, progress=report).run(options)
                caption = str(result.caption_path) if result.caption_path else "disabled"
                summary = (
                    f"### Completed\n"
                    f"- Video: `{result.video_path}`\n"
                    f"- Audio: `{result.audio_path}`\n"
                    f"- Captions: `{caption}`\n"
                    f"- Elapsed: `{result.elapsed_seconds:.1f}s`"
                )
                return str(result.video_path), str(result.audio_path), summary
            except Exception as exc:  # noqa: BLE001 - surface pipeline failures in the UI
                traceback.print_exc()
                tail = "\n".join(logs[-20:])
                return None, None, f"### Failed\n`{type(exc).__name__}: {exc}`\n\n```text\n{tail}\n```"

        run.click(
            generate,
            inputs=[
                story_file, story_upload, story_text, output_name, tts_engine, gemini_voice, gemini_model, gemini_preprocess, gemini_split, gemini_chunk_seconds,
                vibe_speaker, vibe_model, vibe_cfg, vibe_steps, vibe_seed, vibe_device, vibe_dtype,
                background, background_upload, output_format, random_start, captions, caption_theme, caption_max_words, caption_pause,
                whisper_model, whisper_language, whisper_compute, whisper_align, encoder, quality, end_padding,
            ],
            outputs=[video_output, audio_output, status],
            api_name="generate_video",
        )

    return demo
