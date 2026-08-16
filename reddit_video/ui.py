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
from .tts import (
    get_gemini_voice_preview,
    get_vibevoice_voice_preview,
    list_vibevoice_presets,
)
from .tts_models import MAGPIE_SPEAKERS, backend_runtime_status

GEMINI_VOICES = [
    "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede", "Callirrhoe", "Autonoe",
    "Enceladus", "Iapetus", "Umbriel", "Algieba", "Despina", "Erinome", "Algenib", "Rasalgethi",
    "Laomedeia", "Achernar", "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi",
    "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat",
]

AUDIO_TEST_STORY = (
    "I knew something was wrong when my phone lit up at two in the morning. "
    "The message was from my best friend's older brother, a man who barely spoke to me unless we were all together. "
    "He wrote, 'Don't panic, but I need to tell you something before she does.' "
    "I stared at that sentence for a full minute before answering. Then another message appeared: "
    "'She found the letter you left in my jacket last summer.' I actually laughed, because there had never been a letter. "
    "Then I remembered the note I had written as a joke after one very flirtatious night, and suddenly I wasn't laughing anymore."
)

TTS_ENGINE_CHOICES = [
    ("Gemini TTS", "gemini"),
    ("Microsoft VibeVoice 1.5B", "vibevoice"),
    ("Fish Audio S2 Pro", "fish"),
    ("Step Audio EditX", "step"),
    ("NVIDIA Magpie Multilingual 357M", "magpie"),
    ("Chatterbox", "chatterbox"),
    ("Higgs Audio V3 / Higgs TTS 3", "higgs"),
]


def _vibe_choices() -> list[str]:
    installed = list_vibevoice_presets(PROJECT_ROOT / "vendor" / "VibeVoice")
    return installed or ["Alice", "Frank"]


def _first_or_none(values: list[str], preferred: str | None = None) -> str | None:
    if preferred and preferred in values:
        return preferred
    return values[0] if values else None


def _runtime_status_markdown() -> str:
    status = backend_runtime_status(PROJECT_ROOT)
    labels = {
        "fish": "Fish S2 Pro",
        "step": "Step Audio EditX",
        "magpie": "NVIDIA Magpie",
        "chatterbox": "Chatterbox",
        "higgs": "Higgs TTS 3",
    }
    lines = ["**Local TTS runtime status**"]
    for key, label in labels.items():
        value = status[key]
        ready = value.lower().startswith("ready")
        lines.append(f"- {'READY' if ready else 'NOT INSTALLED'} - **{label}**: `{value}`")
    lines.append("Run `./setup_tts_models.ps1 -Backend <fish|step|magpie|chatterbox|higgs|all>` to provision model runtimes.")
    return "\n".join(lines)


def build_ui() -> gr.Blocks:
    stories = list_input_stories()
    backgrounds = list_background_videos()
    vibe_voices = _vibe_choices()

    with gr.Blocks(title="Reddit Romantics Automation") as demo:
        gr.Markdown(
            "# Reddit Romantics Video Automation\n"
            "Pick any narration engine, test only its audio, or generate the full captioned video. "
            "Heavy local TTS models are isolated from the main app environment so they can use their own dependencies."
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
                    choices=TTS_ENGINE_CHOICES,
                    value="gemini",
                    label="Narration engine",
                )

                with gr.Accordion("Gemini TTS settings", open=True):
                    gemini_voice = gr.Dropdown(GEMINI_VOICES, value="Kore", label="Gemini voice")
                    gemini_preview = gr.Audio(label="Gemini voice preview", type="filepath", interactive=False)
                    gemini_model = gr.Textbox(value="gemini-3.1-flash-tts-preview", label="Gemini TTS model")
                    gemini_preprocess = gr.Checkbox(value=True, label="Preprocess text")
                    gemini_split = gr.Checkbox(value=True, label="Treat ------------- as a hard chunk boundary")
                    gemini_chunk_seconds = gr.Slider(60, 240, value=180, step=15, label="Semantic chunk target (seconds)")

                with gr.Accordion("VibeVoice 1.5B settings", open=False):
                    gr.Markdown("VibeVoice generates the whole story in one pass; there is no Gemini-style text chunking.")
                    vibe_speaker = gr.Dropdown(
                        choices=vibe_voices,
                        value=_first_or_none(vibe_voices, "Alice"),
                        allow_custom_value=True,
                        label="Speaker preset",
                    )
                    vibe_preview = gr.Audio(label="VibeVoice preset preview", type="filepath", interactive=False)
                    vibe_model = gr.Textbox(value="microsoft/VibeVoice-1.5B", label="Model")
                    with gr.Row():
                        vibe_cfg = gr.Slider(1.0, 2.0, value=1.3, step=0.05, label="CFG scale")
                        vibe_steps = gr.Slider(4, 30, value=10, step=1, label="Diffusion steps")
                    with gr.Row():
                        vibe_seed = gr.Number(value=42, precision=0, label="Seed")
                        vibe_device = gr.Dropdown(["auto", "cuda", "cpu"], value="auto", label="Device")
                        vibe_dtype = gr.Dropdown(["auto", "bfloat16", "float16", "float32"], value="auto", label="Dtype")

                with gr.Accordion("Fish Audio S2 Pro settings", open=False):
                    gr.Markdown(
                        "**Hybrid is the recommended mode for this 8 GB GPU.** It runs the full, unquantized F16 S2 Pro "
                        "weights through native `s2.cpp`: selected transformer layers + Fast-AR + KV cache + codec on CUDA, "
                        "remaining transformer layers on CPU. The codec falls back to CPU if CUDA memory is unavailable. "
                        "No Q6/Q8/4-bit model is used."
                    )
                    with gr.Row():
                        fish_device = gr.Dropdown(
                            [("Hybrid CPU + CUDA (unquantized F16)", "hybrid"), ("Official CPU / BF16", "cpu"), ("Official full CUDA", "cuda")],
                            value="hybrid",
                            label="Fish runtime",
                        )
                        fish_gpu_layers = gr.Slider(1, 24, value=20, step=1, label="CUDA transformer layers (hybrid)")
                    with gr.Row():
                        fish_half = gr.Checkbox(value=False, label="FP16 on official full-CUDA path only")
                        fish_temperature = gr.Slider(0.2, 1.5, value=1.0, step=0.05, label="Temperature")
                        fish_seed = gr.Number(value=42, precision=0, label="Seed (official path)")
                    fish_reference_audio = gr.File(label="Optional Fish reference voice WAV", file_types=["audio"], type="filepath")
                    fish_reference_text = gr.Textbox(label="Reference clip transcript (required when cloning)", lines=2)

                with gr.Accordion("Step Audio EditX settings", open=False):
                    gr.Markdown(
                        "Step EditX needs a reference clip for zero-shot TTS. Its official runtime is Linux/CUDA-oriented; "
                        "use the AWQ/memory-efficient setup for an 8 GB card."
                    )
                    step_reference_audio = gr.File(label="Step reference voice audio (required)", file_types=["audio"], type="filepath")
                    step_reference_text = gr.Textbox(label="Exact reference clip transcript (required)", lines=2)
                    step_mode = gr.Radio(
                        choices=[("Zero-shot clone", "clone"), ("Paralinguistic tags in target text", "paralinguistic")],
                        value="clone",
                        label="Step generation mode",
                    )

                with gr.Accordion("NVIDIA Magpie settings", open=False):
                    magpie_model = gr.Textbox(value="nvidia/magpie_tts_multilingual_357m", label="Model")
                    with gr.Row():
                        magpie_speaker = gr.Dropdown(MAGPIE_SPEAKERS, value="John", label="Speaker")
                        magpie_language = gr.Textbox(value="en", label="Language code")
                        magpie_device = gr.Dropdown(["auto", "cuda", "cpu"], value="auto", label="Device")
                    with gr.Row():
                        magpie_use_cfg = gr.Checkbox(value=True, label="Use CFG")
                        magpie_cfg_scale = gr.Slider(1.0, 5.0, value=2.5, step=0.1, label="CFG scale")

                with gr.Accordion("Chatterbox settings", open=False):
                    gr.Markdown("Turbo/Nano understand paralinguistic tags such as `[chuckle]`. A reference clip is optional for voice cloning.")
                    with gr.Row():
                        chatterbox_variant = gr.Dropdown(["turbo", "nano", "original"], value="turbo", label="Variant")
                        chatterbox_device = gr.Dropdown(["auto", "cuda", "cpu"], value="auto", label="Device")
                    chatterbox_reference_audio = gr.File(label="Optional Chatterbox reference voice audio", file_types=["audio"], type="filepath")

                with gr.Accordion("Higgs Audio V3 / Higgs TTS 3 settings", open=False):
                    gr.Markdown(
                        "Higgs supports inline `<|emotion:...|>`, `<|style:...|>`, `<|prosody:...|>`, and `<|sfx:...|>` controls. "
                        "Higgs runs through a managed SGLang-Omni Docker server; auto/cpu modes use aggressive CPU offload so the 4B backbone can fit alongside the codec on an 8 GB GPU."
                    )
                    higgs_model = gr.Textbox(value="bosonai/higgs-audio-v3-tts-4b", label="Model")
                    higgs_device = gr.Dropdown(["auto", "cuda", "cpu"], value="auto", label="Device")

        runtime_status = gr.Markdown(_runtime_status_markdown())

        gr.Markdown("## Audio-only TTS test")
        audio_test_text = gr.Textbox(
            value=AUDIO_TEST_STORY,
            lines=7,
            label="Short test story (edit this to try laughs, whispers, pauses, etc.)",
        )
        with gr.Row():
            test_audio = gr.Button("Generate audio only", variant="primary")
            refresh_runtime = gr.Button("Refresh TTS runtime status")
        with gr.Row():
            test_audio_output = gr.Audio(label="TTS test output", type="filepath")
            test_audio_status = gr.Markdown()

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
                choices=[("YouTube Shorts 1080x1920", "shorts"), ("Keep source dimensions", "source")],
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
        preview_status = gr.Markdown()
        status = gr.Markdown()

        def load_gemini_preview(voice: str | None, model: str | None):
            if not voice:
                return None, ""
            try:
                preview = get_gemini_voice_preview(
                    PROJECT_ROOT,
                    voice,
                    model or "gemini-3.1-flash-tts-preview",
                    log=lambda message: print(message, flush=True),
                )
                return str(preview), f"Gemini preview ready: **{voice}**"
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                return None, f"Gemini preview unavailable: `{type(exc).__name__}: {exc}`"

        def load_vibe_preview(speaker: str | None):
            if not speaker:
                return None, ""
            try:
                preview = get_vibevoice_voice_preview(PROJECT_ROOT, speaker)
                return str(preview), f"VibeVoice reference sample ready: **{speaker}**"
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                return None, f"VibeVoice preview unavailable: `{type(exc).__name__}: {exc}`"

        gemini_voice.change(load_gemini_preview, [gemini_voice, gemini_model], [gemini_preview, preview_status], show_progress="minimal")
        gemini_model.change(load_gemini_preview, [gemini_voice, gemini_model], [gemini_preview, preview_status], show_progress="minimal")
        vibe_speaker.change(load_vibe_preview, [vibe_speaker], [vibe_preview, preview_status], show_progress="hidden")
        demo.load(load_gemini_preview, [gemini_voice, gemini_model], [gemini_preview, preview_status], show_progress="minimal")
        demo.load(load_vibe_preview, [vibe_speaker], [vibe_preview, preview_status], show_progress="hidden")

        def refresh_choices(current_story: str | None, current_background: str | None, current_voice: str | None):
            new_stories = list_input_stories()
            new_backgrounds = list_background_videos()
            new_voices = _vibe_choices()
            return (
                gr.Dropdown(choices=new_stories, value=_first_or_none(new_stories, current_story)),
                gr.Dropdown(choices=new_backgrounds, value=_first_or_none(new_backgrounds, current_background)),
                gr.Dropdown(choices=new_voices, value=_first_or_none(new_voices, current_voice)),
            )

        refresh.click(refresh_choices, [story_file, background, vibe_speaker], [story_file, background, vibe_speaker])
        refresh_runtime.click(_runtime_status_markdown, outputs=[runtime_status], show_progress="hidden")

        def make_tts_options(
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
            fish_device_value,
            fish_gpu_layers_value,
            fish_half_value,
            fish_temperature_value,
            fish_seed_value,
            fish_reference_audio_value,
            fish_reference_text_value,
            step_reference_audio_value,
            step_reference_text_value,
            step_mode_value,
            magpie_model_value,
            magpie_speaker_value,
            magpie_language_value,
            magpie_device_value,
            magpie_use_cfg_value,
            magpie_cfg_scale_value,
            chatterbox_variant_value,
            chatterbox_device_value,
            chatterbox_reference_audio_value,
            higgs_model_value,
            higgs_device_value,
            **extra,
        ) -> PipelineOptions:
            return PipelineOptions(
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
                fish_device=fish_device_value,
                fish_gpu_layers=int(fish_gpu_layers_value),
                fish_half=bool(fish_half_value),
                fish_temperature=float(fish_temperature_value),
                fish_seed=int(fish_seed_value),
                fish_reference_audio=fish_reference_audio_value or None,
                fish_reference_text=fish_reference_text_value or "",
                step_reference_audio=step_reference_audio_value or None,
                step_reference_text=step_reference_text_value or "",
                step_mode=step_mode_value,
                magpie_model=magpie_model_value or "nvidia/magpie_tts_multilingual_357m",
                magpie_speaker=magpie_speaker_value or "John",
                magpie_language=magpie_language_value or "en",
                magpie_device=magpie_device_value,
                magpie_use_cfg=bool(magpie_use_cfg_value),
                magpie_cfg_scale=float(magpie_cfg_scale_value),
                chatterbox_variant=chatterbox_variant_value,
                chatterbox_device=chatterbox_device_value,
                chatterbox_reference_audio=chatterbox_reference_audio_value or None,
                higgs_model=higgs_model_value or "bosonai/higgs-audio-v3-tts-4b",
                higgs_device=higgs_device_value,
                **extra,
            )

        tts_inputs = [
            tts_engine,
            gemini_voice, gemini_model, gemini_preprocess, gemini_split, gemini_chunk_seconds,
            vibe_speaker, vibe_model, vibe_cfg, vibe_steps, vibe_seed, vibe_device, vibe_dtype,
            fish_device, fish_gpu_layers, fish_half, fish_temperature, fish_seed, fish_reference_audio, fish_reference_text,
            step_reference_audio, step_reference_text, step_mode,
            magpie_model, magpie_speaker, magpie_language, magpie_device, magpie_use_cfg, magpie_cfg_scale,
            chatterbox_variant, chatterbox_device, chatterbox_reference_audio,
            higgs_model, higgs_device,
        ]

        def generate_audio_test(
            preview_text_value,
            tts_engine_value,
            gemini_voice_value, gemini_model_value, gemini_preprocess_value, gemini_split_value, gemini_chunk_seconds_value,
            vibe_speaker_value, vibe_model_value, vibe_cfg_value, vibe_steps_value, vibe_seed_value, vibe_device_value, vibe_dtype_value,
            fish_device_value, fish_gpu_layers_value, fish_half_value, fish_temperature_value, fish_seed_value, fish_reference_audio_value, fish_reference_text_value,
            step_reference_audio_value, step_reference_text_value, step_mode_value,
            magpie_model_value, magpie_speaker_value, magpie_language_value, magpie_device_value, magpie_use_cfg_value, magpie_cfg_scale_value,
            chatterbox_variant_value, chatterbox_device_value, chatterbox_reference_audio_value,
            higgs_model_value, higgs_device_value,
            progress=gr.Progress(),  # noqa: B008 - Gradio injects this dependency
        ):
            logs: list[str] = []

            def log(message: str) -> None:
                logs.append(message)
                print(message, flush=True)

            def report(value: float, message: str) -> None:
                progress(value, desc=message)

            try:
                options = make_tts_options(
                    preview_text_value,
                    "tts_preview",
                    tts_engine_value,
                    gemini_voice_value, gemini_model_value, gemini_preprocess_value, gemini_split_value, gemini_chunk_seconds_value,
                    vibe_speaker_value, vibe_model_value, vibe_cfg_value, vibe_steps_value, vibe_seed_value, vibe_device_value, vibe_dtype_value,
                    fish_device_value, fish_gpu_layers_value, fish_half_value, fish_temperature_value, fish_seed_value, fish_reference_audio_value, fish_reference_text_value,
                    step_reference_audio_value, step_reference_text_value, step_mode_value,
                    magpie_model_value, magpie_speaker_value, magpie_language_value, magpie_device_value, magpie_use_cfg_value, magpie_cfg_scale_value,
                    chatterbox_variant_value, chatterbox_device_value, chatterbox_reference_audio_value,
                    higgs_model_value, higgs_device_value,
                )
                result = RedditVideoPipeline(log=log, progress=report).run_audio(options)
                summary = (
                    f"### Audio ready\n"
                    f"- Engine: **{tts_engine_value}**\n"
                    f"- Audio: `{result.audio_path}`\n"
                    f"- Duration: `{result.duration_seconds:.1f}s`\n"
                    f"- Generation time: `{result.elapsed_seconds:.1f}s`"
                )
                return str(result.audio_path), summary
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                tail = "\n".join(logs[-25:])
                return None, f"### Audio test failed\n`{type(exc).__name__}: {exc}`\n\n```text\n{tail}\n```"

        test_audio.click(
            generate_audio_test,
            inputs=[audio_test_text, *tts_inputs],
            outputs=[test_audio_output, test_audio_status],
            api_name="generate_audio_only",
        )

        def generate(
            story_file_value,
            story_upload_value,
            story_text_value,
            output_name_value,
            tts_engine_value,
            gemini_voice_value, gemini_model_value, gemini_preprocess_value, gemini_split_value, gemini_chunk_seconds_value,
            vibe_speaker_value, vibe_model_value, vibe_cfg_value, vibe_steps_value, vibe_seed_value, vibe_device_value, vibe_dtype_value,
            fish_device_value, fish_gpu_layers_value, fish_half_value, fish_temperature_value, fish_seed_value, fish_reference_audio_value, fish_reference_text_value,
            step_reference_audio_value, step_reference_text_value, step_mode_value,
            magpie_model_value, magpie_speaker_value, magpie_language_value, magpie_device_value, magpie_use_cfg_value, magpie_cfg_scale_value,
            chatterbox_variant_value, chatterbox_device_value, chatterbox_reference_audio_value,
            higgs_model_value, higgs_device_value,
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
            progress=gr.Progress(),  # noqa: B008 - Gradio injects this dependency
        ):
            logs: list[str] = []

            def log(message: str) -> None:
                logs.append(message)
                print(message, flush=True)

            def report(value: float, message: str) -> None:
                progress(value, desc=message)

            try:
                options = make_tts_options(
                    story_text_value,
                    output_name_value,
                    tts_engine_value,
                    gemini_voice_value, gemini_model_value, gemini_preprocess_value, gemini_split_value, gemini_chunk_seconds_value,
                    vibe_speaker_value, vibe_model_value, vibe_cfg_value, vibe_steps_value, vibe_seed_value, vibe_device_value, vibe_dtype_value,
                    fish_device_value, fish_gpu_layers_value, fish_half_value, fish_temperature_value, fish_seed_value, fish_reference_audio_value, fish_reference_text_value,
                    step_reference_audio_value, step_reference_text_value, step_mode_value,
                    magpie_model_value, magpie_speaker_value, magpie_language_value, magpie_device_value, magpie_use_cfg_value, magpie_cfg_scale_value,
                    chatterbox_variant_value, chatterbox_device_value, chatterbox_reference_audio_value,
                    higgs_model_value, higgs_device_value,
                    story_file=story_upload_value or story_file_value or None,
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
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                tail = "\n".join(logs[-25:])
                return None, None, f"### Failed\n`{type(exc).__name__}: {exc}`\n\n```text\n{tail}\n```"

        run.click(
            generate,
            inputs=[
                story_file, story_upload, story_text, output_name,
                *tts_inputs,
                background, background_upload, output_format, random_start, captions, caption_theme, caption_max_words, caption_pause,
                whisper_model, whisper_language, whisper_compute, whisper_align, encoder, quality, end_padding,
            ],
            outputs=[video_output, audio_output, status],
            api_name="generate_video",
        )

    return demo
