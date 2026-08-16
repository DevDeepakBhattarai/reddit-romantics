from __future__ import annotations

import traceback
from pathlib import Path

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
from .fish import (
    cache_fish_reference_preset,
    fish_runtime_status,
    list_fish_reference_presets,
    resolve_fish_reference_preset,
)
from .tts_text import detect_speakers, provider_speaker_limit

GEMINI_VOICES = [
    "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede", "Callirrhoe", "Autonoe",
    "Enceladus", "Iapetus", "Umbriel", "Algieba", "Despina", "Erinome", "Algenib", "Rasalgethi",
    "Laomedeia", "Achernar", "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi",
    "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat",
]

TTS_ENGINE_CHOICES = [
    ("Gemini TTS", "gemini"),
    ("Microsoft VibeVoice 1.5B", "vibevoice"),
    ("Fish Audio S2 Pro", "fish"),
]
MAX_SPEAKER_SLOTS = 5


def _vibe_choices() -> list[str]:
    installed = list_vibevoice_presets(PROJECT_ROOT / "vendor" / "VibeVoice")
    return installed or ["Alice", "Frank"]


def _first_or_none(values: list[str], preferred: str | None = None) -> str | None:
    if preferred and preferred in values:
        return preferred
    return values[0] if values else None


def _runtime_status_markdown() -> str:
    status = fish_runtime_status(PROJECT_ROOT)
    ready = status.lower().startswith("ready") or status.lower().startswith("official fish ready")
    return (
        "**Local TTS runtime status**\n"
        f"- {'READY' if ready else 'NOT READY'} - **Fish Audio S2 Pro**: `{status}`\n\n"
        "VibeVoice uses its installed local voice presets. Gemini uses the configured Google API key."
    )


def _read_story_for_casting(
    story_text: str | None,
    story_file: str | None,
    story_upload: str | None,
) -> str:
    if story_text and story_text.strip():
        return story_text
    candidate = story_upload or story_file
    if not candidate:
        return ""
    path = Path(candidate)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    try:
        return path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError):
        return ""


def build_ui() -> gr.Blocks:
    stories = list_input_stories()
    backgrounds = list_background_videos()
    vibe_voices = _vibe_choices()
    fish_presets = list_fish_reference_presets(PROJECT_ROOT)

    with gr.Blocks(title="Reddit Romantics Automation") as demo:
        gr.Markdown(
            "# Reddit Romantics Video Automation\n"
            "Multi-speaker narration is available through Gemini, VibeVoice, and Fish Audio. "
            "Paste `Speaker 0: ...`, `Speaker 1: ...` turns and the voice-casting controls appear automatically."
        )

        with gr.Row():
            with gr.Column(scale=3):
                story_file = gr.Dropdown(
                    label="Existing run story (optional when pasting text)",
                    choices=stories,
                    value=_first_or_none(stories),
                    allow_custom_value=True,
                )
                story_upload = gr.File(label="Upload story .txt", file_types=[".txt"], type="filepath")
                story_text = gr.Textbox(
                    label="Story text (takes priority over file)",
                    lines=16,
                    placeholder=(
                        "Speaker 0 - gender=female; Maya; narrator; warm and quick-witted.\n"
                        "Speaker 1 - gender=male; Adrian; main counterpart; deep, controlled voice.\n\n"
                        "Speaker 0: The first time Adrian kissed me...\n"
                        "Speaker 1: Maya."
                    ),
                )
                output_name = gr.Textbox(label="Story / run title (optional)", placeholder="my story")

            with gr.Column(scale=2):
                tts_engine = gr.Radio(
                    choices=TTS_ENGINE_CHOICES,
                    value="fish",
                    label="Narration engine",
                )

                with gr.Group(visible=False) as gemini_settings:
                    gr.Markdown("### Gemini TTS settings")
                    gemini_voice = gr.Dropdown(GEMINI_VOICES, value="Kore", label="Default / single-speaker voice")
                    gemini_preview = gr.Audio(label="Gemini voice preview", type="filepath", interactive=False)
                    gemini_model = gr.Textbox(value="gemini-3.1-flash-tts-preview", label="Gemini TTS model")
                    gemini_preprocess = gr.Checkbox(value=True, label="Preprocess text without removing speaker labels/tags")
                    gemini_split = gr.Checkbox(value=True, label="Treat ------------- as a hard chunk boundary")
                    gemini_chunk_seconds = gr.Slider(60, 240, value=180, step=15, label="Semantic chunk target (seconds)")
                    gr.Markdown(
                        "Gemini accepts up to **2 speakers**. Generic `[laugh]`, `<laugh>`, `[giggle]`, `[pause]`, "
                        "etc. are normalized to Gemini audio tags before generation."
                    )

                with gr.Group(visible=False) as vibe_settings:
                    gr.Markdown("### VibeVoice 1.5B settings")
                    vibe_speaker = gr.Dropdown(
                        choices=vibe_voices,
                        value=_first_or_none(vibe_voices, "Alice"),
                        allow_custom_value=True,
                        label="Default / single-speaker preset",
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
                    gr.Markdown(
                        "VibeVoice accepts up to **4 speakers**. Inline decorator markup is stripped because this "
                        "model does not expose a stable documented tag language."
                    )

                with gr.Group(visible=True) as fish_settings:
                    gr.Markdown("### Fish Audio S2 Pro settings")
                    gr.Markdown("Native Windows **unquantized F16 hybrid** runtime.")
                    with gr.Row():
                        fish_gpu_layers = gr.Slider(1, 36, value=20, step=1, label="CUDA transformer layers")
                        fish_temperature = gr.Slider(0.2, 1.5, value=1.0, step=0.05, label="Temperature")
                    fish_reference_audio = gr.File(
                        label="Optional default Fish reference voice WAV (single-speaker fallback)",
                        file_types=["audio"],
                        type="filepath",
                    )
                    fish_reference_text = gr.Textbox(
                        label="Default reference transcript (required with default reference WAV)",
                        lines=2,
                    )
                    gr.Markdown(
                        "Fish uses native `<|speaker:N|>` turns internally and supports bracketed controls such as "
                        "`[laughing]`, `[chuckle]`, `[whisper]`, and `[pause]`."
                    )

                provider_help = gr.Markdown()

        runtime_status = gr.Markdown(_runtime_status_markdown())

        gr.Markdown("## Speaker voice assignments ? required")
        speaker_status = gr.Markdown(
            "Fish auto-casts from `gender=male` / `gender=female` story metadata. Manual Fish presets are optional overrides."
        )

        speaker_groups: list[gr.Group] = []
        speaker_labels: list[gr.Markdown] = []
        speaker_gemini_voices: list[gr.Dropdown] = []
        speaker_vibe_voices: list[gr.Dropdown] = []
        speaker_fish_presets: list[gr.Dropdown] = []
        speaker_fish_preset_names: list[gr.Textbox] = []
        speaker_fish_audio: list[gr.File] = []
        speaker_fish_text: list[gr.Textbox] = []

        for slot in range(MAX_SPEAKER_SLOTS):
            with gr.Group(visible=False) as speaker_group:
                speaker_label = gr.Markdown(f"**Speaker {slot}**")
                with gr.Row():
                    speaker_gemini = gr.Dropdown(
                        GEMINI_VOICES,
                        value="Kore" if slot == 0 else ("Puck" if slot == 1 else "Kore"),
                        label="Gemini voice",
                        visible=True,
                    )
                    speaker_vibe = gr.Dropdown(
                        choices=vibe_voices,
                        value=_first_or_none(vibe_voices, "Alice" if slot == 0 else "Frank"),
                        allow_custom_value=True,
                        label="VibeVoice preset",
                        visible=False,
                    )
                    speaker_fish_preset = gr.Dropdown(
                        choices=fish_presets,
                        value=None,
                        allow_custom_value=True,
                        label="Fish preset override",
                        info="Leave blank for automatic male/female casting from story metadata. Upload overrides this preset.",
                        visible=False,
                    )
                    speaker_fish_preset_name = gr.Textbox(
                        label="Save uploaded Fish voice as preset",
                        placeholder="e-girl (blank = use uploaded filename)",
                        info="When an upload is used, it is cached permanently under this name for future sessions.",
                        visible=False,
                    )
                    speaker_fish_ref = gr.File(
                        label="Fish reference voice audio",
                        file_types=["audio"],
                        type="filepath",
                        visible=False,
                    )
                    speaker_fish_transcript = gr.Textbox(
                        label="Exact transcript of uploaded Fish reference",
                        lines=2,
                        visible=False,
                    )
            speaker_groups.append(speaker_group)
            speaker_labels.append(speaker_label)
            speaker_gemini_voices.append(speaker_gemini)
            speaker_vibe_voices.append(speaker_vibe)
            speaker_fish_presets.append(speaker_fish_preset)
            speaker_fish_preset_names.append(speaker_fish_preset_name)
            speaker_fish_audio.append(speaker_fish_ref)
            speaker_fish_text.append(speaker_fish_transcript)

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
                choices=[("Keep source dimensions (full video)", "source"), ("Vertical 1080x1920", "shorts")],
                value="source",
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
            refresh_runtime = gr.Button("Refresh Fish status")
            run = gr.Button("Generate video", variant="primary")

        with gr.Row():
            video_output = gr.Video(label="Final video")
            audio_output = gr.Audio(label="Narration", type="filepath")
        preview_status = gr.Markdown()
        status = gr.Markdown()

        def provider_visibility(engine: str):
            descriptions = {
                "gemini": "**Gemini:** native 1â€“2 speaker casting; supported generic tags are normalized to Gemini bracket tags.",
                "vibevoice": "**VibeVoice:** native 1â€“4 speaker casting from WAV voice presets; decorator markup is removed before inference.",
                "fish": "**Fish S2 Pro (default):** story gender metadata auto-selects the male/female presets; manual preset or upload overrides remain available.",
            }
            return (
                gr.update(visible=engine == "gemini"),
                gr.update(visible=engine == "vibevoice"),
                gr.update(visible=engine == "fish"),
                descriptions.get(engine, ""),
            )

        tts_engine.change(
            provider_visibility,
            [tts_engine],
            [gemini_settings, vibe_settings, fish_settings, provider_help],
            show_progress="hidden",
        )

        casting_outputs = [speaker_status]
        for slot in range(MAX_SPEAKER_SLOTS):
            casting_outputs.extend([
                speaker_groups[slot],
                speaker_labels[slot],
                speaker_gemini_voices[slot],
                speaker_vibe_voices[slot],
                speaker_fish_presets[slot],
                speaker_fish_preset_names[slot],
                speaker_fish_audio[slot],
                speaker_fish_text[slot],
            ])

        def _casting_updates(source: str, engine: str):
            speakers = detect_speakers(source)
            count = len(speakers) or 1
            limit = provider_speaker_limit(engine)
            if len(speakers) > limit:
                message = (
                    f"**{len(speakers)} speakers detected, but {engine} supports at most {limit} in this pipeline.** "
                    "Choose a provider with enough speaker capacity before generating."
                )
            elif speakers:
                message = (
                    f"**{len(speakers)} speaker(s) detected.** Fish casts automatically from speaker gender metadata; visible preset rows are overrides."
                )
            else:
                message = (
                    "**Single narrator mode.** Speaker 0 still has an explicit voice assignment; no provider may choose it randomly."
                )

            result: list[object] = [message]
            for slot in range(MAX_SPEAKER_SLOTS):
                active = slot < count
                if active and speakers:
                    speaker = speakers[slot]
                    label = f"**Speaker {speaker.speaker_id}**"
                    if speaker.description:
                        label += f" ? {speaker.description}"
                elif active:
                    label = "**Speaker 0 ? single narrator**"
                else:
                    label = f"**Speaker slot {slot + 1}**"
                provider_slot_supported = slot < limit
                fish_visible = active and engine == "fish" and provider_slot_supported
                result.extend([
                    gr.update(visible=active),
                    gr.update(value=label),
                    gr.update(visible=active and engine == "gemini" and provider_slot_supported),
                    gr.update(visible=active and engine == "vibevoice" and provider_slot_supported),
                    gr.update(visible=fish_visible),
                    gr.update(visible=fish_visible),
                    gr.update(visible=fish_visible),
                    gr.update(visible=fish_visible),
                ])
            return result

        def update_casting(story_text_value, story_file_value, story_upload_value, engine):
            source = _read_story_for_casting(story_text_value, story_file_value, story_upload_value)
            return _casting_updates(source, engine)

        casting_inputs = [story_text, story_file, story_upload, tts_engine]
        for trigger in (story_text, story_file, story_upload, tts_engine):
            trigger.change(update_casting, casting_inputs, casting_outputs, show_progress="hidden")
        demo.load(update_casting, casting_inputs, casting_outputs, show_progress="hidden")

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

        def refresh_choices(
            current_story: str | None,
            current_background: str | None,
            current_voice: str | None,
            *values,
        ):
            speaker_voice_values = values[:MAX_SPEAKER_SLOTS]
            fish_preset_values = values[MAX_SPEAKER_SLOTS:]
            new_stories = list_input_stories()
            new_backgrounds = list_background_videos()
            new_voices = _vibe_choices()
            new_fish_presets = list_fish_reference_presets(PROJECT_ROOT)
            updates: list[object] = [
                gr.update(choices=new_stories, value=_first_or_none(new_stories, current_story)),
                gr.update(choices=new_backgrounds, value=_first_or_none(new_backgrounds, current_background)),
                gr.update(choices=new_voices, value=_first_or_none(new_voices, current_voice)),
            ]
            for value in speaker_voice_values:
                updates.append(gr.update(choices=new_voices, value=_first_or_none(new_voices, value)))
            for value in fish_preset_values:
                updates.append(
                    gr.update(
                        choices=new_fish_presets,
                        value=value if value in new_fish_presets else _first_or_none(new_fish_presets),
                    )
                )
            return updates

        refresh.click(
            refresh_choices,
            [story_file, background, vibe_speaker, *speaker_vibe_voices, *speaker_fish_presets],
            [story_file, background, vibe_speaker, *speaker_vibe_voices, *speaker_fish_presets],
            show_progress="hidden",
        )
        refresh_runtime.click(_runtime_status_markdown, outputs=[runtime_status], show_progress="hidden")

        provider_inputs = [
            tts_engine,
            gemini_voice, gemini_model, gemini_preprocess, gemini_split, gemini_chunk_seconds,
            vibe_speaker, vibe_model, vibe_cfg, vibe_steps, vibe_seed, vibe_device, vibe_dtype,
            fish_gpu_layers, fish_temperature, fish_reference_audio, fish_reference_text,
            *speaker_gemini_voices,
            *speaker_vibe_voices,
            *speaker_fish_presets,
            *speaker_fish_preset_names,
            *speaker_fish_audio,
            *speaker_fish_text,
        ]

        def make_tts_options(
            story_text_value,
            output_name_value,
            *values,
            speaker_source_text: str | None = None,
            **extra,
        ) -> PipelineOptions:
            cursor = 0
            tts_engine_value = values[cursor]; cursor += 1
            gemini_voice_value, gemini_model_value, gemini_preprocess_value, gemini_split_value, gemini_chunk_seconds_value = values[cursor:cursor + 5]; cursor += 5
            vibe_speaker_value, vibe_model_value, vibe_cfg_value, vibe_steps_value, vibe_seed_value, vibe_device_value, vibe_dtype_value = values[cursor:cursor + 7]; cursor += 7
            fish_gpu_layers_value, fish_temperature_value, fish_reference_audio_value, fish_reference_text_value = values[cursor:cursor + 4]; cursor += 4
            gemini_slots = list(values[cursor:cursor + MAX_SPEAKER_SLOTS]); cursor += MAX_SPEAKER_SLOTS
            vibe_slots = list(values[cursor:cursor + MAX_SPEAKER_SLOTS]); cursor += MAX_SPEAKER_SLOTS
            fish_preset_slots = list(values[cursor:cursor + MAX_SPEAKER_SLOTS]); cursor += MAX_SPEAKER_SLOTS
            fish_preset_name_slots = list(values[cursor:cursor + MAX_SPEAKER_SLOTS]); cursor += MAX_SPEAKER_SLOTS
            fish_audio_slots = list(values[cursor:cursor + MAX_SPEAKER_SLOTS]); cursor += MAX_SPEAKER_SLOTS
            fish_text_slots = list(values[cursor:cursor + MAX_SPEAKER_SLOTS]); cursor += MAX_SPEAKER_SLOTS

            speakers = detect_speakers(speaker_source_text if speaker_source_text is not None else (story_text_value or ""))
            speaker_ids = [speaker.speaker_id for speaker in speakers] or [0]
            gemini_speaker_voices = {
                speaker_id: gemini_slots[index]
                for index, speaker_id in enumerate(speaker_ids[:MAX_SPEAKER_SLOTS])
                if gemini_slots[index]
            }
            vibevoice_speaker_voices = {
                speaker_id: vibe_slots[index]
                for index, speaker_id in enumerate(speaker_ids[:MAX_SPEAKER_SLOTS])
                if vibe_slots[index]
            }
            fish_speaker_references: dict[int, tuple[str | Path, str]] = {}
            for index, speaker_id in enumerate(speaker_ids[:MAX_SPEAKER_SLOTS]):
                uploaded_audio = fish_audio_slots[index]
                uploaded_text = (fish_text_slots[index] or "").strip()
                preset_id = (fish_preset_slots[index] or "").strip()
                if uploaded_audio:
                    if not uploaded_text:
                        raise ValueError(
                            f"Speaker {speaker_id}: Fish uploaded reference audio requires its exact transcript."
                        )
                    _saved_name, saved_audio, saved_text = cache_fish_reference_preset(
                        PROJECT_ROOT,
                        uploaded_audio,
                        uploaded_text,
                        (fish_preset_name_slots[index] or "").strip(),
                    )
                    fish_speaker_references[speaker_id] = (saved_audio, saved_text)
                elif preset_id:
                    fish_speaker_references[speaker_id] = resolve_fish_reference_preset(
                        PROJECT_ROOT, preset_id
                    )

            return PipelineOptions(
                story_text=story_text_value or "",
                output_name=output_name_value or "",
                tts_engine=tts_engine_value,
                gemini_voice=gemini_voice_value or "Kore",
                gemini_model=gemini_model_value or "gemini-3.1-flash-tts-preview",
                gemini_preprocess=bool(gemini_preprocess_value),
                gemini_split_on_separator=bool(gemini_split_value),
                gemini_chunk_seconds=int(gemini_chunk_seconds_value),
                gemini_speaker_voices=gemini_speaker_voices,
                vibevoice_model=vibe_model_value or "microsoft/VibeVoice-1.5B",
                vibevoice_speaker=vibe_speaker_value or "Alice",
                vibevoice_cfg_scale=float(vibe_cfg_value),
                vibevoice_diffusion_steps=int(vibe_steps_value),
                vibevoice_seed=int(vibe_seed_value),
                vibevoice_device=vibe_device_value,
                vibevoice_dtype=vibe_dtype_value,
                vibevoice_speaker_voices=vibevoice_speaker_voices,
                fish_gpu_layers=int(fish_gpu_layers_value),
                fish_temperature=float(fish_temperature_value),
                fish_reference_audio=fish_reference_audio_value or None,
                fish_reference_text=fish_reference_text_value or "",
                fish_speaker_references=fish_speaker_references,
                **extra,
            )

        render_inputs = [
            background, background_upload, output_format, random_start,
            captions, caption_theme, caption_max_words, caption_pause,
            whisper_model, whisper_language, whisper_compute, whisper_align,
            encoder, quality, end_padding,
        ]
        provider_input_count = len(provider_inputs)

        def generate(story_file_value, story_upload_value, story_text_value, output_name_value, *values):
            logs: list[str] = []

            def log(message: str) -> None:
                logs.append(message)
                print(message, flush=True)

            try:
                provider_values = values[:provider_input_count]
                render_values = values[provider_input_count:]
                (
                    background_value, background_upload_value, output_format_value, random_start_value,
                    captions_value, caption_theme_value, caption_max_words_value, caption_pause_value,
                    whisper_model_value, whisper_language_value, whisper_compute_value, whisper_align_value,
                    encoder_value, quality_value, end_padding_value,
                ) = render_values

                resolved_story_text = story_text_value or ""
                if not resolved_story_text.strip():
                    resolved_story_text = _read_story_for_casting("", story_file_value, story_upload_value)

                options = make_tts_options(
                    story_text_value or "",
                    output_name_value,
                    *provider_values,
                    speaker_source_text=resolved_story_text,
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
                result = RedditVideoPipeline(log=log).run(options)
                caption = str(result.caption_path) if result.caption_path else "disabled"
                short_summary = (
                    f"\n- Short: `{result.short_video_path}` at `{result.short_end_seconds:.2f}s`"
                    if result.short_video_path and result.short_end_seconds is not None
                    else "\n- Short: skipped (no [[SHORTS_CLIFFHANGER]] marker)"
                )
                summary = (
                    "### Completed\n"
                    f"- Run: `{result.run_dir}`\n"
                    f"- Video: `{result.video_path}`\n"
                    f"- Audio: `{result.audio_path}`\n"
                    f"- Transcript: `{result.whisper_json_path}`\n"
                    f"- Captions: `{caption}`"
                    f"{short_summary}\n"
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
                *provider_inputs,
                *render_inputs,
            ],
            outputs=[video_output, audio_output, status],
            api_name="generate_video",
        )

    return demo
