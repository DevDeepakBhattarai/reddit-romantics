# Reddit Romantics Automation

A local browser + CLI pipeline for turning a Reddit-style story into a narrated, captioned vertical video.

The repository has one reusable Python pipeline with two front ends:

- **Gradio UI** for interactive browser use.
- **CLI / Python entry point** for scripts and automation.

Gemini TTS now uses `gemini-3.1-flash-tts-preview` with automatic semantic ~3-minute chunking. Microsoft `microsoft/VibeVoice-1.5B` is available as a second narration engine. The old CSM implementation has been removed.

## What the pipeline does

1. Accepts a pasted story, uploaded/local `.txt` story, or CLI story file.
2. Generates narration with Gemini TTS or VibeVoice 1.5B.
3. Transcribes the narration with WhisperX for word-level timing.
4. Converts those timings into a selectable ASS caption theme.
5. Loops/crops the selected background footage.
6. Renders a final **1080x1920 YouTube Shorts** video (or keeps source dimensions if selected).

VibeVoice is intentionally **not chunked**. The complete story is sent as one `Speaker 1:` generation so its long-context speech model can maintain continuity across long narration.

## Setup

Requirements outside Python:

- Windows
- Git
- `uv`
- FFmpeg / ffprobe on `PATH`
- NVIDIA GPU recommended for VibeVoice + WhisperX

Run:

```powershell
.\setup.ps1
```

The setup script creates two isolated environments:

- `.venv` — application, Gradio, Gemini, and VibeVoice runtime.
- `.whisperx-venv` — WhisperX in a separate environment to prevent dependency conflicts.

It also installs a pinned VibeVoice community runtime into `vendor/VibeVoice`. Microsoft removed the official TTS inference code from the upstream VibeVoice repository in January 2026, while the `microsoft/VibeVoice-1.5B` model weights remain available. The pinned community runtime preserves the compatible inference implementation required to run those weights.

After setup, put your Gemini key in `.env` when you want Gemini TTS:

```dotenv
GOOGLE_API_KEY=your_key_here
```

VibeVoice does not require the Gemini API key.

### Setup without WhisperX

If you already have WhisperX elsewhere:

```powershell
.\setup.ps1 -SkipWhisperX
```

Then set `WHISPERX_COMMAND` in `.env`, for example:

```dotenv
WHISPERX_COMMAND=conda run -n whisperx whisperx
```

## Gradio UI

```powershell
.\.venv\Scripts\python.exe app.py
```

or:

```powershell
.\.venv\Scripts\python.exe main.py ui
```

Open `http://127.0.0.1:7860`.

The UI exposes:

- story file / pasted story
- Gemini vs VibeVoice
- Gemini 3.1 Flash TTS, all 30 currently documented voice names, preprocessing, explicit separator boundaries, and semantic chunk duration
- VibeVoice speaker, model, CFG, diffusion steps, seed, device, and dtype
- background video
- Shorts vs source-sized output
- random background start
- captions on/off
- caption theme, word count, and pause grouping
- WhisperX model/alignment/compute type
- NVENC/CPU renderer and quality
- end padding

The Generate button is also registered as the Gradio API endpoint `generate_video`, but Gradio is not required for automation; use the CLI below when scripting.

## CLI / script automation

See every option:

```powershell
.\.venv\Scripts\python.exe main.py run --help
```

### Gemini example

```powershell
.\.venv\Scripts\python.exe main.py run `
  --story-file input\my_story.txt `
  --tts gemini `
  --gemini-voice Kore `
  --background videos\minecraft\minecraft.mp4 `
  --caption-theme classic_yellow
```

Gemini automatically splits long stories into roughly 3-minute semantic chunks. It prefers whole paragraphs, then sentence boundaries, then clauses/words only when necessary. Existing `-------------` lines remain optional hard boundaries; `--no-gemini-split` now ignores only those explicit boundaries and does **not** disable safe automatic chunking. Use `--gemini-chunk-seconds` to change the 180-second target.

### VibeVoice example

```powershell
.\.venv\Scripts\python.exe main.py run `
  --story-file input\my_story.txt `
  --tts vibevoice `
  --vibevoice-model microsoft/VibeVoice-1.5B `
  --vibevoice-speaker Alice `
  --background videos\minecraft\minecraft.mp4 `
  --caption-theme story_calm
```

VibeVoice receives the full story in one generation; it does not use Gemini's separator chunking system.

### List available local inputs/options

```powershell
.\.venv\Scripts\python.exe main.py list
```

## Caption themes

The built-in themes are:

- `classic_yellow` — bold white with active yellow word
- `clean_white` — restrained creator-style captions
- `tiktok_pop` — bright pink active word
- `mrbeast_punch` — large uppercase Impact-style captions
- `neon_green` — green active word
- `electric_blue` — blue accent
- `red_alert` — large uppercase red accent
- `minimal_box` — clean lower translucent box
- `karaoke` — lower word-follow captions
- `story_calm` — softer lower captions for long narration

Themes live in `reddit_video/captions.py` and can be extended without touching the render pipeline.

## Backward-compatible batch command

The original command remains usable:

```bat
process.bat my_story.txt minecraft Kore
process.bat my_story.txt asmr Leda
```

It is now a thin compatibility wrapper over `main.py run` instead of containing a second copy of the pipeline.

## Project structure

```text
app.py                         Gradio convenience launcher
main.py                        CLI + UI launcher
process.bat                    Legacy compatibility wrapper
setup.ps1                      Reproducible local setup
reddit_video/
  pipeline.py                  Reusable end-to-end pipeline
  tts.py                       Gemini + VibeVoice adapters
  captions.py                  Caption themes + ASS generation
caption/
  convert_json_to_ass.py       Backward-compatible caption CLI
  ...                          Existing caption assets/tools
gemini-tts/                    Existing Gemini implementation
input/                         Story text files
videos/                        Background footage
output/                        Generated videos/captions (gitignored)
vendor/                        Downloaded VibeVoice runtime (gitignored)
.work/                         Temporary generation files (gitignored)
```

## Notes

- VibeVoice model weights are downloaded by Hugging Face on first use, so the first VibeVoice generation has an additional download/setup cost.
- Very long VibeVoice generations can use substantial VRAM. If CUDA runs out of memory, lower other GPU load or select CPU as a fallback (much slower).
- The render queue is intentionally limited to one Gradio job at a time so multiple browser clicks do not compete for the same GPU.
