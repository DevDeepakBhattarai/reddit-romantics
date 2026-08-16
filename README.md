# Reddit Romantics Automation

Local story-to-video automation for long-form YouTube videos and cliffhanger Shorts.

## Workflow

Each story gets exactly one user-facing workspace:

```text
runs/2026-08-16_16-27-short-story-title/
  story.md
  narration.wav
  transcript.json
  captions.ass
  full.mp4
  short.wav
  short-transcript.json
  short-captions.ass
  short.mp4
  thumbnail.png
```

`runs/` is local and gitignored. Temporary model/runtime work remains under `.work/`.

The intended production flow is:

1. Generate the story.
2. Review/refine it.
3. Save only the refined script to `story.md`.
4. Generate narration once. Fish Audio S2 Pro is the default; Gemini and VibeVoice remain optional overrides.
5. The reviewed story contains exactly one `[[SHORTS_CLIFFHANGER]]` marker on its own line after the chosen cliffhanger sentence.
6. The pipeline strips that marker before TTS, transcribes once with WhisperX, and renders `full.mp4`.
7. It aligns the pre-marker story words to `transcript.json`, derives the cutoff timestamp programmatically, and renders `short.mp4` automatically. TTS and WhisperX are not rerun.
8. Save the final thumbnail as `thumbnail.png` in the same run folder.

There are no root `input/` or `output/` directories in the new workflow.

## Skills

The story skills now live in this repository under `skills/`:

- `reddit-story` - story coordinator.
- `reddit-story-genres` - genre-specific behavior.
- `reddit-story-hooks` - opening-hook guidance.
- `reddit-story-review` - refinement pass.
- `reddit-story-thumbnail` - thumbnail generation guidance.
- `reddit-story-video` - minimal instructions for driving this local pipeline and producing the full video + Short.

## Supported narration engines

Only these three providers are maintained:

- **Gemini TTS** - API-based narration, up to two speakers.
- **Microsoft VibeVoice 1.5B** - local long-form narration with reusable presets.
- **Fish Audio S2 Pro** - local native Windows `s2.cpp`, full unquantized F16 model, 28 CUDA transformer layers by default.

For Fish, production stories carry non-spoken `gender=male` / `gender=female` speaker metadata. The pipeline auto-casts male speakers to `Ethan` and female speakers to `Sarah`. `--speaker-preset ID=PRESET` is an optional override. Gemini and VibeVoice still require explicit per-speaker voices.

## Setup

Requirements: Windows, Python 3.11, `uv`, Git, FFmpeg/FFprobe, and an NVIDIA GPU for the local accelerated paths.

```powershell
.\setup.ps1
```

For Gemini, set `GOOGLE_API_KEY` in `.env`.

## CLI

Create a run first:

```powershell
.\.venv\Scripts\python.exe main.py new-run --title "my story"
```

Write the reviewed story to the printed folder's `story.md`, then generate the full video.

Gemini example:

```powershell
.\.venv\Scripts\python.exe main.py run `
  --run-dir runs\2026-08-16_16-27_my-story `
  --tts gemini `
  --speaker-preset 0=Kore `
  --speaker-preset 1=Puck
```

VibeVoice example:

```powershell
.\.venv\Scripts\python.exe main.py run `
  --run-dir runs\2026-08-16_16-27_my-story `
  --tts vibevoice `
  --speaker-preset 0=Alice `
  --speaker-preset 1=Frank
```

Default Fish example:

```powershell
.\.venv\Scripts\python.exe main.py run `
  --run-dir runs\2026-08-16_16-27_my-story
```

The story metadata decides whether each slot uses the male or female default voice. Add `--speaker-preset ID=PRESET` only when you intentionally want to override that automatic casting.

The same `run` command also creates `short.mp4` automatically when `story.md` contains `[[SHORTS_CLIFFHANGER]]`. There is no separate AI cliffhanger-selection step and no manual timestamp argument.

`main.py list` shows existing runs, backgrounds, caption themes, VibeVoice presets, and saved Fish presets.

## Gradio UI

```powershell
.\.venv\Scripts\python.exe app.py
```

The UI now writes through the same run-folder pipeline. The default full-video mode keeps the background source dimensions.

## Project structure

```text
app.py
main.py
setup.ps1
skills/
reddit_video/
  pipeline.py
  runs.py
  captions.py
  tts.py
  fish.py
  tts_text.py
gemini-tts/
videos/
voice_presets/
patches/
runs/        # generated, gitignored
.work/       # internal, gitignored
vendor/      # runtimes/models, gitignored
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
