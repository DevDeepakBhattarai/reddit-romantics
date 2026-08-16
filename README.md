# Reddit Romantics Automation

A focused local pipeline for turning story text into narrated videos.

## Supported narration engines

Only these three TTS providers are maintained:

- **Gemini TTS** — API-based narration, including up to two speakers.
- **Microsoft VibeVoice 1.5B** — local long-form narration with reusable voice presets.
- **Fish Audio S2 Pro** — local native Windows `s2.cpp` inference using the full unquantized F16 model and configurable partial CUDA offload (28 transformer layers by default).


## What the pipeline does

1. Loads or accepts story text.
2. Detects speaker turns and applies an explicit voice to every speaker.
3. Generates narration with Gemini, VibeVoice, or Fish Audio.
4. Optionally creates WhisperX captions.
5. Renders the narration over background footage with FFmpeg.

## Requirements

- Windows
- Python 3.11
- `uv`
- Git
- FFmpeg / FFprobe
- NVIDIA GPU recommended for VibeVoice, Fish, and WhisperX
- CMake + a CUDA-capable C++ build environment for Fish
- WSL only for the one-time Fish F16 model export performed by `setup.ps1`

## Setup

```powershell
.\setup.ps1
```

That single setup script prepares the application environment, installs the pinned VibeVoice runtime, builds the Fish `s2.cpp` CUDA runtime, exports Fish S2 Pro to full F16 GGUF when needed, and prepares WhisperX.

Useful setup switches:

```powershell
.\setup.ps1 -SkipFish
.\setup.ps1 -SkipWhisperX
```

For Gemini, put your API key in `.env`:

```dotenv
GOOGLE_API_KEY=your_key_here
```

## Gradio UI

```powershell
.\.venv\Scripts\python.exe app.py
```

The UI is video-focused. It contains provider settings, speaker voice assignments, background/caption controls, and the final **Generate video** action. There is no separate audio-only generation page or endpoint.

## CLI

```powershell
.\.venv\Scripts\python.exe main.py run --help
```

Gemini example:

```powershell
.\.venv\Scripts\python.exe main.py run `
  --story-file input\story.txt `
  --tts gemini `
  --gemini-voice Kore `
  --background videos\minecraft\minecraft.mp4
```

VibeVoice example:

```powershell
.\.venv\Scripts\python.exe main.py run `
  --story-file input\story.txt `
  --tts vibevoice `
  --vibevoice-speaker Alice `
  --background videos\minecraft\minecraft.mp4
```

Fish example:

```powershell
.\.venv\Scripts\python.exe main.py run `
  --story-file input\story.txt `
  --tts fish `
  --fish-gpu-layers 28 `
  --fish-reference-audio voice.wav `
  --fish-reference-text "Exact transcript of the reference clip." `
  --background videos\minecraft\minecraft.mp4
```

Fish long-form narration semantically packs complete sentences under a 500-character target while keeping speaker boundaries explicit. Uploaded Fish reference voices can be saved under `voice_presets/fish/` and reused later.

## Project structure

```text
app.py                    Gradio launcher
main.py                   CLI + UI command entry point
setup.ps1                 Single project setup script
reddit_video/
  pipeline.py             End-to-end video pipeline
  tts.py                  Gemini + VibeVoice implementations
  fish.py                 Fish S2 Pro implementation
  tts_text.py             Speaker/tag normalization
  captions.py             WhisperX caption styling/render helpers
gemini-tts/
  gemini_tts.py           Gemini TTS worker
input/                    Story text files
videos/                   Background footage
voice_presets/fish/       Saved Fish reference voices (gitignored)
patches/                   Required s2.cpp long-form patch
output/                    Generated videos (gitignored)
vendor/                    VibeVoice + s2.cpp runtimes (gitignored)
.work/                     Active model/cache/work files (gitignored)
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
