---
name: reddit-story-video
description: Enqueue an existing Reddit story run for video generation.
---

# Reddit Story Video

Work in `D:\Reddit-Romantics\Automation`.

This skill only starts video generation for an **existing** run. Do not create a run, generate/review/edit the story, or inspect story contents.

## Start generation

```powershell
.\.venv\Scripts\python.exe main.py enqueue --run-dir <existing-run>
```

`enqueue` returns immediately and the detached worker processes queued runs one at a time. After enqueue succeeds, the task is done. Do not wait for completion or poll the queue unless the user explicitly asks.

## Common options

All normal pipeline options can be passed to `enqueue`.

- `--tts fish|gemini|vibevoice` — default: `fish`
- `--speaker-preset ID=PRESET` — override a speaker voice; repeat for multiple speakers
- `--background PATH`
- `--caption-theme THEME`
- `--no-captions`
- `--encoder auto|nvenc|cpu`
- `--video-quality N`
- `--audio-bitrate RATE`
- `--fish-gpu-layers N` — default: `20`
- `--fish-temperature N`

For the complete current option list, use:

```powershell
.\.venv\Scripts\python.exe main.py enqueue --help
```

## Voices

Fish is the default. Without overrides, Fish auto-selects the configured male/female presets from the story metadata. Use `--speaker-preset ID=PRESET` only when a specific voice is requested.

Examples:

```powershell
# Default Fish voices
.\.venv\Scripts\python.exe main.py enqueue --run-dir <existing-run>

# Manual Fish voice override
.\.venv\Scripts\python.exe main.py enqueue --run-dir <existing-run> --speaker-preset 0=Ethan --speaker-preset 1=Sarah

# Gemini
.\.venv\Scripts\python.exe main.py enqueue --run-dir <existing-run> --tts gemini --speaker-preset 0=Kore --speaker-preset 1=Puck

# VibeVoice
.\.venv\Scripts\python.exe main.py enqueue --run-dir <existing-run> --tts vibevoice --speaker-preset 0=Alice --speaker-preset 1=Frank
```

## Be careful

- `--run-dir` must point to the already-created run.
- Do not call `new-run` from this skill.
- Do not modify `story.md`.
- Use `enqueue`, not synchronous `run`, unless the user explicitly asks for foreground debugging.
- Enqueue each requested run once; the worker handles sequential execution.
