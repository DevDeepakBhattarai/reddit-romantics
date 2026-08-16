---
name: reddit-story-video
description: Run the local Reddit Romantics story-to-video pipeline for a reviewed story. Use for scheduled production or whenever a story should become a full YouTube video and cliffhanger Short in D:\Reddit-Romantics\Automation.
---

# Reddit Story Video

Work in `D:\Reddit-Romantics\Automation`.

## Existing production run

If the story already lives at `runs/<timestamp>_<title>/story.md`, that folder **is the run**. Reuse it. Do not call `new-run`, do not copy the story into another folder, and do not rename the run. Scheduled review/video production always uses this mode.

Before starting, require a non-empty `story.md` containing exactly one `[[SHORTS_CLIFFHANGER]]` line and one non-spoken casting metadata line for each speaker, such as `Speaker 0 - gender=female; narrator; Maya`. The review stage is responsible for keeping the marker and gender metadata correct.

Run the pipeline with:

`.\.venv\Scripts\python.exe main.py run --run-dir <existing-run>`

Fish Audio S2 Pro is the default. The pipeline reads the story metadata and auto-casts each speaker by the actual character gender, not by speaker number: male -> `Ethan`, female -> `Sarah`. `--speaker-preset ID=PRESET` is only a manual override.

The pipeline strips `[[SHORTS_CLIFFHANGER]]` before TTS, generates narration once, transcribes once, aligns the marker to that transcript, and writes both `full.mp4` and `short.mp4` into the same run folder. Do not ask another AI to choose a Short cutoff. Do not rerun TTS or WhisperX for the Short.

## New story without a run

Only when a reviewed story does **not** already have a run folder, create one with `.\.venv\Scripts\python.exe main.py new-run --title "short title"`, save the reviewed script as `<run>\story.md`, and then run the same command against that new run.

Keep all user-facing artifacts in the same `runs/<timestamp>_<title>/` folder. `.work/` is only for internal temporary/cache files.
