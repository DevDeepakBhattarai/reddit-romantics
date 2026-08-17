---
name: reddit-story-video
description: Queue the local Reddit Romantics story-to-video pipeline for a reviewed story. Use for scheduled production or whenever a story should become a full YouTube video and cliffhanger Short in D:\Reddit-Romantics\Automation.
---

# Reddit Story Video

Work in `D:\Reddit-Romantics\Automation`.

## Existing production run

If the story already lives at `runs/<timestamp>_<title>/story.md`, that folder **is the run**. Reuse it. Do not call `new-run`, do not copy the story into another folder, and do not rename the run. Scheduled review/video production always uses this mode.

Before enqueueing, require a non-empty `story.md` containing exactly one `[[SHORTS_CLIFFHANGER]]` line and one non-spoken casting metadata line for each speaker, such as `Speaker 0 - gender=female; narrator; Maya`. The review stage is responsible for keeping the marker and gender metadata correct.

## Scheduled/background production

Long video generation must not run inside the ChatGPT/scheduled-task session. Enqueue it and return immediately:

`.\.venv\Scripts\python.exe main.py enqueue --run-dir <existing-run>`

`enqueue` is intentionally fast. It ensures a detached worker exists, sends the run into that worker's in-memory FIFO queue, and exits. **Do not wait for the video to finish, do not poll the worker, and do not replace `enqueue` with the synchronous `run` command in scheduled production.** Once every intended run has been enqueued successfully, the AI task is complete and should end.

There may be any number of queued runs while the worker is alive. A single worker owns an in-memory FIFO queue and runs exactly one video pipeline at a time. This prevents two Fish/WhisperX/render jobs from competing for the same machine. When the queue has been drained, the worker exits by itself and the queue is destroyed. If the worker crashes or is killed, every pending queue item dies with it; the next worker always starts empty. Failed-job details remain in the worker/job logs and do not prevent later in-memory jobs from running.

Use `.\.venv\Scripts\python.exe main.py queue-status` only when a user explicitly asks to inspect queue state; scheduled production should not sit around monitoring it.

Fish Audio S2 Pro is the default. The pipeline reads story metadata and auto-casts each speaker by character gender, not by speaker number: male -> `Ethan`, female -> `Sarah`. `--speaker-preset ID=PRESET` remains an optional manual override.

For each queued run, the worker strips `[[SHORTS_CLIFFHANGER]]` before TTS, generates narration once, transcribes once, aligns the marker to that transcript, and writes both `full.mp4` and `short.mp4` into the same run folder. Do not ask another AI to choose a Short cutoff. Do not rerun TTS or WhisperX for the Short.

## Interactive synchronous production

The synchronous command still exists for manual debugging when the caller explicitly wants to stay attached until completion:

`.\.venv\Scripts\python.exe main.py run --run-dir <existing-run>`

Do not use synchronous mode from scheduled tasks.

## New story without a run

Only when a reviewed story does **not** already have a run folder, create one with `.\.venv\Scripts\python.exe main.py new-run --title "short title"`, save the reviewed script as `<run>\story.md`, and then enqueue that run.

Keep all user-facing artifacts in the same `runs/<timestamp>_<title>/` folder. `.work/` is only for internal temporary/cache/queue state.
