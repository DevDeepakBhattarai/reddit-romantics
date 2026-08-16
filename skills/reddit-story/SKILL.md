---
name: reddit-story
description: Write short, addictive Reddit-style stories for narrated YouTube videos and Shorts. Use when the user asks for a Reddit story, dramatic confession, relationship story, revenge story, spicy romance, family or workplace drama, suspense, horror, or another short first-person story meant to keep people listening.
---

# Reddit Story

Write a story that is easy to understand, easy to listen to, and hard to leave.

These stories are made for narration, not literary fiction. Keep the language natural. Let the narrator sound like a real person telling something that still matters to them. Do not over-plan on the page, explain your technique, or turn the story into a writing exercise.

## How to work

Decide the story type first. Use `reddit-story-genres` for the genre behavior, especially when the user asks for revenge, spicy romance, betrayal, confession, family drama, workplace drama, or suspense/horror.

Then write the complete story.

The first spoken line matters more than any other line. The viewer may give the video only a couple of seconds before scrolling. Use `reddit-story-hooks` and open with the interesting thing itself: a discovery, betrayal, threat, confession, contradiction, attraction, humiliation, or other concrete event. Do not begin with background, ages, how people met, weather, or vague setup.

Keep the story simple enough to follow by ear. Move forward naturally. Give background only when the listener needs it. If two scenes do the same job, combine them. Slow down for the moments people came to hear: the confrontation, reveal, romantic tension, revenge, confession, or frightening discovery.

The protagonist should make choices and affect the outcome. Do not let coincidence, a random authority figure, or a convenient stranger solve the story for them.

## Narration format

Return the story as a two-voice script.

For production stories, understand the characters first and add non-spoken casting metadata before the dialogue:

`Speaker 0 - gender=female; narrator; Maya`
`Speaker 1 - gender=male; main counterpart; Adrian`

Choose `male` or `female` from the actual character and narration perspective. Do not assume Speaker 0 is female or Speaker 1 is male. A male-narrated story must mark Speaker 0 male; a female-narrated story must mark Speaker 0 female. This metadata is used only for voice casting and is stripped before TTS.

- `Speaker 0:` is the narrator and stays the narrator for the whole story.
- `Speaker 1:` is normally the main other person in the story.
- Prefer a stable two-person audio relationship. Minor characters can be reported by Speaker 0 instead of constantly taking over Speaker 1.
- Every spoken line starts with a speaker tag.
- Keep lines short enough to sound natural in TTS.
- Write numbers, times, money, and abbreviations the way they should be spoken when that improves narration.

The script is generated with Fish Audio S2 Pro. Use inline performance cues when they make the delivery more alive: `[whisper]`, `[laughing]`, `[sigh]`, `[angry]`,`[flirty]`,`[slow]` ,`[shocked]`, `[pause]`, `[fast]`, `[emphasis]`, `[breath]`, or another short natural instruction the voice can actually perform. Do not decorate every line. Do not spend the opening seconds on a laugh, breath, or sound effect before the hook reaches the listener.

## Refine before returning

For a normal interactive story request, a first draft is not the final answer. After the full story exists, use `reddit-story-review` to review and improve it. Fix the story instead of merely noting problems. The hook gets checked first, then clarity, flow, speaker consistency, momentum, payoff, and performance.

The user should receive only the refined story. Do not expose the outline, rough draft, review notes, scorecard, or internal reasoning unless they explicitly ask for them.

There is one production exception: when the caller explicitly requests a **generation-only staged pass**, do not invoke `reddit-story-review`, thumbnail generation, or video generation in that pass. Generate the complete story, include the required `[[SHORTS_CLIFFHANGER]]` marker, save it as the run's `story.md`, and stop. A later scheduled review stage will review that file exactly once and overwrite it in place if needed.

If the user asks for a thumbnail, finish and refine the story first, then use `reddit-story-thumbnail`.

## Production workflow

The local project root is `D:\Reddit-Romantics\Automation`, and production artifacts live under `runs/`.

Build the Short decision into the story itself. Put exactly one `[[SHORTS_CLIFFHANGER]]` marker on its own line immediately after a complete spoken line that makes a strong unresolved cliffhanger. Aim to place it early enough that normal narration reaches it comfortably before two minutes; roughly the first 110-170 spoken words is a useful target. The line before the marker should create a question, reveal, threat, confession, discovery, or decision that the viewer wants resolved. Do not put the answer immediately before the marker.

The marker is production metadata, not dialogue. Keep it in `story.md`; the video pipeline strips it before TTS and uses it to derive the Short cutoff automatically from the transcript.

There are two valid production modes:

- **Single-pass production:** create the run with `.\.venv\Scripts\python.exe main.py new-run --title "short title"`, generate the story, review it with `reddit-story-review`, save the final reviewed script as that run's `story.md`, then hand that same run to `reddit-story-video`.
- **Staged generation-only production:** generate the complete story first without creating a run. When it is fully written, create the run with `.\.venv\Scripts\python.exe main.py new-run --title "short title"` and write the complete story to that run's `story.md`. Then stop. Do not review it and do not start video generation. This avoids exposing an empty/in-progress `story.md`; a run folder containing only a non-empty `story.md` is the handoff state for the later review/video task. When a scheduled job is generating multiple stories, finish all requested stories before creating any of their run folders so the later scheduled stage sees the batch atomically.

Never create a second run folder for the same story during review or video generation.

## Defaults

When the user leaves details open, make sensible creative decisions instead of asking unnecessary questions. Keep the cast small, the conflict clear, and the story focused on one main emotional problem.

Do not add headings, explanations, or closing commentary. Production casting metadata is the one allowed non-spoken preamble; otherwise return the finished narration only unless the user asked for another artifact such as a thumbnail.
