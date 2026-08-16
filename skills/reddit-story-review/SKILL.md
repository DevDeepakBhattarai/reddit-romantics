---
name: reddit-story-review
description: Review and refine a finished Reddit-style narration script before it is returned. Use automatically after reddit-story creates a draft, or when the user asks to improve, tighten, rewrite, or diagnose an existing story.
---

# Reddit Story Review

Review the finished story as if you did not write it. Fix problems directly instead of producing a long critique unless the user explicitly asked for notes.

Start with the hook. If the first spoken line would not make a cold viewer stop scrolling, rewrite it. It should be clear immediately and open a question worth hearing answered.

Then read the story once for understanding. A listener cannot look back up the page. At every point it should be obvious who is speaking, what just happened, why it matters, and why the next beat follows from it. Simplify confusing references and move necessary context closer to where it is needed.

Check the flow. Remove repeated arguments, repeated discoveries, and explanations that do not change anything. Keep the story moving toward the thing the opening promised. Spend more time on the confrontation, romantic tension, revenge, confession, reveal, or frightening moment than on travel, scheduling, or setup.

Check the protagonist. They should make meaningful choices and help cause the ending. If luck or another person solves the central problem, strengthen the protagonist's role.

Check the two voices. Speaker 0 remains the narrator. Speaker 1 should usually remain the main counterpart. Do not make the listener decode constant voice-slot changes. If a minor character does not need a live line, let Speaker 0 report what they said.

For production scripts, preserve and verify the non-spoken `Speaker N - gender=male|female; ...` casting metadata. The gender must match the actual character using that speaker slot, including the narrator's perspective. Correct the metadata if the rewrite changes who occupies a slot. Do not remove it; Fish Audio uses it for deterministic voice selection.

Check the payoff. The story must actually deliver the moment the listener has been waiting for. Do not build toward a confrontation, revenge, confession, romantic decision, or reveal and then summarize it in one sentence.

Finally, listen for performance. Add or adjust Fish Audio S2 Pro cues where they produce a real audible difference. Use emotion, whispers, laughter, pauses, breathlessness, anger, shock, or changes in speed where the scene earns them. Remove cues that merely decorate ordinary narration or delay the opening hook.

If the script contains `[[SHORTS_CLIFFHANGER]]`, treat it as non-spoken production metadata. There must be exactly one marker, on its own line, immediately after a complete spoken line. Check that the line before it is a genuinely unresolved cliffhanger and that the marker is still early enough to land comfortably under two minutes of narration. Move the marker if the rewrite changes the best cliffhanger; never delete it from an automation script and never place it inside a spoken line.

When you change the story substantially, check the first line one more time. The hook still needs to match the final version.

Return the refined script to `reddit-story`. Do not expose the rough draft or review notes unless the user asked to see them.

## Staged automation

When reviewing a scheduled production story that already exists at `runs/<run>/story.md`, work on that exact file and that exact run. Do not create a new run folder.

Read the whole story once and perform one review pass. If changes are warranted, apply all necessary fixes during that same pass and overwrite `story.md` with the refined version. If the story already meets the review criteria, leave `story.md` unchanged. Do not run a second review pass merely to verify your own edits. Preserve exactly one valid `[[SHORTS_CLIFFHANGER]]` marker.

When the caller also requested video production, hand the same existing run folder to `reddit-story-video` after this review pass finishes.
