---
name: reddit-story
description: Write short, addictive Reddit-style stories for narrated YouTube videos and Shorts. Use when the user asks for a Reddit story, dramatic confession, relationship story, revenge story, spicy romance, family or workplace drama, suspense, horror, or another short first-person story meant to keep people listening.
---

# Reddit Story

Write a story that is easy to understand, easy to listen to, and hard to leave.

These stories are made for narration, not literary fiction. Keep the language natural and simple to understand. Let the narrator sound like a real person telling something that still matters to them. Do not over-plan on the page, explain your technique, or turn the story into a writing exercise.

## How to work

Decide the story type first. Use `reddit-story-genres` skill for the genre behavior, especially when the user asks for revenge, spicy romance, betrayal, confession, family drama, workplace drama, or suspense/horror.

Then write the complete story.

For production stories, ten minutes of finished narration is the minimum, not the target. Before finalizing, make sure the script contains at least 1,700 spoken words after excluding speaker tags, casting metadata, performance cues, and other non-spoken production markers. Aim for roughly 1,700 to 2,000 spoken words, and go longer when the pacing or pauses would otherwise bring the finished audio under ten minutes.

The first spoken line matters more than any other line. The viewer may give the video only a couple of seconds before scrolling. So, I want you to use this information gap theory to create a compelling hook for the story. The hook has to be very easy to digest and actually grasp the idea from. Make sure it gives just the right amount of information to make the listener curious, but not too much that they can guess the ending.

```
George Loewenstein's information-gap theory argues that curiosity appears when we become aware of a gap between what we know and what we want to know. Later research suggests curiosity is often especially strong when people know enough to form a hypothesis, but not enough to resolve it. If you give them nothing, they are confused. If you give them everything, there is no curiosity.
```

# Bad Examples

> My boyfriend had been transferring exactly $300 to the same woman every Friday for eleven months. I assumed he was cheating. Then I saw her last name.

Has too much information and hard to follow.

> My ex walked into the radio booth at one in the morning, locked the door behind her, and said, "You have ten minutes before I marry the wrong man."

Too vague and difficult to understand.

Good Examples

> I made a joke with my husband and it's costing me my marriage with a man i truly love

Has the right balance of information and curiosity.

> My own friend convinced my husband that I cheated on him, he kicked me out of our house.

Make sure the hook is a short, digestible and easy to understand idea that makes the listener curious about what happens next. Don't make it overly complicated with details and overlapping things. Make it straight forward.

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
