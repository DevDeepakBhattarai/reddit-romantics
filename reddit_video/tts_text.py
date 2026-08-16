from __future__ import annotations

import re
from dataclasses import dataclass

_SPEAKER_LINE_RE = re.compile(r"^\s*Speaker\s+(\d+)\s*:\s*(.*)$", re.IGNORECASE)
_SPEAKER_META_RE = re.compile(r"^\s*Speaker\s+(\d+)\s*[-–—]\s*(.+?)\s*$", re.IGNORECASE)
_DECORATOR_RE = re.compile(r"(?P<open>\[|<)(?P<body>[A-Za-z][A-Za-z ,.'!_-]{0,63})(?P<close>\]|>)")


@dataclass(frozen=True)
class SpeakerInfo:
    speaker_id: int
    description: str = ""

    @property
    def name(self) -> str:
        return f"Speaker {self.speaker_id}"

    @property
    def label(self) -> str:
        return f"{self.name} — {self.description}" if self.description else self.name


def detect_speakers(text: str) -> list[SpeakerInfo]:
    """Return speakers in numeric order, using optional `Speaker N - description` metadata."""
    descriptions: dict[int, str] = {}
    found: set[int] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip().strip('"')
        meta = _SPEAKER_META_RE.match(line)
        if meta:
            speaker_id = int(meta.group(1))
            found.add(speaker_id)
            descriptions[speaker_id] = meta.group(2).strip()
            continue
        turn = _SPEAKER_LINE_RE.match(line)
        if turn:
            found.add(int(turn.group(1)))
    return [SpeakerInfo(speaker_id, descriptions.get(speaker_id, "")) for speaker_id in sorted(found)]



def infer_speaker_gender(speaker: SpeakerInfo) -> str | None:
    """Read deterministic gender metadata chosen by the story-generation AI."""
    description = speaker.description.strip().lower()
    if not description:
        return None

    explicit = re.search(r"\bgender\s*[:=]\s*(male|female)\b", description)
    if explicit:
        return explicit.group(1)

    # Backward-compatible support for older descriptive metadata. Prefer the
    # explicit `gender=...` form in newly generated production stories.
    if re.search(r"\b(female|woman|girl)\b", description):
        return "female"
    if re.search(r"\b(male|man|boy)\b", description):
        return "male"
    return None

def strip_speaker_metadata(text: str) -> str:
    """Remove voice-casting metadata while preserving actual `Speaker N:` transcript turns."""
    lines: list[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if stripped in {'"""', "'''"}:
            continue
        if _SPEAKER_META_RE.match(stripped.strip('"')):
            continue
        lines.append(raw_line)
    return "\n".join(lines).strip()


def _canonical_tag(body: str, provider: str) -> str:
    normalized = re.sub(r"\s+", " ", body.strip().lower())
    # Common authoring aliases and frequent typos. Gemini prefers verb forms;
    # Fish accepts free-form natural-language tags.
    aliases = {
        "laught": "laugh",
        "laughter": "laugh",
        "laughing": "laugh",
        "giggle": "giggle",
        "giggles": "giggle",
        "giggling": "giggle",
        "chuckles": "chuckle",
        "chuckling": "chuckle",
        "whisper": "whisper",
        "whispers": "whisper",
        "whispering": "whisper",
        "sigh": "sigh",
        "sighs": "sigh",
        "pause": "pause",
        "short pause": "short pause",
    }
    base = aliases.get(normalized, normalized)
    if provider == "gemini":
        gemini_aliases = {
            "laugh": "laughs",
            "giggle": "giggles",
            "whisper": "whispers",
            "sigh": "sighs",
            "pause": "short pause",
        }
        return gemini_aliases.get(base, base)
    if provider == "fish":
        fish_aliases = {
            "giggle": "chuckle",
            "short pause": "short pause",
        }
        return fish_aliases.get(base, base)
    return base


def normalize_decorators(text: str, provider: str) -> str:
    """Translate generic `[tag]`/`<tag>` markup into provider-safe text.

    Gemini and Fish S2 both document bracketed natural-language audio controls.
    VibeVoice does not document a stable inline decorator syntax, so tag-like
    markup is removed instead of risking it being spoken literally.
    """
    if provider not in {"gemini", "fish", "vibevoice"}:
        raise ValueError(f"Unknown TTS provider for decorator normalization: {provider}")

    def replace(match: re.Match[str]) -> str:
        body = match.group("body")
        if provider == "vibevoice":
            return ""
        return f"[{_canonical_tag(body, provider)}]"

    result = _DECORATOR_RE.sub(replace, text)
    result = re.sub(r"[ \t]{2,}", " ", result)
    result = re.sub(r"[ \t]+\n", "\n", result)
    return result.strip()


def remap_speaker_ids(text: str) -> tuple[str, dict[int, int]]:
    """Map arbitrary speaker IDs to contiguous 0..N-1 IDs for local models."""
    speakers = detect_speakers(text)
    mapping = {speaker.speaker_id: index for index, speaker in enumerate(speakers)}
    if not mapping:
        return text, {}

    lines: list[str] = []
    for raw_line in text.splitlines():
        match = _SPEAKER_LINE_RE.match(raw_line)
        if not match:
            lines.append(raw_line)
            continue
        old_id = int(match.group(1))
        lines.append(f"Speaker {mapping[old_id]}: {match.group(2).strip()}")
    return "\n".join(lines), mapping


def prepare_text_for_provider(text: str, provider: str) -> tuple[str, dict[int, int]]:
    """Strip casting notes, normalize tags, and convert speaker syntax per provider."""
    cleaned = normalize_decorators(strip_speaker_metadata(text), provider)
    if provider == "gemini":
        return cleaned, {speaker.speaker_id: speaker.speaker_id for speaker in detect_speakers(cleaned)}

    remapped, mapping = remap_speaker_ids(cleaned)
    if provider == "fish":
        lines: list[str] = []
        found_turn = False
        for raw_line in remapped.splitlines():
            match = _SPEAKER_LINE_RE.match(raw_line)
            if match:
                found_turn = True
                lines.append(f"<|speaker:{int(match.group(1))}|>{match.group(2).strip()}")
            elif raw_line.strip():
                lines.append(raw_line.strip())
        if not found_turn and lines:
            lines[0] = f"<|speaker:0|>{lines[0]}"
        return "\n".join(lines).strip(), mapping

    # VibeVoice's processor natively parses `Speaker N: text` lines.
    if not detect_speakers(remapped) and remapped.strip():
        remapped = "\n".join(
            f"Speaker 0: {line.strip()}" for line in remapped.splitlines() if line.strip()
        )
    return remapped.strip(), mapping


def provider_speaker_limit(provider: str) -> int:
    return {"gemini": 2, "vibevoice": 4, "fish": 5}[provider]


def validate_speaker_count(text: str, provider: str) -> list[SpeakerInfo]:
    speakers = detect_speakers(text)
    count = len(speakers) or 1
    limit = provider_speaker_limit(provider)
    if count > limit:
        raise ValueError(
            f"{provider} supports at most {limit} speakers in this pipeline, but the story contains {count}."
        )
    return speakers
