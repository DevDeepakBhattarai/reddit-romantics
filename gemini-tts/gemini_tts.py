#!/usr/bin/env python3
"""Gemini text-to-speech generation for Reddit Romantics.

Long stories are automatically split into narration-sized semantic chunks. The
splitter prefers paragraph boundaries, then sentences, then clauses, and only
falls back to word boundaries for pathological long sentences. It never cuts a
word in the middle.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import os
import re
import sys
import tempfile
import time
import warnings
import wave
from pathlib import Path

from google import genai

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print(
        "Warning: python-dotenv not installed. Using system environment variables only."
    )


DEFAULT_MODEL = "gemini-3.1-flash-tts-preview"
DEFAULT_VOICE = "Kore"
DEFAULT_CHUNK_SECONDS = 180
DEFAULT_ESTIMATED_WPM = 140
DEFAULT_MAX_RETRIES = 3
SAMPLE_RATE = 24000
CHANNELS = 1
SAMPLE_WIDTH = 2
_SENTENCE_RE = re.compile(r".+?(?:[.!?]+(?:[\"'”’\)\]]+)?(?=\s+|$)|$)", re.DOTALL)
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[;:,—–])\s+")
_EXPLICIT_SEPARATOR_RE = re.compile(r"(?m)^\s*-{5,}\s*$")


def setup_gemini_client(api_key: str | None = None):
    """Initialize the Gemini client with an API key."""
    api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Google API key not found. Set GOOGLE_API_KEY (or GEMINI_API_KEY) in .env or the environment."
        )

    print("Initializing Gemini TTS client...")
    return genai.Client(api_key=api_key)


def write_wave_file(
    filename: str | os.PathLike[str],
    pcm_data: bytes,
    channels: int = CHANNELS,
    rate: int = SAMPLE_RATE,
    sample_width: int = SAMPLE_WIDTH,
) -> None:
    """Write raw PCM data to a WAV file."""
    with wave.open(str(filename), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm_data)


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _fits_chunk(text: str, max_chars: int, max_words: int) -> bool:
    return len(text) <= max_chars and _word_count(text) <= max_words


def _split_sentences(text: str) -> list[str]:
    """Split prose at sentence endings while keeping punctuation attached."""
    sentences = [match.group(0).strip() for match in _SENTENCE_RE.finditer(text)]
    return [sentence for sentence in sentences if sentence]


def _split_clauses(text: str) -> list[str]:
    return [piece.strip() for piece in _CLAUSE_SPLIT_RE.split(text) if piece.strip()]


def _merge_pieces(
    pieces: list[str], max_chars: int, max_words: int, separator: str = " "
) -> list[str]:
    """Greedily merge already-safe pieces without crossing the chunk budget."""
    merged: list[str] = []
    current = ""

    for piece in pieces:
        candidate = piece if not current else f"{current}{separator}{piece}"
        if current and not _fits_chunk(candidate, max_chars, max_words):
            merged.append(current.strip())
            current = piece
        else:
            current = candidate

    if current.strip():
        merged.append(current.strip())
    return merged


def _split_oversized_text(text: str, max_chars: int, max_words: int) -> list[str]:
    """Recursively split a long paragraph at natural language boundaries.

    Boundary priority mirrors a recursive character splitter, but is deliberately
    narration-aware: sentence -> clause -> word. A single huge token is kept
    intact instead of being cut mid-word.
    """
    text = text.strip()
    if not text or _fits_chunk(text, max_chars, max_words):
        return [text] if text else []

    for splitter in (_split_sentences, _split_clauses):
        pieces = splitter(text)
        if len(pieces) <= 1:
            continue

        safe_pieces: list[str] = []
        for piece in pieces:
            if _fits_chunk(piece, max_chars, max_words):
                safe_pieces.append(piece)
            else:
                safe_pieces.extend(_split_oversized_text(piece, max_chars, max_words))
        return _merge_pieces(safe_pieces, max_chars, max_words)

    words = text.split()
    if len(words) <= 1:
        # Do not introduce an incoherent mid-word split just to satisfy a numeric limit.
        return [text]
    return _merge_pieces(words, max_chars, max_words)


def _chunk_section(section: str, max_chars: int, max_words: int) -> list[str]:
    """Chunk one hard section, preferring complete paragraphs whenever possible."""
    raw_paragraphs = re.split(r"\n\s*\n+", section.strip())
    units: list[tuple[str, int]] = []

    for paragraph_index, raw_paragraph in enumerate(raw_paragraphs):
        # Newlines inside a paragraph are usually line wrapping, not semantic breaks.
        paragraph = re.sub(r"[ \t]*\n[ \t]*", " ", raw_paragraph).strip()
        paragraph = re.sub(r"[ \t]+", " ", paragraph)
        if not paragraph:
            continue

        pieces = _split_oversized_text(paragraph, max_chars, max_words)
        units.extend((piece, paragraph_index) for piece in pieces if piece)

    chunks: list[str] = []
    current = ""
    previous_paragraph: int | None = None

    for unit, paragraph_index in units:
        if not current:
            current = unit
            previous_paragraph = paragraph_index
            continue

        separator = "\n\n" if paragraph_index != previous_paragraph else " "
        candidate = f"{current}{separator}{unit}"
        if _fits_chunk(candidate, max_chars, max_words):
            current = candidate
        else:
            chunks.append(current.strip())
            current = unit
        previous_paragraph = paragraph_index

    if current.strip():
        chunks.append(current.strip())
    return chunks


def split_text_semantically(
    text: str,
    target_seconds: int = DEFAULT_CHUNK_SECONDS,
    estimated_wpm: int = DEFAULT_ESTIMATED_WPM,
    respect_explicit_separator: bool = True,
) -> list[str]:
    """Split text into roughly ``target_seconds`` of speech per Gemini request.

    The duration target is converted to a conservative word/character budget.
    Explicit dashed separator lines are treated as hard boundaries when enabled.
    Automatic semantic chunking is always applied to long sections.
    """
    text = text.strip()
    if not text:
        return []
    if target_seconds <= 0:
        raise ValueError("target_seconds must be greater than zero")
    if estimated_wpm <= 0:
        raise ValueError("estimated_wpm must be greater than zero")

    max_words = max(40, round(target_seconds * estimated_wpm / 60))
    # English prose averages roughly 5 characters/word plus whitespace. Six keeps
    # the character budget aligned with the speech-duration word budget.
    max_chars = max(800, max_words * 6)

    if respect_explicit_separator:
        sections = [
            section.strip()
            for section in _EXPLICIT_SEPARATOR_RE.split(text)
            if section.strip()
        ]
    else:
        sections = [_EXPLICIT_SEPARATOR_RE.sub("\n\n", text)]

    chunks: list[str] = []
    for section in sections:
        chunks.extend(_chunk_section(section, max_chars=max_chars, max_words=max_words))

    if not chunks:
        raise ValueError("No narratable text remained after chunking.")
    return chunks


def preprocess_text(text: str) -> str:
    """Clean narration text without destroying paragraph boundaries."""
    cleaned_lines: list[str] = []
    previous_blank = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if cleaned_lines and not previous_blank:
                cleaned_lines.append("")
            previous_blank = True
            continue

        while line and line[0] in ",-._/@#*%$().":
            line = line[1:].lstrip()
        line = line.replace(";", ", ").replace(":", ", ")
        line = line.replace("‘", "'").replace("’", "'")
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line:
            cleaned_lines.append(line)
            previous_blank = False

    return "\n".join(cleaned_lines).strip()


def read_text_file(file_path: str | os.PathLike[str]) -> str:
    """Read text from a file using common encodings."""
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return Path(file_path).read_text(encoding=encoding).strip()
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not read file {file_path} with any supported encoding")


def _build_tts_prompt(text: str) -> str:
    """Give Gemini explicit recitation instructions to avoid added sounds/text."""
    return (
        "Synthesize the transcript below as natural, continuous story narration. "
        "This transcript may be one chunk of a longer story, so keep the narrator's "
        "voice, pacing, volume, and delivery consistent. Do not add an introduction, "
        "outro, commentary, music, sound effects, or any words not present in the transcript. "
        "Speak only the text under TRANSCRIPT.\n\n"
        "TRANSCRIPT:\n"
        f"{text}"
    )


def _is_retryable_error(exc: Exception) -> bool:
    message = str(exc).lower()
    markers = (
        "429",
        "500",
        "502",
        "503",
        "504",
        "internal",
        "temporar",
        "resource exhausted",
        "deadline",
        "unavailable",
        "invalid_request",
        "invalid argument",
    )
    return any(marker in message for marker in markers)


def _decode_output_audio(interaction) -> bytes:
    output_audio = getattr(interaction, "output_audio", None)
    data = getattr(output_audio, "data", None) if output_audio is not None else None
    if not data:
        raise RuntimeError("Gemini returned no audio data.")

    if isinstance(data, str):
        return base64.b64decode(data)
    if isinstance(data, bytes):
        # Current Interactions responses expose base64 text, but tolerate raw bytes
        # in case the SDK normalizes that representation in a future release.
        try:
            return base64.b64decode(data, validate=True)
        except (ValueError, binascii.Error):
            return data
    raise TypeError(f"Unsupported Gemini audio payload type: {type(data).__name__}")


def generate_tts_audio(
    client,
    text: str,
    voice_name: str = DEFAULT_VOICE,
    model_name: str = DEFAULT_MODEL,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> bytes:
    """Generate one semantic chunk with Gemini 3.1 TTS via Interactions API."""
    prompt = _build_tts_prompt(text)
    print(
        f"Generating Gemini speech: model={model_name}, voice={voice_name}, words={_word_count(text)}"
    )

    for attempt in range(1, max_retries + 1):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Interactions usage is experimental.*",
                    category=UserWarning,
                )
                interaction = client.interactions.create(
                    model=model_name,
                    input=prompt,
                    response_format={"type": "audio"},
                    generation_config={"speech_config": [{"voice": voice_name}]},
                )
            audio_data = _decode_output_audio(interaction)
            if not audio_data:
                raise RuntimeError("Gemini returned an empty audio payload.")
            return audio_data
        except Exception as exc:
            if attempt >= max_retries or not _is_retryable_error(exc):
                raise
            delay = min(8, 2 ** (attempt - 1))
            print(
                f"Transient Gemini TTS error on attempt {attempt}/{max_retries}: {exc}. "
                f"Retrying in {delay}s..."
            )
            time.sleep(delay)

    raise RuntimeError("Gemini TTS generation failed unexpectedly.")


def save_audio(audio_data: bytes, output_path: str | os.PathLike[str]) -> None:
    write_wave_file(output_path, audio_data)


def _pcm_duration_seconds(audio_data: bytes) -> float:
    bytes_per_second = SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH
    return len(audio_data) / bytes_per_second


def combine_audio_files(
    audio_file_paths: list[str | os.PathLike[str]], output_path: str | os.PathLike[str]
) -> None:
    """Concatenate compatible PCM WAV chunks without re-encoding."""
    if not audio_file_paths:
        raise ValueError("No audio chunks were supplied for concatenation.")

    print(f"Combining {len(audio_file_paths)} Gemini audio chunks...")
    with wave.open(str(output_path), "wb") as output:
        output.setnchannels(CHANNELS)
        output.setsampwidth(SAMPLE_WIDTH)
        output.setframerate(SAMPLE_RATE)

        for index, audio_path in enumerate(audio_file_paths, start=1):
            with wave.open(str(audio_path), "rb") as source:
                params = (
                    source.getnchannels(),
                    source.getframerate(),
                    source.getsampwidth(),
                )
                expected = (CHANNELS, SAMPLE_RATE, SAMPLE_WIDTH)
                if params != expected:
                    raise RuntimeError(
                        f"Gemini chunk {index} has incompatible WAV parameters {params}; expected {expected}."
                    )
                output.writeframes(source.readframes(source.getnframes()))


def _resolve_input_path(script_dir: Path, text_file: str) -> Path:
    primary = script_dir / "input" / text_file
    if primary.exists():
        return primary

    fallback = Path("input") / text_file
    if fallback.exists():
        print(f"Using input file from current directory: {fallback}")
        return fallback

    raise FileNotFoundError(f"Text file not found. Checked {primary} and {fallback}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a text file to speech using Gemini 3.1 Flash TTS"
    )
    parser.add_argument("--text_file", required=True, help="Input .txt filename")
    parser.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        help=f"Gemini voice (default: {DEFAULT_VOICE})",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("GEMINI_TTS_MODEL", DEFAULT_MODEL),
        help=f"Gemini TTS model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--preprocess", action="store_true", help="Apply light narration text cleanup"
    )
    parser.add_argument(
        "--api_key", help="Google API key; normally read from GOOGLE_API_KEY"
    )
    parser.add_argument(
        "--high_quality",
        action="store_true",
        help="Compatibility flag; Gemini uses its native output quality",
    )
    parser.add_argument(
        "--no_split",
        action="store_true",
        help=(
            "Ignore explicit dashed separator lines. Automatic semantic ~3-minute chunking remains enabled "
            "for long text."
        ),
    )
    parser.add_argument(
        "--chunk-seconds",
        "--chunk_seconds",
        dest="chunk_seconds",
        type=int,
        default=DEFAULT_CHUNK_SECONDS,
        help=f"Target maximum narration duration per request (default: {DEFAULT_CHUNK_SECONDS}s)",
    )
    parser.add_argument(
        "--estimated-wpm",
        "--estimated_wpm",
        dest="estimated_wpm",
        type=int,
        default=DEFAULT_ESTIMATED_WPM,
        help=f"Conservative speech-rate estimate used for chunk sizing (default: {DEFAULT_ESTIMATED_WPM})",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Retries for transient Gemini API failures (default: {DEFAULT_MAX_RETRIES})",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    input_path = _resolve_input_path(script_dir, args.text_file)
    output_folder = script_dir / "output"
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / f"{Path(args.text_file).stem}.wav"

    try:
        client = setup_gemini_client(args.api_key)
        text = read_text_file(input_path)
        chunks = split_text_semantically(
            text,
            target_seconds=args.chunk_seconds,
            estimated_wpm=args.estimated_wpm,
            respect_explicit_separator=not args.no_split,
        )
        if args.preprocess:
            chunks = [preprocess_text(chunk) for chunk in chunks]

        print(
            f"Semantic chunking produced {len(chunks)} request(s), targeting about "
            f"{args.chunk_seconds}s each at {args.estimated_wpm} WPM."
        )
        for index, chunk in enumerate(chunks, start=1):
            estimated_seconds = _word_count(chunk) / args.estimated_wpm * 60
            print(
                f"  Chunk {index}/{len(chunks)}: {_word_count(chunk)} words, {len(chunk)} chars, "
                f"~{estimated_seconds:.0f}s estimated"
            )

        if len(chunks) == 1:
            audio_data = generate_tts_audio(
                client,
                chunks[0],
                voice_name=args.voice,
                model_name=args.model,
                max_retries=args.max_retries,
            )
            print(
                f"Chunk 1 actual PCM duration: {_pcm_duration_seconds(audio_data):.1f}s"
            )
            save_audio(audio_data, output_path)
        else:
            with tempfile.TemporaryDirectory(prefix="gemini_tts_") as temp_dir_name:
                temp_dir = Path(temp_dir_name)
                temp_audio_files: list[Path] = []

                for index, chunk in enumerate(chunks, start=1):
                    print(f"\n--- Gemini chunk {index}/{len(chunks)} ---")
                    audio_data = generate_tts_audio(
                        client,
                        chunk,
                        voice_name=args.voice,
                        model_name=args.model,
                        max_retries=args.max_retries,
                    )
                    print(
                        f"Chunk {index} actual PCM duration: {_pcm_duration_seconds(audio_data):.1f}s"
                    )
                    temp_path = temp_dir / f"chunk_{index:04d}.wav"
                    save_audio(audio_data, temp_path)
                    temp_audio_files.append(temp_path)

                combine_audio_files(temp_audio_files, output_path)

        print("\nGemini TTS conversion completed successfully.")
        print(f"Output: {output_path}")
        print(f"Model: {args.model}")
        print(f"Voice: {args.voice}")
        print(f"Semantic chunks: {len(chunks)}")
    except Exception as exc:  # noqa: BLE001 - CLI boundary reports API/filesystem failures
        print(f"\nError: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
