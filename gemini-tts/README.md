# Gemini TTS

This folder contains the existing Gemini text-to-speech backend used by the main Reddit Romantics pipeline.

## Recommended usage

For normal use, run Gemini through the repository UI or shared CLI rather than invoking this folder directly:

```powershell
# Browser UI
.\.venv\Scripts\python.exe ..\app.py

# CLI
.\.venv\Scripts\python.exe ..\main.py run --story-file ..\input\my_story.txt --tts gemini --gemini-voice Kore
```

The root pipeline handles audio paths, captions, background footage, and final rendering.

## Standalone usage

The backend can still be run on its own:

```powershell
python gemini_tts.py --text_file test.txt --voice Kore --preprocess --high_quality
```

Input files are read from `gemini-tts/input/` (with the repository `input/` folder retained as a fallback), and generated WAV files are written to `gemini-tts/output/`.

## Configuration

Set your API key in the repository root `.env`:

```dotenv
GOOGLE_API_KEY=your_key_here
```

The default model is the current Gemini Flash TTS model used by the pipeline:

```text
gemini-3.1-flash-tts-preview
```

You can now override it without editing Python:

```powershell
python gemini_tts.py --text_file test.txt --model YOUR_MODEL_NAME --voice Kore
```

or:

```dotenv
GEMINI_TTS_MODEL=YOUR_MODEL_NAME
```

The Gradio UI exposes the model name as an editable field as well.

## Voice selection

The main Gradio UI exposes all 30 currently documented prebuilt Gemini TTS voice names. The standalone script accepts any supported voice through `--voice`.

## Semantic long-story chunking

Long Gemini narration is always split automatically into chunks targeting about **180 seconds** of speech. The splitter is narration-aware and recursive: it keeps complete paragraphs together whenever possible, then falls back to sentence boundaries, then clauses, and finally word boundaries for unusually long sentences. It never intentionally cuts through a word.

A line containing:

```text
-------------
```

is still treated as an explicit hard chunk boundary for compatibility. `--no_split` now means "ignore those explicit separator lines"; it does **not** disable automatic safe chunking.

Change the target when needed:

```powershell
python gemini_tts.py --text_file test.txt --chunk-seconds 180
```

Generated PCM WAV chunks are concatenated without re-encoding. The Gemini call also uses a consistent narration instruction for every chunk and retries transient server failures.

VibeVoice uses a different path in the main pipeline and intentionally receives the complete story in one generation.

## Arguments

```text
--text_file       input text filename (required)
--voice           Gemini prebuilt voice name (default: Kore)
--model           Gemini TTS model name
--preprocess      apply the repository's text cleanup
--api_key         optional API key override
--high_quality    retained compatibility flag
--no_split        ignore explicit ------------- boundaries (semantic chunking stays enabled)
--chunk-seconds    target speech duration per semantic chunk (default: 180)
--estimated-wpm    conservative speech-rate estimate used for sizing (default: 140)
--max-retries      retries for transient Gemini API failures (default: 3)
```
