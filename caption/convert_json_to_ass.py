"""Backward-compatible WhisperX JSON -> themed ASS converter."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reddit_video.captions import CAPTION_THEMES, convert_whisperx_json_to_ass


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert WhisperX JSON to styled ASS captions")
    parser.add_argument("input", help="Input WhisperX JSON file")
    parser.add_argument("output", help="Output ASS file")
    parser.add_argument("--theme", choices=sorted(CAPTION_THEMES), default="classic_yellow")
    parser.add_argument("--max-words", type=int, default=0, help="0 uses the selected theme default")
    parser.add_argument("--pause", type=float, default=0.5, help="Pause threshold in seconds")
    parser.add_argument("--font-size", type=int, default=None, help="Optional legacy font-size override")
    parser.add_argument("--font", default=None, help="Optional legacy font-name override")
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1920)
    args = parser.parse_args()

    theme_name = args.theme
    if args.font_size is not None or args.font is not None:
        base = CAPTION_THEMES[theme_name]
        custom_name = "__legacy_override__"
        CAPTION_THEMES[custom_name] = replace(
            base,
            font=args.font or base.font,
            font_size=args.font_size or base.font_size,
        )
        theme_name = custom_name

    convert_whisperx_json_to_ass(
        args.input,
        args.output,
        theme_name=theme_name,
        max_words=args.max_words or None,
        pause_threshold=args.pause,
        width=args.width,
        height=args.height,
    )
    print(f"Caption file written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
