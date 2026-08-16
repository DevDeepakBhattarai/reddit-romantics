from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", choices=["turbo", "nano", "original"], default="turbo")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--reference-audio")
    args = parser.parse_args()

    import numpy as np
    import torch
    import torchaudio as ta

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    text = Path(args.text_file).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("TTS input text is empty.")

    if args.variant in {"turbo", "nano"}:
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        model = ChatterboxTurboTTS.from_pretrained(device=device, nano=args.variant == "nano")
    else:
        from chatterbox.tts import ChatterboxTTS

        model = ChatterboxTTS.from_pretrained(device=device)

    # Turbo loudness normalization can promote the NumPy reference waveform to
    # float64 on Windows. S3Tokenizer expects float32 audio, so preserve the
    # original speech dtype after normalization rather than disabling it.
    if args.variant in {"turbo", "nano"} and hasattr(model, "norm_loudness"):
        original_norm_loudness = model.norm_loudness

        def norm_loudness_float32(*norm_args, **norm_kwargs):
            return np.asarray(original_norm_loudness(*norm_args, **norm_kwargs), dtype=np.float32)

        model.norm_loudness = norm_loudness_float32

    kwargs = {}
    if args.reference_audio:
        kwargs["audio_prompt_path"] = args.reference_audio
    wav = model.generate(text, **kwargs)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ta.save(str(output), wav.detach().cpu(), model.sr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
