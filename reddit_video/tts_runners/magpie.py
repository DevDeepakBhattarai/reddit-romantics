from __future__ import annotations

import argparse
from pathlib import Path

SPEAKERS = {"Aria": 0, "Jason": 1, "John": 2, "Leo": 3, "Sofia": 4}


def _sample_rate(model) -> int:
    # Magpie decodes through its codec's output sample rate. Prefer that over
    # the codec input/training sample rate when both are exposed.
    for attribute in ("output_sample_rate", "sample_rate"):
        for owner in (model, getattr(model, "cfg", None), getattr(model, "_cfg", None)):
            if owner is None:
                continue
            value = getattr(owner, attribute, None)
            if value:
                return int(value)
    return 22050


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="nvidia/magpie_tts_multilingual_357m")
    parser.add_argument("--speaker", choices=list(SPEAKERS), default="John")
    parser.add_argument("--language", default="en")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--use-cfg", action="store_true")
    parser.add_argument("--cfg-scale", type=float, default=2.5)
    args = parser.parse_args()

    import soundfile as sf
    import torch
    from huggingface_hub import hf_hub_download
    from nemo.collections.tts.models import MagpieTTSModel

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for Magpie, but CUDA is not available in this runtime.")

    text = Path(args.text_file).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("TTS input text is empty.")

    # NVIDIA's own model-support test restores the .nemo checkpoint directly
    # from Hugging Face, then calls do_tts(). This avoids relying on
    # list_available_models()/from_pretrained registration for Magpie.
    model_filename = f"{args.model.rsplit('/', 1)[-1]}.nemo"
    checkpoint = hf_hub_download(repo_id=args.model, filename=model_filename)
    model = MagpieTTSModel.restore_from(checkpoint, map_location="cpu").to(device)
    model.eval()

    # Current NeMo keeps cfg_scale on inference_parameters; do_tts() takes only
    # the boolean CFG switch plus transcript/language/speaker arguments.
    if hasattr(model, "inference_parameters"):
        model.inference_parameters.cfg_scale = float(args.cfg_scale)

    with torch.inference_mode():
        audio, audio_len = model.do_tts(
            transcript=text,
            language=args.language,
            apply_TN=True,
            use_cfg=bool(args.use_cfg),
            speaker_index=SPEAKERS[args.speaker],
        )

    waveform = audio.detach().float().cpu().squeeze().numpy()
    if hasattr(audio_len, "item"):
        length = int(audio_len.item())
        if length > 0 and length <= waveform.shape[-1]:
            waveform = waveform[:length]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), waveform, _sample_rate(model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
