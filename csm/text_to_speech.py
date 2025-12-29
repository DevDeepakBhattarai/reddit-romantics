#!/usr/bin/env python3
"""
Text-to-Speech Script
A command-line script that converts text files to audio using CSM TTS model.

The script automatically:
- Looks for input text files in the 'input' folder
- Saves output audio files to the 'output' folder
- Uses the same filename as input but with .wav extension

Usage:
    python text_to_speech.py --text_file input.txt
    python text_to_speech.py --text_file input.txt --reference_audio voice.wav --reference_text "transcript"
"""

import os
import sys
import argparse
import torch
import torchaudio
import subprocess
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from generator import Segment, Generator
from models import Model, ModelArgs
from tqdm import tqdm
import fix_compile

def get_device():
    """Automatically select the best available device."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def load_csm_model(device="cuda"):
    """Load the CSM TTS model."""
    print("Loading CSM TTS model...")
    
    # Path to model file
    model_path = os.path.join("models", "model.safetensors")
    if not os.path.exists(model_path):
        print("Local model not found. Downloading from Hugging Face...")
        model_path = hf_hub_download(repo_id="sesame/csm-1b", filename="model.safetensors")
    
    # Initialize model
    model_args = ModelArgs(
        backbone_flavor="llama-1B",
        decoder_flavor="llama-100M",
        text_vocab_size=128256,
        audio_vocab_size=2051,
        audio_num_codebooks=32,
    )
    
    model = Model(model_args).to(device=device, dtype=torch.float32)
    state_dict = load_file(model_path, device=device)
    model.load_state_dict(state_dict)
    
    print(f"Model loaded successfully on device: {device}")
    return Generator(model)


def convert_mp3_to_wav(mp3_path, wav_path):
    """Convert MP3 to WAV using ffmpeg."""
    print(f"Converting {mp3_path} to mono WAV...")
    subprocess.call(['ffmpeg', '-i', mp3_path, '-ac', '1', wav_path, '-y'])  # -y to overwrite
    print("Conversion complete.")


def load_audio(audio_path, target_sample_rate):
    """Load and preprocess audio file."""
    # Convert MP3 to WAV if needed
    if audio_path.endswith('.mp3'):
        wav_path = audio_path.replace('.mp3', '.wav')
        convert_mp3_to_wav(audio_path, wav_path)
        audio_path = wav_path
    
    print(f"Loading audio from {audio_path}...")
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if audio_tensor.shape[0] > 1:
        audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)
        print("Converted audio to mono")
    
    # Resample to target sample rate
    if sample_rate != target_sample_rate:
        print(f"Resampling from {sample_rate}Hz to {target_sample_rate}Hz")
        audio_tensor = torchaudio.functional.resample(
            audio_tensor.squeeze(0), 
            orig_freq=sample_rate, 
            new_freq=target_sample_rate
        )
    else:
        audio_tensor = audio_tensor.squeeze(0)
    
    return audio_tensor


def preprocess_text(text):
    """Clean and preprocess text for better TTS output."""
    # Remove punctuation at start of each line
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        # Remove punctuation from start of line
        while line and line[0] in ',-._/@#*%$().':
            line = line[1:]
        line = line.strip()
        if line:  # Only add non-empty lines
            cleaned_lines.append(line)
    
    # Join lines and replace specific punctuation
    text = ', '.join(cleaned_lines)
    # Replace semicolons and colons with comma space
    text = text.replace(';', ', ').replace(':', ', ')
    text = text.replace("'", "").replace("'", "")
    
    return text


def read_text_file(file_path):
    """Read text from file with various encoding attempts."""
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read().strip()
        except UnicodeDecodeError:
            continue
    
    raise ValueError(f"Could not read file {file_path} with any of the attempted encodings")


def generate_speech(generator, text, reference_audio=None, reference_text=None, max_length_ms=50000, temperature=0.9, topk=50):
    """Generate speech from text with optional voice cloning."""
    context = []
    
    if reference_audio and reference_text:
        print("Loading reference voice for cloning...")
        ref_audio_tensor = load_audio(reference_audio, generator.sample_rate)
        context_segment = Segment(
            text=reference_text,
            speaker=0,
            audio=ref_audio_tensor
        )
        context.append(context_segment)
    
    print("Generating speech...")
    generated_audio = generator.generate(
        text=text,
        speaker=0,
        context=context,
        max_audio_length_ms=max_length_ms,
        temperature=temperature,
        topk=topk
    )
    
    return generated_audio


def save_audio(audio, filename, sample_rate):
    """Save generated audio to file."""
    print(f"Saving audio to {filename}...")
    torchaudio.save(filename, audio.unsqueeze(0).cpu(), sample_rate)
    print(f"Audio saved successfully!")


def main():
    parser = argparse.ArgumentParser(description="Convert text file to speech using CSM TTS")
    parser.add_argument("--text_file", required=True, help="Name of input text file (will be looked for in 'input' folder)")
    parser.add_argument("--reference_audio", help="Path to reference audio file for voice cloning")
    parser.add_argument("--reference_text", help="Transcript of reference audio")
    parser.add_argument("--max_length", type=int, default=50000, help="Maximum audio length in milliseconds")
    parser.add_argument("--preprocess", action="store_true", help="Apply text preprocessing")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto", 
                       help="Device to use for inference")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature (lower = more deterministic, higher = more creative). Range: 0.1-1.5")
    parser.add_argument("--topk", type=int, default=50, help="Top-k sampling (lower = more focused, higher = more diverse). Range: 1-100")
    parser.add_argument("--high_quality", action="store_true", help="Use high quality settings (lower temperature, more focused sampling)")
    parser.add_argument("--creative", action="store_true", help="Use creative settings (higher temperature, more diverse sampling)")
    
    args = parser.parse_args()
    
    # Setup input and output paths
    input_folder = "input"
    output_folder = "output"
    
    # Create folders if they don't exist
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    
    # Construct full input path
    input_path = os.path.join(input_folder, args.text_file)
    
    # Generate output filename (same as input but with .wav extension)
    base_name = os.path.splitext(args.text_file)[0]
    output_filename = f"{base_name}.wav"
    output_path = os.path.join(output_folder, output_filename)
    
    # Validate arguments
    if not os.path.exists(input_path):
        print(f"Error: Text file '{input_path}' not found!")
        print(f"Please place your text file in the '{input_folder}' folder.")
        sys.exit(1)
    
    if args.reference_audio and not args.reference_text:
        print("Error: --reference_text is required when using --reference_audio")
        sys.exit(1)
    
    if args.reference_audio and not os.path.exists(args.reference_audio):
        print(f"Error: Reference audio file '{args.reference_audio}' not found!")
        sys.exit(1)
    
    # Select device
    device = get_device() if args.device == "auto" else args.device
    print(f"Using device: {device}")
    
    # Configure generation parameters based on quality settings
    temperature = args.temperature
    topk = args.topk
    
    if args.high_quality:
        temperature = 0.5  # More deterministic, cleaner output
        topk = 15         # More focused sampling
        print("🎯 Using HIGH QUALITY settings: temperature=0.5, top-k=15 (slower but cleaner)")
    elif args.creative:
        temperature = 1.1  # More creative, varied output
        topk = 50         # More diverse sampling
        print("🎨 Using CREATIVE settings: temperature=1.1, top-k=50 (more expressive)")
    else:
        print(f"🔧 Using settings: temperature={temperature}, top-k={topk}")
    
    try:
        # Load model
        generator = load_csm_model(device)
        
        # Read input text
        print(f"Reading text from {input_path}...")
        text = read_text_file(input_path)
        
        if args.preprocess:
            print("Preprocessing text...")
            text = preprocess_text(text)
        
        print(f"Text to convert ({len(text)} characters):")
        print(f"'{text[:100]}{'...' if len(text) > 100 else ''}'")
        
        # Generate speech
        audio = generate_speech(
            generator, 
            text, 
            args.reference_audio, 
            args.reference_text, 
            args.max_length,
            temperature,
            topk
        )
        
        # Save output
        save_audio(audio, output_path, generator.sample_rate)
        
        print("\n✅ Text-to-speech conversion completed successfully!")
        print(f"📁 Output saved to: {output_path}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 