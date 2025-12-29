#!/usr/bin/env python3
"""
Gemini Text-to-Speech Script
A command-line script that converts text files to audio using Google's Gemini TTS API.

The script automatically:
- Looks for input text files in the 'input' folder
- Splits text on "-------------" separator lines
- Generates audio for each segment separately
- Combines all segments into a single audio file
- Saves output audio files to the 'output' folder  
- Uses the same filename as input but with .wav extension

Usage:
    python gemini_tts.py --text_file input.txt
    python gemini_tts.py --text_file input.txt --voice Kore
"""

import os
import sys
import argparse
import wave
import base64
import tempfile
from google import genai
from google.genai import types

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Using system environment variables only.")


def setup_gemini_client(api_key=None):
    """Initialize the Gemini client with API key."""
    if not api_key:
        # Try to get API key from environment variable
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError(
                "Google API key not found! Please:\n"
                "1. Copy 'env_template.txt' to '.env'\n"
                "2. Edit .env and add your API key\n"
                "3. Or set GOOGLE_API_KEY environment variable"
            )
    
    print("Initializing Gemini TTS client...")
    client = genai.Client(api_key=api_key)
    return client


def write_wave_file(filename, pcm_data, channels=1, rate=24000, sample_width=2):
    """Write PCM data to a WAV file."""
    print(f"\nWriting audio file with parameters:")
    print(f"Channels: {channels}")
    print(f"Sample rate: {rate}")
    print(f"Sample width: {sample_width}")
    print(f"Data length: {len(pcm_data)} bytes")

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm_data)


def split_text_on_separator(text, separator="-------------"):
    """Split text into segments based on separator line."""
    segments = text.split(separator)
    # Clean up segments - remove empty lines and strip whitespace
    cleaned_segments = []
    for segment in segments:
        segment = segment.strip()
        if segment:  # Only add non-empty segments
            cleaned_segments.append(segment)
    
    print(f"Text split into {len(cleaned_segments)} segments")
    return cleaned_segments


def combine_audio_files(audio_file_paths, output_path, sample_rate=24000, channels=1, sample_width=2):
    """Combine multiple WAV files into a single WAV file."""
    print(f"Combining {len(audio_file_paths)} audio files...")
    
    combined_audio_data = b""
    
    for i, audio_path in enumerate(audio_file_paths):
        print(f"Reading segment {i+1}: {audio_path}")
        with wave.open(audio_path, 'rb') as wf:
            # Verify audio parameters match
            if wf.getnchannels() != channels:
                print(f"Warning: Channel mismatch in {audio_path}")
            if wf.getframerate() != sample_rate:
                print(f"Warning: Sample rate mismatch in {audio_path}")
            if wf.getsampwidth() != sample_width:
                print(f"Warning: Sample width mismatch in {audio_path}")
            
            # Read audio data and append
            audio_data = wf.readframes(wf.getnframes())
            combined_audio_data += audio_data
    
    # Write combined audio
    print(f"Writing combined audio to: {output_path}")
    write_wave_file(output_path, combined_audio_data, channels, sample_rate, sample_width)
    print("✅ Audio files combined successfully!")


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


def generate_tts_audio(client, text, voice_name="Kore"):
    """Generate TTS audio using Gemini API."""
    print(f"Generating speech with voice: {voice_name}")
    print(f"Text length: {len(text)} characters")
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["audio"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice_name,
                        )
                    )
                ),
            )
        )
        
        # Extract audio data
        audio_data = response.candidates[0].content.parts[0].inline_data.data
        print("✅ Audio generated successfully!")
        
        return audio_data
        
    except Exception as e:
        print(f"❌ Error generating audio: {str(e)}")
        raise


def save_audio(audio_data, output_path, sample_rate=24000):
    """Save audio data to WAV file."""
    print(f"Saving audio to: {output_path}")
    write_wave_file(output_path, audio_data, rate=sample_rate)
    print(f"✅ Audio saved successfully!")


def main():
    parser = argparse.ArgumentParser(description="Convert text file to speech using Gemini TTS")
    parser.add_argument("--text_file", required=True, help="Name of input text file (will be looked for in 'input' folder)")
    parser.add_argument("--voice", default="Kore", help="Voice to use for speech generation (default: Kore)")
    parser.add_argument("--preprocess", action="store_true", help="Apply text preprocessing")
    parser.add_argument("--api_key", help="Google API key (optional, can use GOOGLE_API_KEY env var)")
    parser.add_argument("--high_quality", action="store_true", help="Use high quality settings (currently same as default for Gemini)")
    parser.add_argument("--no_split", action="store_true", help="Don't split text on separator lines, process as single file")
    
    args = parser.parse_args()
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Setup input and output paths relative to script location
    input_folder = os.path.join(script_dir, "input")
    output_folder = os.path.join(script_dir, "output")
    
    # Create folders if they don't exist
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    
    # Construct full input path - check multiple locations for backward compatibility
    input_path = os.path.join(input_folder, args.text_file)
    
    # If file doesn't exist in script's input folder, check the current working directory's input folder
    if not os.path.exists(input_path):
        fallback_input_path = os.path.join("input", args.text_file)
        if os.path.exists(fallback_input_path):
            input_path = fallback_input_path
            print(f"Using input file from current directory: {input_path}")
        else:
            print(f"Error: Text file not found!")
            print(f"Checked: {input_path}")
            print(f"Checked: {fallback_input_path}")
            print(f"Please place your text file in one of these locations.")
            sys.exit(1)
    
    # Generate output filename (same as input but with .wav extension)
    base_name = os.path.splitext(args.text_file)[0]
    output_filename = f"{base_name}.wav"
    output_path = os.path.join(output_folder, output_filename)
    
    # Validate input file exists (redundant check but keeping for clarity)
    if not os.path.exists(input_path):
        print(f"Error: Text file '{input_path}' not found!")
        print(f"Please place your text file in the '{input_folder}' folder.")
        sys.exit(1)
    
    try:
        # Setup Gemini client
        client = setup_gemini_client(args.api_key)
        
        # Read input text
        print(f"Reading text from {input_path}...")
        text = read_text_file(input_path)
        
        # Split text into segments (unless --no_split is specified)
        if args.no_split:
            segments = [text]
            print("Processing as single segment (--no_split specified)")
        else:
            segments = split_text_on_separator(text)
        
        # If only one segment, process normally
        if len(segments) == 1:
            print("Single segment detected, processing normally...")
            segment_text = segments[0]
            
            if args.preprocess:
                print("Preprocessing text...")
                segment_text = preprocess_text(segment_text)
            
            print(f"Text to convert ({len(segment_text)} characters):")
            print(f"'{segment_text[:100]}{'...' if len(segment_text) > 100 else ''}'")
            
            # Generate speech
            audio_data = generate_tts_audio(client, segment_text, args.voice)
            
            # Save output
            save_audio(audio_data, output_path)
            
        else:
            # Multiple segments - generate audio for each and combine
            print(f"Processing {len(segments)} segments...")
            
            temp_audio_files = []
            temp_dir = tempfile.mkdtemp()
            
            try:
                # Generate audio for each segment
                for i, segment in enumerate(segments):
                    print(f"\n--- Processing segment {i+1}/{len(segments)} ---")
                    
                    segment_text = segment
                    if args.preprocess:
                        print(f"Preprocessing segment {i+1}...")
                        segment_text = preprocess_text(segment_text)
                    
                    print(f"Segment {i+1} text ({len(segment_text)} characters):")
                    print(f"'{segment_text[:100]}{'...' if len(segment_text) > 100 else ''}'")
                    
                    # Generate speech for this segment
                    audio_data = generate_tts_audio(client, segment_text, args.voice)
                    
                    # Save temporary audio file
                    temp_audio_path = os.path.join(temp_dir, f"segment_{i+1}.wav")
                    save_audio(audio_data, temp_audio_path)
                    temp_audio_files.append(temp_audio_path)
                
                # Combine all segments
                print(f"\n--- Combining {len(temp_audio_files)} segments ---")
                combine_audio_files(temp_audio_files, output_path)
                
            finally:
                # Clean up temporary files
                print("Cleaning up temporary files...")
                for temp_file in temp_audio_files:
                    try:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    except Exception as e:
                        print(f"Warning: Could not remove temp file {temp_file}: {e}")
                
                try:
                    os.rmdir(temp_dir)
                except Exception as e:
                    print(f"Warning: Could not remove temp directory {temp_dir}: {e}")
        
        print("\n✅ Text-to-speech conversion completed successfully!")
        print(f"📁 Output saved to: {output_path}")
        print(f"🎵 Voice used: {args.voice}")
        if not args.no_split and len(segments) > 1:
            print(f"📊 Segments processed: {len(segments)}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()