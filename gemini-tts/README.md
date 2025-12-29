# Gemini TTS Implementation

This folder contains the Google Gemini Text-to-Speech implementation that replaces the CSM TTS system in the automation workflow.

## Features

- **High-quality speech synthesis** using Google's Gemini 2.5 Flash TTS model
- **Single speaker setup** with multiple voice options
- **Seamless integration** with the existing automation pipeline
- **Same interface** as the previous CSM system for easy migration

## Setup

1. **Install Dependencies**

   ```bash
   pip install google-genai>=1.16.1 python-dotenv
   ```

   Or run the setup script:

   ```bash
   setup.bat
   ```

2. **Configure Google API Key**

   **Option A: Using .env file (Recommended)**

   ```bash
   # The setup script will create .env file for you
   # Edit .env and replace 'your_google_api_key_here' with your actual key
   ```

   **Option B: Environment Variable**

   ```bash
   set GOOGLE_API_KEY=your_api_key_here
   ```

3. **Get Your API Key**
   - Visit: https://aistudio.google.com/app/apikey
   - Create a new API key
   - Copy it to your .env file

## Available Voices

- **Kore** (default) - Natural, balanced voice
- **Leda** - Soft, gentle voice
- **Charon** - Deep, authoritative voice
- **Puck** - Energetic, expressive voice

## Usage

### Standalone Usage

```bash
python gemini_tts.py --text_file your_text.txt --voice Kore --preprocess --high_quality
```

### Integration with Automation

The main automation script (`../process.bat`) automatically uses this Gemini TTS system. Just run:

```bash
process.bat your_story.txt asmr
```

### Quick Test

```bash
test.bat
```

## Folder Structure

```
gemini-tts/
├── gemini_tts.py          # Main TTS script
├── requirements.txt       # Python dependencies
├── setup.bat             # Setup script
├── test.bat              # Test script
├── env_template.txt      # Template for .env file
├── .env                  # Your API key (created by setup)
├── input/                # Input text files
├── output/               # Generated audio files
└── README.md            # This file
```

## Arguments

- `--text_file`: Input text file name (required)
- `--voice`: Voice to use (default: Kore)
- `--preprocess`: Apply text preprocessing for better speech
- `--high_quality`: Use high-quality settings
- `--api_key`: Google API key (optional if set in .env)

## API Key Security

The `.env` file approach is recommended because:

- ✅ **Secure** - API key stays in local file
- ✅ **Persistent** - No need to set environment variables each time
- ✅ **Convenient** - Automatically loaded by the script
- ✅ **Standard** - Industry standard for configuration

**Important**: Never commit your `.env` file to version control!

## Advantages over CSM

- ✅ **No local model download** required (API-based)
- ✅ **Faster setup** - just install one package
- ✅ **Better quality** - Uses Google's latest TTS technology
- ✅ **More reliable** - No GPU/hardware dependencies
- ✅ **Natural sounding** voices with emotional expression
- ✅ **Consistent results** across different machines

## Example

```bash
# Run setup
setup.bat

# Edit .env file with your API key

# Test the setup
test.bat

# Use with your content
echo "This is a test of Gemini TTS." > input/test.txt
python gemini_tts.py --text_file test.txt --voice Kore --preprocess
```
