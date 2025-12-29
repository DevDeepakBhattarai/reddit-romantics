# Reddit Romantics Automation

This project automates the complete video creation workflow for Reddit stories, now powered by **Google Gemini TTS** for superior speech quality.

## 🚀 Quick Start

1. **Install ffmpeg** on your machine
2. **Setup Gemini TTS** (replaces CSM)
3. **Install WhisperX** for captions
4. **Run automation** with your story

## Setup Instructions

### 1. Install ffmpeg

Download and install ffmpeg from https://ffmpeg.org/

### 2. Setup Gemini TTS (New!)

```bash
cd gemini-tts
setup.bat
# Edit the .env file with your Google API key from https://aistudio.google.com/app/apikey
```

### 3. Install WhisperX

a. Install `miniconda`, create a virtual environment named `whisperx`

b. Run the following:

```bash
conda activate whisperx
python -m pip install whisperx==3.3.0
```

c. Follow instructions from: https://github.com/m-bain/whisperX/issues/983#issuecomment-2585510553

## 🎯 Usage

```bash
# Place your story text in input/your_story.txt
process.bat your_story.txt asmr
# or
process.bat your_story.txt minecraft
```

## 🎵 TTS System: Gemini vs CSM

We've **upgraded from CSM to Gemini TTS** for better quality:

| Feature     | Gemini TTS ✅          | CSM (Old)              |
| ----------- | ---------------------- | ---------------------- |
| Setup       | Simple pip install     | Complex model download |
| Quality     | Superior, natural      | Good but robotic       |
| Speed       | API-based, fast        | Local processing       |
| Reliability | Cloud-based, stable    | Hardware dependent     |
| Voices      | 4 high-quality options | Limited options        |

## 📁 Project Structure

```
├── process.bat           # Main automation script
├── gemini-tts/          # New TTS system
│   ├── gemini_tts.py    # TTS implementation
│   ├── setup.bat        # TTS setup
│   └── README.md        # TTS documentation
├── input/               # Your story text files
├── output/              # Final videos
├── videos/              # Background footage
│   ├── asmr/           # ASMR backgrounds
│   └── minecraft/      # Minecraft footage
└── caption/            # Generated subtitles
```

## 🔧 Available Voices

- **Kore** (default) - Natural, balanced
- **Leda** - Soft, gentle
- **Charon** - Deep, authoritative
- **Puck** - Energetic, expressive
