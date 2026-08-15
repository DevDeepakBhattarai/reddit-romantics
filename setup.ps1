param(
    [string]$PythonVersion = "3.11",
    [ValidateSet("cpu", "cu118", "cu121", "cu124", "cu126", "cu128")]
    [string]$TorchBackend = "cu128",
    [switch]$SkipWhisperX
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$VibeVoiceCommit = "631804b9c1f042e381207fe87c54603fe6accbc1"
$VibeVoiceRepo = "https://github.com/vibevoice-community/VibeVoice.git"
$VibeVoiceDir = Join-Path $PSScriptRoot "vendor\VibeVoice"
$AppVenv = Join-Path $PSScriptRoot ".venv"
$WhisperVenv = Join-Path $PSScriptRoot ".whisperx-venv"

function Require-Command([string]$Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found on PATH."
    }
}

Require-Command "uv"
Require-Command "git"
Require-Command "ffmpeg"
Require-Command "ffprobe"

Write-Host "==> Creating application environment ($PythonVersion)"
if (-not (Test-Path (Join-Path $AppVenv "Scripts\python.exe"))) {
    uv venv --python $PythonVersion $AppVenv
}
$AppPython = Join-Path $AppVenv "Scripts\python.exe"

Write-Host "==> Installing CUDA/PyTorch runtime ($TorchBackend)"
uv pip install --python $AppPython --torch-backend $TorchBackend torch torchvision torchaudio

Write-Host "==> Installing application dependencies"
uv pip install --python $AppPython -r requirements.txt

Write-Host "==> Installing pinned VibeVoice runtime"
New-Item -ItemType Directory -Path (Split-Path $VibeVoiceDir) -Force | Out-Null
if (-not (Test-Path (Join-Path $VibeVoiceDir ".git"))) {
    git clone --filter=blob:none --depth 1 --sparse $VibeVoiceRepo $VibeVoiceDir
}
# Only materialize the inference package, top-level package metadata, and WAV voice presets.
# This avoids downloading VibeVoice demo videos and streaming-model preset assets.
git -C $VibeVoiceDir sparse-checkout set --no-cone "/vibevoice/" "/demo/voices/*.wav" "/pyproject.toml" "/README.md"
git -C $VibeVoiceDir fetch origin $VibeVoiceCommit --depth 1
git -C $VibeVoiceDir checkout --detach $VibeVoiceCommit
uv pip install --python $AppPython --torch-backend $TorchBackend -e $VibeVoiceDir

if (-not $SkipWhisperX) {
    Write-Host "==> Creating isolated WhisperX environment"
    if (-not (Test-Path (Join-Path $WhisperVenv "Scripts\python.exe"))) {
        uv venv --python $PythonVersion $WhisperVenv
    }
    $WhisperPython = Join-Path $WhisperVenv "Scripts\python.exe"
    uv pip install --python $WhisperPython --torch-backend $TorchBackend whisperx
}

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "==> Created .env from .env.example; add GOOGLE_API_KEY before using Gemini TTS."
}

Write-Host ""
Write-Host "Setup complete."
Write-Host "Launch UI:  .\.venv\Scripts\python.exe app.py"
Write-Host "CLI help:   .\.venv\Scripts\python.exe main.py run --help"
if ($SkipWhisperX) {
    Write-Host "Captions:   WhisperX installation was skipped. Configure WHISPERX_COMMAND manually before rendering captions."
} else {
    Write-Host "Captions:   local .whisperx-venv is ready and auto-detected by the pipeline."
}
