param(
    [string]$PythonVersion = "3.11",
    [ValidateSet("cpu", "cu118", "cu121", "cu124", "cu126", "cu128")]
    [string]$TorchBackend = "cu128",
    [switch]$SkipFish,
    [switch]$SkipWhisperX
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$VibeVoiceCommit = "631804b9c1f042e381207fe87c54603fe6accbc1"
$VibeVoiceRepo = "https://github.com/vibevoice-community/VibeVoice.git"
$VibeVoiceDir = Join-Path $PSScriptRoot "vendor\VibeVoice"
$S2Dir = Join-Path $PSScriptRoot "vendor\s2.cpp"
$AppVenv = Join-Path $PSScriptRoot ".venv"
$WhisperVenv = Join-Path $PSScriptRoot ".whisperx-venv"
$FishNativeDir = Join-Path $PSScriptRoot ".work\fish-native"
$FishGguf = Join-Path $FishNativeDir "s2-pro-f16.gguf"

function Require-Command([string]$Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found on PATH."
    }
}

function Get-WslPath([string]$WindowsPath) {
    if ($WindowsPath -match '^([A-Za-z]):[\\/](.*)$') {
        $drive = $Matches[1].ToLowerInvariant()
        $rest = $Matches[2].Replace('\', '/')
        return "/mnt/$drive/$rest"
    }
    throw "Expected an absolute Windows path, got: $WindowsPath"
}

function Invoke-Wsl([string]$Command) {
    & wsl.exe bash -lc $Command
    if ($LASTEXITCODE -ne 0) {
        throw "WSL command failed with exit code $LASTEXITCODE"
    }
}

function Ensure-WslUv {
    Require-Command "wsl.exe"
    $uv = (& wsl.exe bash -lc 'command -v uv || true' | Select-Object -First 1).Trim()
    if ($uv) { return $uv }

    Write-Host "==> Installing uv inside WSL for the one-time Fish F16 export"
    & wsl.exe -u root bash -lc 'apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl'
    if ($LASTEXITCODE -ne 0) { throw "Could not install curl inside WSL." }
    Invoke-Wsl 'curl -LsSf https://astral.sh/uv/install.sh | sh'
    $uv = (& wsl.exe bash -lc 'command -v uv || printf %s "$HOME/.local/bin/uv"' | Select-Object -First 1).Trim()
    if (-not $uv) { throw "uv installation inside WSL did not produce an executable." }
    return $uv
}

function Install-FishHybrid {
    Require-Command "git"
    Require-Command "cmake"

    New-Item -ItemType Directory -Force -Path (Split-Path $S2Dir), $FishNativeDir | Out-Null
    if (-not (Test-Path (Join-Path $S2Dir ".git"))) {
        Write-Host "==> Cloning s2.cpp for Fish Audio S2 Pro"
        git clone --recurse-submodules https://github.com/rodrigomatta/s2.cpp.git $S2Dir
        if ($LASTEXITCODE -ne 0) { throw "Failed to clone s2.cpp." }
    } else {
        git -C $S2Dir submodule update --init --recursive
        if ($LASTEXITCODE -ne 0) { throw "Failed to initialize s2.cpp submodules." }
    }

    $patch = Join-Path $PSScriptRoot "patches\s2cpp-batched-segments.patch"
    if (Test-Path $patch) {
        git -C $S2Dir apply --check $patch *> $null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "==> Applying Fish semantic sentence batching patch"
            git -C $S2Dir apply $patch
            if ($LASTEXITCODE -ne 0) { throw "Failed to apply Fish s2.cpp patch." }
        } else {
            git -C $S2Dir apply --reverse --check $patch *> $null
            if ($LASTEXITCODE -ne 0) {
                throw "Fish s2.cpp patch no longer applies cleanly."
            }
        }
    }

    $build = Join-Path $S2Dir "build-cuda"
    Write-Host "==> Building native Fish hybrid runtime"
    cmake -S $S2Dir -B $build -DS2_CUDA=ON -DCMAKE_BUILD_TYPE=Release
    if ($LASTEXITCODE -ne 0) { throw "s2.cpp CUDA configure failed." }
    cmake --build $build --config Release --parallel 4
    if ($LASTEXITCODE -ne 0) { throw "s2.cpp CUDA build failed." }

    $exeDir = Join-Path $build "Release"
    $dllDir = Join-Path $build "bin\Release"
    if (-not (Test-Path (Join-Path $exeDir "s2.exe"))) {
        $alternateExe = Join-Path $dllDir "s2.exe"
        if (Test-Path $alternateExe) {
            New-Item -ItemType Directory -Force -Path $exeDir | Out-Null
            Copy-Item $alternateExe (Join-Path $exeDir "s2.exe") -Force
        } else {
            throw "s2.cpp build completed but s2.exe was not found."
        }
    }
    if (Test-Path $dllDir) {
        Get-ChildItem $dllDir -Filter "*.dll" | Copy-Item -Destination $exeDir -Force
    }

    $needsExport = -not (Test-Path $FishGguf)
    if (-not $needsExport) {
        $needsExport = (Get-Item $FishGguf).Length -lt 9000000000
    }
    if (-not $needsExport) {
        Write-Host "==> Keeping existing full F16 Fish model: $FishGguf"
        return
    }

    Write-Host "==> Downloading and exporting Fish S2 Pro as full unquantized F16"
    $uv = Ensure-WslUv
    $wslHome = (& wsl.exe bash -lc 'printf %s "$HOME"' | Select-Object -First 1).Trim()
    if (-not $wslHome) { throw "Could not resolve WSL home directory." }
    $exportRoot = "$wslHome/.cache/reddit-romantics/fish-export"
    $exportPython = "$exportRoot/.venv/bin/python"
    $checkpoint = "$exportRoot/checkpoints/s2-pro"
    $wslExporter = Get-WslPath (Join-Path $S2Dir "quantize\unified_export_gguf.py")
    $wslGguf = Get-WslPath $FishGguf

    Invoke-Wsl "mkdir -p '$exportRoot'; test -x '$exportPython' || '$uv' venv --python 3.11 '$exportRoot/.venv'"
    Invoke-Wsl "'$uv' pip install --python '$exportPython' --torch-backend cpu torch numpy gguf safetensors huggingface_hub hf_xet"
    $downloadCode = "from huggingface_hub import snapshot_download; snapshot_download('fishaudio/s2-pro', local_dir='$checkpoint')"
    Invoke-Wsl "'$exportPython' -c `"$downloadCode`""
    Invoke-Wsl "'$exportPython' '$wslExporter' --checkpoint-path '$checkpoint' --codec-checkpoint-path '$checkpoint/codec.pth' --output '$wslGguf' --out-dtype f16"

    if (-not (Test-Path $FishGguf) -or (Get-Item $FishGguf).Length -lt 9000000000) {
        throw "Fish F16 export did not produce a complete GGUF at $FishGguf"
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

Write-Host "==> Installing application runtime ($TorchBackend)"
uv pip install --python $AppPython --torch-backend $TorchBackend torch torchvision torchaudio
uv pip install --python $AppPython -r requirements.txt

Write-Host "==> Installing pinned VibeVoice runtime"
New-Item -ItemType Directory -Path (Split-Path $VibeVoiceDir) -Force | Out-Null
if (-not (Test-Path (Join-Path $VibeVoiceDir ".git"))) {
    git clone --filter=blob:none --depth 1 --sparse $VibeVoiceRepo $VibeVoiceDir
}
git -C $VibeVoiceDir sparse-checkout set --no-cone "/vibevoice/" "/demo/voices/*.wav" "/pyproject.toml" "/README.md"
git -C $VibeVoiceDir fetch origin $VibeVoiceCommit --depth 1
git -C $VibeVoiceDir checkout --detach $VibeVoiceCommit
uv pip install --python $AppPython --torch-backend $TorchBackend -e $VibeVoiceDir

if (-not $SkipFish) {
    Install-FishHybrid
}

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
Write-Host "UI:  .\.venv\Scripts\python.exe app.py"
Write-Host "CLI: .\.venv\Scripts\python.exe main.py run --help"
