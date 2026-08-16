param(
    [ValidateSet("fish", "step", "magpie", "chatterbox", "all")]
    [string]$Backend = "all",
    [ValidateSet("cpu", "cu118", "cu121", "cu124", "cu126", "cu128")]
    [string]$TorchBackend = "cu128"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$RuntimeRoot = Join-Path $PSScriptRoot ".tts-venvs"
$VendorRoot = Join-Path $PSScriptRoot "vendor"
$HfHome = Join-Path $PSScriptRoot ".work\hf-cache"
New-Item -ItemType Directory -Force -Path $RuntimeRoot, $VendorRoot, $HfHome | Out-Null
$env:HF_HOME = $HfHome
$env:HF_HUB_DISABLE_XET = "0"

function Require-Command([string]$Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found on PATH."
    }
}

function New-TtsVenv([string]$Name, [string]$PythonVersion) {
    $dir = Join-Path $RuntimeRoot $Name
    $python = Join-Path $dir "Scripts\python.exe"
    if (-not (Test-Path $python)) {
        Write-Host "==> Creating $Name runtime (Python $PythonVersion)"
        uv venv --python $PythonVersion $dir
    }
    return $python
}

function Install-Torch([string]$Python) {
    Write-Host "==> Installing/updating PyTorch ($TorchBackend)"
    uv pip install --python $Python --upgrade --torch-backend $TorchBackend torch torchvision torchaudio
}

function Test-PythonCode([string]$Python, [string]$Code) {
    $previousErrorAction = $ErrorActionPreference
    try {
        $ErrorActionPreference = "SilentlyContinue"
        & $Python -c $Code *> $null
        return ($LASTEXITCODE -eq 0)
    } finally {
        $ErrorActionPreference = $previousErrorAction
    }
}

function Ensure-Torch([string]$Python) {
    if (Test-PythonCode $Python "import torch; assert tuple(map(int, torch.__version__.split('+')[0].split('.')[:2])) >= (2, 7)") {
        $version = & $Python -c "import torch; print(torch.__version__)"
        Write-Host "==> Keeping existing PyTorch $version"
        return
    }
    Install-Torch $Python
}

function Download-Hf([string]$Repo) {
    Write-Host "==> Downloading $Repo into shared HF cache"
    $helper = Join-Path $PSScriptRoot ".work\hf_download.py"
    uv run --with huggingface_hub --with hf_xet python $helper --repo $Repo
    if ($LASTEXITCODE -ne 0) { throw "Hugging Face download failed for $Repo" }
}


function Get-WslPath([string]$WindowsPath) {
    if ($WindowsPath -match '^([A-Za-z]):[\\/](.*)$') {
        $drive = $Matches[1].ToLowerInvariant()
        $rest = $Matches[2].Replace('\', '/')
        return "/mnt/$drive/$rest"
    }
    Require-Command "wsl.exe"
    $escaped = $WindowsPath.Replace("'", "'\''")
    $result = & wsl.exe bash -lc "wslpath -a -u '$escaped'"
    if ($LASTEXITCODE -ne 0 -or -not $result) {
        throw "Could not translate Windows path to WSL: $WindowsPath"
    }
    return ($result | Select-Object -First 1).Trim()
}

function Invoke-Wsl([string]$Command) {
    Write-Host "WSL> $Command"
    $scriptPath = Join-Path $PSScriptRoot (".work\\wsl-command-" + [Guid]::NewGuid().ToString("N") + ".sh")
    $contents = "#!/usr/bin/env bash`nset -e`n$Command`n"
    [System.IO.File]::WriteAllText($scriptPath, $contents, [System.Text.UTF8Encoding]::new($false))
    $wslScript = Get-WslPath $scriptPath
    try {
        & wsl.exe bash $wslScript
        if ($LASTEXITCODE -ne 0) {
            throw "WSL command failed with exit code $LASTEXITCODE"
        }
    } finally {
        Remove-Item $scriptPath -Force -ErrorAction SilentlyContinue
    }
}

function Ensure-WslBase {
    Require-Command "wsl.exe"
    Write-Host "==> Ensuring WSL build/audio dependencies"
    & wsl.exe -u root bash -lc 'apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git git-lfs curl ffmpeg build-essential pkg-config portaudio19-dev libsox-dev libsndfile1-dev python3.12-dev sox'
    if ($LASTEXITCODE -ne 0) { throw "Failed to install WSL system dependencies." }
    Invoke-Wsl 'export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"; command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh; uv --version'
}

function Install-Fish {
    Require-Command "git"
    Ensure-WslBase

    $root = Join-Path $VendorRoot "fish-speech"
    if (-not (Test-Path (Join-Path $root ".git"))) {
        Write-Host "==> Cloning Fish Speech source"
        git clone --filter=blob:none https://github.com/fishaudio/fish-speech.git $root
    } else {
        Write-Host "==> Fish Speech source already present"
    }
    $wslRoot = Get-WslPath $root
    $wslHelper = Get-WslPath (Join-Path $PSScriptRoot ".work\hf_download.py")

    Write-Host "==> Installing Fish runtime on WSL native filesystem (avoids slow /mnt Python imports)"
    Invoke-Wsl "export PATH=`"`$HOME/.local/bin:`$HOME/.cargo/bin:`$PATH`"; export HF_HOME=`"`$HOME/.cache/huggingface`"; NATIVE=`"`$HOME/.cache/reddit-romantics/fish`"; mkdir -p `"`$NATIVE`"; cd '$wslRoot'; UV_PROJECT_ENVIRONMENT=`"`$NATIVE/.venv`" uv sync --python 3.12 --extra cpu"

    Write-Host "==> Downloading Fish S2 Pro weights onto WSL native filesystem"
    Invoke-Wsl "export HF_HOME=`"`$HOME/.cache/huggingface`"; NATIVE=`"`$HOME/.cache/reddit-romantics/fish`"; `"`$NATIVE/.venv/bin/python`" '$wslHelper' --repo fishaudio/s2-pro --local-dir `"`$NATIVE/checkpoints/s2-pro`""
    Set-Content -Path (Join-Path $PSScriptRoot ".work\fish-wsl-native-ready") -Value "wsl-native-v1"

    # The official PyTorch runner cannot split S2 Pro layers across CPU + CUDA.
    # Build s2.cpp instead: it supports exact transformer-layer offload without
    # quantizing the model. Runtime stays native Windows, so there is no WSL
    # startup/import penalty once setup has finished.
    Require-Command "cmake"
    $s2Root = Join-Path $VendorRoot "s2.cpp"
    if (-not (Test-Path (Join-Path $s2Root ".git"))) {
        Write-Host "==> Cloning native s2.cpp hybrid inference engine"
        git clone --recurse-submodules https://github.com/rodrigomatta/s2.cpp.git $s2Root
        if ($LASTEXITCODE -ne 0) { throw "Failed to clone s2.cpp" }
    } else {
        Write-Host "==> s2.cpp source already present; ensuring ggml submodule is initialized"
        git -C $s2Root submodule update --init --recursive
        if ($LASTEXITCODE -ne 0) { throw "Failed to initialize s2.cpp ggml submodule" }
    }

    $s2Build = Join-Path $s2Root "build-cuda"
    Write-Host "==> Building native s2.cpp with CUDA support"
    cmake -S $s2Root -B $s2Build -DS2_CUDA=ON -DCMAKE_BUILD_TYPE=Release
    if ($LASTEXITCODE -ne 0) { throw "s2.cpp CUDA configure failed" }
    cmake --build $s2Build --config Release --parallel 4
    if ($LASTEXITCODE -ne 0) { throw "s2.cpp CUDA build failed" }

    # MSVC places the executable and ggml runtime DLLs in sibling Release
    # directories. Keep them together so s2.exe can launch directly from the
    # application without modifying the user's global PATH.
    $s2ExeDir = Join-Path $s2Build "Release"
    $s2DllDir = Join-Path $s2Build "bin\Release"
    if (-not (Test-Path (Join-Path $s2ExeDir "s2.exe"))) {
        throw "s2.cpp build completed but s2.exe was not found in $s2ExeDir"
    }
    if (Test-Path $s2DllDir) {
        Get-ChildItem $s2DllDir -Filter "*.dll" | Copy-Item -Destination $s2ExeDir -Force
    }

    # Export the already-downloaded official checkpoint to a full F16 GGUF.
    # F16 is the reference-quality, non-quantized base export used by s2.cpp;
    # do NOT run Q8/Q6/Q4 conversion here.
    $fishNativeDir = Join-Path $PSScriptRoot ".work\fish-native"
    New-Item -ItemType Directory -Force -Path $fishNativeDir | Out-Null
    $fishGguf = Join-Path $fishNativeDir "s2-pro-f16.gguf"
    $needsExport = -not (Test-Path $fishGguf)
    if (-not $needsExport) {
        $needsExport = (Get-Item $fishGguf).Length -lt 9000000000
    }
    if ($needsExport) {
        Write-Host "==> Exporting full, unquantized F16 S2 Pro GGUF (~9.9 GB; one-time step)"
        $wslS2Root = Get-WslPath $s2Root
        $wslFishGguf = Get-WslPath $fishGguf
        Invoke-Wsl "export PATH=`"`$HOME/.local/bin:`$HOME/.cargo/bin:`$PATH`"; NATIVE=`"`$HOME/.cache/reddit-romantics/fish`"; uv pip install --python `"`$NATIVE/.venv/bin/python`" numpy gguf safetensors; `"`$NATIVE/.venv/bin/python`" '$wslS2Root/quantize/unified_export_gguf.py' --checkpoint-path `"`$NATIVE/checkpoints/s2-pro`" --codec-checkpoint-path `"`$NATIVE/checkpoints/s2-pro/codec.pth`" --output '$wslFishGguf' --out-dtype f16"
    } else {
        Write-Host "==> Keeping existing full F16 Fish GGUF: $fishGguf"
    }
}

function Install-Step {
    Require-Command "git"
    Ensure-WslBase

    $root = Join-Path $VendorRoot "Step-Audio-EditX"
    if (-not (Test-Path (Join-Path $root ".git"))) {
        Write-Host "==> Cloning Step Audio EditX source"
        git clone --filter=blob:none https://github.com/stepfun-ai/Step-Audio-EditX.git $root
    } else {
        Write-Host "==> Step Audio EditX source already present"
    }
    $wslRoot = Get-WslPath $root
    $wslHelper = Get-WslPath (Join-Path $PSScriptRoot ".work\hf_download.py")

    Write-Host "==> Installing Step Audio runtime on WSL native filesystem"
    Invoke-Wsl "export PATH=`"`$HOME/.local/bin:`$HOME/.cargo/bin:`$PATH`"; export HF_HOME=`"`$HOME/.cache/huggingface`"; NATIVE=`"`$HOME/.cache/reddit-romantics/step`"; mkdir -p `"`$NATIVE`"; cd '$wslRoot'; UV_PROJECT_ENVIRONMENT=`"`$NATIVE/.venv`" uv sync --refresh --python 3.12"

    Write-Host "==> Downloading Step tokenizer + AWQ model onto WSL native filesystem"
    Invoke-Wsl "export HF_HOME=`"`$HOME/.cache/huggingface`"; NATIVE=`"`$HOME/.cache/reddit-romantics/step`"; `"`$NATIVE/.venv/bin/python`" '$wslHelper' --repo stepfun-ai/Step-Audio-Tokenizer --local-dir `"`$NATIVE/models/Step-Audio-Tokenizer`"; `"`$NATIVE/.venv/bin/python`" '$wslHelper' --repo stepfun-ai/Step-Audio-EditX-AWQ-4bit --local-dir `"`$NATIVE/models/Step-Audio-EditX-AWQ-4bit`""

    # Ensure the official English zero-shot prompt is materialized even if git-lfs
    # did not hydrate it during a filtered clone.
    Invoke-Wsl "cd '$wslRoot'; git lfs pull --include='examples/zero_shot_en_prompt.wav' || true"
    Set-Content -Path (Join-Path $PSScriptRoot ".work\step-wsl-native-ready") -Value "wsl-native-v1"
}

function Install-Magpie {
    Ensure-WslBase
    $wslHelper = Get-WslPath (Join-Path $PSScriptRoot ".work\hf_download.py")

    Write-Host "==> Preparing NVIDIA Speech source + Magpie runtime on WSL native filesystem"
    Invoke-Wsl "export PATH=`"`$HOME/.local/bin:`$HOME/.cargo/bin:`$PATH`"; export HF_HOME=`"`$HOME/.cache/huggingface`"; ROOT=`"`$HOME/.cache/reddit-romantics/magpie`"; SRC=`"`$ROOT/Speech`"; mkdir -p `"`$ROOT`"; if [ ! -d `"`$SRC/.git`" ]; then git clone --depth 1 --filter=blob:none --no-checkout https://github.com/NVIDIA-NeMo/Speech.git `"`$SRC`"; cd `"`$SRC`"; git sparse-checkout init --cone; git sparse-checkout set nemo; git checkout main; else cd `"`$SRC`"; git fetch --depth 1 origin main; git reset --hard origin/main; fi; test -x `"`$ROOT/.venv/bin/python`" || uv venv --python 3.12 `"`$ROOT/.venv`""

    Write-Host "==> Installing NVIDIA NeMo Speech TTS runtime from the checked-out main branch"
    Invoke-Wsl "export PATH=`"`$HOME/.local/bin:`$HOME/.cargo/bin:`$PATH`"; export HF_HOME=`"`$HOME/.cache/huggingface`"; ROOT=`"`$HOME/.cache/reddit-romantics/magpie`"; uv pip install --no-sources --python `"`$ROOT/.venv/bin/python`" --torch-backend $TorchBackend --upgrade `"`$ROOT/Speech[tts]`" kaldialign soundfile huggingface_hub hf_xet peft"

    # NeMo only declares torch>=2.6. Without re-applying the selected backend,
    # dependency resolution can upgrade torch while leaving torchvision/torchaudio
    # on incompatible builds (for example cu130 torch with cu128 torchvision).
    Write-Host "==> Aligning Magpie PyTorch/TorchVision/TorchAudio ($TorchBackend)"
    Invoke-Wsl "export PATH=`"`$HOME/.local/bin:`$HOME/.cargo/bin:`$PATH`"; ROOT=`"`$HOME/.cache/reddit-romantics/magpie`"; uv pip install --python `"`$ROOT/.venv/bin/python`" --torch-backend $TorchBackend --upgrade torch torchvision torchaudio"

    Write-Host "==> Verifying NVIDIA Magpie runtime"
    $cudaCheck = if ($TorchBackend -eq "cpu") { "" } else { "; assert torch.cuda.is_available(), 'CUDA is not available to Magpie in WSL2'" }
    Invoke-Wsl "export HF_HOME=`"`$HOME/.cache/huggingface`"; PY=`"`$HOME/.cache/reddit-romantics/magpie/.venv/bin/python`"; `"`$PY`" -c `"import torch, torchvision, torchaudio, soundfile, kaldialign, huggingface_hub, peft; from nemo.collections.tts.models import MagpieTTSModel$cudaCheck; print('Magpie runtime ready:', torch.__version__, 'CUDA:', torch.cuda.is_available(), 'SoundFile:', soundfile.__version__)`""

    Write-Host "==> Downloading Magpie checkpoint onto WSL native filesystem"
    Invoke-Wsl "export HF_HOME=`"`$HOME/.cache/huggingface`"; PY=`"`$HOME/.cache/reddit-romantics/magpie/.venv/bin/python`"; `"`$PY`" '$wslHelper' --repo nvidia/magpie_tts_multilingual_357m"
    Set-Content -Path (Join-Path $PSScriptRoot ".work\magpie-runtime-ready") -Value "wsl-native-v1"
}

function Install-Chatterbox {
    $python = New-TtsVenv "chatterbox" "3.11"
    Write-Host "==> Installing latest official Chatterbox source (includes Nano support)"
    uv pip install --python $python --upgrade "git+https://github.com/resemble-ai/chatterbox.git" huggingface_hub hf_xet
    Write-Host "==> Restoring Chatterbox's pinned Torch 2.6 runtime with $TorchBackend acceleration"
    uv pip install --python $python --torch-backend $TorchBackend --upgrade "torch==2.6.0" "torchaudio==2.6.0" "torchvision==0.21.0"
    Download-Hf "ResembleAI/chatterbox-turbo"
    Download-Hf "ResembleAI/chatterbox-nano"
    Download-Hf "ResembleAI/chatterbox"
}

Require-Command "uv"

$targets = if ($Backend -eq "all") {
    @("fish", "step", "magpie", "chatterbox")
} else {
    @($Backend)
}

$failures = @()
foreach ($target in $targets) {
    Write-Host ""
    Write-Host "========== $target ==========" -ForegroundColor Cyan
    try {
        switch ($target) {
            "fish" { Install-Fish }
            "step" { Install-Step }
            "magpie" { Install-Magpie }
            "chatterbox" { Install-Chatterbox }
        }
        Write-Host "==> $target runtime ready" -ForegroundColor Green
    } catch {
        if ($Backend -ne "all") { throw }
        $failures += "$target`: $($_.Exception.Message)"
        Write-Host "==> $target setup failed; continuing with the other backends." -ForegroundColor Yellow
        Write-Host $_.Exception.Message -ForegroundColor Yellow
    }
}

Write-Host ""
if ($failures.Count -gt 0) {
    Write-Host "Setup completed with some backend failures:" -ForegroundColor Yellow
    $failures | ForEach-Object { Write-Host "  - $_" -ForegroundColor Yellow }
    exit 2
}

Write-Host "Requested TTS runtimes and model weights are ready." -ForegroundColor Green
Write-Host "Launch Gradio: .\.venv\Scripts\python.exe app.py"
Write-Host "Use 'Generate audio only' to test a model without rendering a video."
