[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

# Configuration
$config = @{
    GitInstallerUrl = 'https://github.com/git-for-windows/git/releases/download/v2.42.0.windows.2/Git-2.42.0.2-64-bit.exe'
    GitInstaller    = Join-Path $PWD 'Git-2.42.0-64-bit.exe'
    LocalGitPath    = Join-Path $PWD 'Git'
    RepoUrl         = 'https://github.com/SesameAILabs/csm.git'
    CloneDir        = Join-Path $PWD 'csm'
}

function Test-GitInstallation {
    try {
        # Check in PowerShell
        $null = Get-Command git -ErrorAction Stop
        
        # Check in CMD
        $cmdCheck = cmd /c "git --version" 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Git not accessible in CMD" -ForegroundColor Yellow
            return $false
        }
        
        Write-Host "Git is working properly" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "Git not found in PowerShell" -ForegroundColor Yellow
        return $false
    }
}

try {
    Write-Host "Checking Git installation..." -ForegroundColor Cyan
    
    if (-not (Test-GitInstallation)) {
        Write-Host "Git is not installed. Installing now..." -ForegroundColor Yellow
        
        if (-not (Test-Path $config.GitInstaller)) {
            # Use BITS for faster download with progress
            Import-Module BitsTransfer
            Start-BitsTransfer -Source $config.GitInstallerUrl -Destination $config.GitInstaller
        }

        # Verify download
        if (-not (Test-Path $config.GitInstaller) -or (Get-Item $config.GitInstaller).Length -lt 1MB) {
            throw "Download failed or file is corrupted"
        }

        Write-Host "Installing Git silently..." -ForegroundColor Yellow
        $args = @(
            '/VERYSILENT',
            '/SUPPRESSMSGBOXES',
            '/NORESTART',
            '/NOCANCEL',
            "/DIR=`"$($config.LocalGitPath)`""
        )
        
        Start-Process -FilePath $config.GitInstaller -ArgumentList $args -Wait
        
        # Refresh PATH
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + 
                    [System.Environment]::GetEnvironmentVariable("Path", "User")
        
        # Verify installation
        if (-not (Test-GitInstallation)) {
            throw "Git installation failed verification"
        }
    }

    # # Now that Git is confirmed working, check for project
    # Write-Host "Checking for existing project..." -ForegroundColor Cyan
    # if (Test-Path $config.CloneDir) {
    #     # Verify if it's a valid git repository
    #     Push-Location $config.CloneDir
    #     $isGitRepo = git rev-parse --git-dir 2>$null
    #     if ($LASTEXITCODE -eq 0 -and $isGitRepo) {
    #         Write-Host "Repository directory exists and is a valid Git repo. Updating..." -ForegroundColor Yellow
    #         git pull
    #     } else {
    #         Pop-Location
    #         Write-Host "Directory exists but is not a Git repository. Removing and cloning..." -ForegroundColor Yellow
    #         Remove-Item $config.CloneDir -Recurse -Force
    #         git clone $config.RepoUrl $config.CloneDir
    #     }
    #     Pop-Location
    # } else {
    #     Write-Host "Cloning repository..." -ForegroundColor Yellow
    #     git clone $config.RepoUrl $config.CloneDir
    # }

    Write-Host "`nAll tasks completed successfully!" -ForegroundColor Green
}
catch {
    Write-Host "`nError: $_" -ForegroundColor Red
    Write-Host "Script encountered an error." -ForegroundColor Red
    exit 1
}
finally {
    Write-Host "`n"
}
