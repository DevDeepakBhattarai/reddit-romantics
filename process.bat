@echo off
setlocal

rem Backward-compatible wrapper for the original automation entry point.
rem Advanced/new usage lives in: python main.py run --help
rem Usage: process.bat input_text_file.txt video_type [GeminiVoice]
rem video_type must be asmr or minecraft.

if "%~1"=="" (
    echo Error: Please provide the input text file name.
    echo Usage: process.bat input_text_file.txt [asmr^|minecraft] [GeminiVoice]
    exit /b 1
)

if "%~2"=="" (
    echo Error: Please specify video type.
    echo Usage: process.bat input_text_file.txt [asmr^|minecraft] [GeminiVoice]
    exit /b 1
)

if /i not "%~2"=="asmr" if /i not "%~2"=="minecraft" (
    echo Error: Video type must be either "asmr" or "minecraft".
    exit /b 1
)

set "STORY=%~1"
set "BACKGROUND=%~2"
set "VOICE=%~3"
if "%VOICE%"=="" set "VOICE=Kore"

if not exist "%STORY%" (
    if exist "input\%STORY%" (
        set "STORY=input\%STORY%"
    ) else (
        echo Error: story file not found: %~1
        exit /b 1
    )
)

set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

"%PYTHON%" "%~dp0main.py" run ^
    --story-file "%STORY%" ^
    --background "%BACKGROUND%" ^
    --tts gemini ^
    --gemini-voice "%VOICE%"

exit /b %ERRORLEVEL%
