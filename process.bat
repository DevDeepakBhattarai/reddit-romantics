@echo off
setlocal enabledelayedexpansion

:: This script automates the complete video creation workflow
:: Usage: process.bat input_text_file.txt video_type [voice]
:: video_type should be either "asmr" or "minecraft"
:: voice is optional and defaults to "Kore" if not provided

:: Check if required parameters were provided
if "%~1"=="" (
    echo Error: Please provide the input text file name.
    echo Usage: process.bat input_text_file.txt [asmr^|minecraft] [voice]
    echo Example: process.bat my_story.txt asmr
    echo Example: process.bat my_story.txt asmr Nova
    exit /b 1
)

if "%~2"=="" (
    echo Error: Please specify video type.
    echo Usage: process.bat input_text_file.txt [asmr^|minecraft] [voice]
    echo Example: process.bat my_story.txt asmr
    echo Example: process.bat my_story.txt asmr Nova
    exit /b 1
)

:: Validate video type parameter
if /i not "%~2"=="asmr" if /i not "%~2"=="minecraft" (
    echo Error: Video type must be either "asmr" or "minecraft"
    echo Usage: process.bat input_text_file.txt [asmr^|minecraft] [voice]
    exit /b 1
)

:: Store parameters
set "input_text_file=%~1"
set "video_type=%~2"
set "voice=%~3"

:: Set default voice if not provided
if "%voice%"=="" set "voice=Kore"

:: Extract filename without extension for output naming
for %%F in ("%input_text_file%") do set "base_filename=%%~nF"

:: Navigate to the project root directory
echo Navigating to project directory...
cd /d "D:\Reddit-Romantics\Automation"

:: Validate input file exists
if not exist "input\%input_text_file%" (
    echo Error: Input text file 'input\%input_text_file%' not found!
    echo Please place your text file in the 'input' folder.
    exit /b 1
)

:: Define paths - Updated for Gemini TTS
set "audio_output=gemini-tts\output\%base_filename%.wav"
set "temp_video=temp_%base_filename%_video.mp4"
set "final_video=temp_%base_filename%_final.mp4"
set "output_video=output\%base_filename%_final.mp4"

:: Create output directory if it doesn't exist
if not exist "output" mkdir output

echo ========================================
echo STARTING AUTOMATED VIDEO CREATION
echo ========================================
echo Input Text File: %input_text_file%
echo Video Type: %video_type%
echo Base Filename: %base_filename%
echo TTS System: Gemini TTS
echo Voice: %voice%
echo ========================================

:: Step 1: Convert text to speech using Gemini TTS
echo.
echo Step 1/4: Converting text to speech with Gemini TTS...
echo ========================================

echo Running Gemini TTS conversion...
python gemini-tts\gemini_tts.py --text_file %input_text_file% --preprocess --high_quality --voice %voice%
if !errorlevel! neq 0 (
    echo Error: Gemini TTS conversion failed!
    exit /b 1
)

:: Check if audio was generated
if not exist "%audio_output%" (
    echo Error: Audio file was not generated at %audio_output%
    exit /b 1
)

echo ✅ Audio generated successfully with Gemini TTS: %audio_output%

:: Step 2: Get audio duration
echo.
echo Step 2/4: Analyzing audio duration...
echo ========================================

for /f "tokens=*" %%i in ('ffprobe -v quiet -show_entries format^=duration -of csv^=p^=0 "%audio_output%"') do set "audio_duration=%%i"
echo Audio duration: !audio_duration! seconds

:: Step 3: Create video with looped background footage
echo.
echo Step 3/4: Creating video with %video_type% background...
echo ========================================

:: Select video source based on type
if /i "%video_type%"=="asmr" (
    :: Use the first video file found in asmr folder
    for %%F in (videos\asmr\*.mp4) do (
        set "source_video=%%F"
        goto :found_video
    )
    echo Error: No video files found in videos\asmr\ folder
    exit /b 1
) else (
    :: Use minecraft video
    set "source_video=videos\minecraft\minecraft.mp4"
    if not exist "!source_video!" (
        echo Error: Minecraft video not found at !source_video!
        exit /b 1
    )
)

:found_video
echo Using source video: !source_video!

:: Get source video duration
for /f "tokens=*" %%i in ('ffprobe -v quiet -show_entries format^=duration -of csv^=p^=0 "!source_video!"') do set "source_duration=%%i"
echo Source video duration: !source_duration! seconds

:: Calculate video duration (audio + 3 seconds)
for /f "tokens=*" %%i in ('powershell -command "%audio_duration% + 3"') do set "video_duration=%%i"
echo Video duration (audio + 3 sec): !video_duration! seconds

:: Calculate how many loops we need for the extended video duration
for /f "tokens=*" %%i in ('powershell -command "[math]::Ceiling(%video_duration% / %source_duration%)"') do set "loops_needed=%%i"
echo Loops needed: !loops_needed!

:: Create concat file for efficient looping
set "concat_file=temp_concat_%base_filename%.txt"
echo Creating concat file: !concat_file!

:: Generate concat file with the source video repeated
(
    for /l %%i in (1,1,!loops_needed!) do (
        echo file '!source_video!'
    )
) > "!concat_file!"

:: Create looped video using concat demuxer (much faster - no re-encoding)
echo Creating looped video efficiently...
ffmpeg -y -f concat -safe 0 -i "!concat_file!" -c copy -t !video_duration! "!temp_video!"
if !errorlevel! neq 0 (
    echo Error: Failed to create looped video!
    del "!concat_file!"
    exit /b 1
)

:: Clean up concat file
del "!concat_file!"

echo ✅ Looped video created efficiently: !temp_video!

:: Step 4: Combine audio with video
echo.
echo Combining audio with video (with 3 sec silence at end)...
ffmpeg -y -i "!temp_video!" -i "%audio_output%" -filter_complex "[1:a]apad=pad_dur=3[audio_padded]" -c:v copy -c:a aac -map 0:v:0 -map "[audio_padded]" "!final_video!"
if !errorlevel! neq 0 (
    echo Error: Failed to combine audio with video!
    exit /b 1
)

echo ✅ Audio and video combined: !final_video!

:: Step 4: Generate captions
echo.
echo Step 4/4: Generating captions...
echo ========================================

:: Initialize conda and run WhisperX for subtitle generation
%WINDIR%\System32\WindowsPowerShell\v1.0\powershell.exe -ExecutionPolicy ByPass -Command "& { & 'C:\Users\deepak_bhattarai\miniconda3\shell\condabin\conda-hook.ps1'; conda activate 'C:\Users\deepak_bhattarai\miniconda3'; conda activate whisperx; Write-Host 'Generating subtitles with WhisperX...'; whisperx '!final_video!' --model large-v2 --output_format json --align_model WAV2VEC2_ASR_LARGE_LV60K_960H --highlight_words True --compute_type float16 --output_dir 'D:\Reddit-Romantics\Automation\caption'; Write-Host 'Converting JSON to ASS format...'; python .\caption\convert_json_to_ass.py 'D:\Reddit-Romantics\Automation\caption\temp_%base_filename%_final.json' 'D:\Reddit-Romantics\Automation\caption\temp_%base_filename%_final.ass' --max-words 4 --pause 0.5 --font-size 140; exit ;}"

if !errorlevel! neq 0 (
    echo Error: Caption generation failed!
    exit /b 1
)

echo ✅ Captions generated successfully

:: Step 5: Create final captioned video
echo.
echo Creating final captioned video (optimized for speed and high-res captions)...
echo ========================================

ffmpeg -y -hwaccel cuda -i "!final_video!" -vf "subtitles='./caption/temp_%base_filename%_final.ass'" -c:v h264_nvenc -preset p7 -rc constqp/cbr/vbr -cq 20 -b:v 5M -maxrate 10M -c:a copy "!output_video!"
if !errorlevel! neq 0 (
    echo Error: Failed to create final captioned video!
    exit /b 1
)

:: Clean up temporary files
echo.
echo Cleaning up temporary files...
if exist "!temp_video!" del "!temp_video!"
if exist "!final_video!" del "!final_video!"

echo ========================================
echo ✅ PROCESS COMPLETED SUCCESSFULLY! ✅
echo ========================================
echo.
echo 📁 Final output: !output_video!
echo 🎵 Audio: %audio_output% (Generated with Gemini TTS)
echo 📄 Captions: caption\temp_%base_filename%_final.ass
echo 🗣️ Voice: %voice% (Gemini TTS)
echo.
echo Your video is ready for upload!
echo ========================================

pause