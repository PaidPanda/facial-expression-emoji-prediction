@echo off

title Jetson Emotion Recognition Demo

:: FORCE working directory to the folder where this .bat is located
cd /d "%~dp0"

echo ======================================
echo       Jetson Emotion Recognition
echo ======================================
echo 1) Run Jetson webcam demo (jetson_demo.py)
echo 2) Exit

echo --------------------------------------
set /p choice="Enter your choice (1-2): "

if "%choice%"=="1" (
    echo Running Jetson webcam demo...
    echo Ensure 'emotionnet.engine' is in the project root
    python jetson_demo.py
    pause
    exit
)

if "%choice%"=="2" (
    echo Exiting...
    exit
)

echo Invalid choice.
pause
exit
