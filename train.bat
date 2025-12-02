@echo off
title Facial Expression Trainer

:: FORCE working directory to the folder where this .bat is located
cd /d "%~dp0"

echo ======================================
echo       Facial Expression Trainer
echo ======================================
echo 1) Run multi-model training (main.py)
echo 2) Run final CNN training (cNN.py)
echo 3) Run webcam demo (webcam_demo.py)
echo 4) Exit
echo --------------------------------------
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo Running multi-model training...
    python -m training_models.main
    pause
    exit
)

if "%choice%"=="2" (
    echo Running final CNN training...
    python cNN.py
    pause
    exit
)

if "%choice%"=="3" (
    echo Running webcam demo...
    echo Ensure 'my_emotion_recognizer_best_69.4%.keras' is in the root folder
    python webcam_demo.py
    pause
    exit
)

echo Invalid choice.
pause
exit
