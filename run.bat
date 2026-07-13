@echo off
title Instagram Auto-Drawing Tool
echo ===================================================
echo   Starting Instagram Auto-Drawing Tool Setup...
echo ===================================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in your system PATH.
    echo Please install Python from https://www.python.org/ and check
    echo the box to "Add Python to PATH" during installation.
    echo.
    pause
    exit /b
)

:: Install dependencies
echo [1/3] Checking and installing dependencies...
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [WARNING] Failed to install dependencies. Make sure you have internet access.
)
echo.

:: Check if adb is connected
echo [2/3] Checking ADB connection...
adb devices > temp_devices.txt
findstr /i "device" temp_devices.txt >nul
if %errorlevel% neq 0 (
    echo [WARNING] No Android device detected.
    echo Please connect your phone via USB with USB Debugging enabled.
)
del temp_devices.txt
echo.

:: Prompt for image path
echo [3/3] Ready to run!
set /p img_path="Enter the path to your image (or drag and drop it here): "

:: Remove quotes if drag-and-dropped
set img_path=%img_path:"=%

if not exist "%img_path%" (
    echo [ERROR] Image file does not exist at: %img_path%
    pause
    exit /b
)

:: Run script
echo Starting the interactive draw process...
python draw_interactive.py "%img_path%"

echo.
echo Process complete.
pause
