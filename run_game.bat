@echo off
setlocal
cd /d "%~dp0"

set "RUNNER="
set "LOCAL_PY=%~dp0venv\Scripts\python.exe"

REM Prefer the project's virtual environment when its Python installation is valid.
if exist "%LOCAL_PY%" (
    "%LOCAL_PY%" -c "import cv2, tensorflow; from fer.fer import FER" >nul 2>&1
    if not errorlevel 1 set "RUNNER="%LOCAL_PY%""
)

REM Fall back to installed Python versions that already have the required packages.
if not defined RUNNER (
    py -3.12 -c "import cv2, tensorflow; from fer.fer import FER" >nul 2>&1
    if not errorlevel 1 set "RUNNER=py -3.12"
)
if not defined RUNNER (
    py -3.11 -c "import cv2, tensorflow; from fer.fer import FER" >nul 2>&1
    if not errorlevel 1 set "RUNNER=py -3.11"
)
if not defined RUNNER (
    python -c "import cv2, tensorflow; from fer.fer import FER" >nul 2>&1
    if not errorlevel 1 set "RUNNER=python"
)

if not defined RUNNER (
    echo [ERROR] No Python installation with OpenCV, FER, and TensorFlow was found.
    echo.
    echo The game dependencies are not installed for Python 3.11 or 3.12.
    echo From this folder, set up the project environment with:
    echo   py -3.11 -m venv venv
    echo   venv\Scripts\python.exe -m pip install --upgrade pip
    echo   venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    echo After installation, run this file again.
    pause
    exit /b 1
)

echo Starting EMO RUSH!!!
%RUNNER% "%~dp0DRAFT0_EC.py"
if errorlevel 1 (
    echo.
    echo [ERROR] The game exited with an error. Check the message above.
    pause
)

endlocal
