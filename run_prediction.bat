@echo off
echo ===================================================
echo Starting Real-Time Prediction (Python 3.11 Environment)
echo ===================================================
echo.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    echo Please run the setup script first.
    pause
    exit /b
)

echo Activating environment...
echo Running predict_realtime.py...
.venv\Scripts\python predict_realtime.py
if %errorlevel% neq 0 (
    echo.
    echo Application exited with error code %errorlevel%
    pause
)
