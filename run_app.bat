@echo off
echo ===================================================
echo Starting AI Sign Language App (Python 3.11 Environment)
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
echo Running main.py...
.venv\Scripts\python main.py
if %errorlevel% neq 0 (
    echo.
    echo Application exited with error code %errorlevel%
    pause
)
