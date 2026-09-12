@echo off
setlocal

echo ============================================
echo  Face Attendance System - Windows Setup
echo ============================================
echo.

set "PY_LAUNCH=py -3.10"

echo Using: %PY_LAUNCH%
echo.

if not exist venv (
    echo Creating virtual environment in .\venv ...
    %PY_LAUNCH% -m venv venv
) else (
    echo Virtual environment .\venv already exists, skipping creation.
)

call venv\Scripts\activate.bat

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing dependencies from requirements.txt ...
pip install -r requirements.txt

echo.
echo ============================================
echo  Setup complete.
echo  Run "run.bat" to start the application.
echo ============================================
pause