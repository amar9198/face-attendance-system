@echo off
setlocal

if not exist venv\Scripts\activate.bat (
    echo Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo Starting Flask app...
echo Open http://localhost:5000 in your browser once it says "Running on".
echo Press CTRL+C to stop the server.
echo.

python app\main.py

pause
