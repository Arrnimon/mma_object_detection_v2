@echo off
echo ========================================
echo Video Object Detection Setup
echo ========================================
echo.

echo Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH.
    echo Please install Python 3.8+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo Python found!
python --version

echo.
echo Installing required packages...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo Installation failed. Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installation completed successfully!
echo ========================================
echo.
echo To run the GUI application:
echo   python gui_detector.py
echo.
echo To run command line detection:
echo   python object_detector.py your_video.mp4
echo.
echo To process multiple videos:
echo   python batch_processor.py input_folder output_folder
echo.
pause 