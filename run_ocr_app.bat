@echo off
REM Start the Advanced OCR System Streamlit App

echo ========================================
echo Starting Advanced OCR System
echo ========================================
echo.
echo Opening in your default browser...
echo Press Ctrl+C to stop the application
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found
    echo Please run setup_complete.bat first
    pause
    exit /b 1
)

REM Check if Streamlit is installed
pip show streamlit >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Streamlit not installed
    echo Please run setup_complete.bat first
    pause
    exit /b 1
)

REM Start the Streamlit app
streamlit run advanced_ocr_system.py

echo.
echo Application stopped.
pause
