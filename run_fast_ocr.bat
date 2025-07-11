@echo off
REM Fast OCR System - Quick Start

echo ========================================
echo Fast OCR System - Speed Optimized
echo ========================================
echo.

echo ⚡ Starting Fast OCR System...
echo This version is optimized for speed:
echo • Only 2 image preprocessing methods
echo • Only 3 OCR configurations  
echo • Early stopping on good results
echo • Single AI model for cleaning
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found
    echo Please run quick_install.bat first
    pause
    exit /b 1
)

REM Start the fast OCR system
streamlit run fast_ocr_system.py

echo.
echo Fast OCR system stopped.
pause
