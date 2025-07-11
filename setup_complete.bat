@echo off
REM Advanced OCR System - Complete Setup and Run Script
REM This script will install dependencies, setup models, and run the system

echo ========================================
echo Advanced OCR Code Extraction System
echo Complete Setup and Installation
echo ========================================
echo.

REM Check if Python is installed
echo [1/8] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)
echo ✅ Python is installed
echo.

REM Check if pip is available
echo [2/8] Checking pip...
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: pip is not available
    echo Please ensure pip is installed with Python
    pause
    exit /b 1
)
echo ✅ pip is available
echo.

REM Upgrade pip
echo [3/8] Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install Python dependencies
echo [4/8] Installing Python packages...
echo This may take a few minutes...
pip install streamlit>=1.28.0
pip install pytesseract>=0.3.10
pip install opencv-python>=4.8.0
pip install ollama>=0.1.7
pip install numpy>=1.24.0
pip install Pillow>=10.0.0
pip install guesslang>=2.2.1
pip install pygments>=2.15.0
pip install regex>=2023.6.3
pip install requests>=2.31.0

echo ✅ Python packages installed
echo.

REM Check Tesseract installation
echo [5/8] Checking Tesseract OCR...
tesseract --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  WARNING: Tesseract OCR not found
    echo Please install Tesseract from:
    echo https://github.com/UB-Mannheim/tesseract/wiki
    echo.
    echo After installation, add Tesseract to your PATH
    echo Common path: C:\Program Files\Tesseract-OCR
    echo.
) else (
    echo ✅ Tesseract OCR is installed
)
echo.

REM Check Ollama installation
echo [6/8] Checking Ollama...
ollama list >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  WARNING: Ollama not found or not running
    echo Please:
    echo 1. Install Ollama from https://ollama.ai
    echo 2. Start Ollama service
    echo 3. Run this script again
    echo.
    pause
) else (
    echo ✅ Ollama is running
    echo.
    
    REM Pull required models
    echo [7/8] Downloading AI models...
    echo This will take a while (several GB of downloads)...
    echo.
    
    echo Pulling CodeLlama 13B...
    ollama pull codellama:13b
    
    echo Pulling DeepSeek Coder V2 16B...
    ollama pull deepseek-coder-v2:16b
    
    echo Pulling WizardCoder 34B...
    ollama pull wizardcoder:34b
    
    echo Pulling Phi-3 Medium...
    ollama pull phi3:medium
    
    echo Pulling Llama 3.2 Vision 11B...
    ollama pull llama3.2-vision:11b
    
    echo ✅ All models downloaded
)
echo.

REM Run setup script
echo [8/8] Running system setup...
python setup_advanced_ocr.py
echo.

REM Create test images
echo Creating test images...
python test_advanced_system.py
echo.

echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. To start the application: run_ocr_app.bat
echo 2. To test the system: test_ocr_system.bat
echo 3. To see all available commands: help_commands.bat
echo.
pause
