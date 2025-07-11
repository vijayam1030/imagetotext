@echo off
REM Help and Commands Reference for Advanced OCR System

echo ========================================
echo Advanced OCR System - Command Reference
echo ========================================
echo.

echo 🚀 MAIN COMMANDS:
echo ----------------
echo setup_complete.bat     - Complete system setup (run this first)
echo run_ocr_app.bat        - Start the full OCR application
echo run_fast_ocr.bat       - Start the FAST OCR application (recommended)
echo test_ocr_system.bat    - Test the system
echo help_commands.bat      - Show this help (current file)
echo.

echo 📦 INDIVIDUAL SETUP COMMANDS:
echo -----------------------------
echo python -m pip install --upgrade pip
echo pip install streamlit>=1.28.0
echo pip install pytesseract>=0.3.10
echo pip install opencv-python>=4.8.0
echo pip install ollama>=0.1.7
echo pip install numpy>=1.24.0
echo pip install Pillow>=10.0.0
echo pip install guesslang>=2.2.1
echo pip install pygments>=2.15.0
echo pip install regex>=2023.6.3
echo.

echo 🤖 OLLAMA MODEL COMMANDS:
echo -------------------------
echo ollama pull codellama:13b
echo ollama pull deepseek-coder-v2:16b
echo ollama pull wizardcoder:34b
echo ollama pull phi3:medium
echo ollama pull llama3.2-vision:11b
echo.

echo 🔧 MANUAL COMMANDS:
echo ------------------
echo python setup_advanced_ocr.py      - Run setup script
echo python test_advanced_system.py    - Create test images
echo streamlit run advanced_ocr_system.py - Start full app manually
echo streamlit run fast_ocr_system.py     - Start fast app manually
echo.

echo 📋 CHECK COMMANDS:
echo -----------------
echo python --version                  - Check Python version
echo pip --version                     - Check pip version
echo tesseract --version               - Check Tesseract OCR
echo ollama list                       - List available models
echo pip list                          - Show installed packages
echo.

echo 🔍 TROUBLESHOOTING:
echo ------------------
echo If Tesseract not found:
echo   - Download from: https://github.com/UB-Mannheim/tesseract/wiki
echo   - Add to PATH: C:\Program Files\Tesseract-OCR
echo.
echo If Ollama not found:
echo   - Download from: https://ollama.ai
echo   - Start Ollama service after installation
echo.
echo If Python packages fail:
echo   - Try: pip install --upgrade setuptools wheel
echo   - Try: pip install --no-cache-dir [package-name]
echo.

echo 📁 FILES CREATED:
echo ----------------
echo advanced_ocr_system.py           - Main application
echo setup_advanced_ocr.py            - Setup script
echo test_advanced_system.py          - Test script
echo requirements_advanced.txt        - Dependencies
echo README_Advanced_OCR.md           - Documentation
echo.
echo Test images (created by test script):
echo   test_sql_code.png
echo   test_python_code.png
echo   test_javascript_code.png
echo.
echo Extracted files will be saved in:
echo   extracted_code/ directory
echo.

echo ========================================
echo Quick Start (FAST VERSION):
echo 1. Run: quick_install.bat
echo 2. Run: run_fast_ocr.bat
echo 3. Upload an image and test!
echo ========================================
echo.
echo Quick Start (FULL VERSION):
echo 1. Run: setup_complete.bat
echo 2. Run: run_ocr_app.bat  
echo 3. Upload an image and test!
echo ========================================
echo.
pause
