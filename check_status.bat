@echo off
REM Check System Status for Advanced OCR System

echo ========================================
echo Advanced OCR System - Status Check
echo ========================================
echo.

echo 🔍 SYSTEM STATUS:
echo ----------------

REM Check Python
echo Checking Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python: Not installed
) else (
    echo ✅ Python: Installed
    python --version
)
echo.

REM Check pip
echo Checking pip...
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ pip: Not available
) else (
    echo ✅ pip: Available
    pip --version
)
echo.

REM Check Tesseract
echo Checking Tesseract OCR...
tesseract --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Tesseract: Not installed or not in PATH
) else (
    echo ✅ Tesseract: Installed
    tesseract --version 2>nul | findstr "tesseract"
)
echo.

REM Check Ollama
echo Checking Ollama...
ollama list >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Ollama: Not running or not installed
) else (
    echo ✅ Ollama: Running
    echo Available models:
    ollama list
)
echo.

REM Check Python packages
echo Checking Python packages...
pip show streamlit >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Streamlit: Not installed
) else (
    echo ✅ Streamlit: Installed
)

pip show pytesseract >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ PyTesseract: Not installed
) else (
    echo ✅ PyTesseract: Installed
)

pip show opencv-python >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ OpenCV: Not installed
) else (
    echo ✅ OpenCV: Installed
)

pip show ollama >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Ollama Python: Not installed
) else (
    echo ✅ Ollama Python: Installed
)

pip show guesslang >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Guesslang: Not installed
) else (
    echo ✅ Guesslang: Installed
)

pip show pygments >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Pygments: Not installed
) else (
    echo ✅ Pygments: Installed
)
echo.

REM Check files
echo Checking system files...
if exist advanced_ocr_system.py (
    echo ✅ Main application: Found
) else (
    echo ❌ Main application: Missing
)

if exist setup_advanced_ocr.py (
    echo ✅ Setup script: Found
) else (
    echo ❌ Setup script: Missing
)

if exist test_advanced_system.py (
    echo ✅ Test script: Found
) else (
    echo ❌ Test script: Missing
)

if exist requirements_advanced.txt (
    echo ✅ Requirements file: Found
) else (
    echo ❌ Requirements file: Missing
)
echo.

REM Check test images
echo Checking test images...
if exist test_sql_code.png (
    echo ✅ SQL test image: Found
) else (
    echo ❌ SQL test image: Missing
)

if exist test_python_code.png (
    echo ✅ Python test image: Found
) else (
    echo ❌ Python test image: Missing
)

if exist test_javascript_code.png (
    echo ✅ JavaScript test image: Found
) else (
    echo ❌ JavaScript test image: Missing
)
echo.

echo ========================================
echo Status Check Complete
echo ========================================
echo.
echo If any items show ❌, run setup_complete.bat
echo.
pause
