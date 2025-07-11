@echo off
echo ===============================================
echo   Guesslang Installation Helper
echo ===============================================
echo.

echo This script helps install guesslang for enhanced language detection.
echo Guesslang has strict TensorFlow dependencies that may conflict.
echo.

echo Choose installation method:
echo 1. Try compatible TensorFlow version (recommended)
echo 2. Force specific versions (may break other packages)
echo 3. Create separate environment for guesslang
echo 4. Skip guesslang (system works well without it)
echo.

set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo.
    echo Installing with compatible TensorFlow...
    pip install "tensorflow>=2.5.0,<2.16.0"
    pip install guesslang
    echo.
    echo Testing installation...
    python -c "from guesslang import Guess; print('✅ Guesslang installed successfully!')"
) else if "%choice%"=="2" (
    echo.
    echo Installing specific versions...
    pip install tensorflow==2.5.0
    pip install guesslang==2.2.1
    echo.
    echo Testing installation...
    python -c "from guesslang import Guess; print('✅ Guesslang installed successfully!')"
) else if "%choice%"=="3" (
    echo.
    echo Creating separate environment...
    python -m venv venv_guesslang
    call venv_guesslang\Scripts\activate
    pip install -r requirements_with_guesslang.txt
    echo.
    echo ✅ Separate environment created: venv_guesslang
    echo To use: call venv_guesslang\Scripts\activate
    echo Then run: streamlit run advanced_ocr_system.py
) else if "%choice%"=="4" (
    echo.
    echo ✅ Skipping guesslang installation.
    echo Your system will use pygments + pattern matching for language detection.
    echo This provides good accuracy without dependency conflicts.
) else (
    echo Invalid choice. Please run again and choose 1-4.
)

echo.
echo ===============================================
echo Installation complete!
echo ===============================================
echo.

if "%choice%"=="1" or "%choice%"=="2" (
    echo You can now run:
    echo   streamlit run advanced_ocr_system.py
    echo   streamlit run fast_ocr_system.py
    echo.
)

echo To test the system:
echo   python test_enhanced_system.py
echo.

pause
