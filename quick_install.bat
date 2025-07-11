@echo off
REM Quick Install - Essential packages only

echo ========================================
echo Quick Install - Essential Packages
echo ========================================
echo.

echo Installing essential packages only...
echo This is faster than the full setup.
echo.

pip install streamlit
pip install pytesseract
pip install opencv-python
pip install ollama
pip install numpy
pip install Pillow

echo.
echo ✅ Essential packages installed
echo.
echo For full setup with all features, run: setup_complete.bat
echo To start the app: run_ocr_app.bat
echo.
pause
