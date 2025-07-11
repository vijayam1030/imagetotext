@echo off
REM Test the Advanced OCR System

echo ========================================
echo Testing Advanced OCR System
echo ========================================
echo.

REM Run the test script
echo Running system tests...
python test_advanced_system.py

echo.
echo ========================================
echo Manual Testing Instructions:
echo ========================================
echo.
echo 1. Start the app: run_ocr_app.bat
echo 2. Upload one of these test images:
echo    - test_sql_code.png
echo    - test_python_code.png  
echo    - test_javascript_code.png
echo 3. Click "Extract and Clean Code"
echo 4. Review the results
echo.

REM Check if test images exist
if exist test_sql_code.png (
    echo ✅ SQL test image available
) else (
    echo ⚠️  SQL test image not found
)

if exist test_python_code.png (
    echo ✅ Python test image available
) else (
    echo ⚠️  Python test image not found
)

if exist test_javascript_code.png (
    echo ✅ JavaScript test image available
) else (
    echo ⚠️  JavaScript test image not found
)

echo.
pause
