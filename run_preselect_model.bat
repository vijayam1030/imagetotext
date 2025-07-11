@echo off
echo ===================================================
echo Advanced OCR System - Pre-Select Model Version
echo ===================================================
echo.
echo 🎯 NEW FEATURE: PRE-ANALYSIS MODEL SELECTION
echo.
echo   • Select your AI model BEFORE uploading any image
echo   • See model size, speed, and best-use info upfront
echo   • Compare all available models side-by-side
echo   • Your selected model is used for all analysis tasks
echo.
echo 📊 Benefits:
echo   • Make informed decisions about processing time
echo   • Choose based on your specific code type
echo   • See resource requirements before starting
echo   • Consistent model across all analysis tasks
echo.
echo 🤖 Model Selection Features:
echo   • Size information (1.1GB to 19GB)
echo   • Speed indicators (Very Fast to Slow)
echo   • Best-for recommendations (Python, SQL, Web, etc.)
echo   • Quality ratings and descriptions
echo.
echo 📋 Workflow:
echo   1. Select your preferred AI model
echo   2. Upload image with code
echo   3. OCR extraction and language detection
echo   4. AI analysis with your chosen model
echo   5. Four expandable output sections
echo.
echo Starting pre-select model application...
echo Press Ctrl+C to stop
echo.
streamlit run advanced_ocr_preselect.py
pause
