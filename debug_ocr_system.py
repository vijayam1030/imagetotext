#!/usr/bin/env python3
"""
Debug version of OCR system to identify UI issues
"""

import streamlit as st
import pytesseract
import cv2
import numpy as np
from PIL import Image
import ollama
from datetime import datetime
from pathlib import Path

# Simplified imports
try:
    from advanced_ocr_system import (
        AdvancedLanguageDetector, 
        TesseractOCREngine, 
        OllamaCodeCleaner, 
        CodeSaver
    )
    st.success("✅ All imports successful")
except Exception as e:
    st.error(f"❌ Import error: {e}")

def main():
    """Debug OCR app to test tab functionality"""
    st.set_page_config(
        page_title="Debug OCR System",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Debug OCR System - Tab Testing")
    st.markdown("**This is a debug version to test the Cleaned Code tab**")
    
    # Test components
    st.subheader("🧪 Component Tests")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        try:
            detector = AdvancedLanguageDetector()
            st.success("✅ Language Detector")
        except Exception as e:
            st.error(f"❌ Detector: {e}")
    
    with col2:
        try:
            ocr_engine = TesseractOCREngine()
            st.success("✅ OCR Engine")
        except Exception as e:
            st.error(f"❌ OCR: {e}")
    
    with col3:
        try:
            cleaner = OllamaCodeCleaner()
            if hasattr(cleaner, 'clean_code_only'):
                st.success("✅ Code Cleaner + clean_code_only")
            else:
                st.error("❌ Missing clean_code_only method")
        except Exception as e:
            st.error(f"❌ Cleaner: {e}")
    
    with col4:
        try:
            saver = CodeSaver()
            st.success("✅ File Saver")
        except Exception as e:
            st.error(f"❌ Saver: {e}")
    
    # Test tabs creation
    st.subheader("📋 Tab Creation Test")
    
    # Create test tabs exactly like the main app
    tabs = st.tabs([
        "📊 Detailed Overview", 
        "💬 Line-by-Line Comments", 
        "🔧 Cleaned Code", 
        "📄 Original Text"
    ])
    
    # Test each tab
    with tabs[0]:
        st.markdown("### 📊 Test Overview Tab")
        st.markdown("This is the **Detailed Overview** tab content.")
        st.code("# This is test overview content\nprint('Overview tab working!')")
    
    with tabs[1]:
        st.markdown("### 💬 Test Comments Tab")
        st.markdown("This is the **Line-by-Line Comments** tab content.")
        st.code("""# This is a comment for line 1
print('Hello World')
# This is a comment for line 2  
result = 42""")
    
    with tabs[2]:
        st.markdown("### 🔧 Test Cleaned Code Tab")
        st.markdown("This is the **Cleaned Code** tab content.")
        
        if st.button("🧪 Test clean_code_only method"):
            try:
                cleaner = OllamaCodeCleaner()
                test_code = "print('hello world')\nresult = 1 + 1"
                
                with st.spinner("Testing code cleaning..."):
                    cleaned = cleaner.clean_code_only(test_code, "python", "phi3:medium")
                
                st.success("✅ clean_code_only method worked!")
                st.code(cleaned, language="python")
                
            except Exception as e:
                st.error(f"❌ clean_code_only failed: {e}")
        
        st.code("""# This is test cleaned code
print('Hello World')
result = 42
print(f'Result: {result}')""")
    
    with tabs[3]:
        st.markdown("### 📄 Test Original Tab")
        st.markdown("This is the **Original Text** tab content.")
        st.code("This is the raw OCR output text...")
    
    # Instructions
    st.subheader("🎯 Debug Instructions")
    st.markdown("""
    **Check the following:**
    
    1. **Can you see all 4 tabs above?**
       - 📊 Detailed Overview
       - 💬 Line-by-Line Comments  
       - 🔧 Cleaned Code
       - 📄 Original Text
    
    2. **Can you click on the "🔧 Cleaned Code" tab?**
    
    3. **Does the "Test clean_code_only method" button work?**
    
    4. **Are all components showing ✅ green checkmarks above?**
    
    If any of these fail, that's where the issue is!
    """)
    
    # Model check
    st.subheader("🤖 Model Availability")
    try:
        models = ollama.list()
        available_models = [model['name'] for model in models['models']]
        
        if available_models:
            st.success(f"✅ Found {len(available_models)} Ollama models")
            with st.expander("📋 Available Models"):
                for model in available_models[:10]:  # Show first 10
                    st.markdown(f"• `{model}`")
                if len(available_models) > 10:
                    st.markdown(f"• ... and {len(available_models) - 10} more")
        else:
            st.warning("⚠️ No Ollama models found")
            
    except Exception as e:
        st.error(f"❌ Ollama connection failed: {e}")

if __name__ == "__main__":
    main()
