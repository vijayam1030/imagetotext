#!/usr/bin/env python3
"""
Advanced OCR System with Expandable UI Sections
Enhanced version with expandable sections instead of tabs
"""

import streamlit as st
import pytesseract
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import ollama
import base64
import io
import re
import json
import time
from datetime import datetime
import os
from pathlib import Path

# Language detection imports
try:
    from guesslang import Guess
    GUESSLANG_AVAILABLE = True
except ImportError:
    GUESSLANG_AVAILABLE = False
    st.info("ℹ️ Guesslang not available (has TensorFlow dependencies). Using enhanced pattern matching + Pygments for language detection.")

try:
    from pygments.lexers import guess_lexer, get_lexer_by_name
    from pygments.util import ClassNotFound
    from pygments.formatters import get_formatter_by_name
    PYGMENTS_AVAILABLE = True
except ImportError:
    PYGMENTS_AVAILABLE = False
    st.warning("⚠️ Pygments not available. Install with: pip install pygments")

# Import classes from the original advanced system
from advanced_ocr_system import (
    AdvancedLanguageDetector,
    TesseractOCREngine, 
    OllamaCodeCleaner,
    CodeSaver
)

def main():
    """Main Streamlit application with expandable sections"""
    st.set_page_config(
        page_title="Advanced OCR Code Extractor",
        page_icon="🔧",
        layout="wide"
    )
    
    st.title("🔧 Advanced OCR Code Extractor")
    st.markdown("**Tesseract OCR + Multiple AI Models + Language Detection + Code Cleanup**")
    st.markdown("*Now with expandable sections for better viewing!*")
    
    # Initialize components
    detector = AdvancedLanguageDetector()
    ocr_engine = TesseractOCREngine()
    cleaner = OllamaCodeCleaner()
    saver = CodeSaver()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("🖼️ Image Display")
        image_display_width = st.slider(
            "Display Width", 
            min_value=300, 
            max_value=1200, 
            value=800, 
            step=100,
            help="Adjust image display width"
        )
        
        show_image_details = st.checkbox("Show image analysis", value=True)
        
        st.subheader("🤖 Available Models")
        if cleaner.available_models:
            for model in cleaner.available_models[:5]:  # Show first 5
                st.markdown(f"• `{model}`")
            if len(cleaner.available_models) > 5:
                st.markdown(f"• ... and {len(cleaner.available_models) - 5} more")
        else:
            st.error("❌ No Ollama models found")
        
        st.subheader("🔍 Language Detection")
        if GUESSLANG_AVAILABLE:
            st.markdown("• **Guesslang**: ✅ (ML-based)")
        else:
            st.markdown("• **Guesslang**: ❌ (TensorFlow conflicts)")
        st.markdown(f"• **Pygments**: {'✅' if PYGMENTS_AVAILABLE else '❌'} (Lexer-based)")
        st.markdown("• **Pattern Matching**: ✅ (Enhanced rules)")
        
        if not GUESSLANG_AVAILABLE:
            with st.expander("ℹ️ About Guesslang"):
                st.markdown("""
                Guesslang is a machine learning library for language detection but has strict TensorFlow dependencies that can conflict with other packages.
                
                **Our system works excellently without it** using:
                - **Pygments lexer analysis** (very accurate)
                - **Enhanced pattern matching** (SQL, Python, JavaScript, Java, C#)
                - **Confidence scoring** system
                
                You can install guesslang in a separate environment if needed, but it's not required for good results.
                """)
        
        st.subheader("📁 File Saving")
        auto_save = st.checkbox("Auto-save extracted code", value=True)
        
        st.subheader("🎛️ OCR Settings")
        show_ocr_details = st.checkbox("Show OCR attempt details", value=False)
    
    # Main interface
    uploaded_file = st.file_uploader(
        "📤 Upload an image containing code",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'gif'],
        help="Upload an image with code or script content"
    )
    
    if uploaded_file is not None:
        # Load and process image
        image = Image.open(uploaded_file)
        
        # Get image info
        width, height = image.size
        file_size = len(uploaded_file.getvalue()) / 1024  # KB
        
        # Display image with better quality settings
        if image_display_width < width:
            st.image(
                image, 
                caption=f"📸 Uploaded Image - {width}×{height}px ({file_size:.1f}KB)",
                width=image_display_width
            )
        else:
            st.image(
                image, 
                caption=f"📸 Uploaded Image - {width}×{height}px ({file_size:.1f}KB)",
                use_column_width=True
            )
        
        # Show image details
        if show_image_details:
            with st.expander("📊 Image Analysis", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Width", f"{width}px")
                with col2:
                    st.metric("Height", f"{height}px")
                with col3:
                    st.metric("Size", f"{file_size:.1f}KB")
                with col4:
                    st.metric("Format", image.format or "Unknown")
                
                # Image quality recommendations
                if width < 500 or height < 300:
                    st.warning("⚠️ Low resolution detected. Consider using a higher resolution image for better OCR accuracy.")
                elif width > 2000 or height > 2000:
                    st.info("💡 High resolution image. This may take longer to process but should give better results.")
                
                if file_size > 5000:
                    st.warning("⚠️ Large file size. Processing may be slower.")
        
        if st.button("🚀 Extract and Clean Code", type="primary"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Step 1: OCR Extraction
            st.subheader("📝 Step 1: OCR Text Extraction")
            with st.spinner("Extracting text with Tesseract OCR..."):
                ocr_results = ocr_engine.extract_text(image)
            
            if not ocr_results:
                st.error("❌ No text could be extracted from the image")
                return
            
            # Show best OCR result
            best_ocr = ocr_results[0]
            raw_text = best_ocr['text']
            
            st.success(f"✅ Extracted {len(raw_text)} characters using {best_ocr['preprocessing']} + {best_ocr['config']}")
            
            if show_ocr_details:
                with st.expander("🔍 OCR Attempt Details", expanded=False):
                    for i, result in enumerate(ocr_results[:5]):  # Show top 5
                        st.markdown(f"**#{i+1}: {result['preprocessing']} + {result['config']} ({result['length']} chars)**")
                        st.code(result['text'][:200] + "..." if len(result['text']) > 200 else result['text'])
            
            # Step 2: Language Detection
            st.subheader("🔍 Step 2: Language Detection")
            with st.spinner("Detecting programming language..."):
                detected_lang, confidence, all_scores = detector.detect_language(raw_text)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("🎯 Detected Language", detected_lang.upper())
                st.metric("📊 Confidence", f"{confidence:.2f}")
            
            with col2:
                st.markdown("**🏆 Detection Scores:**")
                for lang, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:5]:
                    st.markdown(f"• **{lang.title()}**: {score:.3f}")
            
            # Step 3: AI Analysis and Code Processing
            st.subheader("🧹 Step 3: AI Analysis with Language-Specific Models")
            
            if not cleaner.available_models:
                st.error("❌ No Ollama models available for analysis")
                st.info("💡 Please install Ollama models like: `ollama pull codellama:7b`")
                
                # Show only raw text if no AI available
                with st.expander("📋 Raw Extracted Code", expanded=True):
                    st.code(raw_text, language=detected_lang)
                    
                    if auto_save:
                        filepath = saver.save_code(raw_text, detected_lang, "raw_ocr", timestamp)
                        if filepath:
                            st.success(f"💾 Saved to: {filepath}")
                    
                    st.download_button(
                        label="📄 Download Raw Text",
                        data=raw_text,
                        file_name=f"raw_ocr_{timestamp}.txt",
                        mime="text/plain"
                    )
                return
            
            # Select best model for the detected language
            selected_model = cleaner.select_best_model_for_language(detected_lang)
            
            if not selected_model:
                st.error("❌ No suitable AI model found")
                return
            
            st.info(f"🎯 Using **{selected_model}** for {detected_lang.upper()} code analysis")
            
            # Generate analysis sections
            col1, col2 = st.columns(2)
            
            with col1:
                with st.spinner("📊 Creating detailed overview..."):
                    detailed_overview = cleaner.create_detailed_overview(raw_text, detected_lang, selected_model)
                
            with col2:
                with st.spinner("💬 Creating line-by-line comments..."):
                    line_comments = cleaner.create_line_by_line_comments(raw_text, detected_lang, selected_model)
            
            # Display results in expandable sections
            st.subheader("📋 Analysis Results")
            st.markdown("*Click on each section to expand/collapse*")
            
            # Section 1: Detailed Overview (expanded by default)
            with st.expander("📊 Detailed Code Overview", expanded=True):
                st.markdown(f"**Analyzed by:** {selected_model}")
                st.markdown(f"**Language:** {detected_lang.upper()}")
                st.markdown("---")
                st.markdown(detailed_overview)
                
                # Save and download options
                col1, col2 = st.columns(2)
                with col1:
                    if auto_save:
                        overview_filepath = saver.save_code(
                            detailed_overview, 'md', 
                            f"overview_{detected_lang}", timestamp
                        )
                        if overview_filepath:
                            st.success(f"💾 Overview saved to: {overview_filepath}")
                
                with col2:
                    st.download_button(
                        label="📊 Download Overview",
                        data=detailed_overview,
                        file_name=f"code_overview_{detected_lang}_{timestamp}.md",
                        mime="text/markdown"
                    )
            
            # Section 2: Line-by-Line Comments
            with st.expander("💬 Line-by-Line Educational Comments", expanded=False):
                st.markdown(f"**Documented by:** {selected_model}")
                st.markdown(f"**Language:** {detected_lang.upper()}")
                st.markdown("---")
                st.code(line_comments, language=detected_lang)
                
                # Save and download options
                col1, col2 = st.columns(2)
                with col1:
                    if auto_save:
                        comments_filepath = saver.save_code(
                            line_comments, detected_lang, 
                            f"commented_{detected_lang}", timestamp
                        )
                        if comments_filepath:
                            st.success(f"💾 Comments saved to: {comments_filepath}")
                
                with col2:
                    extension = saver.extensions.get(detected_lang, '.txt')
                    st.download_button(
                        label="💬 Download Comments",
                        data=line_comments,
                        file_name=f"commented_code_{detected_lang}_{timestamp}{extension}",
                        mime="text/plain"
                    )
            
            # Section 3: Cleaned Code
            with st.expander("🔧 Cleaned Code (No Comments)", expanded=False):
                st.markdown(f"**Processed by:** {selected_model}")
                st.markdown(f"**Language:** {detected_lang.upper()}")
                st.markdown("---")
                
                # Create cleaned version of the raw code
                with st.spinner("Cleaning extracted code..."):
                    cleaned_code = cleaner.clean_code_only(raw_text, detected_lang, selected_model)
                
                st.code(cleaned_code, language=detected_lang)
                
                # Save and download options
                col1, col2 = st.columns(2)
                with col1:
                    if auto_save:
                        cleaned_filepath = saver.save_code(
                            cleaned_code, detected_lang, 
                            f"cleaned_{detected_lang}", timestamp
                        )
                        if cleaned_filepath:
                            st.success(f"💾 Cleaned code saved to: {cleaned_filepath}")
                
                with col2:
                    extension = saver.extensions.get(detected_lang, '.txt')
                    st.download_button(
                        label="🔧 Download Clean Code",
                        data=cleaned_code,
                        file_name=f"cleaned_code_{detected_lang}_{timestamp}{extension}",
                        mime="text/plain"
                    )
            
            # Section 4: Original Text
            with st.expander("📄 Raw OCR Output", expanded=False):
                st.markdown(f"**Extracted by:** {best_ocr['preprocessing']} + {best_ocr['config']}")
                st.markdown(f"**Characters:** {len(raw_text)}")
                st.markdown("---")
                st.code(raw_text, language='text')
                
                # Save and download options
                col1, col2 = st.columns(2)
                with col1:
                    if auto_save:
                        raw_filepath = saver.save_code(
                            raw_text, detected_lang, 
                            "raw_ocr", timestamp
                        )
                        if raw_filepath:
                            st.success(f"💾 Raw text saved to: {raw_filepath}")
                
                with col2:
                    st.download_button(
                        label="📄 Download Raw Text",
                        data=raw_text,
                        file_name=f"raw_ocr_{timestamp}.txt",
                        mime="text/plain"
                    )
    
    # Footer
    st.markdown("---")
    st.markdown("**🔧 System Components:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**OCR Engine:**")
        st.markdown("• Tesseract Core")
        st.markdown("• PyTesseract Wrapper")
        st.markdown("• 7 OCR Configurations")
        st.markdown("• 6 Image Preprocessing")
    
    with col2:
        st.markdown("**Language Detection:**")
        st.markdown("• Guesslang ML Model")
        st.markdown("• Pygments Lexer")
        st.markdown("• Pattern Matching")
        st.markdown("• Confidence Scoring")
    
    with col3:
        st.markdown("**AI Code Cleanup:**")
        st.markdown("• CodeLlama")
        st.markdown("• DeepSeek Coder")
        st.markdown("• WizardCoder")
        st.markdown("• Phi-3")

if __name__ == "__main__":
    main()
