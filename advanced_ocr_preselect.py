#!/usr/bin/env python3
"""
Advanced OCR System with Pre-Analysis Model Selection
Enhanced version where users can select the AI model before starting analysis
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

try:
    from pygments.lexers import guess_lexer, get_lexer_by_name
    from pygments.util import ClassNotFound
    from pygments.formatters import get_formatter_by_name
    PYGMENTS_AVAILABLE = True
except ImportError:
    PYGMENTS_AVAILABLE = False

# Import classes from the original advanced system
from advanced_ocr_system import (
    AdvancedLanguageDetector,
    TesseractOCREngine, 
    OllamaCodeCleaner,
    CodeSaver
)

def get_model_info_with_sizes():
    """Get model information with sizes and descriptions"""
    model_info = {
        'codellama:7b': {
            'size': '3.8GB',
            'speed': 'Fast',
            'description': 'Good general coding model, fast inference',
            'best_for': ['Python', 'JavaScript', 'General']
        },
        'codellama:13b': {
            'size': '7.3GB', 
            'speed': 'Medium',
            'description': 'Better quality, balanced speed/performance',
            'best_for': ['Python', 'Java', 'C++']
        },
        'codellama:34b': {
            'size': '19GB',
            'speed': 'Slow',
            'description': 'Highest quality, slower inference',
            'best_for': ['Complex code', 'Architecture']
        },
        'deepseek-coder-v2:16b': {
            'size': '9.1GB',
            'speed': 'Medium',
            'description': 'Excellent for enterprise languages',
            'best_for': ['SQL', 'Java', 'C#', 'Enterprise']
        },
        'qwen2.5-coder:7b': {
            'size': '4.2GB',
            'speed': 'Fast',
            'description': 'Great for web development',
            'best_for': ['JavaScript', 'TypeScript', 'Web']
        },
        'qwen2.5-coder:1.5b': {
            'size': '1.1GB',
            'speed': 'Very Fast',
            'description': 'Lightweight, very fast',
            'best_for': ['Quick analysis', 'Low memory']
        },
        'wizardcoder:34b': {
            'size': '19GB',
            'speed': 'Slow', 
            'description': 'High quality instruction following',
            'best_for': ['Complex tasks', 'Documentation']
        },
        'phi3:medium': {
            'size': '7.9GB',
            'speed': 'Fast',
            'description': 'Microsoft model, good general purpose',
            'best_for': ['General', 'Fast analysis']
        },
        'phi3:mini': {
            'size': '2.3GB',
            'speed': 'Very Fast',
            'description': 'Compact but capable',
            'best_for': ['Quick tasks', 'Resource limited']
        }
    }
    return model_info

def main():
    """Main Streamlit application with pre-analysis model selection"""
    st.set_page_config(
        page_title="Advanced OCR Code Extractor",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔍 Advanced OCR Code Extractor")
    st.markdown("**Tesseract OCR + Multiple AI Models + Language Detection + Code Cleanup**")
    
    # Initialize core components
    detector = AdvancedLanguageDetector()
    ocr_engine = TesseractOCREngine()
    cleaner = OllamaCodeCleaner()
    saver = CodeSaver()
    
    # Check available models first
    if not cleaner.available_models:
        st.error("❌ No Ollama models found. Please install and start Ollama first.")
        st.info("💡 Install Ollama from https://ollama.ai and run: `ollama pull codellama:7b`")
        return
    
    # PRE-ANALYSIS MODEL SELECTION (MAIN FEATURE)
    st.subheader("🤖 Step 1: Choose Your AI Model")
    st.markdown("**Select the AI model that will analyze your code before uploading any image:**")
    
    model_info = get_model_info_with_sizes()
    
    # Create model options with detailed info
    model_options = {}
    default_model = cleaner.available_models[0]  # First available as default
    
    for model in cleaner.available_models:
        base_name = model.split(':')[0] + ':' + model.split(':')[1] if ':' in model else model
        
        if base_name in model_info:
            info = model_info[base_name]
            display_name = f"{model} ({info['size']}, {info['speed']}) - {', '.join(info['best_for'][:2])}"
            model_options[display_name] = model
        else:
            display_name = f"{model} (Size unknown)"
            model_options[display_name] = model
    
    # Model selection at the top
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_display = st.selectbox(
            "🎯 Select AI Model for Analysis:",
            options=list(model_options.keys()),
            index=0,
            help="This model will be used for all AI analysis tasks: overview, comments, and code cleaning"
        )
        selected_model = model_options[selected_display]
        
        # Store in session state
        st.session_state.selected_model = selected_model
    
    with col2:
        # Show detailed model information
        base_name = selected_model.split(':')[0] + ':' + selected_model.split(':')[1] if ':' in selected_model else selected_model
        if base_name in model_info:
            info = model_info[base_name]
            st.markdown("**📋 Selected Model:**")
            st.markdown(f"**Size:** {info['size']}")
            st.markdown(f"**Speed:** {info['speed']}")
            st.markdown(f"**Best for:** {', '.join(info['best_for'])}")
            
            with st.expander("📖 Model Description"):
                st.markdown(info['description'])
        else:
            st.markdown("**📋 Selected Model:**")
            st.markdown(f"**Name:** {selected_model}")
            st.markdown("**Info:** Available for use")
    
    # Model comparison guide
    with st.expander("📊 Compare All Available Models", expanded=False):
        st.markdown("### 🤖 Model Comparison:")
        
        # Create comparison table
        col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 2, 1])
        
        with col1:
            st.markdown("**Model**")
        with col2:
            st.markdown("**Size**")
        with col3:
            st.markdown("**Speed**")
        with col4:
            st.markdown("**Best For**")
        with col5:
            st.markdown("**Quality**")
        
        st.markdown("---")
        
        for model in cleaner.available_models:
            base_name = model.split(':')[0] + ':' + model.split(':')[1] if ':' in model else model
            
            col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 2, 1])
            
            with col1:
                if model == selected_model:
                    st.markdown(f"**{model}** ⭐")
                else:
                    st.markdown(f"**{model}**")
            
            if base_name in model_info:
                info = model_info[base_name]
                with col2:
                    st.markdown(info['size'])
                with col3:
                    st.markdown(info['speed'])
                with col4:
                    st.markdown(', '.join(info['best_for'][:2]))
                with col5:
                    if 'Very Fast' in info['speed']:
                        st.markdown('Good')
                    elif 'Fast' in info['speed']:
                        st.markdown('Very Good')
                    elif 'Medium' in info['speed']:
                        st.markdown('Excellent')
                    else:
                        st.markdown('Outstanding')
            else:
                with col2:
                    st.markdown('Unknown')
                with col3:
                    st.markdown('Unknown')
                with col4:
                    st.markdown('General')
                with col5:
                    st.markdown('Unknown')
    
    # Configuration sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Show selected model prominently
        st.subheader("🎯 Selected Model")
        st.success(f"**{selected_model}**")
        if base_name in model_info:
            info = model_info[base_name]
            st.info(f"Size: {info['size']} | Speed: {info['speed']}")
        
        # Image display settings
        st.subheader("🖼️ Image Display")
        image_display_width = st.slider("Image display width", 200, 800, 400)
        show_image_details = st.checkbox("Show image details", value=False)
        
        # File saving settings
        st.subheader("📁 File Saving")
        auto_save = st.checkbox("Auto-save extracted code", value=True)
        
        # OCR settings
        st.subheader("🎛️ OCR Settings")
        show_ocr_details = st.checkbox("Show OCR attempt details", value=False)
        
        # Language detection info
        if not GUESSLANG_AVAILABLE:
            with st.expander("ℹ️ About Language Detection"):
                st.markdown("""
                Using enhanced pattern matching + Pygments for language detection.
                This provides excellent accuracy without TensorFlow dependencies.
                """)
    
    # Main interface
    st.subheader("📤 Step 2: Upload Image with Code")
    uploaded_file = st.file_uploader(
        "Upload an image containing code",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'gif'],
        help="Upload an image with code or script content"
    )
    
    if uploaded_file is not None:
        # Load and process image
        image = Image.open(uploaded_file)
        
        # Get image info
        width, height = image.size
        file_size = len(uploaded_file.getvalue()) / 1024  # KB
        
        # Display image
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
        
        # Show image details if requested
        if show_image_details:
            with st.expander("📊 Image Analysis", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📏 Width", f"{width}px")
                with col2:
                    st.metric("📐 Height", f"{height}px")
                with col3:
                    st.metric("💾 File Size", f"{file_size:.1f}KB")
                with col4:
                    st.metric("🎨 Mode", image.mode)
        
        # Processing button with selected model info
        if st.button(f"🚀 Extract & Analyze with {selected_model}", type="primary"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Step 3: OCR Processing
            st.subheader("🔍 Step 3: OCR Text Extraction")
            
            with st.spinner("Extracting text from image..."):
                ocr_results = ocr_engine.extract_text(image)
                
                if not ocr_results:
                    st.error("❌ No text could be extracted from the image")
                    return
                
                best_ocr = ocr_results[0]
                raw_text = best_ocr['text']
                
                if show_ocr_details:
                    with st.expander("🔍 OCR Attempt Details", expanded=False):
                        st.markdown(f"**Best Configuration:** {best_ocr['preprocessing']} + {best_ocr['config']}")
                        st.markdown(f"**Characters Extracted:** {len(raw_text)}")
                        st.markdown(f"**Lines:** {best_ocr.get('lines', 'N/A')}")
                        
                        if len(ocr_results) > 1:
                            st.markdown("**All OCR Attempts:**")
                            for i, result in enumerate(ocr_results[:5]):
                                st.markdown(f"*Attempt {i+1}:* {result['preprocessing']} + {result['config']} → {result['length']} chars")
            
            if not raw_text.strip():
                st.error("❌ No text detected in the image. Try adjusting the image quality.")
                return
            
            st.success(f"✅ Extracted {len(raw_text)} characters")
            
            # Step 4: Language Detection
            st.subheader("🔍 Step 4: Programming Language Detection")
            
            with st.spinner("Detecting programming language..."):
                detected_lang, confidence, all_scores = detector.detect_language(raw_text)
            
            if detected_lang:
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"🎯 Detected Language: **{detected_lang.upper()}**")
                    st.metric("📊 Confidence", f"{confidence:.2f}")
                
                with col2:
                    st.markdown("**🏆 Detection Scores:**")
                    for lang, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:5]:
                        st.markdown(f"• **{lang.title()}**: {score:.3f}")
                
                # Step 5: AI Analysis
                st.subheader(f"🧹 Step 5: AI Analysis with {selected_model}")
                st.info(f"🎯 Using your pre-selected model: **{selected_model}** for {detected_lang.upper()} code analysis")
                
                # Generate analysis sections
                with st.spinner("Generating comprehensive analysis..."):
                    try:
                        detailed_overview = cleaner.create_detailed_overview(raw_text, detected_lang, selected_model)
                    except Exception as e:
                        st.error(f"Error generating overview: {e}")
                        detailed_overview = "Error generating detailed overview."
                    
                    try:
                        line_comments = cleaner.create_line_by_line_comments(raw_text, detected_lang, selected_model)
                    except Exception as e:
                        st.error(f"Error generating comments: {e}")
                        line_comments = raw_text
                
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
        st.markdown("• Multiple Configurations")
        st.markdown("• Advanced Preprocessing")
    
    with col2:
        st.markdown("**Language Detection:**")
        st.markdown("• Pattern Matching")
        st.markdown("• Pygments Lexer")
        st.markdown("• Confidence Scoring")
    
    with col3:
        st.markdown("**AI Analysis:**")
        st.markdown(f"• Selected: {selected_model}")
        st.markdown("• Overview Generation")
        st.markdown("• Code Documentation")
        st.markdown("• Error Correction")

if __name__ == "__main__":
    main()
