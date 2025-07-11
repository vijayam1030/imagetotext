#!/usr/bin/env python3
"""
Advanced OCR System with Multi-File Upload Support
Enhanced version that can process multiple images simultaneously
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
from concurrent.futures import ThreadPoolExecutor
import zipfile

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

def process_single_image(image, filename, detector, ocr_engine, cleaner, selected_model, show_ocr_details=False):
    """Process a single image and return results"""
    try:
        # OCR Processing
        ocr_results = ocr_engine.extract_text(image)
        
        if not ocr_results:
            return {
                'filename': filename,
                'status': 'error',
                'error': 'No text could be extracted from the image',
                'raw_text': '',
                'detected_lang': 'unknown',
                'confidence': 0.0
            }
        
        best_ocr = ocr_results[0]
        raw_text = best_ocr['text']
        
        if not raw_text.strip():
            return {
                'filename': filename,
                'status': 'error',
                'error': 'No text detected in the image',
                'raw_text': '',
                'detected_lang': 'unknown',
                'confidence': 0.0
            }
        
        # Language Detection
        detected_lang, confidence, all_scores = detector.detect_language(raw_text)
        
        # AI Analysis
        try:
            detailed_overview = cleaner.create_detailed_overview(raw_text, detected_lang, selected_model)
        except Exception as e:
            detailed_overview = f"Error generating overview: {str(e)}"
        
        try:
            line_comments = cleaner.create_line_by_line_comments(raw_text, detected_lang, selected_model)
        except Exception as e:
            line_comments = raw_text
        
        try:
            cleaned_code = cleaner.clean_code_only(raw_text, detected_lang, selected_model)
        except Exception as e:
            cleaned_code = raw_text
        
        return {
            'filename': filename,
            'status': 'success',
            'raw_text': raw_text,
            'detected_lang': detected_lang,
            'confidence': confidence,
            'all_scores': all_scores,
            'detailed_overview': detailed_overview,
            'line_comments': line_comments,
            'cleaned_code': cleaned_code,
            'ocr_info': best_ocr,
            'characters': len(raw_text),
            'lines': len(raw_text.split('\n'))
        }
        
    except Exception as e:
        return {
            'filename': filename,
            'status': 'error',
            'error': str(e),
            'raw_text': '',
            'detected_lang': 'unknown',
            'confidence': 0.0
        }

def create_batch_download_zip(results, timestamp):
    """Create a ZIP file with all processed results"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Create summary file
        summary_content = f"# Batch OCR Processing Summary\n"
        summary_content += f"**Processing Time:** {timestamp}\n"
        summary_content += f"**Total Files:** {len(results)}\n\n"
        
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'error']
        
        summary_content += f"**Successful:** {len(successful)}\n"
        summary_content += f"**Failed:** {len(failed)}\n\n"
        
        if successful:
            summary_content += "## Successfully Processed Files:\n"
            for result in successful:
                summary_content += f"- **{result['filename']}** ({result['detected_lang']}, {result['characters']} chars)\n"
        
        if failed:
            summary_content += "\n## Failed Files:\n"
            for result in failed:
                summary_content += f"- **{result['filename']}** - {result['error']}\n"
        
        zip_file.writestr("BATCH_SUMMARY.md", summary_content)
        
        # Add individual file results
        for result in results:
            if result['status'] == 'success':
                base_name = os.path.splitext(result['filename'])[0]
                
                # Overview
                zip_file.writestr(f"{base_name}_overview.md", result['detailed_overview'])
                
                # Comments
                extension = '.py' if result['detected_lang'] == 'python' else '.txt'
                zip_file.writestr(f"{base_name}_comments{extension}", result['line_comments'])
                
                # Cleaned code
                zip_file.writestr(f"{base_name}_cleaned{extension}", result['cleaned_code'])
                
                # Raw text
                zip_file.writestr(f"{base_name}_raw.txt", result['raw_text'])
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def main():
    """Main Streamlit application with multi-file support"""
    st.set_page_config(
        page_title="Advanced OCR Code Extractor - Multi-File",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔍 Advanced OCR Code Extractor - Multi-File Support")
    st.markdown("**Process Multiple Images Simultaneously with Batch Analysis**")
    
    # Initialize core components
    detector = AdvancedLanguageDetector()
    ocr_engine = TesseractOCREngine()
    cleaner = OllamaCodeCleaner()
    saver = CodeSaver()
    
    # Check available models
    if not cleaner.available_models:
        st.error("❌ No Ollama models found. Please install and start Ollama first.")
        st.info("💡 Install Ollama from https://ollama.ai and run: `ollama pull codellama:7b`")
        return
    
    # Model selection
    st.subheader("🤖 Step 1: Choose Your AI Model")
    model_info = get_model_info_with_sizes()
    
    # Create model options
    model_options = {}
    for model in cleaner.available_models:
        base_name = model.split(':')[0] + ':' + model.split(':')[1] if ':' in model else model
        
        if base_name in model_info:
            info = model_info[base_name]
            display_name = f"{model} ({info['size']}, {info['speed']}) - {', '.join(info['best_for'][:2])}"
            model_options[display_name] = model
        else:
            display_name = f"{model} (Size unknown)"
            model_options[display_name] = model
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_display = st.selectbox(
            "🎯 Select AI Model for Batch Analysis:",
            options=list(model_options.keys()),
            index=0,
            help="This model will be used for all uploaded images"
        )
        selected_model = model_options[selected_display]
    
    with col2:
        # Show model details
        base_name = selected_model.split(':')[0] + ':' + selected_model.split(':')[1] if ':' in selected_model else selected_model
        if base_name in model_info:
            info = model_info[base_name]
            st.markdown("**📋 Selected Model:**")
            st.markdown(f"**Size:** {info['size']}")
            st.markdown(f"**Speed:** {info['speed']}")
            st.markdown(f"**Best for:** {', '.join(info['best_for'])}")
    
    # Configuration sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Selected model display
        st.subheader("🎯 Selected Model")
        st.success(f"**{selected_model}**")
        if base_name in model_info:
            info = model_info[base_name]
            st.info(f"Size: {info['size']} | Speed: {info['speed']}")
        
        # Processing settings
        st.subheader("🔄 Processing Settings")
        max_concurrent = st.slider("Max concurrent processing", 1, 5, 2, help="Number of images to process simultaneously")
        show_individual_progress = st.checkbox("Show individual image progress", value=True)
        
        # File saving settings
        st.subheader("📁 Output Settings")
        auto_save = st.checkbox("Auto-save all results", value=True)
        create_zip_download = st.checkbox("Create batch download ZIP", value=True)
        
        # Display settings
        st.subheader("🖼️ Display Settings")
        show_image_previews = st.checkbox("Show image previews", value=True)
        preview_width = st.slider("Preview width", 100, 400, 200)
        show_ocr_details = st.checkbox("Show OCR details", value=False)
    
    # Multi-file upload
    st.subheader("📤 Step 2: Upload Multiple Images")
    uploaded_files = st.file_uploader(
        "Upload multiple images containing code",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'gif'],
        accept_multiple_files=True,
        help="Select multiple images to process in batch"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} images uploaded")
        
        # Show file summary
        total_size = sum(len(f.getvalue()) for f in uploaded_files) / 1024  # KB
        st.info(f"📊 Total size: {total_size:.1f}KB | Average: {total_size/len(uploaded_files):.1f}KB per image")
        
        # Preview uploaded images
        if show_image_previews:
            with st.expander(f"🖼️ Preview {len(uploaded_files)} Uploaded Images", expanded=False):
                cols = st.columns(min(4, len(uploaded_files)))
                for i, uploaded_file in enumerate(uploaded_files):
                    with cols[i % 4]:
                        image = Image.open(uploaded_file)
                        st.image(image, caption=uploaded_file.name, width=preview_width)
                        st.caption(f"{image.size[0]}×{image.size[1]}px")
        
        # Processing button
        if st.button(f"🚀 Process {len(uploaded_files)} Images with {selected_model}", type="primary"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Results container
            results = []
            
            # Process images
            if max_concurrent == 1:
                # Sequential processing
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name} ({i+1}/{len(uploaded_files)})")
                    
                    image = Image.open(uploaded_file)
                    result = process_single_image(image, uploaded_file.name, detector, ocr_engine, cleaner, selected_model, show_ocr_details)
                    results.append(result)
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    if show_individual_progress:
                        if result['status'] == 'success':
                            st.success(f"✅ {uploaded_file.name}: {result['detected_lang']} ({result['characters']} chars)")
                        else:
                            st.error(f"❌ {uploaded_file.name}: {result['error']}")
            
            else:
                # Parallel processing
                status_text.text("Processing images in parallel...")
                
                def process_wrapper(uploaded_file):
                    image = Image.open(uploaded_file)
                    return process_single_image(image, uploaded_file.name, detector, ocr_engine, cleaner, selected_model, show_ocr_details)
                
                with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                    # Submit all tasks
                    future_to_file = {executor.submit(process_wrapper, f): f for f in uploaded_files}
                    
                    # Collect results as they complete
                    completed = 0
                    for future in future_to_file:
                        result = future.result()
                        results.append(result)
                        completed += 1
                        
                        progress_bar.progress(completed / len(uploaded_files))
                        
                        if show_individual_progress:
                            if result['status'] == 'success':
                                st.success(f"✅ {result['filename']}: {result['detected_lang']} ({result['characters']} chars)")
                            else:
                                st.error(f"❌ {result['filename']}: {result['error']}")
            
            # Processing complete
            status_text.text("🎉 Batch processing complete!")
            
            # Results summary
            successful = [r for r in results if r['status'] == 'success']
            failed = [r for r in results if r['status'] == 'error']
            
            st.subheader("📊 Batch Processing Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📁 Total Files", len(uploaded_files))
            with col2:
                st.metric("✅ Successful", len(successful))
            with col3:
                st.metric("❌ Failed", len(failed))
            with col4:
                success_rate = (len(successful) / len(uploaded_files)) * 100
                st.metric("📈 Success Rate", f"{success_rate:.1f}%")
            
            # Language distribution
            if successful:
                lang_counts = {}
                for result in successful:
                    lang = result['detected_lang']
                    lang_counts[lang] = lang_counts.get(lang, 0) + 1
                
                st.subheader("🔍 Detected Languages")
                lang_cols = st.columns(min(4, len(lang_counts)))
                for i, (lang, count) in enumerate(lang_counts.items()):
                    with lang_cols[i % 4]:
                        st.metric(f"📝 {lang.upper()}", count)
            
            # Batch download
            if create_zip_download and successful:
                st.subheader("📦 Batch Download")
                zip_data = create_batch_download_zip(results, timestamp)
                st.download_button(
                    label=f"📦 Download All Results ({len(successful)} files)",
                    data=zip_data,
                    file_name=f"batch_ocr_results_{timestamp}.zip",
                    mime="application/zip"
                )
            
            # Individual results
            st.subheader("📋 Individual Results")
            
            # Create tabs for each successful result
            if successful:
                tab_names = [f"📄 {result['filename']}" for result in successful[:10]]  # Limit to 10 tabs
                if len(successful) > 10:
                    tab_names.append(f"📋 +{len(successful) - 10} more")
                
                tabs = st.tabs(tab_names)
                
                for i, result in enumerate(successful[:10]):  # Show first 10 in tabs
                    with tabs[i]:
                        st.markdown(f"### 📄 {result['filename']}")
                        
                        # File summary
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("🔤 Language", result['detected_lang'].upper())
                        with col2:
                            st.metric("📊 Confidence", f"{result['confidence']:.2f}")
                        with col3:
                            st.metric("📏 Characters", result['characters'])
                        
                        # Analysis sections
                        with st.expander("📊 Detailed Overview", expanded=True):
                            st.markdown(result['detailed_overview'])
                            
                            if auto_save:
                                overview_filepath = saver.save_code(
                                    result['detailed_overview'], 'md', 
                                    f"overview_{os.path.splitext(result['filename'])[0]}", timestamp
                                )
                        
                        with st.expander("💬 Line-by-Line Comments", expanded=False):
                            st.code(result['line_comments'], language=result['detected_lang'])
                            
                            if auto_save:
                                comments_filepath = saver.save_code(
                                    result['line_comments'], result['detected_lang'], 
                                    f"commented_{os.path.splitext(result['filename'])[0]}", timestamp
                                )
                        
                        with st.expander("🔧 Cleaned Code", expanded=False):
                            st.code(result['cleaned_code'], language=result['detected_lang'])
                            
                            if auto_save:
                                cleaned_filepath = saver.save_code(
                                    result['cleaned_code'], result['detected_lang'], 
                                    f"cleaned_{os.path.splitext(result['filename'])[0]}", timestamp
                                )
                        
                        with st.expander("📄 Raw OCR Output", expanded=False):
                            st.code(result['raw_text'], language='text')
                            
                            if auto_save:
                                raw_filepath = saver.save_code(
                                    result['raw_text'], result['detected_lang'], 
                                    f"raw_{os.path.splitext(result['filename'])[0]}", timestamp
                                )
                
                # Show remaining results in expander if more than 10
                if len(successful) > 10:
                    with tabs[-1]:
                        st.markdown(f"### 📋 Remaining {len(successful) - 10} Results")
                        for result in successful[10:]:
                            with st.expander(f"📄 {result['filename']}", expanded=False):
                                st.markdown(f"**Language:** {result['detected_lang'].upper()}")
                                st.markdown(f"**Characters:** {result['characters']}")
                                st.markdown(f"**Confidence:** {result['confidence']:.2f}")
                                
                                # Quick preview of overview
                                overview_preview = result['detailed_overview'][:500] + "..." if len(result['detailed_overview']) > 500 else result['detailed_overview']
                                st.markdown("**Overview Preview:**")
                                st.markdown(overview_preview)
            
            # Failed results
            if failed:
                st.subheader("❌ Failed Processes")
                for result in failed:
                    st.error(f"**{result['filename']}**: {result['error']}")
    
    # Footer
    st.markdown("---")
    st.markdown("**🔧 Multi-File Processing System:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Batch Features:**")
        st.markdown("• Multiple file upload")
        st.markdown("• Parallel processing")
        st.markdown("• Progress tracking")
        st.markdown("• Batch downloads")
    
    with col2:
        st.markdown("**OCR & Analysis:**")
        st.markdown("• Tesseract OCR engine")
        st.markdown("• Language detection")
        st.markdown("• AI-powered analysis")
        st.markdown("• Error handling")
    
    with col3:
        st.markdown("**Output Options:**")
        st.markdown("• Individual results")
        st.markdown("• ZIP batch download")
        st.markdown("• Auto-save files")
        st.markdown("• Processing summary")

if __name__ == "__main__":
    main()
