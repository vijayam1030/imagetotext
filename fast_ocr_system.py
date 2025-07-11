#!/usr/bin/env python3
"""
Fast OCR System - Optimized for Speed
Streamlined version with the most effective configurations only
"""

import streamlit as st
import pytesseract
import cv2
import numpy as np
from PIL import Image
import ollama
import base64
import io
import re
import time
from datetime import datetime
from pathlib import Path

# Language detection imports
try:
    from guesslang import Guess
    GUESSLANG_AVAILABLE = True
except ImportError:
    GUESSLANG_AVAILABLE = False

try:
    from pygments.lexers import guess_lexer
    from pygments.util import ClassNotFound
    PYGMENTS_AVAILABLE = True
except ImportError:
    PYGMENTS_AVAILABLE = False

class FastLanguageDetector:
    """Fast language detection with minimal overhead"""
    
    def __init__(self):
        self.guesslang_model = None
        if GUESSLANG_AVAILABLE:
            try:
                self.guesslang_model = Guess()
            except:
                pass
        
        self.comment_styles = {
            'sql': '-- ',
            'python': '# ',
            'javascript': '// ',
            'java': '// ',
            'csharp': '// ',
            'cpp': '// ',
            'c': '// ',
            'text': '# '
        }
    
    def detect_language_fast(self, code):
        """Quick language detection using enhanced pattern matching"""
        # Try guesslang first (most accurate) if available
        if self.guesslang_model:
            try:
                language = self.guesslang_model.language_name(code)
                confidence = self.guesslang_model.language_confidence(code)
                if confidence > 0.5:
                    return language.lower(), confidence
            except:
                pass
        
        # Enhanced pattern matching as fallback
        code_upper = code.upper()
        
        # SQL patterns (expanded)
        sql_patterns = ['SELECT', 'FROM', 'WHERE', 'INSERT', 'UPDATE', 'DELETE', 'CREATE TABLE', 'ALTER', 'DROP', 'JOIN']
        sql_score = sum(1 for pattern in sql_patterns if pattern in code_upper)
        
        # Python patterns (expanded)
        python_patterns = ['DEF ', 'IMPORT ', 'PRINT(', 'IF __NAME__', 'CLASS ', 'FOR ', 'WHILE ', 'TRY:', 'EXCEPT:']
        python_score = sum(1 for pattern in python_patterns if pattern in code_upper)
        
        # JavaScript patterns (expanded)
        js_patterns = ['FUNCTION', 'VAR ', 'LET ', 'CONST ', 'CONSOLE.LOG', '=>', 'DOCUMENT.', 'WINDOW.']
        js_score = sum(1 for pattern in js_patterns if pattern in code_upper)
        
        # Java patterns
        java_patterns = ['PUBLIC CLASS', 'PUBLIC STATIC', 'SYSTEM.OUT', 'STRING ', 'VOID MAIN']
        java_score = sum(1 for pattern in java_patterns if pattern in code_upper)
        
        # C# patterns
        cs_patterns = ['USING SYSTEM', 'NAMESPACE ', 'CONSOLE.WRITELINE', 'PUBLIC CLASS']
        cs_score = sum(1 for pattern in cs_patterns if pattern in code_upper)
        
        scores = {
            'sql': sql_score, 
            'python': python_score, 
            'javascript': js_score,
            'java': java_score,
            'csharp': cs_score
        }
        
        if max(scores.values()) > 0:
            best_lang = max(scores, key=scores.get)
            confidence = min(scores[best_lang] / 3.0, 1.0)
            return best_lang, confidence
        
        return 'text', 0.5

class FastOCREngine:
    """Optimized OCR engine with only the most effective configurations"""
    
    def __init__(self):
        # Only the 3 most effective configurations
        self.fast_configs = [
            {'name': 'Default', 'config': r'--oem 3 --psm 6'},
            {'name': 'Single Column', 'config': r'--oem 3 --psm 4'},
            {'name': 'Code Optimized', 'config': r'--oem 3 --psm 6 -c preserve_interword_spaces=1'}
        ]
    
    def preprocess_image_fast(self, image):
        """Fast image preprocessing - only 2 most effective methods"""
        img_array = np.array(image)
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array.copy()
        
        # Only use 2 most effective preprocessing methods
        preprocessed = []
        
        # Method 1: Original with slight enhancement
        enhanced = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)
        preprocessed.append(('Enhanced', enhanced))
        
        # Method 2: Adaptive threshold (best for code)
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        preprocessed.append(('Adaptive', adaptive))
        
        return preprocessed
    
    def extract_text_fast(self, image):
        """Fast OCR extraction with early stopping"""
        preprocessed_images = self.preprocess_image_fast(image)
        
        best_result = ""
        best_length = 0
        
        # Try preprocessing methods
        for img_name, img in preprocessed_images:
            # Try OCR configurations
            for config in self.fast_configs:
                try:
                    text = pytesseract.image_to_string(img, config=config['config'])
                    text = text.strip()
                    
                    if len(text) > best_length:
                        best_result = text
                        best_length = len(text)
                        
                        # Early stopping if we get good result
                        if len(text) > 100:  # Good enough result
                            return {
                                'text': text,
                                'method': f"{img_name} + {config['name']}",
                                'length': len(text)
                            }
                            
                except Exception as e:
                    continue
        
        return {
            'text': best_result,
            'method': 'Best attempt',
            'length': len(best_result)
        }

class FastCodeCleaner:
    """Fast code cleaning with language-specific model selection"""
    
    def __init__(self):
        # Language-specific model preferences
        self.language_models = {
            'sql': [
                'deepseek-coder-v2:16b',  # Best for SQL
                'codellama:13b',
                'qwen2.5-coder:7b'
            ],
            'python': [
                'codellama:13b',          # Best for Python
                'deepseek-coder-v2:16b',
                'qwen2.5-coder:7b'
            ],
            'javascript': [
                'qwen2.5-coder:7b',       # Good for JS/web
                'codellama:13b',
                'deepseek-coder-v2:16b'
            ],
            'java': [
                'deepseek-coder-v2:16b',  # Good for enterprise languages
                'codellama:13b',
                'wizardcoder:15b'
            ],
            'csharp': [
                'deepseek-coder-v2:16b',  # Good for Microsoft stack
                'wizardcoder:15b',
                'codellama:13b'
            ],
            'default': [
                'phi3:medium',            # Fast general purpose
                'codellama:7b',
                'qwen2.5-coder:1.5b'
            ]
        }
        
        self.available_models = self.get_available_models()
        
    def get_available_models(self):
        """Get list of available Ollama models"""
        try:
            models = ollama.list()
            return [model['name'] for model in models['models']]
        except Exception as e:
            st.warning(f"Could not check models: {e}")
            return []
    
    def select_best_model_for_language(self, language):
        """Select the best available model for the detected language"""
        # Get preferred models for this language
        preferred = self.language_models.get(language, self.language_models['default'])
        
        # Find first available preferred model
        for model_name in preferred:
            for available in self.available_models:
                if model_name.lower() in available.lower():
                    return available
        
        # Fallback to any available model
        if self.available_models:
            return self.available_models[0]
        
        return None
    
    def create_detailed_overview(self, code, language, model):
        """Create comprehensive code overview"""
        if not model:
            return "No AI model available for analysis"
        
        try:
            prompt = f"""You are an expert {language.upper()} code analyst. Provide a comprehensive overview of this code.

ANALYSIS SECTIONS:
1. **Purpose & Functionality**: What does this code accomplish?
2. **Technical Architecture**: How is the code structured?
3. **Key Components**: Main functions, classes, or queries
4. **Data Flow**: How data moves through the code
5. **Dependencies**: Required libraries, databases, or systems
6. **Complexity Assessment**: Technical difficulty level
7. **Best Practices**: Code quality observations
8. **Potential Issues**: Any problems or improvements needed

Be detailed and technical. Focus on {language.upper()}-specific aspects.

{language.upper()} Code to analyze:
```{language}
{code}
```

Provide detailed analysis:"""

            response = ollama.chat(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    'num_gpu': -1,
                    'num_thread': 4,
                    'temperature': 0.1,
                    'num_predict': 2000,
                    'num_ctx': 4096,
                    'top_p': 0.9,
                },
                stream=False
            )
            
            return response['message']['content']
            
        except Exception as e:
            return f"Error creating overview: {str(e)}"
    
    def create_line_by_line_comments(self, code, language, model):
        """Create line-by-line code comments"""
        if not model:
            return "No AI model available for commenting"
        
        try:
            # Determine comment style
            comment_styles = {
                'sql': '-- ',
                'python': '# ',
                'javascript': '// ',
                'java': '// ',
                'csharp': '// ',
                'cpp': '// ',
                'c': '// ',
                'text': '# '
            }
            
            comment_prefix = comment_styles.get(language, '# ')
            
            prompt = f"""You are a {language.upper()} code documentation expert. Add a detailed comment above EVERY SINGLE LINE of code.

REQUIREMENTS:
1. For each line of code, write: {comment_prefix}[Explanation] then the actual code line
2. Comment EVERY line individually - no grouping
3. Be specific about what each line does
4. Use {comment_prefix.strip()} for comments (standard for {language.upper()})
5. Preserve original code exactly
6. Explain technical details, parameters, logic

FORMAT EXAMPLE:
{comment_prefix}Create a connection to the database server
connection = create_connection()
{comment_prefix}Execute a SELECT query to retrieve all user records
result = connection.execute("SELECT * FROM users")

{language.upper()} code to document:
```{language}
{code}
```

Add line-by-line comments:"""

            response = ollama.chat(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    'num_gpu': -1,
                    'num_thread': 4,
                    'temperature': 0.0,  # Very deterministic for consistency
                    'num_predict': 3000,
                    'num_ctx': 4096,
                    'top_p': 0.1,
                },
                stream=False
            )
            
            return response['message']['content']
            
        except Exception as e:
            return f"Error creating comments: {str(e)}"
    
    def clean_code_only(self, code, language, model):
        """Clean and fix code without adding comments"""
        if not model:
            return "No AI model available for code cleaning"
        
        try:
            prompt = f"""You are an expert {language.upper()} developer. Clean this OCR-extracted code by fixing errors and formatting issues.

CLEANING REQUIREMENTS:
1. Fix OCR character recognition errors
2. Correct syntax and formatting
3. Add proper indentation
4. Fix misread keywords and symbols
5. Follow {language.upper()} best practices
6. DO NOT add comments - only clean the code
7. Return ONLY the corrected code

{language.upper()} Code to clean:
```{language}
{code}
```

Cleaned code:"""

            response = ollama.chat(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    'num_gpu': -1,
                    'num_thread': 4,
                    'temperature': 0.1,
                    'num_predict': 1500,
                    'num_ctx': 4096,
                    'top_p': 0.9,
                },
                stream=False
            )
            
            return response['message']['content']
            
        except Exception as e:
            return f"Error cleaning code: {str(e)}"

class FastCodeSaver:
    """Simple file saving with basic extensions"""
    
    def __init__(self):
        self.save_dir = Path("extracted_code")
        self.save_dir.mkdir(exist_ok=True)
        
        self.extensions = {
            'sql': '.sql',
            'python': '.py', 
            'javascript': '.js',
            'java': '.java',
            'csharp': '.cs',
            'text': '.txt'
        }
    
    def save_code(self, code, language, prefix="extracted"):
        """Save code with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        extension = self.extensions.get(language, '.txt')
        filename = f"{prefix}_{language}_{timestamp}{extension}"
        filepath = self.save_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(code)
            return str(filepath)
        except Exception as e:
            return None

def main():
    """Fast OCR Streamlit app"""
    st.set_page_config(
        page_title="Fast OCR Code Extractor",
        page_icon="⚡",
        layout="wide"
    )
    
    st.title("⚡ Fast OCR Code Extractor")
    st.markdown("**Optimized for Speed - Essential Features Only**")
    
    # Initialize components
    detector = FastLanguageDetector()
    ocr_engine = FastOCREngine()
    cleaner = FastCodeCleaner()
    saver = FastCodeSaver()
    
    # Sidebar
    with st.sidebar:
        st.header("⚡ Fast Mode Settings")
        
        st.subheader("🖼️ Image Display")
        image_quality = st.selectbox(
            "Display Quality",
            ["High Quality", "Medium Quality", "Low Quality"],
            index=0
        )
        
        show_original_size = st.checkbox("Show full resolution", value=False)
        
        st.subheader("🚀 Performance")
        st.markdown("**Speed Optimizations:**")
        st.markdown("• 2 image preprocessing methods")
        st.markdown("• 3 OCR configurations") 
        st.markdown("• Early stopping on good results")
        st.markdown("• Language-specific AI model selection")
        
        st.subheader("🤖 Model Selection")
        st.markdown("**Language-Specific Models:**")
        st.markdown("• **SQL**: DeepSeek Coder (best for databases)")
        st.markdown("• **Python**: CodeLlama (optimized for Python)")
        st.markdown("• **JavaScript**: Qwen2.5 Coder (web-focused)")
        st.markdown("• **Java/C#**: DeepSeek (enterprise languages)")
        
        available_count = len(cleaner.available_models)
        st.markdown(f"**Available Models**: {available_count}")
        
        if available_count == 0:
            st.warning("No Ollama models available")
        
        auto_save = st.checkbox("Auto-save results", value=True)
        show_timing = st.checkbox("Show timing info", value=True)
        show_preprocessing = st.checkbox("Show preprocessing preview", value=False)
    
    # Main interface
    uploaded_file = st.file_uploader(
        "📤 Upload code image",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload image containing code or text"
    )
    
    if uploaded_file is not None:
        # Load and process image
        image = Image.open(uploaded_file)
        
        # Get image info
        width, height = image.size
        file_size = len(uploaded_file.getvalue()) / 1024  # KB
        
        # Display image with quality settings
        if show_original_size:
            # Show full resolution
            st.image(
                image, 
                caption=f"📸 Uploaded Image - {width}×{height}px ({file_size:.1f}KB)",
                use_column_width=False
            )
        else:
            # Determine display width based on quality setting
            if image_quality == "High Quality":
                display_width = min(800, width)
            elif image_quality == "Medium Quality":
                display_width = min(600, width)
            else:
                display_width = min(400, width)
            
            st.image(
                image, 
                caption=f"📸 Uploaded Image - {width}×{height}px ({file_size:.1f}KB)",
                width=display_width
            )
        
        # Show image details
        with st.expander("📊 Image Details"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Width", f"{width}px")
            with col2:
                st.metric("Height", f"{height}px")
            with col3:
                st.metric("Size", f"{file_size:.1f}KB")
            with col4:
                st.metric("Format", image.format or "Unknown")
        
        # Recommend optimal settings
        if width < 500 or height < 300:
            st.warning("⚠️ Low resolution image detected. OCR accuracy may be reduced.")
            st.info("💡 For best results, use images with at least 500×300 pixels.")
        elif file_size > 5000:  # > 5MB
            st.warning("⚠️ Large file size detected. Processing may be slower.")
            st.info("💡 Consider compressing the image for faster processing.")
        
        if st.button("⚡ Fast Extract & Clean", type="primary"):
            start_time = time.time()
            
            # Step 1: OCR
            with st.spinner("🔍 Extracting text..."):
                ocr_start = time.time()
                
                # Show preprocessing preview if requested
                if show_preprocessing:
                    st.subheader("🔧 Image Preprocessing Preview")
                    preprocessed_images = ocr_engine.preprocess_image_fast(image)
                    
                    cols = st.columns(len(preprocessed_images))
                    for i, (method_name, processed_img) in enumerate(preprocessed_images):
                        with cols[i]:
                            # Convert numpy array back to PIL Image for display
                            if isinstance(processed_img, np.ndarray):
                                if len(processed_img.shape) == 2:  # Grayscale
                                    pil_img = Image.fromarray(processed_img, mode='L')
                                else:
                                    pil_img = Image.fromarray(processed_img)
                            else:
                                pil_img = processed_img
                            
                            st.image(
                                pil_img, 
                                caption=f"🔧 {method_name}",
                                width=200
                            )
                
                ocr_result = ocr_engine.extract_text_fast(image)
                ocr_time = time.time() - ocr_start
                
                raw_text = ocr_result['text']
                
                if show_timing:
                    st.success(f"✅ OCR completed in {ocr_time:.2f}s using {ocr_result['method']}")
                
                if not raw_text.strip():
                    st.error("❌ No text extracted from image")
                    return
            
            # Step 2: Language Detection & Model Selection
            with st.spinner("🔍 Detecting language and selecting AI model..."):
                lang_start = time.time()
                detected_lang, confidence = detector.detect_language_fast(raw_text)
                
                # Select best model for detected language
                selected_model = cleaner.select_best_model_for_language(detected_lang)
                
                lang_time = time.time() - lang_start
                
                if show_timing:
                    st.success(f"✅ Language detection: {lang_time:.2f}s")
            
            # Display results
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                st.metric("🎯 Language", detected_lang.upper())
            with col2:
                st.metric("📊 Confidence", f"{confidence:.2f}")
            with col3:
                if selected_model:
                    st.metric("🤖 Selected Model", selected_model)
                else:
                    st.warning("❌ No AI model available")
            
            # Step 3: Display raw text
            st.subheader("📋 Extracted Text")
            st.code(raw_text, language=detected_lang)
            
            # Step 4: AI Analysis (Two Sections)
            if selected_model:
                # Section 1: Detailed Overview
                with st.spinner("📖 Creating detailed code overview..."):
                    overview_start = time.time()
                    overview = cleaner.create_detailed_overview(raw_text, detected_lang, selected_model)
                    overview_time = time.time() - overview_start
                    
                    if show_timing:
                        st.success(f"✅ Overview creation: {overview_time:.2f}s")
                
                st.subheader("📖 Detailed Code Overview")
                st.markdown(overview)
                
                # Section 2: Line-by-Line Comments
                with st.spinner("💬 Creating line-by-line comments..."):
                    comments_start = time.time()
                    commented_code = cleaner.create_line_by_line_comments(raw_text, detected_lang, selected_model)
                    comments_time = time.time() - comments_start
                    
                    if show_timing:
                        st.success(f"✅ Line-by-line comments: {comments_time:.2f}s")
                
                st.subheader("💬 Line-by-Line Commented Code")
                st.code(commented_code, language=detected_lang)
                
                # Save files
                if auto_save:
                    raw_path = saver.save_code(raw_text, detected_lang, "raw")
                    overview_path = saver.save_code(overview, detected_lang, "overview")
                    comments_path = saver.save_code(commented_code, detected_lang, "commented")
                    
                    saved_files = []
                    if raw_path: saved_files.append(raw_path)
                    if overview_path: saved_files.append(overview_path)
                    if comments_path: saved_files.append(comments_path)
                    
                    if saved_files:
                        st.success(f"💾 Files saved: {', '.join(saved_files)}")
                
                # Download buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.download_button(
                        "📥 Download Raw Code",
                        raw_text,
                        f"raw_{detected_lang}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{saver.extensions.get(detected_lang, 'txt')[1:]}",
                        mime="text/plain"
                    )
                with col2:
                    st.download_button(
                        "📥 Download Overview",
                        overview,
                        f"overview_{detected_lang}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/plain"
                    )
                with col3:
                    st.download_button(
                        "📥 Download Commented", 
                        commented_code,
                        f"commented_{detected_lang}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{saver.extensions.get(detected_lang, 'txt')[1:]}",
                        mime="text/plain"
                    )
            else:
                st.warning("⚠️ No AI models available for code cleaning")
                
                # Save raw text only
                if auto_save:
                    raw_path = saver.save_code(raw_text, detected_lang, "raw")
                    if raw_path:
                        st.success(f"💾 Raw text saved: {raw_path}")
                
                st.download_button(
                    "📥 Download Raw Text",
                    raw_text,
                    f"raw_{detected_lang}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{saver.extensions.get(detected_lang, 'txt')[1:]}",
                    mime="text/plain"
                )
            
            # Show total time
            total_time = time.time() - start_time
            if show_timing:
                st.info(f"⏱️ Total processing time: {total_time:.2f} seconds")
    
    # Footer
    st.markdown("---")
    st.markdown("**⚡ Fast Mode Features:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**OCR Optimization:**")
        st.markdown("• 2 preprocessing methods")
        st.markdown("• 3 OCR configurations")
        st.markdown("• Early stopping")
    
    with col2:
        st.markdown("**Language Detection:**")
        if GUESSLANG_AVAILABLE:
            st.markdown("• Guesslang ✅")
        else:
            st.markdown("• Enhanced patterns ✅")
        st.markdown("• Pygments fallback")
        st.markdown("• Fast confidence scoring")
    
    with col3:
        st.markdown("**AI Cleaning:**")
        st.markdown("• Single best model")
        st.markdown("• Optimized prompts")
        st.markdown("• Reduced context size")

if __name__ == "__main__":
    main()
