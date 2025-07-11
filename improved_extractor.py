#!/usr/bin/env python3
"""
Improved Image to Text Extractor with Better Language Detection and Commenting
"""
import streamlit as st
import requests
import base64
import io
from PIL import Image, ImageEnhance, ImageFilter
import ollama
import time
import pytesseract
import cv2
import numpy as np
import re

class LanguageDetector:
    """Enhanced language detection with better accuracy"""
    
    def __init__(self):
        self.patterns = {
            'sql': {
                'keywords': [
                    r'\bSELECT\b', r'\bFROM\b', r'\bWHERE\b', r'\bINSERT\b', r'\bUPDATE\b', 
                    r'\bDELETE\b', r'\bCREATE\s+TABLE\b', r'\bALTER\s+TABLE\b', r'\bDROP\b',
                    r'\bJOIN\b', r'\bINNER\s+JOIN\b', r'\bLEFT\s+JOIN\b', r'\bRIGHT\s+JOIN\b',
                    r'\bGROUP\s+BY\b', r'\bORDER\s+BY\b', r'\bHAVING\b', r'\bUNION\b'
                ],
                'comment_style': '-- ',
                'score_multiplier': 2
            },
            'python': {
                'keywords': [
                    r'\bdef\s+\w+\(', r'\bclass\s+\w+', r'\bimport\s+\w+', r'\bfrom\s+\w+\s+import\b',
                    r'\bif\s+__name__\s*==\s*["\']__main__["\']', r'\bprint\s*\(', r'\breturn\b',
                    r'\bfor\s+\w+\s+in\b', r'\bwhile\s+', r'\btry\s*:', r'\bexcept\b', r'\bfinally\b'
                ],
                'comment_style': '# ',
                'score_multiplier': 1.5
            },
            'javascript': {
                'keywords': [
                    r'\bfunction\s+\w+', r'\bvar\s+\w+', r'\blet\s+\w+', r'\bconst\s+\w+',
                    r'\bconsole\.log\s*\(', r'\breturn\s+', r'\bif\s*\(', r'\bfor\s*\(',
                    r'\bwhile\s*\(', r'=>', r'\bdocument\.', r'\bwindow\.'
                ],
                'comment_style': '// ',
                'score_multiplier': 1
            },
            'java': {
                'keywords': [
                    r'\bpublic\s+class\b', r'\bprivate\s+', r'\bprotected\s+', r'\bpublic\s+static\s+void\s+main\b',
                    r'\bSystem\.out\.println\s*\(', r'\bString\s+\w+', r'\bint\s+\w+', r'\bboolean\s+\w+'
                ],
                'comment_style': '// ',
                'score_multiplier': 1
            },
            'csharp': {
                'keywords': [
                    r'\busing\s+System', r'\bpublic\s+class\b', r'\bprivate\s+', r'\bprotected\s+',
                    r'\bConsole\.WriteLine\s*\(', r'\bstring\s+\w+', r'\bint\s+\w+', r'\bnamespace\s+\w+'
                ],
                'comment_style': '// ',
                'score_multiplier': 1
            }
        }
    
    def detect_language(self, text):
        """Detect programming language with improved accuracy"""
        text_upper = text.upper()
        scores = {}
        
        for lang, config in self.patterns.items():
            score = 0
            for pattern in config['keywords']:
                matches = len(re.findall(pattern, text, re.IGNORECASE | re.MULTILINE))
                score += matches * config['score_multiplier']
            scores[lang] = score
        
        # Additional heuristics
        if ';' in text and text.count(';') > text.count('\n') * 0.3:
            scores['sql'] = scores.get('sql', 0) * 1.5
            scores['javascript'] = scores.get('javascript', 0) * 1.2
            scores['java'] = scores.get('java', 0) * 1.2
            scores['csharp'] = scores.get('csharp', 0) * 1.2
        
        if ':' in text and text.count(':') > text.count(';'):
            scores['python'] = scores.get('python', 0) * 1.3
        
        detected_lang = max(scores, key=scores.get) if any(scores.values()) else 'unknown'
        confidence = scores.get(detected_lang, 0)
        
        return detected_lang, confidence, scores

class ImprovedExtractor:
    """Enhanced text extraction with multiple approaches"""
    
    def __init__(self):
        self.detector = LanguageDetector()
    
    def enhance_image_for_ocr(self, image):
        """Advanced image preprocessing for better OCR"""
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Convert to grayscale
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array.copy()
            
            # Multiple enhancement approaches
            enhanced_versions = []
            
            # Version 1: Standard enhancement
            enhanced1 = cv2.convertScaleAbs(gray, alpha=1.2, beta=20)
            enhanced_versions.append(enhanced1)
            
            # Version 2: High contrast
            enhanced2 = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)
            enhanced_versions.append(enhanced2)
            
            # Version 3: Adaptive threshold
            adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            enhanced_versions.append(adaptive)
            
            # Version 4: Morphological operations
            kernel = np.ones((1,1), np.uint8)
            morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            enhanced_versions.append(morph)
            
            return enhanced_versions
            
        except Exception as e:
            return [np.array(image.convert('L'))]
    
    def extract_with_multiple_ocr(self, image):
        """Try multiple OCR configurations"""
        enhanced_images = self.enhance_image_for_ocr(image)
        
        # Different OCR configurations
        ocr_configs = [
            r'--oem 3 --psm 6',  # Single uniform block
            r'--oem 3 --psm 4',  # Single column
            r'--oem 3 --psm 3',  # Fully automatic
            r'--oem 3 --psm 1',  # Automatic with OSD
            r'--oem 3 --psm 11', # Sparse text
            r'--oem 3 --psm 13'  # Raw line
        ]
        
        best_result = ""
        best_length = 0
        
        for img in enhanced_images:
            for config in ocr_configs:
                try:
                    result = pytesseract.image_to_string(img, config=config)
                    if len(result.strip()) > best_length:
                        best_result = result.strip()
                        best_length = len(result.strip())
                except:
                    continue
        
        return best_result
    
    def extract_with_vision_model(self, image, detected_language="unknown"):
        """Enhanced vision model extraction with language hints"""
        try:
            image_b64 = self.encode_image_to_base64(image)
            
            language_hint = ""
            if detected_language != "unknown":
                language_hint = f"This appears to be {detected_language.upper()} code. "
            
            prompt = f"""{language_hint}You are a precise OCR system. Extract ALL visible text exactly as it appears.

CRITICAL INSTRUCTIONS:
1. Read EVERY character, symbol, and line of text
2. Preserve EXACT formatting, indentation, and spacing
3. Include ALL punctuation, brackets, quotes, and special characters
4. Maintain original line breaks and structure
5. Do NOT interpret, explain, or modify anything
6. Do NOT add comments or explanations
7. Copy text character-for-character as it appears
8. If you see code, copy it exactly including all syntax

Extract the text:"""
            
            response = ollama.chat(
                model="llama3.2-vision:11b",
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                        'images': [image_b64]
                    }
                ],
                options={
                    'num_gpu': -1,
                    'num_thread': 2,
                    'temperature': 0.0,
                    'num_predict': 1500,
                    'num_ctx': 2048,
                    'top_p': 0.05,
                    'repeat_penalty': 1.0,
                },
                stream=False
            )
            
            return response['message']['content'].strip()
            
        except Exception as e:
            return f"Vision model error: {str(e)}"
    
    def encode_image_to_base64(self, image):
        """Convert PIL image to base64"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    def extract_text_hybrid(self, image):
        """Hybrid extraction using both OCR and vision model"""
        print("🔍 Starting hybrid text extraction...")
        
        # Step 1: Try OCR first
        print("📝 Attempting OCR extraction...")
        ocr_result = self.extract_with_multiple_ocr(image)
        
        # Step 2: Quick language detection on OCR result
        if ocr_result and len(ocr_result.strip()) > 10:
            detected_lang, confidence, scores = self.detector.detect_language(ocr_result)
            print(f"🔍 Detected language: {detected_lang} (confidence: {confidence})")
        else:
            detected_lang = "unknown"
        
        # Step 3: Try vision model
        print("👁️ Attempting vision model extraction...")
        vision_result = self.extract_with_vision_model(image, detected_lang)
        
        # Step 4: Choose best result
        if len(vision_result) > len(ocr_result) and not vision_result.startswith("Vision model error"):
            print("✅ Using vision model result")
            final_result = vision_result
        elif ocr_result and len(ocr_result.strip()) > 5:
            print("✅ Using OCR result")
            final_result = ocr_result
        else:
            print("⚠️ Both methods produced minimal results")
            final_result = vision_result if vision_result else ocr_result
        
        return final_result, detected_lang
    
    def create_detailed_comments(self, text, language):
        """Create detailed line-by-line comments"""
        if not text or not text.strip():
            return "No text to comment"
        
        try:
            lang_config = self.detector.patterns.get(language, self.detector.patterns['sql'])
            comment_style = lang_config['comment_style']
            
            # Create very specific prompt
            prompt = f"""You are an expert {language.upper()} code commentator. Add a detailed comment above EVERY SINGLE LINE of this {language.upper()} code.

EXACT FORMAT REQUIRED:
{comment_style}[Detailed explanation of what this specific line does]
[THE ACTUAL CODE LINE EXACTLY AS WRITTEN]

RULES:
1. Comment EVERY line individually
2. Keep original code EXACTLY the same
3. Use {comment_style.strip()} for comments
4. Be specific about what each line does
5. Do NOT summarize or group lines
6. Do NOT change any original text

EXAMPLE:
{comment_style}Create a table named 'users' with specified columns
CREATE TABLE users (
{comment_style}Define an integer column 'id' as the primary key
    id INT PRIMARY KEY,
{comment_style}Define a variable character column 'name' with max 100 characters
    name VARCHAR(100)
{comment_style}Close the table definition statement
);

Now comment this {language.upper()} code:
{text}"""
            
            response = ollama.chat(
                model="qwen2.5-coder:3b",
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                options={
                    'num_gpu': -1,
                    'num_thread': 2,
                    'temperature': 0.0,
                    'num_predict': 3000,
                    'num_ctx': 4096,
                    'top_p': 0.1,
                    'repeat_penalty': 1.0,
                },
                stream=False
            )
            
            return response['message']['content']
            
        except Exception as e:
            return f"Error creating comments: {str(e)}"
    
    def create_code_overview(self, text, language):
        """Create a comprehensive code overview"""
        try:
            prompt = f"""Analyze this {language.upper()} code and provide a detailed technical overview:

1. **Purpose**: What does this code accomplish?
2. **Language & Version**: Specific {language.upper()} features used
3. **Structure**: Main components and organization
4. **Key Operations**: Important functions, queries, or logic
5. **Dependencies**: Required libraries, databases, or systems
6. **Data Flow**: How data moves through the code
7. **Complexity**: Technical complexity assessment

Code to analyze:
{text}"""
            
            response = ollama.chat(
                model="qwen2.5-coder:3b",
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                options={
                    'num_gpu': -1,
                    'num_thread': 2,
                    'temperature': 0.1,
                    'num_predict': 1500,
                    'num_ctx': 3072,
                    'top_p': 0.9,
                },
                stream=False
            )
            
            return response['message']['content']
            
        except Exception as e:
            return f"Error creating overview: {str(e)}"

def main():
    """Streamlit app main function"""
    st.set_page_config(
        page_title="Improved Image to Text Extractor",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Improved Image to Text Extractor")
    st.markdown("Advanced text extraction with enhanced language detection and commenting")
    
    extractor = ImprovedExtractor()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        st.subheader("📊 Language Detection")
        show_detection_details = st.checkbox("Show detection details", value=True)
        
        st.subheader("💻 Models")
        st.markdown("**OCR**: Tesseract with multiple configurations")
        st.markdown("**Vision**: llama3.2-vision:11b")
        st.markdown("**Analysis**: qwen2.5-coder:3b")
    
    # Main interface
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload an image containing code or text"
    )
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("🚀 Extract and Analyze", type="primary"):
            with st.spinner("Extracting text..."):
                # Extract text
                extracted_text, detected_language = extractor.extract_text_hybrid(image)
                
                # Language detection details
                if show_detection_details:
                    lang, confidence, scores = extractor.detector.detect_language(extracted_text)
                    
                    st.subheader("🔍 Language Detection Results")
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.metric("Detected Language", lang.upper())
                        st.metric("Confidence Score", f"{confidence}")
                    
                    with col2:
                        st.markdown("**Detection Scores:**")
                        for language, score in scores.items():
                            st.markdown(f"- **{language.upper()}**: {score}")
                
                # Display extracted text
                st.subheader("📝 Extracted Text")
                st.code(extracted_text, language=detected_language)
                
                # Create tabs for analysis
                tab1, tab2 = st.tabs(["📋 Code Overview", "💬 Line-by-Line Comments"])
                
                with tab1:
                    with st.spinner("Creating overview..."):
                        overview = extractor.create_code_overview(extracted_text, detected_language)
                        st.markdown(overview)
                
                with tab2:
                    with st.spinner("Adding comments..."):
                        comments = extractor.create_detailed_comments(extracted_text, detected_language)
                        st.code(comments, language=detected_language)
    
    # Footer
    st.markdown("---")
    st.markdown("**Enhanced Features:**")
    st.markdown("• Multi-configuration OCR extraction")
    st.markdown("• Advanced language detection with confidence scoring")
    st.markdown("• Hybrid OCR + Vision model approach")
    st.markdown("• Improved line-by-line commenting")

if __name__ == "__main__":
    main()
