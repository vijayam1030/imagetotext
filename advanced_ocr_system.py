#!/usr/bin/env python3
"""
Advanced OCR System with Multiple AI Models for Code Extraction and Cleanup
Uses Tesseract, PyTesseract, Guesslang, Pygments, and multiple Ollama models
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

class AdvancedLanguageDetector:
    """Multi-method language detection using guesslang and pygments"""
    
    def __init__(self):
        self.guesslang_model = None
        if GUESSLANG_AVAILABLE:
            try:
                self.guesslang_model = Guess()
            except Exception as e:
                st.warning(f"Failed to initialize Guesslang: {e}")
        
        # Language mappings for file extensions
        self.extension_map = {
            'sql': ['.sql', '.ddl', '.dml'],
            'python': ['.py', '.pyw', '.pyi'],
            'javascript': ['.js', '.jsx', '.mjs'],
            'typescript': ['.ts', '.tsx'],
            'java': ['.java'],
            'csharp': ['.cs'],
            'cpp': ['.cpp', '.cc', '.cxx', '.c++'],
            'c': ['.c', '.h'],
            'php': ['.php', '.phtml'],
            'ruby': ['.rb', '.ruby'],
            'go': ['.go'],
            'rust': ['.rs'],
            'kotlin': ['.kt', '.kts'],
            'swift': ['.swift'],
            'r': ['.r', '.R'],
            'matlab': ['.m'],
            'perl': ['.pl', '.pm'],
            'shell': ['.sh', '.bash', '.zsh'],
            'powershell': ['.ps1', '.psm1'],
            'yaml': ['.yml', '.yaml'],
            'json': ['.json'],
            'xml': ['.xml', '.xsd', '.xsl'],
            'html': ['.html', '.htm'],
            'css': ['.css', '.scss', '.sass'],
        }
        
        # Comment styles for each language
        self.comment_styles = {
            'sql': '-- ',
            'python': '# ',
            'javascript': '// ',
            'typescript': '// ',
            'java': '// ',
            'csharp': '// ',
            'cpp': '// ',
            'c': '// ',
            'php': '// ',
            'ruby': '# ',
            'go': '// ',
            'rust': '// ',
            'kotlin': '// ',
            'swift': '// ',
            'r': '# ',
            'matlab': '% ',
            'perl': '# ',
            'shell': '# ',
            'powershell': '# ',
            'yaml': '# ',
            'json': '',
            'xml': '<!-- ',
            'html': '<!-- ',
            'css': '/* ',
        }
    
    def detect_with_guesslang(self, code):
        """Detect language using guesslang"""
        if not self.guesslang_model:
            return None, 0.0
        
        try:
            language = self.guesslang_model.language_name(code)
            confidence = self.guesslang_model.language_confidence(code)
            return language.lower(), confidence
        except Exception as e:
            st.warning(f"Guesslang detection failed: {e}")
            return None, 0.0
    
    def detect_with_pygments(self, code):
        """Detect language using pygments"""
        if not PYGMENTS_AVAILABLE:
            return None, 0.0
        
        try:
            lexer = guess_lexer(code)
            return lexer.name.lower(), 0.8  # Assume high confidence for pygments
        except ClassNotFound:
            return None, 0.0
        except Exception as e:
            st.warning(f"Pygments detection failed: {e}")
            return None, 0.0
    
    def detect_with_patterns(self, code):
        """Enhanced pattern-based language detection"""
        patterns = {
            'sql': [
                r'\b(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|TRUNCATE)\b',
                r'\b(FROM|WHERE|JOIN|GROUP BY|ORDER BY|HAVING|UNION)\b',
                r'\b(TABLE|DATABASE|INDEX|VIEW|PROCEDURE|FUNCTION)\b',
                r'\b(PRIMARY KEY|FOREIGN KEY|REFERENCES|CONSTRAINT)\b',
                r'\b(INNER JOIN|LEFT JOIN|RIGHT JOIN|FULL JOIN)\b',
                r'\b(VARCHAR|INT|CHAR|DATE|DATETIME|DECIMAL)\b'
            ],
            'python': [
                r'\bdef\s+\w+\s*\(',
                r'\bimport\s+\w+',
                r'\bfrom\s+\w+\s+import\b',
                r'\bif\s+__name__\s*==\s*["\']__main__["\']',
                r'\bprint\s*\(',
                r'\bclass\s+\w+\s*\(',
                r'\bfor\s+\w+\s+in\s+',
                r'\bwith\s+\w+\s*\(',
                r'\belif\b',
                r'\bexcept\s+\w*Exception'
            ],
            'javascript': [
                r'\bfunction\s+\w+\s*\(',
                r'\b(var|let|const)\s+\w+',
                r'\bconsole\.log\s*\(',
                r'=>',
                r'\bdocument\.',
                r'\bwindow\.',
                r'\bjQuery|\$\(',
                r'\basync\s+function',
                r'\bawait\s+',
                r'\b(setTimeout|setInterval)\s*\('
            ],
            'java': [
                r'\bpublic\s+class\s+\w+',
                r'\bpublic\s+static\s+void\s+main\b',
                r'\bSystem\.out\.println\s*\(',
                r'\bString\s+\w+',
                r'\bpublic\s+static\s+',
                r'\bprivate\s+\w+\s+\w+',
                r'\bprotected\s+\w+\s+\w+',
                r'\bimport\s+java\.',
                r'\bthrows\s+\w+Exception',
                r'\bnew\s+\w+\s*\('
            ],
            'csharp': [
                r'\busing\s+System',
                r'\bpublic\s+class\s+\w+',
                r'\bConsole\.WriteLine\s*\(',
                r'\bnamespace\s+\w+',
                r'\bpublic\s+static\s+void\s+Main',
                r'\bprivate\s+\w+\s+\w+',
                r'\bprotected\s+\w+\s+\w+',
                r'\bvar\s+\w+\s*=',
                r'\bstring\s+\w+',
                r'\bint\s+\w+\s*='
            ],
            'cpp': [
                r'#include\s*<\w+>',
                r'\bint\s+main\s*\(',
                r'\bstd::cout\s*<<',
                r'\bstd::\w+',
                r'\busing\s+namespace\s+std',
                r'\bclass\s+\w+\s*{',
                r'\btemplate\s*<',
                r'\bpublic\s*:',
                r'\bprivate\s*:',
                r'\bprotected\s*:'
            ],
            'c': [
                r'#include\s*<\w+\.h>',
                r'\bint\s+main\s*\(',
                r'\bprintf\s*\(',
                r'\bscanf\s*\(',
                r'\bmalloc\s*\(',
                r'\bfree\s*\(',
                r'\bstruct\s+\w+\s*{',
                r'\btypedef\s+\w+',
                r'\bstatic\s+\w+\s+\w+',
                r'\bextern\s+\w+\s+\w+'
            ],
            'php': [
                r'<\?php',
                r'\$\w+\s*=',
                r'\becho\s+',
                r'\bprint\s+',
                r'\bfunction\s+\w+\s*\(',
                r'\bclass\s+\w+\s*{',
                r'\bpublic\s+function\s+',
                r'\bprivate\s+\$\w+',
                r'\brequire\s+',
                r'\binclude\s+'
            ],
            'html': [
                r'<html',
                r'<head>',
                r'<body>',
                r'<div\s+',
                r'<p>',
                r'<script',
                r'<style',
                r'<title>',
                r'<meta\s+',
                r'<link\s+'
            ],
            'css': [
                r'{\s*$',
                r'}\s*$',
                r':\s*\w+;',
                r'#\w+\s*{',
                r'\.\w+\s*{',
                r'@media\s+',
                r'@import\s+',
                r'font-family\s*:',
                r'background-color\s*:',
                r'margin\s*:'
            ]
        }
        
        scores = {}
        code_lines = code.split('\n')
        total_lines = len(code_lines)
        
        for lang, lang_patterns in patterns.items():
            score = 0
            for pattern in lang_patterns:
                matches = len(re.findall(pattern, code, re.IGNORECASE | re.MULTILINE))
                score += matches
            
            # Normalize score by code length
            if total_lines > 0:
                scores[lang] = score / total_lines
            else:
                scores[lang] = 0
        
        if not any(scores.values()):
            return None, 0.0
        
        best_lang = max(scores, key=scores.get)
        max_score = scores[best_lang]
        
        # Convert to confidence (0-1 scale)
        confidence = min(max_score * 2, 1.0)  # Scale and cap at 1.0
        
        return best_lang, confidence
    
    def detect_language(self, code):
        """Comprehensive language detection using multiple methods"""
        results = []
        
        # Method 1: Guesslang
        if GUESSLANG_AVAILABLE:
            gl_lang, gl_conf = self.detect_with_guesslang(code)
            if gl_lang:
                results.append(('guesslang', gl_lang, gl_conf))
        
        # Method 2: Pygments
        if PYGMENTS_AVAILABLE:
            pg_lang, pg_conf = self.detect_with_pygments(code)
            if pg_lang:
                results.append(('pygments', pg_lang, pg_conf))
        
        # Method 3: Pattern matching
        pt_lang, pt_conf = self.detect_with_patterns(code)
        if pt_lang:
            results.append(('patterns', pt_lang, pt_conf))
        
        if not results:
            return 'text', 0.0, {}
        
        # Combine results with weighted scoring
        language_scores = {}
        method_weights = {'guesslang': 0.4, 'pygments': 0.4, 'patterns': 0.2}
        
        for method, lang, conf in results:
            weight = method_weights.get(method, 0.2)
            weighted_score = conf * weight
            
            if lang in language_scores:
                language_scores[lang] += weighted_score
            else:
                language_scores[lang] = weighted_score
        
        best_language = max(language_scores, key=language_scores.get)
        confidence = language_scores[best_language]
        
        return best_language, confidence, language_scores

class TesseractOCREngine:
    """Advanced Tesseract OCR with multiple configurations"""
    
    def __init__(self):
        self.ocr_configs = [
            {'name': 'Default', 'config': r'--oem 3 --psm 6'},
            {'name': 'Single Block', 'config': r'--oem 3 --psm 6 -c preserve_interword_spaces=1'},
            {'name': 'Single Column', 'config': r'--oem 3 --psm 4'},
            {'name': 'Vertical Text', 'config': r'--oem 3 --psm 5'},
            {'name': 'Sparse Text', 'config': r'--oem 3 --psm 11'},
            {'name': 'Raw Line', 'config': r'--oem 3 --psm 13'},
            {'name': 'Code Optimized', 'config': r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!@#$%^&*()_+-={}[]|\\:";\'<>?,./ \t\n'},
        ]
    
    def preprocess_image(self, image):
        """Advanced image preprocessing for better OCR"""
        img_array = np.array(image)
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array.copy()
        
        preprocessed_images = []
        
        # Original
        preprocessed_images.append(('Original', gray))
        
        # High contrast
        contrast = cv2.convertScaleAbs(gray, alpha=1.5, beta=20)
        preprocessed_images.append(('High Contrast', contrast))
        
        # Adaptive threshold
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        preprocessed_images.append(('Adaptive Threshold', adaptive))
        
        # Morphological operations
        kernel = np.ones((1, 1), np.uint8)
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        preprocessed_images.append(('Morphological', morph))
        
        # Gaussian blur + threshold
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images.append(('Gaussian + Otsu', thresh))
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=30)
        preprocessed_images.append(('Denoised', denoised))
        
        return preprocessed_images
    
    def extract_text(self, image):
        """Extract text using optimized OCR configurations"""
        # Use only 2 most effective preprocessing methods for speed
        img_array = np.array(image)
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array.copy()
        
        # Only use 2 most effective preprocessing methods
        fast_preprocessed = [
            ('Enhanced', cv2.convertScaleAbs(gray, alpha=1.2, beta=10)),
            ('Adaptive', cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            ))
        ]
        
        # Use only 3 most effective OCR configurations
        fast_configs = [
            {'name': 'Default', 'config': r'--oem 3 --psm 6'},
            {'name': 'Single Column', 'config': r'--oem 3 --psm 4'},
            {'name': 'Code Optimized', 'config': r'--oem 3 --psm 6 -c preserve_interword_spaces=1'}
        ]
        
        results = []
        best_length = 0
        
        for img_name, img in fast_preprocessed:
            for config in fast_configs:
                try:
                    text = pytesseract.image_to_string(img, config=config['config'])
                    if text.strip():
                        text_length = len(text.strip())
                        results.append({
                            'preprocessing': img_name,
                            'config': config['name'],
                            'text': text.strip(),
                            'length': text_length,
                            'lines': len(text.strip().split('\n'))
                        })
                        
                        # Early stopping if we get a good result
                        if text_length > 100:
                            results.sort(key=lambda x: x['length'], reverse=True)
                            return results
                            
                except Exception as e:
                    continue
        
        # Sort by length (longer is usually better)
        results.sort(key=lambda x: x['length'], reverse=True)
        
        return results

class OllamaCodeCleaner:
    """Advanced code analysis with language-specific model selection"""
    
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
                'wizardcoder:34b'
            ],
            'csharp': [
                'deepseek-coder-v2:16b',  # Good for Microsoft stack
                'wizardcoder:34b',
                'codellama:13b'
            ],
            'default': [
                'phi3:medium',            # Fast general purpose
                'codellama:7b',
                'qwen2.5-coder:1.5b'
            ]
        }
        
        self.available_models = self.check_available_models()
    
    def check_available_models(self):
        """Check which models are available in Ollama"""
        try:
            models = ollama.list()
            available = [model['name'] for model in models['models']]
            st.sidebar.success(f"✅ Found {len(available)} Ollama models")
            return available
        except Exception as e:
            st.warning(f"Could not check available models: {e}")
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
            prompt = f"""You are an expert {language.upper()} code analyst. Provide a comprehensive technical overview of this code.

DETAILED ANALYSIS SECTIONS:
1. **Purpose & Functionality**: What does this code accomplish? What problem does it solve?
2. **Technical Architecture**: How is the code structured? What design patterns are used?
3. **Key Components**: Main functions, classes, queries, or modules and their roles
4. **Data Flow**: How data moves through the code, inputs and outputs
5. **Dependencies**: Required libraries, frameworks, databases, or external systems
6. **Complexity Assessment**: Technical difficulty, performance considerations
7. **{language.upper()} Specific Features**: Language-specific constructs, best practices used
8. **Code Quality**: Adherence to standards, maintainability, readability
9. **Potential Issues**: Any problems, vulnerabilities, or areas for improvement
10. **Usage Context**: When and how this code would typically be used

Be thorough and technical. Focus on {language.upper()}-specific aspects and industry best practices.

{language.upper()} Code to analyze:
```{language}
{code}
```

Provide detailed technical analysis:"""

            response = ollama.chat(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    'num_gpu': -1,
                    'num_thread': 2,
                    'temperature': 0.1,
                    'num_predict': 2500,
                    'num_ctx': 4096,
                    'top_p': 0.9,
                },
                stream=False
            )
            
            return response['message']['content']
            
        except Exception as e:
            return f"Error creating overview: {str(e)}"
    
    def create_line_by_line_comments(self, code, language, model):
        """Create detailed line-by-line code comments"""
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
                'php': '// ',
                'ruby': '# ',
                'go': '// ',
                'rust': '// ',
                'text': '# '
            }
            
            comment_prefix = comment_styles.get(language, '# ')
            
            prompt = f"""You are a senior {language.upper()} developer and code documentation expert. Add detailed, educational comments above EVERY SINGLE LINE of this code.

COMMENTING REQUIREMENTS:
1. Add {comment_prefix}[Detailed explanation] above each line of code
2. Comment EVERY line individually - no skipping or grouping
3. Explain what each line does technically
4. Include parameter explanations, return values, logic flow
5. Mention {language.upper()}-specific features being used
6. Preserve the original code exactly as written
7. Use {comment_prefix.strip()} for all comments (standard for {language.upper()})
8. Be educational - help someone learn from this code

EXAMPLE FORMAT:
{comment_prefix}Import the database connection module for handling SQL operations
import database
{comment_prefix}Create a connection object using the default connection parameters
connection = database.connect()
{comment_prefix}Execute a SELECT statement to retrieve all records from the users table
result = connection.execute("SELECT * FROM users")

Now document this {language.upper()} code with detailed line-by-line comments:

```{language}
{code}
```

Add comprehensive line-by-line documentation:"""

            response = ollama.chat(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    'num_gpu': -1,
                    'num_thread': 2,
                    'temperature': 0.0,  # Very deterministic
                    'num_predict': 4000,
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
            prompt = f"""You are an expert {language.upper()} developer. Clean and fix this OCR-extracted code.

CLEANING REQUIREMENTS:
1. Fix OCR errors (character misrecognition, spacing issues)
2. Correct syntax errors and formatting
3. Add proper indentation and structure
4. Fix variable names and keywords that were misread
5. Ensure the code follows {language.upper()} best practices
6. DO NOT add any comments - only clean the code
7. Return ONLY the corrected code, nothing else

{language.upper()} Code to clean:
```{language}
{code}
```

Return the cleaned code only:"""

            response = ollama.chat(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    'num_gpu': -1,
                    'num_thread': 2,
                    'temperature': 0.1,
                    'num_predict': 2000,
                    'num_ctx': 4096,
                    'top_p': 0.9,
                },
                stream=False
            )
            
            return response['message']['content']
            
        except Exception as e:
            return f"Error cleaning code: {str(e)}"

class CodeSaver:
    """Save extracted and cleaned code with proper file extensions"""
    
    def __init__(self):
        self.save_dir = Path("extracted_code")
        self.save_dir.mkdir(exist_ok=True)
        
        self.extensions = {
            'sql': '.sql',
            'python': '.py',
            'javascript': '.js',
            'typescript': '.ts',
            'java': '.java',
            'csharp': '.cs',
            'cpp': '.cpp',
            'c': '.c',
            'php': '.php',
            'ruby': '.rb',
            'go': '.go',
            'rust': '.rs',
            'kotlin': '.kt',
            'swift': '.swift',
            'r': '.r',
            'matlab': '.m',
            'perl': '.pl',
            'shell': '.sh',
            'powershell': '.ps1',
            'yaml': '.yml',
            'json': '.json',
            'xml': '.xml',
            'html': '.html',
            'css': '.css',
            'md': '.md',
            'markdown': '.md',
            'text': '.txt'
        }
    
    def save_code(self, code, language, prefix="extracted", timestamp=None):
        """Save code to file with appropriate extension"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        extension = self.extensions.get(language, '.txt')
        filename = f"{prefix}_{language}_{timestamp}{extension}"
        filepath = self.save_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(code)
            return str(filepath)
        except Exception as e:
            st.error(f"Error saving file: {e}")
            return None
    
    def create_download_link(self, code, language, filename):
        """Create download link for Streamlit"""
        extension = self.extensions.get(language, '.txt')
        if not filename.endswith(extension):
            filename = f"{filename}{extension}"
        
        return st.download_button(
            label=f"💾 Download {language.upper()} code",
            data=code,
            file_name=filename,
            mime="text/plain"
        )

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Advanced OCR Code Extractor",
        page_icon="🔧",
        layout="wide"
    )
    
    st.title("🔧 Advanced OCR Code Extractor")
    st.markdown("**Tesseract OCR + Multiple AI Models + Language Detection + Code Cleanup**")
    
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
                
                # Show image properties
                st.markdown("**Image Properties:**")
                st.markdown(f"• Color mode: {image.mode}")
                st.markdown(f"• Has transparency: {'Yes' if 'A' in image.mode else 'No'}")
                if hasattr(image, 'info') and 'dpi' in image.info:
                    st.markdown(f"• DPI: {image.info['dpi']}")
                else:
                    st.markdown("• DPI: Not specified (assuming 72 DPI)")
        
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
                st.subheader("🔍 OCR Attempt Details")
                for i, result in enumerate(ocr_results[:5]):  # Show top 5
                    with st.expander(f"#{i+1}: {result['preprocessing']} + {result['config']} ({result['length']} chars)"):
                        st.code(result['text'][:500] + "..." if len(result['text']) > 500 else result['text'])
            
            # Display raw extracted text
            st.subheader("📋 Raw Extracted Text")
            st.code(raw_text, language='text')
            
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
                for lang, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
                    st.markdown(f"• **{lang.title()}**: {score:.3f}")
            
            # Step 3: AI Analysis and Code Processing
            st.subheader("🧹 Step 3: AI Analysis with Language-Specific Models")
            
            if not cleaner.available_models:
                st.error("❌ No Ollama models available for analysis")
                st.info("💡 Please install Ollama models like: `ollama pull codellama:7b`")
                
                # Show only raw text if no AI available
                st.subheader("📋 Raw Extracted Code")
                st.code(raw_text, language=detected_lang)
                
                if auto_save:
                    filepath = saver.save_code(raw_text, detected_lang, "raw_ocr", timestamp)
                    if filepath:
                        st.success(f"💾 Saved to: {filepath}")
                
                saver.create_download_link(raw_text, detected_lang, f"raw_ocr_{timestamp}")
                return
            
            # Select best model for the detected language
            selected_model = cleaner.select_best_model_for_language(detected_lang)
            
            if not selected_model:
                st.error("❌ No suitable AI model found")
                return
            
            st.info(f"🎯 Using **{selected_model}** for {detected_lang.upper()} code analysis")
            
            # Create two analysis sections
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Creating Code Overview...")
                overview_placeholder = st.empty()
                
            with col2:
                st.markdown("### 💬 Creating Line Comments...")
                comments_placeholder = st.empty()
            
            # Generate detailed overview
            with st.spinner("Generating comprehensive code overview..."):
                detailed_overview = cleaner.create_detailed_overview(raw_text, detected_lang, selected_model)
            
            # Generate line-by-line comments
            with st.spinner("Creating line-by-line documentation..."):
                line_comments = cleaner.create_line_by_line_comments(raw_text, detected_lang, selected_model)
            
            # Display results in organized tabs
            st.subheader("📋 Analysis Results")
            
            tabs = st.tabs([
                "� Detailed Overview", 
                "💬 Line-by-Line Comments", 
                "🔧 Cleaned Code", 
                "📄 Original Text"
            ])
            
            # Tab 1: Detailed Overview
            with tabs[0]:
                st.markdown("### 📊 Comprehensive Code Analysis")
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
            
            # Tab 2: Line-by-Line Comments
            with tabs[1]:
                st.markdown("### 💬 Educational Line-by-Line Documentation")
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
            
            # Tab 3: Cleaned Code (without comments)
            with tabs[2]:
                st.markdown("### 🔧 Cleaned Code")
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
            
            # Tab 4: Original Text
            with tabs[3]:
                st.markdown("### 📄 Original OCR Output")
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
