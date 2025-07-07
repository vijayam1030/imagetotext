import streamlit as st
import requests
import base64
import io
from PIL import Image
import ollama
import time
import pytesseract
import cv2
import numpy as np

def encode_image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# REMOVED - Using extract_text_fast instead

def preprocess_image_for_ocr(image):
    """Preprocess image for better OCR accuracy"""
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply image processing for better OCR
        # Increase contrast
        enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced)
        
        # Resize if image is too small (OCR works better on larger images)
        height, width = denoised.shape
        if height < 300 or width < 300:
            scale_factor = max(300/height, 300/width)
            new_height = int(height * scale_factor)
            new_width = int(width * scale_factor)
            denoised = cv2.resize(denoised, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        return denoised
    except Exception as e:
        # Return original image as numpy array if preprocessing fails
        return np.array(image.convert('L'))

def extract_text_with_ocr(image):
    """Extract text using Tesseract OCR"""
    try:
        # Preprocess image
        processed_img = preprocess_image_for_ocr(image)
        
        # Configure Tesseract for code/text recognition
        custom_config = r'--oem 3 --psm 6'
        
        # Extract text using Tesseract
        ocr_text = pytesseract.image_to_string(processed_img, config=custom_config)
        
        return ocr_text.strip()
    except Exception as e:
        return f"OCR Error: {str(e)}"

def extract_text_fast(image, model_name="llama3.2-vision:11b"):
    """Tesseract OCR with vision model fallback"""
    try:
        # Primary: Use Tesseract OCR
        ocr_result = extract_text_with_ocr(image)
        
        # Check if OCR produced good results
        if ocr_result and not ocr_result.startswith("OCR Error:") and len(ocr_result.strip()) > 5:
            return ocr_result
        
        # Fallback: Use vision model if OCR fails or produces minimal text
        try:
            # Convert image to base64
            image_b64 = encode_image_to_base64(image)
            
            # Enhanced prompt for better text extraction
            prompt = """You are a precise OCR system. Look at this image and extract ALL visible text exactly as it appears.

INSTRUCTIONS:
1. Read every character, word, and line of text in the image
2. Copy the text EXACTLY - do not change, interpret, or explain anything
3. Maintain original formatting, spacing, indentation, and line breaks
4. Include ALL text: code, comments, file names, error messages, UI text, etc.
5. If you see programming code, copy it character-for-character
6. Start from top-left, work systematically to bottom-right
7. Do NOT add your own commentary or explanations

Extract the text:"""
            
            # Call vision model with optimized parameters for speed
            response = ollama.chat(
                model=model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                        'images': [image_b64]
                    }
                ],
                options={
                    'num_gpu': -1,
                    'num_thread': 1,
                    'temperature': 0.0,  # Most deterministic
                    'num_predict': 800,   # Reduced for speed
                    'num_ctx': 1024,
                    'top_p': 0.1,  # Very focused
                    'repeat_penalty': 1.0,  # No penalty to allow exact repetition
                },
                stream=False
            )
            
            vision_result = response['message']['content']
            return f"Vision fallback: {vision_result}"
            
        except Exception as vision_error:
            # If both OCR and vision fail, return OCR error (it was tried first)
            return ocr_result if ocr_result else f"Error: OCR failed and vision model unavailable ({str(vision_error)})"
    
    except Exception as e:
        return f"Error processing image: {str(e)}"

# REMOVED - Using get_code_overview_fast instead

# REMOVED - Using explain_code_fast instead

def get_code_overview_fast(extracted_text, model_name="qwen2.5-coder:1.5b"):
    """Fast code overview without streaming"""
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': f'''Analyze this code and provide a detailed overview:

1. Main purpose and functionality
2. Programming language and key components
3. Important functions/methods
4. Overall architecture

Code:
{extracted_text}'''
                }
            ],
            options={
                'num_gpu': -1,
                'num_thread': 1,
                'temperature': 0.1,
                'num_predict': 800,
                'num_ctx': 2048,
                'top_p': 0.9,
            },
            stream=False
        )
        return response['message']['content']
    except Exception as e:
        return f"Error getting code overview: {str(e)}"

def explain_code_fast(extracted_text, model_name="qwen2.5-coder:1.5b"):
    """Fast line-by-line explanation without streaming"""
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': f'''Explain this code line by line with detailed comments:

Format:
```
// Comment explaining what this line does
code line

// Comment explaining what this line does  
code line
```

Code:
{extracted_text}'''
                }
            ],
            options={
                'num_gpu': -1,
                'num_thread': 1,
                'temperature': 0.1,
                'num_predict': 1500,
                'num_ctx': 2048,
                'top_p': 0.9,
            },
            stream=False
        )
        return response['message']['content']
    except Exception as e:
        return f"Error explaining code: {str(e)}"

def main():
    st.set_page_config(
        page_title="Image to Text Extractor",
        page_icon="📸",
        layout="wide"
    )
    
    st.title("📸 Image to Text Extractor")
    st.markdown("Upload an image and extract text using Tesseract OCR with vision model fallback")
    
    # Sidebar for model selection
    with st.sidebar:
        st.header("Model Settings")
        
        # Vision model selection
        st.subheader("🔍 Text Extraction Model")
        vision_model_options = ["llama3.2-vision:11b", "llava:13b", "llava:7b"]
        selected_vision_model = st.selectbox("Select Vision Model", vision_model_options, key="vision_model")
        
        # Code analysis model selection
        st.subheader("💻 Code Analysis Model")
        code_model_options = [
            "qwen2.5-coder:1.5b", 
            "qwen2.5-coder:3b", 
            "deepseek-r1:1.5b",
            "deepseek-coder-v2:16b",
            "qwen2.5:14b", 
            "codellama:13b", 
            "llama3.1:8b"
        ]
        selected_code_model = st.selectbox("Select Code Model", code_model_options, key="code_model")
        
        st.markdown("---")
        st.markdown("**Model Usage:**")
        st.markdown("• **Tesseract OCR**: Primary text extraction")
        st.markdown("• **Vision**: Fallback for OCR failures")
        st.markdown("• **Code**: Analyzes and explains code")
        
        st.markdown("---")
        st.markdown("**Instructions:**")
        st.markdown("1. Upload one or multiple image files")
        st.markdown("2. Click 'Process All Images' button")
        st.markdown("3. View extracted text and AI analysis")
        st.markdown("4. Copy results as needed")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload Image")
        uploaded_files = st.file_uploader(
            "Choose image files",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            # Show all uploaded files
            st.write(f"📁 **{len(uploaded_files)} file(s) uploaded:**")
            for file in uploaded_files:
                st.write(f"📄 {file.name}")
            
            # Display first image as preview
            image = Image.open(uploaded_files[0])
            st.image(image, caption=f"Preview: {uploaded_files[0].name}", use_column_width=True)
            
            # Separate buttons for extraction and analysis
            col_btn1, col_btn2 = st.columns([1, 1])
            
            with col_btn1:
                extract_button = st.button("📸 Extract Text Only", type="primary")
            
            with col_btn2:
                analyze_button = st.button("🔍 Analyze Extracted Text", type="secondary")
            
            if extract_button:
                # Text extraction only
                st.write("---")
                st.subheader("📸 Extracting Text from Images")
                all_extracted_texts = []
                extraction_progress = st.progress(0)
                extraction_status = st.empty()
                
                # Process images efficiently without streaming for speed
                for idx, file in enumerate(uploaded_files):
                    extraction_status.text(f"📷 Extracting text from {file.name} ({idx+1}/{len(uploaded_files)})...")
                    extraction_progress.progress((idx + 1) / len(uploaded_files))
                    
                    img = Image.open(file)
                    extracted_text = extract_text_fast(img, selected_vision_model)
                    
                    if extracted_text and extracted_text != "No text found":
                        all_extracted_texts.append({
                            'filename': file.name,
                            'text': extracted_text
                        })
                
                extraction_progress.empty()
                extraction_status.empty()
                
                if all_extracted_texts:
                    # Create combined text with file separators
                    combined_text = ""
                    individual_texts_display = ""
                    
                    for item in all_extracted_texts:
                        combined_text += f"// ===== FILE: {item['filename']} =====\n"
                        combined_text += item['text'] + "\n\n"
                        
                        # For display purposes
                        individual_texts_display += f"📄 **{item['filename']}:**\n"
                        individual_texts_display += item['text'] + "\n" + "="*50 + "\n\n"
                    
                    # Store results (text extraction only)
                    st.session_state.all_individual_texts = all_extracted_texts
                    st.session_state.combined_extracted_text = combined_text
                    st.session_state.individual_texts_display = individual_texts_display
                    
                    # Clear previous analysis results
                    if 'combined_overview' in st.session_state:
                        del st.session_state.combined_overview
                    if 'combined_line_explanation' in st.session_state:
                        del st.session_state.combined_line_explanation
                    
                    st.success(f"✅ Text extracted from {len(all_extracted_texts)} images! Use 'Analyze Extracted Text' to run analysis.")
                else:
                    st.error("❌ No text found in any of the uploaded images.")
                    
            elif analyze_button:
                # Analysis only (requires existing extracted text)
                if hasattr(st.session_state, 'combined_extracted_text') and st.session_state.combined_extracted_text:
                    st.write("---")
                    st.subheader("🔍 Analyzing Extracted Text")
                    
                    combined_text = st.session_state.combined_extracted_text
                    
                    analysis_progress = st.progress(0)
                    analysis_status = st.empty()
                    
                    # Step 1: Get code overview
                    analysis_status.text(f"💡 Analyzing code overview with {selected_code_model}...")
                    analysis_progress.progress(33)
                    combined_overview = get_code_overview_fast(combined_text, selected_code_model)
                    st.session_state.combined_overview = combined_overview
                    
                    # Step 2: Get line-by-line explanation
                    analysis_status.text(f"🔍 Generating line-by-line explanations with {selected_code_model}...")
                    analysis_progress.progress(66)
                    combined_line_explanation = explain_code_fast(combined_text, selected_code_model)
                    st.session_state.combined_line_explanation = combined_line_explanation
                    
                    analysis_progress.progress(100)
                    analysis_progress.empty()
                    analysis_status.empty()
                    
                    st.success(f"✅ Analysis completed using {selected_code_model}!")
                else:
                    st.error("❌ No extracted text found! Please run 'Extract Text Only' first.")
    
    with col2:
        st.header("Final Results")
        
        # Show results for processed images
        if hasattr(st.session_state, 'combined_extracted_text'):
            # Show file summary
            if hasattr(st.session_state, 'all_individual_texts'):
                st.subheader(f"📁 Processed {len(st.session_state.all_individual_texts)} Images")
                for idx, item in enumerate(st.session_state.all_individual_texts, 1):
                    st.write(f"{idx}. 📄 {item['filename']}")
                st.write("---")
            
            # A. Text Extraction (Individual Files)
            st.subheader("A. 📝 Text Extraction by File")
            if hasattr(st.session_state, 'individual_texts_display'):
                st.text_area(
                    "Extracted text from each image:",
                    value=st.session_state.individual_texts_display,
                    height=300,
                    disabled=True,
                    key="individual_texts_display"
                )
                
                if st.button("📋 Copy Individual Texts"):
                    st.code(st.session_state.individual_texts_display)
                    st.success("Individual texts copied!")
            
            # B. Code Overview (All Combined)
            if hasattr(st.session_state, 'combined_overview'):
                st.subheader("B. 💡 Complete Code Overview")
                st.text_area(
                    "Overall analysis of all code together:",
                    value=st.session_state.combined_overview,
                    height=200,
                    disabled=True,
                    key="combined_overview"
                )
                
                if st.button("📋 Copy Overview"):
                    st.code(st.session_state.combined_overview)
                    st.success("Overview copied!")
            
            # C. Line-by-Line Comments (All Combined)
            if hasattr(st.session_state, 'combined_line_explanation'):
                st.subheader("C. 🔍 Line-by-Line Comments")
                st.text_area(
                    "Detailed comments for all code:",
                    value=st.session_state.combined_line_explanation,
                    height=400,
                    disabled=True,
                    key="combined_line_explanation"
                )
                
                if st.button("📋 Copy All Comments"):
                    st.code(st.session_state.combined_line_explanation)
                    st.success("All comments copied!")
        
        elif hasattr(st.session_state, 'extracted_text'):
            # A. Text Extraction Section
            st.subheader("A. 📝 Text Extraction")
            st.text_area(
                "Extracted text as it appears in the image:",
                value=st.session_state.extracted_text,
                height=200,
                disabled=True,
                key="final_extracted_text"
            )
            
            if st.button("📋 Copy Extracted Text"):
                st.code(st.session_state.extracted_text)
                st.success("Text copied!")
            
            # B. Code Overview Section
            if hasattr(st.session_state, 'overview'):
                st.subheader("B. 💡 Code Overview")
                st.text_area(
                    "Detailed explanation of what the code does:",
                    value=st.session_state.overview,
                    height=200,
                    disabled=True,
                    key="final_overview"
                )
                
                if st.button("📋 Copy Code Overview"):
                    st.code(st.session_state.overview)
                    st.success("Overview copied!")
            
            # C. Line-by-Line Explanation Section
            if hasattr(st.session_state, 'line_explanation'):
                st.subheader("C. 🔍 Line-by-Line Code Explanation")
                st.text_area(
                    "Each line explained with comments:",
                    value=st.session_state.line_explanation,
                    height=300,
                    disabled=True,
                    key="final_line_explanation"
                )
                
                if st.button("📋 Copy Line Explanations"):
                    st.code(st.session_state.line_explanation)
                    st.success("Line explanations copied!")
        else:
            st.info("Upload an image and click 'Process Image' to see results here.")
    
    # Footer
    st.markdown("---")
    st.markdown("**Note:** Make sure Ollama is running and you have the required models installed.")
    st.markdown("Install required models:")
    st.markdown("• `ollama pull llava:13b` (recommended for accuracy)")
    st.markdown("• `ollama pull llava:34b` (best accuracy, slower)")
    st.markdown("• `ollama pull deepseek-v3` (latest)")
    st.markdown("• `ollama pull deepseek-coder-v2:16b` (for code analysis)")
    st.markdown("**Models used:** Tesseract OCR for primary extraction, LLaVA for fallback, DeepSeek for code analysis")

if __name__ == "__main__":
    main()