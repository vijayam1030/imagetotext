import streamlit as st
import requests
import base64
import io
from PIL import Image
import ollama
import time

def encode_image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# REMOVED - Using extract_text_fast instead

def extract_text_fast(image, model_name=" llava:13b"):
    """Fast text extraction without streaming for bulk processing"""
    try:
        # Convert image to base64
        image_b64 = encode_image_to_base64(image)
        
        # Call Ollama API without streaming for maximum speed
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': 'You are an expert OCR system. Examine this image and transcribe EVERY character of text you see. Include all code, comments, function names, variable names, strings, numbers, and any written content. Preserve exact spacing, indentation, and line breaks. Start from the very top of the image and work systematically to the bottom. Do not interpret or explain - just transcribe exactly what you see.',
                    'images': [image_b64]
                }
            ],
            options={
                'num_gpu': -1,  # Use all available GPUs
                'num_thread': 1,  # Single thread for faster processing
                'temperature': 0.0,  # Deterministic output
                'num_predict': 500,  # Limit response length
                'num_ctx': 1024,  # Smaller context for speed
                'top_p': 0.9,  # Focus on high probability tokens
                'repeat_penalty': 1.1,  # Avoid repetition
            },
            stream=False  # NO STREAMING = MUCH FASTER
        )
        
        return response['message']['content']
    
    except Exception as e:
        return f"Error processing image: {str(e)}"

# REMOVED - Using get_code_overview_fast instead

# REMOVED - Using explain_code_fast instead

def get_code_overview_fast(extracted_text):
    """Fast code overview without streaming"""
    try:
        response = ollama.chat(
            model="deepseek-coder-v2:16b",
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

def explain_code_fast(extracted_text):
    """Fast line-by-line explanation without streaming"""
    try:
        response = ollama.chat(
            model="deepseek-coder-v2:16b",
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
    st.markdown("Upload an image and extract text using LLaVA vision model")
    
    # Sidebar for model selection
    with st.sidebar:
        st.header("Settings")
        model_options = ["llava:13b",  "deepseek-v3"]
        selected_model = st.selectbox("Select Vision Model", model_options)
        
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
            
            # Single process button
            if st.button("🚀 Process All Images", type="primary"):
                # Performance-optimized processing
                st.write("---")
                st.subheader("🚀 Processing All Images")
                
                # Step 1: Extract text from ALL images efficiently
                st.write("📝 **Step 1: Extracting text from all images...**")
                all_extracted_texts = []
                extraction_progress = st.progress(0)
                extraction_status = st.empty()
                
                # Process images efficiently without streaming for speed
                for idx, file in enumerate(uploaded_files):
                    extraction_status.text(f"📷 Extracting text from {file.name} ({idx+1}/{len(uploaded_files)})...")
                    extraction_progress.progress((idx + 1) / len(uploaded_files))
                    
                    img = Image.open(file)
                    extracted_text = extract_text_fast(img, selected_model)
                    
                    if extracted_text and extracted_text != "No text found":
                        all_extracted_texts.append({
                            'filename': file.name,
                            'text': extracted_text
                        })
                
                extraction_progress.empty()
                extraction_status.empty()
                
                # Step 2: Process all extracted text efficiently
                if all_extracted_texts:
                    st.write("🔗 **Step 2: Combining and analyzing all code...**")
                    
                    # Create combined text with file separators
                    combined_text = ""
                    individual_texts_display = ""
                    
                    for item in all_extracted_texts:
                        combined_text += f"// ===== FILE: {item['filename']} =====\n"
                        combined_text += item['text'] + "\n\n"
                        
                        # For display purposes
                        individual_texts_display += f"📄 **{item['filename']}:**\n"
                        individual_texts_display += item['text'] + "\n" + "="*50 + "\n\n"
                    
                    # Store results
                    st.session_state.all_individual_texts = all_extracted_texts
                    st.session_state.combined_extracted_text = combined_text
                    st.session_state.individual_texts_display = individual_texts_display
                    
                    # Step 3: Get code overview efficiently
                    analysis_progress = st.progress(0)
                    analysis_status = st.empty()
                    
                    analysis_status.text("💡 Analyzing combined code overview...")
                    analysis_progress.progress(50)
                    combined_overview = get_code_overview_fast(combined_text)
                    st.session_state.combined_overview = combined_overview
                    
                    # Step 4: Get line-by-line explanation efficiently
                    analysis_status.text("🔍 Generating line-by-line explanations...")
                    analysis_progress.progress(100)
                    combined_line_explanation = explain_code_fast(combined_text)
                    st.session_state.combined_line_explanation = combined_line_explanation
                    
                    analysis_progress.empty()
                    analysis_status.empty()
                    
                    st.success(f"✅ Successfully processed {len(all_extracted_texts)} images with optimized performance!")
                else:
                    st.error("❌ No text found in any of the uploaded images.")
    
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
    st.markdown("**Models used:** LLaVA for image extraction, DeepSeek for code analysis")

if __name__ == "__main__":
    main()