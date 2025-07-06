import streamlit as st
import requests
import base64
import io
from PIL import Image
import ollama

def encode_image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def extract_text_from_image(image, model_name="llava:7b"):
    """Extract text from image using Ollama vision model with streaming"""
    try:
        # Convert image to base64
        image_b64 = encode_image_to_base64(image)
        
        # Initialize streaming placeholder
        stream_placeholder = st.empty()
        full_response = ""
        
        # Call Ollama API for text extraction with streaming
        stream = ollama.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': 'Extract all text from this image exactly as it appears. Only return the text content, nothing else. If there is no text, return "No text found".',
                    'images': [image_b64]
                }
            ],
            options={
                'num_gpu': -1,  # Use all available GPUs
                'num_thread': 2,  # Minimal CPU threads to force GPU usage
                'temperature': 0.1,  # Lower temperature for more accurate text extraction
                'num_predict': 300,  # Limit response length
                'num_ctx': 2048,  # Reduce context window for faster processing
            },
            stream=True
        )
        
        # Stream the response
        chunk_counter = 0
        for chunk in stream:
            if 'message' in chunk and 'content' in chunk['message']:
                full_response += chunk['message']['content']
                chunk_counter += 1
                stream_placeholder.text_area(
                    "Extracted Text (Streaming):",
                    value=full_response,
                    height=150,
                    disabled=True,
                    key=f"text_stream_{chunk_counter}"
                )
        
        return full_response
    
    except Exception as e:
        return f"Error processing image: {str(e)}"

def explain_image_content(image, model_name="llava:7b"):
    """Explain what the code/content in the image is doing with streaming"""
    try:
        # Convert image to base64
        image_b64 = encode_image_to_base64(image)
        
        # Initialize streaming placeholder
        stream_placeholder = st.empty()
        full_response = ""
        
        # Call Ollama API for explanation with streaming
        stream = ollama.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': 'Briefly describe what this image contains. If it has code, explain what it does (2-3 sentences).',
                    'images': [image_b64]
                }
            ],
            options={
                'num_gpu': -1,  # Use all available GPUs
                'num_thread': 2,  # Minimal CPU threads to force GPU usage
                'temperature': 0.3,  # Slightly higher temperature for more descriptive explanations
                'num_predict': 200,  # Limit response length
                'num_ctx': 2048,  # Reduce context window for faster processing
            },
            stream=True
        )
        
        # Stream the response
        chunk_counter = 0
        for chunk in stream:
            if 'message' in chunk and 'content' in chunk['message']:
                full_response += chunk['message']['content']
                chunk_counter += 1
                stream_placeholder.text_area(
                    "Visual Analysis (Streaming):",
                    value=full_response,
                    height=120,
                    disabled=True,
                    key=f"visual_stream_{chunk_counter}"
                )
        
        return full_response
    
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

def explain_code_with_codellama(extracted_text):
    """Use CodeLlama to explain code line by line with streaming"""
    try:
        # Initialize streaming placeholder
        stream_placeholder = st.empty()
        full_response = ""
        chunk_counter = 0
        
        # Call Ollama API with CodeLlama model using streaming
        stream = ollama.chat(
            model="codellama:13b",
            messages=[
                {
                    'role': 'user',
                    'content': f'''Analyze this code briefly:
1. What the code does (1-2 sentences)
2. Key functions/methods
3. Important concepts

Code:
{extracted_text}'''
                }
            ],
            options={
                'num_gpu': -1,
                'num_thread': 2,  # Minimal CPU threads to force GPU usage
                'temperature': 0.2,
                'num_predict': 400,  # Reduced response length
                'num_ctx': 2048,  # Reduce context window for faster processing
            },
            stream=True
        )
        
        # Stream the response
        for chunk in stream:
            if 'message' in chunk and 'content' in chunk['message']:
                full_response += chunk['message']['content']
                chunk_counter += 1
                stream_placeholder.text_area(
                    "CodeLlama Analysis (Streaming):",
                    value=full_response,
                    height=200,
                    disabled=True,
                    key=f"codellama_stream_{chunk_counter}"
                )
        
        return full_response
    
    except Exception as e:
        return f"Error analyzing code with CodeLlama: {str(e)}"

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
        model_options = ["llava:7b", "llama3.2-vision:11b", "llama3.2-vision:90b"]
        selected_model = st.selectbox("Select LLaVA Model", model_options)
        
        st.markdown("---")
        st.markdown("**Instructions:**")
        st.markdown("1. Upload an image file")
        st.markdown("2. Click 'Process Image' button")
        st.markdown("3. View text and explanation")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
            accept_multiple_files=False
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Extract text button
            if st.button("🔍 Process Image", type="primary"):
                # Create progress bar and status
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Extract text
                status_text.text("📝 Extracting text from image... (33%)")
                progress_bar.progress(33)
                extracted_text = extract_text_from_image(image, selected_model)
                st.session_state.extracted_text = extracted_text
                
                # Step 2: Visual analysis
                status_text.text("💡 Analyzing image content... (66%)")
                progress_bar.progress(66)
                explanation = explain_image_content(image, selected_model)
                st.session_state.explanation = explanation
                
                # Step 3: CodeLlama analysis (if text found)
                if extracted_text and extracted_text != "No text found":
                    status_text.text("🔍 Analyzing code with CodeLlama... (90%)")
                    progress_bar.progress(90)
                    code_analysis = explain_code_with_codellama(extracted_text)
                    st.session_state.code_analysis = code_analysis
                
                # Complete
                status_text.text("✅ Processing complete!")
                progress_bar.progress(100)
                
                # Clear progress indicators after a short delay
                import time
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
    
    with col2:
        st.header("Results")
        if hasattr(st.session_state, 'extracted_text'):
            # Image to Text Section
            st.subheader("📝 Image to Text")
            st.text_area(
                "Extracted Text:",
                value=st.session_state.extracted_text,
                height=200,
                disabled=True,
                key="final_extracted_text"
            )
            
            # Copy text button
            if st.button("📋 Copy Text"):
                st.code(st.session_state.extracted_text)
                st.success("Text copied!")
            
            # Explanation Section
            if hasattr(st.session_state, 'explanation'):
                st.subheader("💡 Visual Analysis")
                st.text_area(
                    "What the image contains:",
                    value=st.session_state.explanation,
                    height=150,
                    disabled=True,
                    key="final_visual_analysis"
                )
                
                # Copy explanation button
                if st.button("📋 Copy Visual Analysis"):
                    st.code(st.session_state.explanation)
                    st.success("Visual analysis copied!")
            
            # CodeLlama Code Analysis Section
            if hasattr(st.session_state, 'code_analysis'):
                st.subheader("🔍 Code Analysis (CodeLlama)")
                st.text_area(
                    "Code explanation:",
                    value=st.session_state.code_analysis,
                    height=250,
                    disabled=True,
                    key="final_code_analysis"
                )
                
                # Copy code analysis button
                if st.button("📋 Copy Code Analysis"):
                    st.code(st.session_state.code_analysis)
                    st.success("Code analysis copied!")
        else:
            st.info("Upload an image and click 'Process Image' to see results here.")
    
    # Footer
    st.markdown("---")
    st.markdown("**Note:** Make sure Ollama is running and you have a LLaVA model installed.")
    st.markdown("Install required models: `ollama pull llama3.2-vision:11b` and `ollama pull codellama:13b`")

if __name__ == "__main__":
    main()