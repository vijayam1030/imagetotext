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

def extract_text_from_image(image, model_name="llama3.2-vision:11b"):
    """Extract text from image using Ollama vision model"""
    try:
        # Convert image to base64
        image_b64 = encode_image_to_base64(image)
        
        # Call Ollama API for text extraction
        text_response = ollama.chat(
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
                'num_thread': 8,  # Adjust based on your CPU cores
                'temperature': 0.1,  # Lower temperature for more accurate text extraction
            }
        )
        
        return text_response['message']['content']
    
    except Exception as e:
        return f"Error processing image: {str(e)}"

def explain_image_content(image, model_name="llama3.2-vision:11b"):
    """Explain what the code/content in the image is doing"""
    try:
        # Convert image to base64
        image_b64 = encode_image_to_base64(image)
        
        # Call Ollama API for explanation
        explanation_response = ollama.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': 'Analyze this image and explain what the code or content is doing. If it contains code, explain its functionality, purpose, and how it works. If it\'s not code, describe what you see and its purpose.',
                    'images': [image_b64]
                }
            ],
            options={
                'num_gpu': -1,  # Use all available GPUs
                'num_thread': 8,  # Adjust based on your CPU cores
                'temperature': 0.3,  # Slightly higher temperature for more descriptive explanations
            }
        )
        
        return explanation_response['message']['content']
    
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

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
        model_options = ["llama3.2-vision:11b", "llama3.2-vision:90b"]
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
                with st.spinner("Processing image..."):
                    extracted_text = extract_text_from_image(image, selected_model)
                    explanation = explain_image_content(image, selected_model)
                    st.session_state.extracted_text = extracted_text
                    st.session_state.explanation = explanation
    
    with col2:
        st.header("Results")
        if hasattr(st.session_state, 'extracted_text'):
            # Image to Text Section
            st.subheader("📝 Image to Text")
            st.text_area(
                "Extracted Text:",
                value=st.session_state.extracted_text,
                height=200,
                disabled=True
            )
            
            # Copy text button
            if st.button("📋 Copy Text"):
                st.code(st.session_state.extracted_text)
                st.success("Text copied!")
            
            # Explanation Section
            if hasattr(st.session_state, 'explanation'):
                st.subheader("💡 Code/Content Explanation")
                st.text_area(
                    "What the code/content is doing:",
                    value=st.session_state.explanation,
                    height=200,
                    disabled=True
                )
                
                # Copy explanation button
                if st.button("📋 Copy Explanation"):
                    st.code(st.session_state.explanation)
                    st.success("Explanation copied!")
        else:
            st.info("Upload an image and click 'Process Image' to see results here.")
    
    # Footer
    st.markdown("---")
    st.markdown("**Note:** Make sure Ollama is running and you have a LLaVA model installed.")
    st.markdown("Install Llama Vision model: `ollama pull llama3.2-vision:11b`")

if __name__ == "__main__":
    main()