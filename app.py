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

def extract_text_from_image(image, model_name="llava"):
    """Extract text from image using Ollama LLaVA model"""
    try:
        # Convert image to base64
        image_b64 = encode_image_to_base64(image)
        
        # Call Ollama API
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': 'Extract all text from this image. If there is no text, describe what you see in the image.',
                    'images': [image_b64]
                }
            ]
        )
        
        return response['message']['content']
    
    except Exception as e:
        return f"Error processing image: {str(e)}"

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
        model_options = ["llava", "llava:7b", "llava:13b", "llava:34b"]
        selected_model = st.selectbox("Select LLaVA Model", model_options)
        
        st.markdown("---")
        st.markdown("**Instructions:**")
        st.markdown("1. Upload an image file")
        st.markdown("2. Click 'Extract Text' button")
        st.markdown("3. View the extracted text")
    
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
            if st.button("🔍 Extract Text", type="primary"):
                with st.spinner("Processing image..."):
                    extracted_text = extract_text_from_image(image, selected_model)
                    st.session_state.extracted_text = extracted_text
    
    with col2:
        st.header("Extracted Text")
        if hasattr(st.session_state, 'extracted_text'):
            st.text_area(
                "Result:",
                value=st.session_state.extracted_text,
                height=400,
                disabled=True
            )
            
            # Copy to clipboard button
            if st.button("📋 Copy to Clipboard"):
                st.code(st.session_state.extracted_text)
                st.success("Text copied to clipboard!")
        else:
            st.info("Upload an image and click 'Extract Text' to see results here.")
    
    # Footer
    st.markdown("---")
    st.markdown("**Note:** Make sure Ollama is running and you have a LLaVA model installed.")
    st.markdown("Install LLaVA model: `ollama pull llava`")

if __name__ == "__main__":
    main()