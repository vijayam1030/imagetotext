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

def extract_text_from_image(image, model_name="llama3.2-vision:11b"):
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
                    'content': 'Carefully examine this entire image from TOP TO BOTTOM and extract ALL text exactly as it appears. Pay special attention to the very top and bottom edges of the image. Include every line of code, comments, imports, headers, and any text that appears anywhere in the image. Do not skip any lines. Return only the text content in the exact order it appears.',
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
                
                # Update streaming display
                stream_placeholder.text_area(
                    "Extracted Text:",
                    value=full_response,
                    height=150,
                    disabled=True,
                    key=f"text_stream_{chunk_counter}"
                )
                
                # Auto-scroll JavaScript injection
                st.markdown(f"""
                <script>
                setTimeout(function() {{
                    var textAreas = document.querySelectorAll('textarea');
                    if (textAreas.length > 0) {{
                        var lastTextArea = textAreas[textAreas.length - 1];
                        lastTextArea.scrollTop = lastTextArea.scrollHeight;
                    }}
                }}, 50);
                </script>
                """, unsafe_allow_html=True)
                
                # Small delay for smoother animation
                time.sleep(0.05)
        
        return full_response
    
    except Exception as e:
        return f"Error processing image: {str(e)}"

def get_code_overview(extracted_text):
    """Get detailed overview of what the code does"""
    try:
        # Initialize streaming placeholder
        stream_placeholder = st.empty()
        full_response = ""
        
        # Call Ollama API for code overview using DeepSeek
        stream = ollama.chat(
            model="deepseek-coder-v2:16b",
            messages=[
                {
                    'role': 'user',
                    'content': f'''Analyze this code and provide a detailed overview of what it does:

1. What is the main purpose of this code?
2. What programming language is it?
3. What are the key functions/methods?
4. What does the code accomplish?
5. Any important algorithms or patterns used?

Code:
{extracted_text}'''
                }
            ],
            options={
                'num_gpu': -1,
                'num_thread': 2,
                'temperature': 0.2,
                'num_predict': 600,
                'num_ctx': 2048,
            },
            stream=True
        )
        
        # Stream the response
        chunk_counter = 0
        for chunk in stream:
            if 'message' in chunk and 'content' in chunk['message']:
                full_response += chunk['message']['content']
                chunk_counter += 1
                
                # Update streaming display
                stream_placeholder.text_area(
                    "Code Overview:",
                    value=full_response,
                    height=200,
                    disabled=True,
                    key=f"overview_stream_{chunk_counter}"
                )
                
                # Auto-scroll JavaScript injection
                st.markdown(f"""
                <script>
                setTimeout(function() {{
                    var textAreas = document.querySelectorAll('textarea');
                    if (textAreas.length > 0) {{
                        var lastTextArea = textAreas[textAreas.length - 1];
                        lastTextArea.scrollTop = lastTextArea.scrollHeight;
                    }}
                }}, 50);
                </script>
                """, unsafe_allow_html=True)
                
                # Small delay for smoother animation
                time.sleep(0.05)
        
        return full_response
    
    except Exception as e:
        return f"Error getting code overview: {str(e)}"

def explain_code_with_codellama(extracted_text):
    """Use CodeLlama to explain code line by line with streaming"""
    try:
        # Initialize streaming placeholder
        stream_placeholder = st.empty()
        full_response = ""
        chunk_counter = 0
        
        # Call Ollama API with DeepSeek Coder model using streaming
        stream = ollama.chat(
            model="deepseek-coder-v2:16b",
            messages=[
                {
                    'role': 'user',
                    'content': f'''Explain this code line by line. For each line, provide:
1. The original code line
2. A detailed comment explaining what that line does

Format like this:
```
// Comment explaining what this line does
original code line

// Comment explaining what this line does  
original code line
```

Code to analyze:
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
                
                # Update streaming display
                stream_placeholder.text_area(
                    "Line-by-Line Explanation:",
                    value=full_response,
                    height=300,
                    disabled=True,
                    key=f"linebyline_stream_{chunk_counter}"
                )
                
                # Auto-scroll JavaScript injection
                st.markdown(f"""
                <script>
                setTimeout(function() {{
                    var textAreas = document.querySelectorAll('textarea');
                    if (textAreas.length > 0) {{
                        var lastTextArea = textAreas[textAreas.length - 1];
                        lastTextArea.scrollTop = lastTextArea.scrollHeight;
                    }}
                }}, 50);
                </script>
                """, unsafe_allow_html=True)
                
                # Small delay for smoother animation
                time.sleep(0.05)
        
        return full_response
    
    except Exception as e:
        return f"Error explaining code line by line: {str(e)}"

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
        model_options = ["llama3.2-vision:11b"]
        selected_model = st.selectbox("Select Vision Model", model_options)
        
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
                
                # Create containers for streaming outputs
                st.write("---")
                st.subheader("🔄 Live Processing")
                
                text_container = st.container()
                visual_container = st.container()  
                code_container = st.container()
                
                # Step 1: Extract text
                status_text.text("📝 Extracting text from image... (33%)")
                progress_bar.progress(33)
                with text_container:
                    st.write("📝 **Text Extraction:**")
                    extracted_text = extract_text_from_image(image, selected_model)
                    st.session_state.extracted_text = extracted_text
                
                # Step 2: Code overview (if text found)
                if extracted_text and extracted_text != "No text found":
                    status_text.text("💡 Getting code overview... (66%)")
                    progress_bar.progress(66)
                    with visual_container:
                        st.write("💡 **Code Overview:**")
                        overview = get_code_overview(extracted_text)
                        st.session_state.overview = overview
                    
                    # Step 3: Line-by-line explanation
                    status_text.text("🔍 Explaining code line by line... (90%)")
                    progress_bar.progress(90)
                    with code_container:
                        st.write("🔍 **Line-by-Line Explanation:**")
                        line_explanation = explain_code_with_codellama(extracted_text)
                        st.session_state.line_explanation = line_explanation
                
                # Complete
                status_text.text("✅ Processing complete!")
                progress_bar.progress(100)
                
                # Clear progress indicators after a short delay
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
    
    with col2:
        st.header("Final Results")
        if hasattr(st.session_state, 'extracted_text'):
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
    st.markdown("Install required models: `ollama pull llama3.2-vision:11b` and `ollama pull deepseek-coder-v2:16b`")
    st.markdown("**Models used:** Llama 3.2 Vision for image extraction, DeepSeek Coder V2 for code analysis")

if __name__ == "__main__":
    main()