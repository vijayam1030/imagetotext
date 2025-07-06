# Image to Code Analysis Workflow

## Three Sections Implementation

### 1. 📝 Text/Code Extraction
- **Model Used:** `llama3.2-vision:11b`
- **Function:** `extract_text_from_image(image, model_name)`
- **Purpose:** Extract exact text/code from uploaded image as it appears
- **Input:** Raw image uploaded by user
- **Output:** Raw text/code exactly as shown in image
- **Prompt:** "Extract ALL text from this image exactly as it appears. Include code, comments, variable names, function names, and any written content. Return only the text content, nothing else."

### 2. 💡 Code Overview (Detailed Explanation)
- **Model Used:** `deepseek-coder-v2:16b`
- **Function:** `get_code_overview(extracted_text)`
- **Purpose:** Explain in detail what the code is doing
- **Input:** Extracted text from Section 1
- **Output:** Detailed analysis covering:
  - Main purpose of the code
  - Programming language
  - Key functions/methods
  - What the code accomplishes
  - Important algorithms or patterns used

### 3. 🔍 Line-by-Line Code Explanation
- **Model Used:** `deepseek-coder-v2:16b`
- **Function:** `explain_code_with_codellama(extracted_text)` (renamed but uses DeepSeek)
- **Purpose:** Explain each line of code with detailed comments
- **Input:** Extracted text from Section 1
- **Output:** Line-by-line breakdown with format:
  ```
  // Comment explaining what this line does
  original code line
  
  // Comment explaining what this line does
  original code line
  ```

## Data Flow
```
User uploads image → 
Section 1: llama3.2-vision:11b extracts text → 
Section 2: deepseek-coder-v2:16b analyzes overall code → 
Section 3: deepseek-coder-v2:16b explains line by line
```

## Key Points
✅ Only 3 sections as requested
✅ Image is used in Section 1 for text extraction
✅ Extracted text feeds into Sections 2 & 3
✅ Llama 3.2 Vision 11B for image processing
✅ DeepSeek Coder V2 16B for both code analysis sections
✅ Real-time streaming for all sections
✅ Auto-scroll functionality
✅ Copy buttons for each section