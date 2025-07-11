# Simple LLM Usage Flow - Text Diagram

## 🔄 MAIN APPLICATION FLOW

```
┌─────────────────────────────────────────────────────────────────┐
│                        OCR APPLICATION FLOW                     │
└─────────────────────────────────────────────────────────────────┘

📤 USER UPLOADS IMAGE
         │
         ▼
🖼️  IMAGE PROCESSING & DISPLAY
         │
         ▼
🔍 STEP 1: OCR TEXT EXTRACTION (No LLM)
   ├── Tesseract with 7 different configurations
   ├── 6 image preprocessing methods  
   └── Best result selected
         │
         ▼
🎯 STEP 2: LANGUAGE DETECTION (Minimal LLM)
   ├── Guesslang ML Model (optional)
   ├── Pygments Lexer Analysis
   └── Pattern Matching
         │
         ▼
🤖 STEP 3: AI MODEL SELECTION
   ├── Language-specific model priority
   ├── User dropdown selection with size info
   └── Model availability check
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   🦙 OLLAMA LLM PROCESSING                      │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   LLM CALL 1    │  │   LLM CALL 2    │  │   LLM CALL 3    │ │
│  │ Detailed        │  │ Line-by-Line    │  │ Code Cleaning   │ │
│  │ Overview        │  │ Comments        │  │ (No Comments)   │ │
│  │                 │  │                 │  │                 │ │
│  │ Temperature:0.1 │  │ Temperature:0.0 │  │ Temperature:0.1 │ │
│  │ Tokens: 2500    │  │ Tokens: 4000    │  │ Tokens: 2000    │ │
│  │ Focus: Analysis │  │ Focus: Education│  │ Focus: Cleanup  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
📊 SECTION 1          💬 SECTION 2          🔧 SECTION 3          📄 SECTION 4
Detailed Overview     Line-by-Line         Cleaned Code         Raw OCR Output
(LLM Generated)       Comments             (LLM Generated)      (No LLM)
                      (LLM Generated)
         │                       │                       │                │
         └───────────────────────┼───────────────────────┼────────────────┘
                                 ▼
                        💾 SAVE & DOWNLOAD OPTIONS
                                 │
                                 ▼
                        ✅ COMPLETE ANALYSIS READY
```

## 🤖 LLM USAGE SUMMARY

### **WHERE LLMs ARE USED:**

1. **Language Detection (Optional)**
   - Guesslang ML model for code language detection
   - Only if guesslang package is installed
   - Fallback: Pygments + Pattern matching

2. **Detailed Code Overview (MAIN LLM CALL #1)**
   - **Input:** Raw OCR text + Detected language
   - **Process:** Comprehensive technical analysis
   - **Output:** 10-section detailed overview
   - **Model:** Language-specific best model
   - **Tokens:** Up to 2500

3. **Line-by-Line Comments (MAIN LLM CALL #2)**
   - **Input:** Raw OCR text + Detected language  
   - **Process:** Educational documentation
   - **Output:** Code with detailed comments on every line
   - **Model:** Same as overview
   - **Tokens:** Up to 4000

4. **Code Cleaning (MAIN LLM CALL #3)**
   - **Input:** Raw OCR text + Detected language
   - **Process:** Fix OCR errors, format code
   - **Output:** Clean, formatted code without comments
   - **Model:** Same as overview
   - **Tokens:** Up to 2000

### **WHERE LLMs ARE NOT USED:**

❌ **OCR Text Extraction** - Pure Tesseract OCR
❌ **Image Processing** - OpenCV/PIL operations  
❌ **File I/O Operations** - Standard Python file operations
❌ **UI Rendering** - Streamlit framework
❌ **Raw Text Display** - Direct OCR output shown as-is

### **TOTAL LLM RESOURCE USAGE PER IMAGE:**
- **API Calls:** 3 calls to Ollama
- **Total Tokens:** ~8,500 tokens (2500 + 4000 + 2000)
- **Processing Time:** 30 seconds to 5 minutes (depends on model size)
- **Memory Usage:** Depends on selected model (1.1GB to 19GB)

### **MODEL SELECTION INTELLIGENCE:**
```
SQL Code      → deepseek-coder-v2:16b (Best for enterprise)
Python Code   → codellama:13b (Best for Python)
JavaScript    → qwen2.5-coder:7b (Best for web dev)
Java/C#       → deepseek-coder-v2:16b (Enterprise languages)
Unknown/Mixed → phi3:medium (Fast general purpose)
```

### **PERFORMANCE CHARACTERISTICS:**
- **Small Models (1.5b-7b):** Fast, Good quality
- **Medium Models (13b-16b):** Balanced, Excellent quality  
- **Large Models (34b):** Slow, Outstanding quality

The application provides **intelligent model selection** and **comprehensive LLM-powered analysis** while maintaining **fast OCR extraction** and **responsive UI**.
