# Advanced OCR Code Extractor with AI Analysis

**Transform code images into documented, analyzed, and cleaned code with AI-powered language detection and line-by-line explanations.**

## 🚀 Features

### Core Functionality
- **Advanced OCR**: Multiple Tesseract configurations with image preprocessing
- **Smart Language Detection**: Guesslang, Pygments, and pattern matching
- **AI Code Analysis**: Language-specific model selection for optimal results
- **Two-Section Output**: Detailed overview + line-by-line comments
- **File Management**: Auto-save and download capabilities

### AI Models Integration
- **CodeLlama**: Best for Python development
- **DeepSeek Coder**: Excellent for SQL and enterprise languages
- **Qwen2.5 Coder**: Optimized for web development (JavaScript)
- **WizardCoder**: Strong for complex codebases
- **Phi-3**: Fast general-purpose model

### Output Sections
1. **📊 Detailed Code Overview**: Comprehensive technical analysis including:
   - Purpose & functionality
   - Technical architecture
   - Key components
   - Data flow
   - Dependencies
   - Complexity assessment
   - Best practices analysis
   - Potential improvements

2. **💬 Line-by-Line Comments**: Educational documentation with:
   - Individual line explanations
   - Parameter details
   - Logic flow descriptions
   - Language-specific feature explanations
   - Learning-focused commentary

## 📋 Prerequisites

### Required Software
- **Python 3.8+**
- **Tesseract OCR** (for text extraction)
- **Ollama** (for AI models)

### System Dependencies

#### Windows
```bash
# Install Tesseract
winget install UB-Mannheim.TesseractOCR

# Install Ollama
winget install Ollama.Ollama
```

#### macOS
```bash
# Install Tesseract
brew install tesseract

# Install Ollama
brew install ollama
```

#### Linux
```bash
# Install Tesseract
sudo apt-get install tesseract-ocr

# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh
```

## 🔧 Installation

### 1. Clone and Setup
```bash
git clone <repository-url>
cd imagetotext
pip install -r requirements.txt
```

### 2. Install AI Models
```bash
# Essential models (choose based on your needs)
ollama pull codellama:7b          # Best for Python
ollama pull deepseek-coder:6.7b   # Best for SQL/databases
ollama pull qwen2.5-coder:7b      # Best for JavaScript/web
ollama pull phi3:medium           # Fast general-purpose

# Optional advanced models
ollama pull codellama:13b         # More capable Python model
ollama pull deepseek-coder:33b    # Enterprise-grade model
ollama pull wizardcoder:15b       # Complex code analysis
```

### 3. Test Installation
```bash
# Run comprehensive tests
python test_enhanced_system.py

# Or use batch file (Windows)
test_system.bat
```

## 🎯 Usage

### Advanced System (Full Features)
```bash
streamlit run advanced_ocr_system.py
```

### Fast System (Optimized for Speed)
```bash
streamlit run fast_ocr_system.py
```

## 📖 How to Use

1. **Upload Image**: Select an image containing code
2. **Configure Settings**: Adjust image display, quality, and OCR settings
3. **Extract & Analyze**: Click "Extract and Clean Code"
4. **Review Results**: Get four sections:
   - **Detailed Overview**: Comprehensive technical analysis
   - **Line-by-Line Comments**: Educational documentation
   - **Cleaned Code**: Fixed and formatted code
   - **Original Text**: Raw OCR output

## 🛠️ Advanced Features

### Language-Specific Model Selection
The system automatically selects the best AI model for each detected language:

- **SQL**: DeepSeek Coder (optimized for database queries)
- **Python**: CodeLlama (specialized for Python development)
- **JavaScript**: Qwen2.5 Coder (web development focused)
- **Java/C#**: DeepSeek Coder (enterprise language support)
- **General**: Phi-3 (fast, versatile model)

### OCR Optimization
- **7 OCR Configurations**: Different Tesseract settings for various text types
- **6 Image Preprocessing**: Enhancement techniques for better recognition
- **Early Stopping**: Stops when good results are achieved
- **Quality Analysis**: Provides image quality recommendations

### File Management
- **Auto-Save**: Automatically saves all extracted content
- **Download Options**: Individual download for each section
- **Multiple Formats**: Supports various file extensions based on language
- **Timestamp Tracking**: Organized file naming with timestamps

## 📊 Performance Comparison

| Feature | Advanced System | Fast System |
|---------|----------------|-------------|
| OCR Configurations | 7 | 3 |
| Image Preprocessing | 6 methods | 2 methods |
| Language Detection | 3 methods | 2 methods |
| AI Model Selection | Full logic | Simplified |
| Processing Time | 15-30 seconds | 5-15 seconds |
| Accuracy | Highest | Good |

## 🎨 UI Features

### Sidebar Controls
- **Image Display**: Width adjustment and quality settings
- **OCR Settings**: Preprocessing preview and configuration details
- **Model Information**: Available AI models and selection logic
- **File Management**: Auto-save options and download preferences

### Main Interface
- **Image Analysis**: Resolution, size, and quality recommendations
- **Progress Tracking**: Real-time status updates and timing information
- **Results Tabs**: Organized display of all output sections
- **Download Center**: Easy access to all generated content

## 🔍 Language Support

### Fully Supported Languages
- **Python**: Advanced analysis with CodeLlama
- **SQL**: Database-optimized with DeepSeek Coder
- **JavaScript**: Web-focused with Qwen2.5 Coder
- **Java**: Enterprise support with DeepSeek Coder
- **C#**: Microsoft stack with specialized models
- **C/C++**: Systems programming support

### Additional Languages
- TypeScript, PHP, Ruby, Go, Rust, Kotlin, Swift
- R, MATLAB, Perl, Shell scripting
- HTML, CSS, JSON, XML, YAML

## 🚨 Troubleshooting

### Common Issues

#### "No text extracted"
- **Cause**: Low image quality or complex formatting
- **Solution**: Use higher resolution images, ensure good contrast
- **Alternative**: Try different image preprocessing options

#### "No AI models available"
- **Cause**: Ollama not running or no models installed
- **Solution**: 
  ```bash
  # Start Ollama service
  ollama serve
  
  # Install basic model
  ollama pull codellama:7b
  ```

#### "Language detection failed"
- **Cause**: Missing guesslang or pygments
- **Solution**: 
  ```bash
  pip install guesslang pygments
  ```

#### Poor OCR accuracy
- **Cause**: Image quality issues
- **Solutions**:
  - Use images with at least 500×300 resolution
  - Ensure good contrast between text and background
  - Avoid skewed or rotated text
  - Use clear, readable fonts

### Performance Optimization

#### For Speed
- Use Fast System (`fast_ocr_system.py`)
- Choose smaller AI models (phi3:medium, codellama:7b)
- Reduce image resolution for faster processing

#### For Accuracy
- Use Advanced System (`advanced_ocr_system.py`)
- Install larger models (codellama:13b, deepseek-coder:33b)
- Use high-resolution images
- Enable all preprocessing options

## 📁 File Structure

```
imagetotext/
├── advanced_ocr_system.py      # Full-featured system
├── fast_ocr_system.py          # Speed-optimized system
├── test_enhanced_system.py     # Comprehensive test suite
├── test_system.bat             # Windows test runner
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── extracted_code/             # Output directory
    ├── raw_*.txt              # Original OCR output
    ├── overview_*.md          # Detailed analysis
    ├── commented_*.py         # Line-by-line comments
    └── cleaned_*.py           # Cleaned code
```

## 🤝 Contributing

1. **Test Changes**: Run `python test_enhanced_system.py`
2. **Follow Standards**: Use proper Python formatting and documentation
3. **Update Documentation**: Keep README.md current with new features
4. **Performance**: Consider both accuracy and speed implications

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Tesseract OCR**: Open-source OCR engine
- **Ollama**: Local AI model management
- **Streamlit**: Web interface framework
- **OpenCV**: Image processing
- **Guesslang**: Machine learning language detection
- **Pygments**: Syntax highlighting and language detection

---

**Ready to transform your code images into documented, analyzed, and cleaned code!** 🚀
