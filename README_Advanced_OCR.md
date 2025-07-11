# Advanced OCR Code Extraction System

A comprehensive OCR system for extracting and cleaning code from images using multiple AI models and advanced language detection.

## 🌟 Features

### 🔧 Core Technologies
- **Tesseract OCR Engine** with PyTesseract wrapper
- **Multiple Image Preprocessing** techniques (6 different approaches)
- **Advanced Language Detection** using Guesslang, Pygments, and pattern matching
- **AI Code Cleanup** with 4 different Ollama models
- **Automatic File Saving** with proper extensions
- **Streamlit Web Interface** for easy use

### 🤖 AI Models Used
- **CodeLlama 13B** - Code understanding and cleanup
- **DeepSeek Coder V2 16B** - Advanced code analysis
- **WizardCoder 34B** - Code optimization and formatting
- **Phi-3 Medium** - Lightweight code processing
- **Llama 3.2 Vision 11B** - Image-to-text fallback

### 🔍 Language Support
- SQL, Python, JavaScript, TypeScript
- Java, C#, C++, C, PHP, Ruby
- Go, Rust, Kotlin, Swift, R
- MATLAB, Perl, Shell, PowerShell
- YAML, JSON, XML, HTML, CSS

## 📋 Prerequisites

### System Requirements
- Python 3.8 or higher
- Windows, macOS, or Linux
- At least 8GB RAM (16GB recommended for larger models)
- Internet connection for model downloads

### Required Software
1. **Tesseract OCR**
   - Windows: Download from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
   - macOS: `brew install tesseract`
   - Linux: `sudo apt-get install tesseract-ocr`

2. **Ollama**
   - Download from [ollama.ai](https://ollama.ai)
   - Start the Ollama service after installation

## 🚀 Installation

### Step 1: Clone/Download Files
Download all the files to your working directory:
- `advanced_ocr_system.py` - Main application
- `setup_advanced_ocr.py` - Setup script
- `test_advanced_system.py` - Test script
- `requirements_advanced.txt` - Python dependencies

### Step 2: Install Python Dependencies
```bash
pip install -r requirements_advanced.txt
```

### Step 3: Run Setup Script
```bash
python setup_advanced_ocr.py
```

This will:
- Check all dependencies
- Download required Ollama models
- Create test images
- Verify system configuration

### Step 4: Test the System
```bash
python test_advanced_system.py
```

## 🎯 Usage

### Starting the Application
```bash
streamlit run advanced_ocr_system.py
```

### Using the Web Interface
1. **Upload Image**: Choose an image file containing code
2. **Extract Text**: Click "Extract and Clean Code"
3. **Review Results**: 
   - See OCR extraction details
   - Check language detection results
   - Compare cleaned versions from different AI models
4. **Download/Save**: Save cleaned code with proper file extensions

### Supported Image Formats
- PNG, JPG, JPEG, BMP, TIFF, GIF
- Recommended: High contrast, clear text
- Best results: 300+ DPI, black text on white background

## 🔧 System Components

### OCR Engine Features
- **7 OCR Configurations**: Different Tesseract PSM modes
- **6 Image Preprocessing**: Contrast, threshold, morphological operations
- **Automatic Best Selection**: Chooses highest quality extraction
- **Error Recovery**: Fallback options for difficult images

### Language Detection
- **Guesslang**: ML-based language detection
- **Pygments**: Lexer-based detection
- **Pattern Matching**: Regex-based keyword detection
- **Confidence Scoring**: Weighted combination of all methods

### AI Code Cleanup
Each model specializes in different aspects:
- **CodeLlama**: General code understanding
- **DeepSeek**: Advanced syntax correction
- **WizardCoder**: Code optimization
- **Phi-3**: Fast processing for simple fixes

## 📁 File Management

### Automatic Saving
- Raw OCR text saved as `raw_ocr_[language]_[timestamp].[ext]`
- Cleaned code saved as `cleaned_[model]_[language]_[timestamp].[ext]`
- Files saved in `extracted_code/` directory

### File Extensions
- `.sql` for SQL scripts
- `.py` for Python code
- `.js` for JavaScript
- `.java` for Java
- And many more based on detected language

## 🔍 Troubleshooting

### Common Issues

**OCR Not Working**
```
Error: pytesseract not installed
```
- Install: `pip install pytesseract`
- Install Tesseract OCR executable
- Add Tesseract to system PATH

**Models Not Available**
```
Error: Model not found
```
- Check Ollama is running: `ollama list`
- Pull missing models: `ollama pull codellama:13b`
- Wait for model downloads to complete

**Language Detection Issues**
```
Warning: Guesslang not available
```
- Install: `pip install guesslang`
- System will use Pygments and pattern matching as fallback

**Memory Issues**
```
Error: Out of memory
```
- Use smaller models (phi3:medium instead of wizardcoder:34b)
- Close other applications
- Process smaller images

### Performance Tips

1. **For Best OCR Results**:
   - Use high-resolution images (300+ DPI)
   - Ensure good contrast
   - Avoid skewed or rotated text
   - Clean, clear fonts work best

2. **For Faster Processing**:
   - Use smaller AI models
   - Process one image at a time
   - Disable auto-save for large batches

3. **For Better Language Detection**:
   - Include more code context
   - Use typical language patterns
   - Include imports/declarations

## 📊 Model Comparison

| Model | Size | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| Phi-3 Medium | ~14GB | Fast | Good | Quick fixes |
| CodeLlama 13B | ~7GB | Medium | Very Good | General code |
| DeepSeek 16B | ~9GB | Slow | Excellent | Complex syntax |
| WizardCoder 34B | ~19GB | Very Slow | Excellent | Production code |

## 🤝 Contributing

To extend the system:

1. **Add New Languages**: Update `AdvancedLanguageDetector.patterns`
2. **Add OCR Configs**: Extend `TesseractOCREngine.ocr_configs`
3. **Add AI Models**: Update `OllamaCodeCleaner.models`
4. **Add File Types**: Update `CodeSaver.extensions`

## 📄 License

This project is open source. Feel free to modify and distribute.

## 🔗 Links

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [Ollama](https://ollama.ai)
- [Streamlit](https://streamlit.io)
- [Guesslang](https://github.com/yoeo/guesslang)
- [Pygments](https://pygments.org)

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Verify all dependencies are installed
3. Test with the provided test images
4. Check Ollama model availability

---

**🎉 Enjoy extracting and cleaning your code with AI!**
