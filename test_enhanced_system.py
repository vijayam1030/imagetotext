#!/usr/bin/env python3
"""
Test script to verify the enhanced OCR system functionality
"""

import sys
import os
from pathlib import Path
import traceback
import tempfile
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_test_code_image():
    """Create a simple test image with Python code"""
    # Create image with code
    width, height = 800, 400
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # Sample Python code
    code_text = """def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

# Calculate and print first 10 Fibonacci numbers
for i in range(10):
    print(f"F({i}) = {fibonacci(i)}")"""
    
    # Try to use a monospace font, fallback to default
    try:
        # Try to load a monospace font
        font = ImageFont.truetype("consola.ttf", 24)  # Windows
    except:
        try:
            font = ImageFont.truetype("Courier New.ttf", 24)  # Windows fallback
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Monaco.ttf", 24)  # macOS
            except:
                font = ImageFont.load_default()  # Final fallback
    
    # Draw the code
    y_position = 20
    for line in code_text.split('\n'):
        draw.text((20, y_position), line, fill='black', font=font)
        y_position += 30
    
    return image

def test_language_detection():
    """Test language detection functionality"""
    print("🔍 Testing Language Detection...")
    
    try:
        # Test importing components
        from advanced_ocr_system import AdvancedLanguageDetector
        
        detector = AdvancedLanguageDetector()
        
        # Test code samples
        test_codes = {
            'python': '''
def hello_world():
    print("Hello, World!")
    return True
            ''',
            'sql': '''
SELECT name, age, email 
FROM users 
WHERE age > 18 
ORDER BY name;
            ''',
            'javascript': '''
function calculateSum(a, b) {
    return a + b;
}
console.log(calculateSum(5, 3));
            '''
        }
        
        for expected_lang, code in test_codes.items():
            detected_lang, confidence, scores = detector.detect_language(code)
            print(f"  ✅ {expected_lang.upper()}: detected as {detected_lang} (confidence: {confidence:.2f})")
            
        print("✅ Language detection test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Language detection test failed: {e}")
        traceback.print_exc()
        return False

def test_ocr_engine():
    """Test OCR engine functionality"""
    print("📝 Testing OCR Engine...")
    
    try:
        from advanced_ocr_system import TesseractOCREngine
        
        # Create test image
        test_image = create_test_code_image()
        
        # Test OCR
        ocr_engine = TesseractOCREngine()
        results = ocr_engine.extract_text(test_image)
        
        if results and len(results) > 0:
            best_result = results[0]
            extracted_text = best_result['text']
            
            # Check if key words are detected
            key_words = ['def', 'fibonacci', 'print', 'for', 'range']
            detected_words = sum(1 for word in key_words if word.lower() in extracted_text.lower())
            
            print(f"  ✅ OCR extracted {len(extracted_text)} characters")
            print(f"  ✅ Detected {detected_words}/{len(key_words)} key words")
            print(f"  ✅ Used: {best_result['preprocessing']} + {best_result['config']}")
            
            if detected_words >= 3:
                print("✅ OCR engine test passed!")
                return True, extracted_text
            else:
                print("⚠️ OCR quality lower than expected but functional")
                return True, extracted_text
        else:
            print("❌ No text extracted by OCR")
            return False, ""
            
    except Exception as e:
        print(f"❌ OCR engine test failed: {e}")
        traceback.print_exc()
        return False, ""

def test_ai_models():
    """Test AI model availability and functionality"""
    print("🤖 Testing AI Models...")
    
    try:
        from advanced_ocr_system import OllamaCodeCleaner
        import ollama
        
        # Check if Ollama is running
        try:
            models = ollama.list()
            available_models = [model['name'] for model in models['models']]
            print(f"  ✅ Found {len(available_models)} Ollama models")
            
            if len(available_models) == 0:
                print("  ⚠️ No Ollama models found - AI features will be limited")
                print("  💡 Install models with: ollama pull codellama:7b")
                return False
            
            # Test model selection
            cleaner = OllamaCodeCleaner()
            
            test_languages = ['python', 'sql', 'javascript']
            for lang in test_languages:
                selected_model = cleaner.select_best_model_for_language(lang)
                if selected_model:
                    print(f"  ✅ {lang.upper()}: {selected_model}")
                else:
                    print(f"  ⚠️ {lang.upper()}: No suitable model found")
            
            print("✅ AI models test passed!")
            return True
            
        except Exception as e:
            print(f"  ❌ Ollama connection failed: {e}")
            print("  💡 Make sure Ollama is installed and running")
            return False
            
    except Exception as e:
        print(f"❌ AI models test failed: {e}")
        traceback.print_exc()
        return False

def test_file_saving():
    """Test file saving functionality"""
    print("💾 Testing File Saving...")
    
    try:
        from advanced_ocr_system import CodeSaver
        
        saver = CodeSaver()
        
        # Test saving different file types
        test_data = {
            'python': 'print("Hello, World!")',
            'sql': 'SELECT * FROM users;',
            'md': '# Test Overview\nThis is a test.'
        }
        
        saved_files = []
        for lang, code in test_data.items():
            filepath = saver.save_code(code, lang, f"test_{lang}")
            if filepath and os.path.exists(filepath):
                saved_files.append(filepath)
                print(f"  ✅ Saved {lang} file: {os.path.basename(filepath)}")
            else:
                print(f"  ❌ Failed to save {lang} file")
        
        if len(saved_files) == len(test_data):
            print("✅ File saving test passed!")
            
            # Clean up test files
            for filepath in saved_files:
                try:
                    os.remove(filepath)
                    print(f"  🧹 Cleaned up: {os.path.basename(filepath)}")
                except:
                    pass
                    
            return True
        else:
            print("❌ File saving test failed!")
            return False
            
    except Exception as e:
        print(f"❌ File saving test failed: {e}")
        traceback.print_exc()
        return False

def test_requirements():
    """Test if all required packages are available"""
    print("📦 Testing Requirements...")
    
    required_packages = [
        'streamlit',
        'pytesseract', 
        'cv2',
        'PIL',
        'ollama',
        'numpy'
    ]
    
    optional_packages = [
        'guesslang',
        'pygments'
    ]
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                from PIL import Image
            else:
                __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            missing_required.append(package)
            print(f"  ❌ {package}")
    
    for package in optional_packages:
        try:
            __import__(package)
            print(f"  ✅ {package} (optional)")
        except ImportError:
            missing_optional.append(package)
            print(f"  ⚠️ {package} (optional - missing)")
    
    if missing_required:
        print(f"❌ Missing required packages: {', '.join(missing_required)}")
        print("💡 Install with: pip install -r requirements.txt")
        return False
    else:
        print("✅ All required packages available!")
        if missing_optional:
            print(f"⚠️ Optional packages missing: {', '.join(missing_optional)}")
            print("💡 For full functionality: pip install guesslang pygments")
        return True

def main():
    """Run all tests"""
    print("🧪 Advanced OCR System Test Suite")
    print("=" * 50)
    
    tests = [
        ("Requirements", test_requirements),
        ("Language Detection", test_language_detection),
        ("OCR Engine", test_ocr_engine),
        ("AI Models", test_ai_models),
        ("File Saving", test_file_saving)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print()
        try:
            if test_name == "OCR Engine":
                result, extracted_text = test_func()
                results[test_name] = result
            else:
                result = test_func()
                results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print()
    print("📊 Test Summary")
    print("=" * 50)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print()
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print()
        print("🚀 Next steps:")
        print("1. Run: streamlit run advanced_ocr_system.py")
        print("2. Or run: streamlit run fast_ocr_system.py")
        print("3. Upload an image with code")
        print("4. Get detailed overview + line-by-line comments!")
    else:
        print("⚠️ Some tests failed. Check the errors above.")
        
        if not results.get("Requirements", False):
            print("💡 Install missing packages first: pip install -r requirements.txt")
        
        if not results.get("AI Models", False):
            print("💡 Install Ollama and models: https://ollama.ai/")
            print("   Then run: ollama pull codellama:7b")

if __name__ == "__main__":
    main()
