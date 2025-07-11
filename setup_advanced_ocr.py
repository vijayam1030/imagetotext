#!/usr/bin/env python3
"""
Setup script for Advanced OCR System
Downloads required Ollama models and checks dependencies
"""

import subprocess
import sys
import ollama
import time

def run_command(command, description):
    """Run a command and show progress"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error during {description}: {e}")
        return False

def install_python_packages():
    """Install required Python packages"""
    packages = [
        "streamlit>=1.28.0",
        "pytesseract>=0.3.10", 
        "opencv-python>=4.8.0",
        "ollama>=0.1.7",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "guesslang>=2.2.1",
        "pygments>=2.15.0",
        "regex>=2023.6.3"
    ]
    
    print("📦 Installing Python packages...")
    for package in packages:
        success = run_command(f"pip install {package}", f"Installing {package}")
        if not success:
            print(f"⚠️ Failed to install {package}, continuing...")

def pull_ollama_models():
    """Pull required Ollama models"""
    models = [
        "codellama:13b",
        "deepseek-coder-v2:16b", 
        "wizardcoder:34b",
        "phi3:medium",
        "llama3.2-vision:11b"
    ]
    
    print("\n🤖 Pulling Ollama models...")
    print("This may take a while depending on your internet connection...")
    
    for model in models:
        print(f"\n🔄 Pulling {model}...")
        try:
            # Use ollama.pull() method
            ollama.pull(model)
            print(f"✅ Successfully pulled {model}")
        except Exception as e:
            print(f"❌ Failed to pull {model}: {e}")
            print(f"You can manually pull it later with: ollama pull {model}")

def check_tesseract():
    """Check if Tesseract is installed"""
    print("\n🔍 Checking Tesseract installation...")
    try:
        import pytesseract
        result = subprocess.run(["tesseract", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Tesseract is installed and accessible")
            print(f"Version: {result.stdout.split()[1]}")
            return True
        else:
            print("❌ Tesseract not found in PATH")
            return False
    except ImportError:
        print("❌ pytesseract not installed")
        return False
    except FileNotFoundError:
        print("❌ Tesseract executable not found")
        print("Please install Tesseract OCR:")
        print("Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        print("macOS: brew install tesseract")
        print("Linux: sudo apt-get install tesseract-ocr")
        return False

def check_ollama():
    """Check if Ollama is running"""
    print("\n🦙 Checking Ollama...")
    try:
        models = ollama.list()
        print("✅ Ollama is running")
        print(f"Available models: {len(models['models'])}")
        for model in models['models']:
            print(f"  • {model['name']}")
        return True
    except Exception as e:
        print(f"❌ Ollama not accessible: {e}")
        print("Please make sure Ollama is installed and running:")
        print("1. Install from: https://ollama.ai")
        print("2. Start Ollama service")
        return False

def create_test_image():
    """Create a test image with code"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        img = Image.new('RGB', (500, 300), color='white')
        draw = ImageDraw.Draw(img)
        
        code_text = """CREATE TABLE users (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(255)
);

SELECT * FROM users 
WHERE id > 0
ORDER BY name;"""
        
        try:
            font = ImageFont.load_default()
        except:
            font = None
        
        y_pos = 30
        for line in code_text.split('\n'):
            draw.text((20, y_pos), line, fill='black', font=font)
            y_pos += 25
        
        img.save('test_code_image.png')
        print("✅ Created test image: test_code_image.png")
        return True
    except Exception as e:
        print(f"❌ Failed to create test image: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Advanced OCR System Setup")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print(f"❌ Python 3.8+ required, found {python_version.major}.{python_version.minor}")
        return
    else:
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Install Python packages
    install_python_packages()
    
    # Check Tesseract
    tesseract_ok = check_tesseract()
    
    # Check Ollama
    ollama_ok = check_ollama()
    
    # Pull models if Ollama is available
    if ollama_ok:
        pull_ollama_models()
    
    # Create test image
    create_test_image()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Setup Summary:")
    print(f"Python packages: ✅")
    print(f"Tesseract OCR: {'✅' if tesseract_ok else '❌'}")
    print(f"Ollama: {'✅' if ollama_ok else '❌'}")
    
    if tesseract_ok and ollama_ok:
        print("\n🎉 Setup completed successfully!")
        print("Run the system with: streamlit run advanced_ocr_system.py")
    else:
        print("\n⚠️ Some components need manual installation")
        if not tesseract_ok:
            print("• Install Tesseract OCR")
        if not ollama_ok:
            print("• Install and start Ollama")

if __name__ == "__main__":
    main()
