#!/usr/bin/env python3
"""
Quick test script for the Advanced OCR System
"""

import sys
import os
from PIL import Image, ImageDraw, ImageFont

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_test_images():
    """Create test images with different types of code"""
    
    # SQL Test Image
    sql_img = Image.new('RGB', (600, 400), color='white')
    draw = ImageDraw.Draw(sql_img)
    
    sql_code = """-- Create users table
CREATE TABLE users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample data
INSERT INTO users (username, email) 
VALUES ('john_doe', 'john@example.com'),
       ('jane_smith', 'jane@example.com');

-- Query users
SELECT id, username, email 
FROM users 
WHERE created_at > '2024-01-01'
ORDER BY username;"""
    
    y_pos = 20
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    for line in sql_code.split('\n'):
        draw.text((20, y_pos), line, fill='black', font=font)
        y_pos += 18
    
    sql_img.save('test_sql_code.png')
    print("✅ Created SQL test image: test_sql_code.png")
    
    # Python Test Image
    python_img = Image.new('RGB', (600, 400), color='white')
    draw = ImageDraw.Draw(python_img)
    
    python_code = """# Python function to process data
import json
import requests

def fetch_user_data(user_id):
    \"\"\"Fetch user data from API\"\"\"
    url = f"https://api.example.com/users/{user_id}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def main():
    user_data = fetch_user_data(123)
    if user_data:
        print(f"User: {user_data['name']}")
    
if __name__ == "__main__":
    main()"""
    
    y_pos = 20
    for line in python_code.split('\n'):
        draw.text((20, y_pos), line, fill='black', font=font)
        y_pos += 18
    
    python_img.save('test_python_code.png')
    print("✅ Created Python test image: test_python_code.png")
    
    # JavaScript Test Image
    js_img = Image.new('RGB', (600, 400), color='white')
    draw = ImageDraw.Draw(js_img)
    
    js_code = """// JavaScript function to handle user login
const login = async (username, password) => {
    try {
        const response = await fetch('/api/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ username, password })
        });
        
        if (!response.ok) {
            throw new Error('Login failed');
        }
        
        const data = await response.json();
        localStorage.setItem('token', data.token);
        return data;
    } catch (error) {
        console.error('Login error:', error);
        return null;
    }
};"""
    
    y_pos = 20
    for line in js_code.split('\n'):
        draw.text((20, y_pos), line, fill='black', font=font)
        y_pos += 18
    
    js_img.save('test_javascript_code.png')
    print("✅ Created JavaScript test image: test_javascript_code.png")

def test_language_detection():
    """Test language detection functionality"""
    try:
        from advanced_ocr_system import AdvancedLanguageDetector
        
        detector = AdvancedLanguageDetector()
        
        test_codes = {
            'SQL': "SELECT * FROM users WHERE id > 0 ORDER BY name;",
            'Python': "def hello(): print('Hello World')",
            'JavaScript': "function hello() { console.log('Hello World'); }"
        }
        
        print("\n🔍 Testing Language Detection:")
        print("-" * 40)
        
        for lang_name, code in test_codes.items():
            detected, confidence, scores = detector.detect_language(code)
            print(f"{lang_name} code:")
            print(f"  Detected: {detected} (confidence: {confidence:.2f})")
            print(f"  All scores: {scores}")
            print()
            
    except ImportError as e:
        print(f"❌ Could not test language detection: {e}")

def test_imports():
    """Test if all required packages can be imported"""
    print("📦 Testing Package Imports:")
    print("-" * 30)
    
    packages = [
        ('streamlit', 'Streamlit'),
        ('pytesseract', 'PyTesseract'),
        ('cv2', 'OpenCV'),
        ('ollama', 'Ollama'),
        ('numpy', 'NumPy'),
        ('PIL', 'Pillow'),
        ('guesslang', 'Guesslang'),
        ('pygments', 'Pygments')
    ]
    
    for package, name in packages:
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - needs installation")

def main():
    """Main test function"""
    print("🧪 Advanced OCR System - Quick Test")
    print("=" * 40)
    
    # Test imports
    test_imports()
    
    # Create test images
    print("\n🖼️ Creating Test Images:")
    print("-" * 25)
    create_test_images()
    
    # Test language detection
    test_language_detection()
    
    print("\n📋 Next Steps:")
    print("1. Install missing packages: pip install -r requirements_advanced.txt")
    print("2. Install Tesseract OCR")
    print("3. Install and start Ollama")
    print("4. Run setup: python setup_advanced_ocr.py")
    print("5. Start the app: streamlit run advanced_ocr_system.py")
    print("6. Upload one of the test images created above")

if __name__ == "__main__":
    main()
