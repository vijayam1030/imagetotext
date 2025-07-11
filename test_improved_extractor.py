#!/usr/bin/env python3
"""
Test script for the improved extraction system
"""
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import sys
import os

# Add the current directory to path to import our extractor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from improved_extractor import ImprovedExtractor
    print("✅ Successfully imported ImprovedExtractor")
except ImportError as e:
    print(f"❌ Failed to import ImprovedExtractor: {e}")
    sys.exit(1)

def create_test_sql_image():
    """Create a test image with SQL code"""
    img = Image.new('RGB', (600, 400), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    sql_code = """CREATE TABLE users (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(255)
);

SELECT * FROM users 
WHERE id > 0
ORDER BY name;

INSERT INTO users (id, name, email) 
VALUES (1, 'John Doe', 'john@example.com');"""
    
    # Draw the SQL code
    y_position = 30
    for line in sql_code.split('\n'):
        draw.text((20, y_position), line, fill='black', font=font)
        y_position += 25
    
    return img

def test_extraction():
    """Test the new extraction system"""
    print("🧪 Testing Improved Extraction System")
    print("=" * 50)
    
    # Create test image
    print("🖼️ Creating test SQL image...")
    test_img = create_test_sql_image()
    
    # Initialize extractor
    print("⚡ Initializing extractor...")
    extractor = ImprovedExtractor()
    
    # Test language detection
    print("\n🔍 Testing language detection...")
    test_sql = "SELECT * FROM users WHERE id > 0;"
    detected_lang, confidence, scores = extractor.detector.detect_language(test_sql)
    
    print(f"Language: {detected_lang}")
    print(f"Confidence: {confidence}")
    print(f"Scores: {scores}")
    
    # Test OCR extraction (if available)
    print("\n📝 Testing OCR extraction...")
    try:
        ocr_result = extractor.extract_with_multiple_ocr(test_img)
        print(f"OCR Result: {ocr_result[:100]}...")
    except Exception as e:
        print(f"OCR not available: {e}")
    
    # Test vision model extraction
    print("\n👁️ Testing vision model extraction...")
    try:
        vision_result = extractor.extract_with_vision_model(test_img, "sql")
        print(f"Vision Result: {vision_result[:100]}...")
    except Exception as e:
        print(f"Vision model error: {e}")
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    test_extraction()
