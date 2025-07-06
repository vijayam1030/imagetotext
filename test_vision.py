#!/usr/bin/env python3
"""
Test script to verify LLaVA vision model is working
"""
import ollama
import base64
from PIL import Image, ImageDraw, ImageFont
import io

def create_test_image():
    """Create a simple test image with text"""
    img = Image.new('RGB', (400, 200), color='white')
    draw = ImageDraw.Draw(img)
    
    # Add some text
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    draw.text((50, 50), "Hello World!", fill='black', font=font)
    draw.text((50, 100), "def test():", fill='blue', font=font)
    draw.text((70, 120), "    print('Test')", fill='blue', font=font)
    
    return img

def test_ollama_vision():
    """Test if Ollama can process images with LLaVA"""
    print("🔍 Creating test image...")
    test_img = create_test_image()
    
    # Convert to base64
    buffered = io.BytesIO()
    test_img.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    
    print("📸 Testing LLaVA vision model...")
    try:
        response = ollama.chat(
            model='llava:7b',
            messages=[
                {
                    'role': 'user',
                    'content': 'What text do you see in this image?',
                    'images': [img_b64]
                }
            ],
            options={
                'num_gpu': -1,
                'num_thread': 2,
                'temperature': 0.1,
            }
        )
        
        print("✅ Model response:")
        print(response['message']['content'])
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Ollama LLaVA Vision Model")
    print("=" * 40)
    
    # Check if model is loaded
    try:
        models = ollama.list()
        print("📋 Available models:")
        for model in models['models']:
            print(f"  - {model['name']}")
    except Exception as e:
        print(f"❌ Error listing models: {e}")
    
    print("\n" + "=" * 40)
    success = test_ollama_vision()
    
    if success:
        print("\n✅ Vision model is working!")
    else:
        print("\n❌ Vision model test failed!")