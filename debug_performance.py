#!/usr/bin/env python3
"""
Debug script to check Ollama performance issues
"""
import ollama
import time
import base64
from PIL import Image, ImageDraw, ImageFont
import io

def create_simple_test_image():
    """Create a very simple test image"""
    img = Image.new('RGB', (200, 100), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "print('hello')", fill='black')
    return img

def test_model_speed():
    """Test model loading and response time"""
    print("🔍 Testing Ollama Performance...")
    
    # Test 1: Check if model is loaded
    print("\n1. Checking loaded models...")
    try:
        result = ollama.list()
        models = [m['name'] for m in result.get('models', [])]
        print(f"Available models: {models}")
        
        if 'llama3.2-vision:11b' not in models:
            print("❌ llama3.2-vision:11b not found!")
            return False
    except Exception as e:
        print(f"❌ Error listing models: {e}")
        return False
    
    # Test 2: Simple text-only test (should be very fast)
    print("\n2. Testing text-only speed...")
    start_time = time.time()
    try:
        response = ollama.chat(
            model='llama3.2-vision:11b',
            messages=[{'role': 'user', 'content': 'Say hello'}],
            options={'num_gpu': -1, 'num_thread': 1}
        )
        text_time = time.time() - start_time
        print(f"✅ Text response time: {text_time:.2f} seconds")
        if text_time > 10:
            print("⚠️  WARNING: Text response is slow - GPU may not be working")
    except Exception as e:
        print(f"❌ Text test failed: {e}")
        return False
    
    # Test 3: Image processing test
    print("\n3. Testing image processing speed...")
    test_img = create_simple_test_image()
    buffered = io.BytesIO()
    test_img.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    
    start_time = time.time()
    try:
        response = ollama.chat(
            model='llama3.2-vision:11b',
            messages=[{
                'role': 'user',
                'content': 'What text do you see?',
                'images': [img_b64]
            }],
            options={
                'num_gpu': -1,
                'num_thread': 1,
                'temperature': 0.0,
                'num_predict': 50,
                'num_ctx': 512
            }
        )
        image_time = time.time() - start_time
        print(f"✅ Image response time: {image_time:.2f} seconds")
        print(f"Response: {response['message']['content']}")
        
        if image_time > 60:
            print("❌ CRITICAL: Image processing is extremely slow!")
            print("This suggests GPU is not being used properly.")
            return False
        elif image_time > 20:
            print("⚠️  WARNING: Image processing is slow but functional")
            
    except Exception as e:
        print(f"❌ Image test failed: {e}")
        return False
    
    print(f"\n✅ Performance test completed!")
    print(f"Expected time per image: ~{image_time:.1f} seconds")
    return True

if __name__ == "__main__":
    print("🧪 Ollama Performance Diagnostic")
    print("=" * 50)
    
    success = test_model_speed()
    
    if not success:
        print("\n❌ PERFORMANCE ISSUES DETECTED!")
        print("\nPossible solutions:")
        print("1. Restart Ollama: ollama serve")
        print("2. Check GPU: nvidia-smi")
        print("3. Reload model: ollama run llama3.2-vision:11b")
        print("4. Try smaller model: ollama pull llava:7b")
    else:
        print(f"\n✅ Ollama is working correctly!")
        
    print("\n" + "=" * 50)