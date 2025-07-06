#!/usr/bin/env python3
"""
Quick script to check GPU usage and optimize Ollama for better performance
"""
import subprocess
import sys

def check_gpu_usage():
    """Check if GPU is being used by Ollama"""
    try:
        # Check if nvidia-smi is available
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected")
            print(result.stdout)
            
            # Check if ollama is using GPU
            if 'ollama' in result.stdout.lower():
                print("🚀 Ollama is using GPU!")
            else:
                print("⚠️  Ollama might not be using GPU")
        else:
            print("❌ No NVIDIA GPU detected or nvidia-smi not available")
            
    except FileNotFoundError:
        print("❌ nvidia-smi not found - checking for AMD GPU...")
        try:
            result = subprocess.run(['rocm-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ AMD GPU detected")
                print(result.stdout)
            else:
                print("❌ No AMD GPU detected")
        except FileNotFoundError:
            print("❌ No GPU tools found")

def check_ollama_status():
    """Check if Ollama is running and which models are loaded"""
    try:
        result = subprocess.run(['ollama', 'ps'], capture_output=True, text=True)
        if result.returncode == 0:
            print("📊 Ollama Model Status:")
            print(result.stdout)
        else:
            print("❌ Ollama not running or not accessible")
    except FileNotFoundError:
        print("❌ Ollama command not found")

if __name__ == "__main__":
    print("🔍 Checking GPU and Ollama status...\n")
    check_gpu_usage()
    print("\n" + "="*50 + "\n")
    check_ollama_status()
    
    print("\n💡 Tips for faster performance:")
    print("1. Make sure Ollama is using GPU: ollama serve")
    print("2. Keep models loaded: ollama run llama3.2-vision:11b")
    print("3. Monitor GPU usage: nvidia-smi -l 1")
    print("4. Use smaller models if speed is critical")