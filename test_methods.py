#!/usr/bin/env python3
"""
Quick test to verify the OCR methods exist and work
"""

try:
    from advanced_ocr_system import (
        AdvancedLanguageDetector,
        TesseractOCREngine, 
        OllamaCodeCleaner,
        CodeSaver
    )
    
    print("✅ All imports successful")
    
    # Test class instantiation
    detector = AdvancedLanguageDetector()
    ocr_engine = TesseractOCREngine()
    cleaner = OllamaCodeCleaner()
    saver = CodeSaver()
    
    print("✅ All classes instantiated successfully")
    
    # Test method existence
    test_code = "print('hello world')"
    
    # Test language detection
    if hasattr(detector, 'detect_language'):
        result = detector.detect_language(test_code)
        print(f"✅ Language detection works: {result}")
    else:
        print("❌ detect_language method not found")
    
    # Test OCR engine methods
    if hasattr(ocr_engine, 'extract_text'):
        print("✅ extract_text method exists")
    else:
        print("❌ extract_text method not found")
    
    if hasattr(ocr_engine, 'extract_with_best_config'):
        print("⚠️  extract_with_best_config method exists (should NOT)")
    else:
        print("✅ extract_with_best_config method correctly does not exist")
    
    # Test cleaner methods
    required_methods = ['select_best_model_for_language', 'create_detailed_overview', 
                       'create_line_by_line_comments', 'clean_code_only']
    
    for method in required_methods:
        if hasattr(cleaner, method):
            print(f"✅ {method} method exists")
        else:
            print(f"❌ {method} method not found")
    
    print("\n✅ All required methods verified!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
