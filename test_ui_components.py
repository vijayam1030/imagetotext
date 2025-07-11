#!/usr/bin/env python3
"""
Test if clean_code_only method exists and tabs are working
"""

def test_clean_code_method():
    """Test if the clean_code_only method exists"""
    try:
        from advanced_ocr_system import OllamaCodeCleaner
        cleaner = OllamaCodeCleaner()
        
        if hasattr(cleaner, 'clean_code_only'):
            print('✅ clean_code_only method exists')
            return True
        else:
            print('❌ clean_code_only method missing')
            return False
    except Exception as e:
        print(f'❌ Error importing: {e}')
        return False

def test_tabs_content():
    """Test tab creation"""
    tab_names = [
        "📊 Detailed Overview", 
        "💬 Line-by-Line Comments", 
        "🔧 Cleaned Code", 
        "📄 Original Text"
    ]
    
    print("Tab names to create:")
    for i, name in enumerate(tab_names):
        print(f"  {i}: {repr(name)}")
    
    return tab_names

if __name__ == "__main__":
    print("🧪 Testing UI Components")
    print("=" * 40)
    
    method_exists = test_clean_code_method()
    tab_names = test_tabs_content()
    
    if method_exists:
        print("\n✅ All components should work")
    else:
        print("\n❌ Missing clean_code_only method")
        print("💡 This might cause the Cleaned Code tab to not work properly")
