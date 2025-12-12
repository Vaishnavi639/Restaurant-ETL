from restaurant_etl.extractors.universal_extractor import UniversalExtractor

def test_extraction(file_path: str):
    """Test extraction on any file."""
    
    print(f"\n{'='*60}")
    print(f"Testing: {file_path}")
    print(f"{'='*60}\n")
    
    extractor = UniversalExtractor()
    
    try:
        result = extractor.extract(file_path)
        
        print(f"✅ Extraction successful!")
        print(f"📊 Method: {result['extraction_method']}")
        print(f"📊 Characters: {result['char_count']}")
        print(f"\n📝 First 500 characters:\n{'-'*60}")
        print(result['text'][:500])
        print(f"{'-'*60}\n")
        
        return result
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


if __name__ == "__main__":
    # Test with your files
    test_extraction("great_india.pdf")
    # test_extraction("path/to/your/menu.jpg")
