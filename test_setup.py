"""
Quick test script to verify Day 1 setup
"""

import sys
sys.path.append('.')

def test_imports():
    """Test all required imports"""
    print("🧪 Testing imports...")
    
    try:
        import ee
        print("  ✅ earthengine-api")
    except ImportError as e:
        print(f"  ❌ earthengine-api: {e}")
        return False
    
    try:
        import torch
        print(f"  ✅ torch (version {torch.__version__})")
        print(f"     GPU available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"  ❌ torch: {e}")
        return False
    
    try:
        import cv2
        print("  ✅ opencv-python")
    except ImportError as e:
        print(f"  ❌ opencv-python: {e}")
        return False
    
    try:
        import rasterio
        print("  ✅ rasterio")
    except ImportError as e:
        print(f"  ❌ rasterio: {e}")
        return False
    
    try:
        import albumentations
        print("  ✅ albumentations")
    except ImportError as e:
        print(f"  ❌ albumentations: {e}")
        return False
    
    try:
        import yaml
        print("  ✅ pyyaml")
    except ImportError as e:
        print(f"  ❌ pyyaml: {e}")
        return False
    
    return True


def test_gee_connection():
    """Test Google Earth Engine connection"""
    print("\n🌍 Testing GEE connection...")
    
    try:
        import ee
        ee.Initialize(project='projectklymo')
        print("  ✅ Successfully connected to GEE project: projectklymo")
        
        # Test a simple query
        image = ee.Image('COPERNICUS/S2_SR/20230101T000439_20230101T000436_T43QFU')
        info = image.getInfo()
        print(f"  ✅ Test query successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ GEE connection failed: {e}")
        print("     Run: earthengine authenticate")
        return False


def test_config():
    """Test config file loading"""
    print("\n⚙️ Testing config...")
    
    try:
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"  ✅ Config loaded successfully")
        print(f"     Project ID: {config['gee']['project_id']}")
        print(f"     Scale factor: {config['data']['scale_factor']}x")
        print(f"     Patch size: {config['data']['hr_patch_size']}px")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Config loading failed: {e}")
        return False


def test_modules():
    """Test custom modules"""
    print("\n📦 Testing custom modules...")
    
    try:
        from src.data.gee_fetcher import GEEFetcher
        print("  ✅ GEEFetcher imported")
        
        from src.data.preprocessing import DataPreprocessor
        print("  ✅ DataPreprocessor imported")
        
        from src.data.dataset import SentinelSRDataset
        print("  ✅ SentinelSRDataset imported")
        
        from src.data.augmentation import get_training_augmentation
        print("  ✅ Augmentation imported")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Module import failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("🚀 DAY 1 SETUP VERIFICATION")
    print("=" * 60)
    
    results = []
    
    # Test imports
    results.append(("Imports", test_imports()))
    
    # Test GEE connection
    results.append(("GEE Connection", test_gee_connection()))
    
    # Test config
    results.append(("Config", test_config()))
    
    # Test modules
    results.append(("Custom Modules", test_modules()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Day 1 setup is complete.")
        print("\n📝 Next steps:")
        print("  1. Open notebooks/01_day1_data_ingestion.ipynb")
        print("  2. Run the notebook to fetch training data")
        print("  3. Start with 50 samples for testing")
    else:
        print("\n⚠️ Some tests failed. Please fix the issues above.")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
