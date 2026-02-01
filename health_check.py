"""
Quick health check - Run this anytime to verify system status
"""
import sys
sys.path.append('.')

def check_imports():
    """Check if all critical imports work"""
    try:
        import torch
        import numpy as np
        import cv2
        
        import ee
        from src.data.gee_fetcher import GEEFetcher
        from src.data.dataset import GEEStreamingDataset
        return True, "All imports OK"
    except Exception as e:
        return False, f"Import failed: {e}"

def check_gee():
    """Check GEE connection"""
    try:
        import ee
        ee.Initialize(project='projectklymo')
        return True, "GEE connected"
    except Exception as e:
        return False, f"GEE failed: {e}"

def check_config():
    """Check configuration"""
    try:
        import yaml
        config = yaml.safe_load(open('config.yaml'))
        return True, f"Config OK (source: {config['data']['primary_source']})"
    except Exception as e:
        return False, f"Config failed: {e}"

def main():
    print("🏥 Health Check")
    print("=" * 40)
    
    tests = [
        ("Imports", check_imports),
        ("GEE Connection", check_gee),
        ("Configuration", check_config),
    ]
    
    all_passed = True
    for name, test_func in tests:
        passed, msg = test_func()
        status = "✅" if passed else "❌"
        print(f"{status} {name}: {msg}")
        if not passed:
            all_passed = False
    
    print("=" * 40)
    if all_passed:
        print("✅ System Healthy")
        return 0
    else:
        print("❌ System Has Issues")
        return 1

if __name__ == "__main__":
    exit(main())
