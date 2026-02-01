"""
Quick test script to download WorldStrat dataset from Kaggle
"""

print("=" * 60)
print("🚀 WorldStrat Dataset Download Test")
print("=" * 60)

# Test 1: Check kagglehub installation
print("\n1️⃣ Checking kagglehub installation...")
try:
    import kagglehub
    print(f"   ✅ kagglehub version: {kagglehub.__version__}")
except ImportError:
    print("   ❌ kagglehub not installed")
    print("   Run: pip install kagglehub")
    exit(1)

# Test 2: Check Kaggle credentials
print("\n2️⃣ Checking Kaggle credentials...")
import os
from pathlib import Path

kaggle_json_locations = [
    Path.home() / ".kaggle" / "kaggle.json",
    Path(os.environ.get("KAGGLE_CONFIG_DIR", "")) / "kaggle.json" if os.environ.get("KAGGLE_CONFIG_DIR") else None
]

kaggle_json_found = False
for loc in kaggle_json_locations:
    if loc and loc.exists():
        print(f"   ✅ Found kaggle.json at: {loc}")
        kaggle_json_found = True
        break

if not kaggle_json_found:
    print("   ⚠️ kaggle.json not found!")
    print("   📝 Setup instructions:")
    print("      1. Go to https://www.kaggle.com/settings")
    print("      2. Click 'Create New API Token'")
    print("      3. Save kaggle.json to: " + str(Path.home() / ".kaggle" / "kaggle.json"))
    print("\n   Attempting download anyway (may fail)...")

# Test 3: Download WorldStrat dataset
print("\n3️⃣ Downloading WorldStrat dataset...")
print("   ⏳ This may take a few minutes on first download...")

try:
    path = kagglehub.dataset_download("jucor1/worldstrat")
    print(f"\n   ✅ Dataset downloaded successfully!")
    print(f"   📂 Location: {path}")
    
    # Test 4: Inspect contents
    print("\n4️⃣ Inspecting dataset structure...")
    dataset_path = Path(path)
    
    print(f"\n   📁 Contents of {dataset_path.name}:")
    for item in sorted(dataset_path.iterdir()):
        if item.is_dir():
            num_files = sum(1 for _ in item.rglob('*') if _.is_file())
            print(f"      📂 {item.name}/ ({num_files} files)")
        else:
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"      📄 {item.name} ({size_mb:.2f} MB)")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Ready to train.")
    print("=" * 60)
    print("\n💡 Next steps:")
    print("   - Open notebooks/01_day1_data_ingestion.ipynb")
    print("   - Run the cells to process data")
    print("   - Move to Day 2: Model training")
    
except Exception as e:
    print(f"\n   ❌ Download failed: {e}")
    print("\n   🔍 Troubleshooting:")
    print("      - Ensure kaggle.json is in the correct location")
    print("      - Check internet connection")
    print("      - Verify dataset exists: https://www.kaggle.com/datasets/jucor1/worldstrat")
    exit(1)
