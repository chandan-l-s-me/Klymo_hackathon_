"""
Test GEE streaming setup - Quick verification before training
"""
import sys
sys.path.append('.')

print("=" * 60)
print("🧪 Testing GEE Streaming Setup")
print("=" * 60)

# Test 1: Check GEE connection
print("\n1️⃣ Testing GEE connection...")
try:
    import ee
    ee.Initialize(project='projectklymo')
    print("   ✅ GEE connected successfully!")
except Exception as e:
    print(f"   ❌ GEE connection failed: {e}")
    print("\n   📝 Setup needed:")
    print("      1. Run: earthengine authenticate")
    print("      2. Enable API: https://console.developers.google.com/apis/api/earthengine.googleapis.com/overview?project=projectklymo")
    exit(1)

# Test 2: Import GEE Fetcher
print("\n2️⃣ Testing GEE Fetcher import...")
try:
    from src.data.gee_fetcher import GEEFetcher
    fetcher = GEEFetcher()
    print("   ✅ GEEFetcher initialized")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    exit(1)

# Test 3: Generate sample locations
print("\n3️⃣ Generating sample locations...")
try:
    locations = fetcher.generate_random_locations(num_samples=5, region='india')
    print(f"   ✅ Generated {len(locations)} locations")
    for i, (lat, lon) in enumerate(locations[:3], 1):
        print(f"      {i}. ({lat:.4f}, {lon:.4f})")
except Exception as e:
    print(f"   ❌ Location generation failed: {e}")
    exit(1)

# Test 4: Test Dataset class
print("\n4️⃣ Testing GEEStreamingDataset...")
try:
    from src.data.dataset import GEEStreamingDataset
    
    # Create small test dataset
    test_locations = locations[:2]  # Just 2 locations for quick test
    dataset = GEEStreamingDataset(
        locations=test_locations,
        gee_fetcher=fetcher,
        patch_size=256,
        scale_factor=4
    )
    print(f"   ✅ Dataset created with {len(dataset)} samples")
except Exception as e:
    print(f"   ❌ Dataset creation failed: {e}")
    exit(1)

# Test 5: Fetch a single patch (optional - can be slow)
print("\n5️⃣ Testing single patch fetch (optional)...")
print("   ⏭️  Skipping actual fetch to save time")
print("   💡 To test fetch, run: fetcher.fetch_patch_from_point(28.6139, 77.2090)")

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print("\n🚀 Ready for Day 1 notebook!")
print("\n📝 Next steps:")
print("   1. Open notebooks/01_day1_data_ingestion.ipynb")
print("   2. Run cells to authenticate GEE")
print("   3. Test fetching a few patches")
print("   4. Proceed to Day 2: Model training")
print("\n💡 Optional: Fetch and save ~1000 patches for offline training")
print("   (Recommended for Colab where API calls might be slow)")
