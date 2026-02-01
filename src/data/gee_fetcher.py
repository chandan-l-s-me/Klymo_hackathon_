"""
Google Earth Engine Data Fetcher for Sentinel-2 Imagery
Streams data via API - NO massive downloads required
"""

import ee
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import yaml
from pathlib import Path
from tqdm import tqdm
import time
import rasterio
from rasterio.io import MemoryFile


class GEEFetcher:
    """Fetch Sentinel-2 imagery from Google Earth Engine via API"""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize GEE connection"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.gee_config = self.config['gee']
        self.data_config = self.config['data']
        
        # Initialize Earth Engine
        try:
            ee.Initialize(project=self.gee_config['project_id'])
            print(f"✅ Connected to GEE project: {self.gee_config['project_id']}")
        except Exception as e:
            print(f"❌ GEE initialization failed: {e}")
            print("Run: earthengine authenticate")
            raise
    
    def get_sentinel2_collection(self, roi, date_start, date_end, max_cloud_cover=10):
        """
        Get filtered Sentinel-2 collection for a region of interest
        
        Args:
            roi: ee.Geometry - Region of interest
            date_start: str - Start date (YYYY-MM-DD)
            date_end: str - End date (YYYY-MM-DD)
            max_cloud_cover: int - Maximum cloud cover percentage
        
        Returns:
            ee.ImageCollection - Filtered Sentinel-2 collection
        """
        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(roi) \
            .filterDate(date_start, date_end) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_cover))
        
        return collection
    
    def get_rgb_composite(self, image):
        """
        Extract RGB bands from Sentinel-2 image
        
        Args:
            image: ee.Image - Sentinel-2 image
        
        Returns:
            ee.Image - RGB composite (B4, B3, B2)
        """
        rgb = image.select(self.data_config['sentinel_bands'])
        
        # Normalize 16-bit to 8-bit range
        # Sentinel-2 values: 0-10000, typical max: ~3000
        normalized = rgb.multiply(255.0 / self.data_config['normalize_max']).clamp(0, 255).uint8()
        
        return normalized
    
    def fetch_patch_from_point(self, lat, lon, buffer=1280, date_start=None, date_end=None):
        """
        Fetch a single patch centered at a point
        
        Args:
            lat: float - Latitude
            lon: float - Longitude
            buffer: int - Patch size in meters (default 1280m = 128 pixels at 10m)
            date_start: str - Start date
            date_end: str - End date
        
        Returns:
            numpy.ndarray - RGB image (H, W, 3)
        """
        if date_start is None:
            date_start = self.data_config['date_start']
        if date_end is None:
            date_end = self.data_config['date_end']
        
        # Create point geometry
        point = ee.Geometry.Point([lon, lat])
        roi = point.buffer(buffer)
        
        # Get image collection
        collection = self.get_sentinel2_collection(roi, date_start, date_end)
        
        # Check if collection has any images
        count = collection.size().getInfo()
        if count == 0:
            print(f"⚠️ No images found for ({lat:.4f}, {lon:.4f})")
            return None
        
        # Get median composite (reduces cloud artifacts)
        image = collection.median()
        
        # Extract RGB
        rgb_image = self.get_rgb_composite(image)
        
        # Get thumbnail URL with proper dimensions
        dimensions = buffer * 2 // self.data_config['input_resolution']  # pixels
        url = rgb_image.getThumbURL({
            'region': roi.bounds().getInfo(),
            'dimensions': dimensions,
            'format': 'png'
        })
        
        # Download and convert to numpy with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=60)  # Increased to 60s
                response.raise_for_status()
                
                # Read PNG from memory
                img = Image.open(BytesIO(response.content))
                image_array = np.array(img)
                
                # Ensure RGB (drop alpha if present)
                if len(image_array.shape) == 3 and image_array.shape[2] == 4:
                    image_array = image_array[:, :, :3]
                        
                return image_array
                
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    print(f"⏳ Timeout on attempt {attempt + 1}, retrying...")
                    time.sleep(2)
                    continue
                else:
                    print(f"❌ Failed after {max_retries} attempts: Timeout")
                    return None
            except Exception as e:
                print(f"Error downloading patch at ({lat}, {lon}): {e}")
                return None
    
    def generate_random_locations(self, num_samples=100, region='india'):
        """
        Generate random lat/lon coordinates within a region
        
        Args:
            num_samples: int - Number of locations
            region: str - Region name (default: india)
        
        Returns:
            list of tuples - [(lat, lon), ...]
        """
        # India bounding box (approximate)
        if region == 'india':
            lat_min, lat_max = 8.0, 35.0
            lon_min, lon_max = 68.0, 97.0
        else:
            # Default: global
            lat_min, lat_max = -60.0, 60.0
            lon_min, lon_max = -180.0, 180.0
        
        locations = []
        for _ in range(num_samples):
            lat = np.random.uniform(lat_min, lat_max)
            lon = np.random.uniform(lon_min, lon_max)
            locations.append((lat, lon))
        
        return locations
    
    def fetch_dataset(self, num_samples=1000, save_dir="./data/raw", region='india'):
        """
        Fetch multiple patches and save to disk
        
        Args:
            num_samples: int - Number of patches to fetch
            save_dir: str - Directory to save patches
            region: str - Region to sample from
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🌍 Fetching {num_samples} Sentinel-2 patches from {region}...")
        
        # Generate random locations
        locations = self.generate_random_locations(num_samples * 2, region)  # 2x for failures
        
        successful = 0
        failed = 0
        
        with tqdm(total=num_samples) as pbar:
            for idx, (lat, lon) in enumerate(locations):
                if successful >= num_samples:
                    break
                
                # Fetch patch
                patch = self.fetch_patch_from_point(lat, lon)
                
                if patch is not None and patch.shape[0] > 0 and patch.shape[1] > 0:
                    # Save as PNG
                    filename = save_dir / f"patch_{successful:05d}_lat{lat:.4f}_lon{lon:.4f}.png"
                    Image.fromarray(patch.astype(np.uint8)).save(filename)
                    
                    successful += 1
                    pbar.update(1)
                else:
                    failed += 1
                
                # Rate limiting (GEE quota)
                if idx % 10 == 0:
                    time.sleep(1)
        
        print(f"\n✅ Successfully fetched: {successful} patches")
        print(f"❌ Failed: {failed} patches")
        print(f"📁 Saved to: {save_dir}")
        
        return successful
    
    def fetch_specific_location(self, location_name, save_path=None):
        """
        Fetch patch for a specific test location from config
        
        Args:
            location_name: str - Name from config (e.g., 'Delhi')
            save_path: str - Path to save image
        
        Returns:
            numpy.ndarray - Image array
        """
        # Find location in config
        locations = self.config['test_locations']
        location = next((loc for loc in locations if loc['name'] == location_name), None)
        
        if location is None:
            raise ValueError(f"Location '{location_name}' not found in config")
        
        print(f"📍 Fetching {location_name} ({location['lat']}, {location['lon']})")
        
        # Fetch patch
        patch = self.fetch_patch_from_point(
            location['lat'], 
            location['lon'], 
            buffer=location['buffer']
        )
        
        if patch is not None and save_path:
            Image.fromarray(patch.astype(np.uint8)).save(save_path)
            print(f"✅ Saved to {save_path}")
        
        return patch


if __name__ == "__main__":
    # Test GEE connection
    print("🧪 Testing GEE Fetcher...")
    
    fetcher = GEEFetcher()
    
    # Test single patch fetch
    print("\n1️⃣ Testing single patch fetch (Delhi)...")
    patch = fetcher.fetch_specific_location('Delhi', save_path='./test_delhi.png')
    
    if patch is not None:
        print(f"✅ Patch shape: {patch.shape}")
        print(f"✅ Value range: [{patch.min()}, {patch.max()}]")
    
    # Test small dataset fetch (10 samples)
    print("\n2️⃣ Testing small dataset fetch (10 samples)...")
    fetcher.fetch_dataset(num_samples=10, save_dir="./data/test_raw", region='india')
    
    print("\n✅ GEE Fetcher test complete!")
