"""
WorldStrat Dataset Downloader and Processor
Better than synthetic degradation - uses real paired LR/HR data
"""

import os
import shutil
from pathlib import Path
import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm


class WorldStratDownloader:
    """Download and process WorldStrat dataset from Kaggle"""
    
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
    
    def download_worldstrat(self, save_dir="./data/worldstrat"):
        """
        Download WorldStrat dataset from Kaggle
        
        WorldStrat contains:
        - LR: Sentinel-2 imagery (10m/pixel)
        - HR: SPOT/Airbus imagery (1.5m/pixel)
        - Real paired data with ~7x quality difference
        
        Args:
            save_dir: Directory to save dataset
        
        Returns:
            Path to downloaded dataset
        """
        print("📥 Downloading WorldStrat dataset from Kaggle...")
        print("⚠️ This may take several minutes on first download...\n")
        
        try:
            import kagglehub
            
            # Download latest version
            path = kagglehub.dataset_download("jucor1/worldstrat")
            
            print(f"\n✅ Downloaded to: {path}")
            
            # Check structure
            self.inspect_dataset(path)
            
            return path
            
        except ImportError:
            print("❌ kagglehub not installed.")
            print("Run: pip install kagglehub")
            raise
        
        except Exception as e:
            print(f"❌ Download failed: {e}")
            print("\n⚠️ Make sure you have Kaggle credentials set up:")
            print("   1. Go to https://www.kaggle.com/settings")
            print("   2. Create API token (downloads kaggle.json)")
            print("   3. Place in ~/.kaggle/kaggle.json (Linux/Mac) or C:\\Users\\<user>\\.kaggle\\kaggle.json (Windows)")
            raise
    
    def inspect_dataset(self, dataset_path):
        """
        Inspect WorldStrat dataset structure
        
        Expected structure:
        worldstrat/
        ├── train/
        │   ├── lr/  (Sentinel-2)
        │   └── hr/  (SPOT/Airbus)
        ├── val/
        │   ├── lr/
        │   └── hr/
        └── test/
            ├── lr/
            └── hr/
        """
        print("\n🔍 Inspecting dataset structure...")
        
        dataset_path = Path(dataset_path)
        
        # List top-level directories
        print(f"\nDataset root: {dataset_path}")
        print("\nContents:")
        for item in sorted(dataset_path.iterdir()):
            if item.is_dir():
                num_files = len(list(item.rglob('*.*')))
                print(f"  📁 {item.name}/ ({num_files} files)")
            else:
                print(f"  📄 {item.name}")
        
        # Check for train/val/test splits
        for split in ['train', 'val', 'test']:
            split_dir = dataset_path / split
            if split_dir.exists():
                print(f"\n{split.upper()} split:")
                lr_dir = split_dir / 'lr'
                hr_dir = split_dir / 'hr'
                
                if lr_dir.exists():
                    lr_count = len(list(lr_dir.glob('*.*')))
                    print(f"  LR images: {lr_count}")
                
                if hr_dir.exists():
                    hr_count = len(list(hr_dir.glob('*.*')))
                    print(f"  HR images: {hr_count}")
        
        return dataset_path
    
    def process_worldstrat(self, dataset_path, output_dir="./data/processed"):
        """
        Process WorldStrat dataset for training
        
        Steps:
        1. Load LR/HR pairs
        2. Resize/crop to consistent patch size
        3. Normalize
        4. Save in organized structure
        
        Args:
            dataset_path: Path to downloaded dataset
            output_dir: Output directory
        """
        print("\n⚙️ Processing WorldStrat dataset...")
        
        dataset_path = Path(dataset_path)
        output_dir = Path(output_dir)
        
        for split in ['train', 'val']:
            print(f"\nProcessing {split} split...")
            
            # Input directories
            lr_input = dataset_path / split / 'lr'
            hr_input = dataset_path / split / 'hr'
            
            # Output directories
            lr_output = output_dir / split / 'lr'
            hr_output = output_dir / split / 'hr'
            lr_output.mkdir(parents=True, exist_ok=True)
            hr_output.mkdir(parents=True, exist_ok=True)
            
            # Check if directories exist
            if not lr_input.exists() or not hr_input.exists():
                print(f"  ⚠️ Skipping {split} - directories not found")
                continue
            
            # Get file lists
            lr_files = sorted(list(lr_input.glob('*.png')) + list(lr_input.glob('*.jpg')) + list(lr_input.glob('*.tif')))
            hr_files = sorted(list(hr_input.glob('*.png')) + list(hr_input.glob('*.jpg')) + list(hr_input.glob('*.tif')))
            
            print(f"  Found {len(lr_files)} LR and {len(hr_files)} HR images")
            
            # Process pairs
            patch_count = 0
            for lr_file, hr_file in tqdm(zip(lr_files, hr_files), total=len(lr_files)):
                try:
                    # Load images
                    lr_img = np.array(Image.open(lr_file))
                    hr_img = np.array(Image.open(hr_file))
                    
                    # Resize to target size (if needed)
                    target_size = self.data_config['hr_patch_size']
                    
                    if lr_img.shape[0] != target_size or lr_img.shape[1] != target_size:
                        lr_img = np.array(Image.fromarray(lr_img).resize((target_size, target_size)))
                    
                    if hr_img.shape[0] != target_size or hr_img.shape[1] != target_size:
                        hr_img = np.array(Image.fromarray(hr_img).resize((target_size, target_size)))
                    
                    # Save processed images
                    lr_save = lr_output / f"patch_{patch_count:05d}.png"
                    hr_save = hr_output / f"patch_{patch_count:05d}.png"
                    
                    Image.fromarray(lr_img).save(lr_save)
                    Image.fromarray(hr_img).save(hr_save)
                    
                    patch_count += 1
                    
                except Exception as e:
                    print(f"    ⚠️ Failed to process {lr_file.name}: {e}")
                    continue
            
            print(f"  ✅ Processed {patch_count} pairs")
        
        print(f"\n✅ Dataset processed and saved to {output_dir}")
        return output_dir
    
    def get_dataset_stats(self, dataset_path):
        """Get statistics about the dataset"""
        dataset_path = Path(dataset_path)
        
        stats = {}
        
        for split in ['train', 'val', 'test']:
            lr_dir = dataset_path / split / 'lr'
            hr_dir = dataset_path / split / 'hr'
            
            if lr_dir.exists():
                lr_files = list(lr_dir.glob('*.*'))
                if lr_files:
                    sample = np.array(Image.open(lr_files[0]))
                    stats[f'{split}_lr'] = {
                        'count': len(lr_files),
                        'shape': sample.shape,
                        'dtype': sample.dtype
                    }
            
            if hr_dir.exists():
                hr_files = list(hr_dir.glob('*.*'))
                if hr_files:
                    sample = np.array(Image.open(hr_files[0]))
                    stats[f'{split}_hr'] = {
                        'count': len(hr_files),
                        'shape': sample.shape,
                        'dtype': sample.dtype
                    }
        
        return stats


if __name__ == "__main__":
    print("=" * 60)
    print("🌍 WORLDSTRAT DATASET SETUP")
    print("=" * 60)
    
    downloader = WorldStratDownloader()
    
    # Download dataset
    dataset_path = downloader.download_worldstrat()
    
    if dataset_path:
        # Process for training
        processed_path = downloader.process_worldstrat(dataset_path)
        
        # Get stats
        stats = downloader.get_dataset_stats(processed_path)
        
        print("\n📊 Dataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\n✅ WorldStrat dataset ready!")
        print(f"📁 Location: {processed_path}")
        print("\n🚀 Next: Use this data with PyTorch DataLoader")
        print("   from src.data.dataset import get_dataloader")
        print(f"   loader = get_dataloader('{processed_path}/train/lr', '{processed_path}/train/hr')")
    else:
        print("\n⚠️ Failed to download WorldStrat")
        print("Fallback: Use GEE synthetic data pipeline")
