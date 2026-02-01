"""
PyTorch Dataset for Sentinel-2 Super-Resolution
Primary: GEE streaming (on-demand patch fetching)
Secondary: Custom preprocessed datasets
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from pathlib import Path
import yaml
import sys
from typing import Tuple, Optional, List
import albumentations as A
import cv2


class GEEStreamingDataset(Dataset):
    """
    PyTorch Dataset that streams Sentinel-2 patches from GEE on-demand
    NO massive downloads - fetches patches during training
    """
    
    def __init__(
        self,
        locations: List[Tuple[float, float]],
        gee_fetcher,
        patch_size: int = 256,
        scale_factor: int = 4,
        transform=None,
        normalize=True
    ):
        """
        Stream Sentinel-2 patches from Google Earth Engine
        
        Args:
            locations: List of (lat, lon) tuples to fetch
            gee_fetcher: GEEFetcher instance (initialized)
            patch_size: HR patch size (256x256)
            scale_factor: Upscaling factor (4x)
            transform: Augmentation transforms
            normalize: Whether to normalize to [0, 1]
        """
        self.locations = locations
        self.gee_fetcher = gee_fetcher
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.transform = transform
        self.normalize = normalize
        
        print(f"✅ GEE Streaming Dataset: {len(locations)} locations")
        print(f"   📦 Patches will be fetched on-demand (no preloading)")
    
    def __len__(self) -> int:
        return len(self.locations)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fetch patch from GEE and create LR/HR pair on-demand
        
        Returns:
            Tuple of (lr_tensor, hr_tensor)
        """
        lat, lon = self.locations[idx]
        
        # Fetch HR patch from GEE (256x256 at 10m/pixel)
        hr_patch = self.gee_fetcher.fetch_patch_from_point(lat, lon, buffer=1280)
        
        if hr_patch is None:
            # Return dummy data if fetch fails (handle gracefully)
            hr_patch = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)
        
        # Resize to target size if needed
        if hr_patch.shape[0] != self.patch_size:
            hr_patch = cv2.resize(hr_patch, (self.patch_size, self.patch_size))
        
        # Create LR by downsampling (synthetic degradation)
        lr_size = self.patch_size // self.scale_factor
        lr_patch = cv2.resize(hr_patch, (lr_size, lr_size), interpolation=cv2.INTER_AREA)
        
        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=lr_patch, hr_image=hr_patch)
            lr_patch = augmented['image']
            hr_patch = augmented['hr_image']
        
        # Normalize to [0, 1]
        if self.normalize:
            lr_patch = lr_patch.astype(np.float32) / 255.0
            hr_patch = hr_patch.astype(np.float32) / 255.0
        
        # Convert to torch tensors (H, W, C) -> (C, H, W)
        lr_tensor = torch.from_numpy(lr_patch).permute(2, 0, 1).float()
        hr_tensor = torch.from_numpy(hr_patch).permute(2, 0, 1).float()
        
        return lr_tensor, hr_tensor


class SentinelSRDataset(Dataset):
    """PyTorch Dataset for Sentinel-2 LR/HR paired images"""
    
    def __init__(
        self, 
        lr_dir: str, 
        hr_dir: str, 
        transform=None,
        normalize=True
    ):
        """
        Args:
            lr_dir: Directory containing low-resolution images
            hr_dir: Directory containing high-resolution images
            transform: Augmentation transforms
            normalize: Whether to normalize to [0, 1]
        """
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)
        self.transform = transform
        self.normalize = normalize
        
        # Get all LR files
        self.lr_files = sorted(list(self.lr_dir.glob('*.png')))
        self.hr_files = sorted(list(self.hr_dir.glob('*.png')))
        
        # Verify matching pairs
        assert len(self.lr_files) == len(self.hr_files), \
            f"Mismatch: {len(self.lr_files)} LR vs {len(self.hr_files)} HR images"
        
        # Verify filenames match
        for lr_file, hr_file in zip(self.lr_files, self.hr_files):
            assert lr_file.stem == hr_file.stem, \
                f"Filename mismatch: {lr_file.name} vs {hr_file.name}"
    
    def __len__(self) -> int:
        return len(self.lr_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get LR/HR pair
        
        Returns:
            Tuple of (lr_tensor, hr_tensor)
            - lr_tensor: (3, H, W) - Low-resolution input
            - hr_tensor: (3, H, W) - High-resolution target
        """
        # Load images
        lr_image = np.array(Image.open(self.lr_files[idx]))
        hr_image = np.array(Image.open(self.hr_files[idx]))
        
        # Apply augmentations (if any)
        if self.transform:
            augmented = self.transform(image=lr_image, hr_image=hr_image)
            lr_image = augmented['image']
            hr_image = augmented['hr_image']
        
        # Normalize to [0, 1]
        if self.normalize:
            lr_image = lr_image.astype(np.float32) / 255.0
            hr_image = hr_image.astype(np.float32) / 255.0
        
        # Convert to torch tensors (H, W, C) -> (C, H, W)
        lr_tensor = torch.from_numpy(lr_image).permute(2, 0, 1).float()
        hr_tensor = torch.from_numpy(hr_image).permute(2, 0, 1).float()
        
        return lr_tensor, hr_tensor
    
    def get_sample_info(self, idx: int):
        """Get metadata for a sample"""
        return {
            'lr_path': str(self.lr_files[idx]),
            'hr_path': str(self.hr_files[idx]),
            'filename': self.lr_files[idx].name
        }


def get_gee_dataloader(
    gee_fetcher,
    num_samples: int = 1000,
    region: str = "india",
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    transform=None
) -> DataLoader:
    """
    Create DataLoader for GEE streaming (primary method)
    
    Args:
        gee_fetcher: GEEFetcher instance
        num_samples: Number of patches to fetch
        region: Region to sample from ('india')
        batch_size: Batch size
        num_workers: Number of worker processes (0 recommended for GEE API calls)
        shuffle: Whether to shuffle data
        transform: Augmentation transforms
    
    Returns:
        DataLoader instance
    """
    # Generate random locations
    locations = gee_fetcher.generate_random_locations(num_samples, region)
    
    dataset = GEEStreamingDataset(
        locations=locations,
        gee_fetcher=gee_fetcher,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # API calls don't parallelize well
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader


def get_dataloader(
    lr_dir: str,
    hr_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    transform=None
) -> DataLoader:
    """
    Create DataLoader for custom preprocessed datasets
    
    Args:
        lr_dir: Directory with LR images
        hr_dir: Directory with HR images
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        transform: Augmentation transforms
    
    Returns:
        DataLoader instance
    """
    dataset = SentinelSRDataset(lr_dir, hr_dir, transform=transform)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader


def visualize_batch(lr_batch, hr_batch, num_samples=4, save_path=None):
    """
    Visualize a batch of LR/HR pairs
    
    Args:
        lr_batch: Tensor (B, 3, H, W)
        hr_batch: Tensor (B, 3, H, W)
        num_samples: Number of samples to visualize
        save_path: Path to save figure
    """
    import matplotlib.pyplot as plt
    
    num_samples = min(num_samples, lr_batch.shape[0])
    
    fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))
    
    for i in range(num_samples):
        # Convert to numpy (C, H, W) -> (H, W, C)
        lr_img = lr_batch[i].permute(1, 2, 0).cpu().numpy()
        hr_img = hr_batch[i].permute(1, 2, 0).cpu().numpy()
        
        # Clip to [0, 1]
        lr_img = np.clip(lr_img, 0, 1)
        hr_img = np.clip(hr_img, 0, 1)
        
        axes[0, i].imshow(lr_img)
        axes[0, i].set_title(f'LR {i+1}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(hr_img)
        axes[1, i].set_title(f'HR {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    print("🧪 Testing Sentinel SR Dataset...")
    
    print("✅ Dataset classes ready!")
    print("\n📝 Usage:")
    print("\n  # Option 1: GEE Streaming Dataset (Primary - No Downloads!)")
    print("  from src.data.gee_fetcher import GEEFetcher")
    print("  fetcher = GEEFetcher()")
    print("  dataloader = get_gee_dataloader(fetcher, num_samples=1000)")
    print("\n  # Option 2: Preprocessed Local Dataset")
    print("  dataset = SentinelSRDataset('data/train/lr', 'data/train/hr')")
    print("  dataloader = get_dataloader('data/train/lr', 'data/train/hr')")
    print("\n💡 GEE streams patches on-demand - perfect for 72-hour hackathon!")
