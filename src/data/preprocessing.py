"""
Preprocessing utilities for Sentinel-2 super-resolution
- Create LR/HR pairs
- Patch extraction
- Normalization
"""

import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import yaml


class DataPreprocessor:
    """Preprocess Sentinel-2 patches for training"""
    
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.scale = self.data_config['scale_factor']
        self.hr_size = self.data_config['hr_patch_size']
        self.lr_size = self.data_config['lr_patch_size']
    
    def create_lr_hr_pair(self, hr_image, scale=None):
        """
        Create synthetic LR/HR pair by degradation
        
        Args:
            hr_image: numpy.ndarray - High-resolution image (H, W, 3)
            scale: int - Downscaling factor (default from config)
        
        Returns:
            tuple - (lr_image, hr_image)
                lr_image: Low-resolution image (H//scale, W//scale, 3) - actual LR size
                hr_image: Original high-resolution image (H, W, 3)
        """
        if scale is None:
            scale = self.scale
        
        h, w = hr_image.shape[:2]
        
        # Downscale to create LR image (don't upscale back)
        lr_h, lr_w = h // scale, w // scale
        lr_image = cv2.resize(hr_image, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
        
        return lr_image, hr_image
    
    def extract_patches(self, image, patch_size=256, stride=None):
        """
        Extract patches from a large image
        
        Args:
            image: numpy.ndarray - Input image
            patch_size: int - Size of patches
            stride: int - Stride for sliding window (default: patch_size)
        
        Returns:
            list of numpy.ndarray - List of patches
        """
        if stride is None:
            stride = patch_size
        
        h, w = image.shape[:2]
        patches = []
        
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = image[y:y+patch_size, x:x+patch_size]
                patches.append(patch)
        
        return patches
    
    def normalize_image(self, image, input_range=(0, 255), output_range=(0, 1)):
        """
        Normalize image to target range
        
        Args:
            image: numpy.ndarray - Input image
            input_range: tuple - (min, max) of input
            output_range: tuple - (min, max) of output
        
        Returns:
            numpy.ndarray - Normalized image
        """
        image = image.astype(np.float32)
        
        # Normalize to [0, 1]
        image = (image - input_range[0]) / (input_range[1] - input_range[0])
        
        # Scale to output range
        image = image * (output_range[1] - output_range[0]) + output_range[0]
        
        return image
    
    def denormalize_image(self, image, input_range=(0, 1), output_range=(0, 255)):
        """
        Denormalize image back to original range
        
        Args:
            image: numpy.ndarray - Normalized image
            input_range: tuple - (min, max) of normalized range
            output_range: tuple - (min, max) of target range
        
        Returns:
            numpy.ndarray - Denormalized image (uint8)
        """
        # Scale from input range to [0, 1]
        image = (image - input_range[0]) / (input_range[1] - input_range[0])
        
        # Scale to output range
        image = image * (output_range[1] - output_range[0]) + output_range[0]
        
        return np.clip(image, output_range[0], output_range[1]).astype(np.uint8)
    
    def process_dataset(self, input_dir, output_dir, split='train'):
        """
        Process entire dataset: create LR/HR pairs and extract patches
        
        Args:
            input_dir: str - Directory with raw images
            output_dir: str - Directory to save processed data
            split: str - 'train' or 'val'
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        # Create output directories
        lr_dir = output_dir / split / 'lr'
        hr_dir = output_dir / split / 'hr'
        lr_dir.mkdir(parents=True, exist_ok=True)
        hr_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all images
        image_files = list(input_dir.glob('*.png')) + list(input_dir.glob('*.jpg'))
        
        print(f"📦 Processing {len(image_files)} images for {split} set...")
        
        patch_count = 0
        
        for img_path in tqdm(image_files):
            # Load image
            hr_image = np.array(Image.open(img_path))
            
            # Handle images of various sizes
            h, w = hr_image.shape[:2]
            
            # Skip if too small (less than half the target size)
            if h < self.hr_size // 2 or w < self.hr_size // 2:
                continue
            
            # Resize to exact patch size if needed (for images close to target size)
            if h != self.hr_size or w != self.hr_size:
                hr_image = cv2.resize(hr_image, (self.hr_size, self.hr_size), interpolation=cv2.INTER_CUBIC)
            
            # Process the full image as one patch
            hr_patches = [hr_image]
            
            for patch_idx, hr_patch in enumerate(hr_patches):
                # Create LR/HR pair
                lr_patch, hr_patch = self.create_lr_hr_pair(hr_patch)
                
                # Save patches
                lr_filename = lr_dir / f"{img_path.stem}_patch{patch_idx:03d}.png"
                hr_filename = hr_dir / f"{img_path.stem}_patch{patch_idx:03d}.png"
                
                Image.fromarray(lr_patch.astype(np.uint8)).save(lr_filename)
                Image.fromarray(hr_patch.astype(np.uint8)).save(hr_filename)
                
                patch_count += 1
        
        print(f"✅ Created {patch_count} LR/HR patch pairs")
        print(f"📁 Saved to {output_dir}/{split}/")
        
        return patch_count
    
    def visualize_lr_hr_comparison(self, lr_image, hr_image, sr_image=None, save_path=None):
        """
        Visualize LR, HR, and optionally SR images side-by-side
        
        Args:
            lr_image: numpy.ndarray - Low-resolution input
            hr_image: numpy.ndarray - High-resolution ground truth
            sr_image: numpy.ndarray - Super-resolved output (optional)
            save_path: str - Path to save visualization
        """
        import matplotlib.pyplot as plt
        
        num_images = 3 if sr_image is not None else 2
        fig, axes = plt.subplots(1, num_images, figsize=(5*num_images, 5))
        
        axes[0].imshow(lr_image)
        axes[0].set_title('LR Input (Blurry)')
        axes[0].axis('off')
        
        axes[1].imshow(hr_image)
        axes[1].set_title('HR Ground Truth')
        axes[1].axis('off')
        
        if sr_image is not None:
            axes[2].imshow(sr_image)
            axes[2].set_title('SR Output (Model)')
            axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Saved visualization to {save_path}")
        
        plt.show()


if __name__ == "__main__":
    # Test preprocessing
    print("🧪 Testing Data Preprocessor...")
    
    preprocessor = DataPreprocessor()
    
    # Test with a dummy image
    dummy_hr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Create LR/HR pair
    lr, hr = preprocessor.create_lr_hr_pair(dummy_hr)
    
    print(f"✅ LR shape: {lr.shape}")
    print(f"✅ HR shape: {hr.shape}")
    
    # Test normalization
    normalized = preprocessor.normalize_image(hr)
    denormalized = preprocessor.denormalize_image(normalized)
    
    print(f"✅ Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    print(f"✅ Denormalized range: [{denormalized.min()}, {denormalized.max()}]")
    
    print("\n✅ Preprocessor test complete!")
