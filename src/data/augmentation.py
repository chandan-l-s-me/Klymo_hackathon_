"""
Data Augmentation for Satellite Imagery
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def get_training_augmentation(patch_size=256):
    """
    Get training augmentation pipeline
    
    Augmentations for satellite imagery:
    - Horizontal/Vertical flips
    - 90-degree rotations
    - Slight brightness/contrast adjustments
    
    NO geometric distortions (preserve spatial accuracy)
    """
    transform = A.Compose([
        # Geometric augmentations (preserve spatial relationships)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        
        # Subtle color augmentations
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=5,
                sat_shift_limit=10,
                val_shift_limit=10,
                p=1.0
            ),
        ], p=0.3),
        
        # Slight blur to simulate atmospheric effects
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 3), p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
        ], p=0.1),
        
    ], additional_targets={'hr_image': 'image'})
    
    return transform


def get_validation_augmentation():
    """
    Get validation augmentation (minimal)
    Usually just normalization, no random transforms
    """
    transform = A.Compose([
        # No augmentation for validation
    ], additional_targets={'hr_image': 'image'})
    
    return transform


def get_test_augmentation():
    """
    Get test/inference augmentation (none)
    """
    return None


# Albumentations-compatible paired augmentation
class PairedAugmentation:
    """
    Custom augmentation that applies same transform to both LR and HR
    """
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, lr_image, hr_image):
        """
        Apply same augmentation to both images
        
        Args:
            lr_image: numpy.ndarray (H, W, 3)
            hr_image: numpy.ndarray (H, W, 3)
        
        Returns:
            tuple: (augmented_lr, augmented_hr)
        """
        # Albumentations handles paired images via additional_targets
        augmented = self.transform(image=lr_image, hr_image=hr_image)
        
        return augmented['image'], augmented['hr_image']


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    
    print("🧪 Testing Augmentation Pipeline...")
    
    # Create dummy images
    dummy_lr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    dummy_hr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Get augmentation
    aug = get_training_augmentation()
    
    # Apply augmentation
    augmented = aug(image=dummy_lr, hr_image=dummy_hr)
    aug_lr = augmented['image']
    aug_hr = augmented['hr_image']
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    axes[0, 0].imshow(dummy_lr)
    axes[0, 0].set_title('Original LR')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(aug_lr)
    axes[0, 1].set_title('Augmented LR')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(dummy_hr)
    axes[1, 0].set_title('Original HR')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(aug_hr)
    axes[1, 1].set_title('Augmented HR')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation_test.png')
    print("✅ Saved augmentation test to augmentation_test.png")
    
    print("\n✅ Augmentation pipeline ready!")
