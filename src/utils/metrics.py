"""
Evaluation metrics for super-resolution
PSNR and SSIM implementations
"""

import torch
import torch.nn.functional as F
import numpy as np
from math import log10


def calculate_psnr(img1, img2, max_val=1.0):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR)
    
    Args:
        img1: torch.Tensor - First image (B, C, H, W) in range [0, 1]
        img2: torch.Tensor - Second image (B, C, H, W) in range [0, 1]
        max_val: float - Maximum pixel value (default: 1.0)
    
    Returns:
        float - PSNR value in dB
    """
    mse = F.mse_loss(img1, img2)
    
    if mse == 0:
        return float('inf')
    
    psnr = 20 * log10(max_val) - 10 * log10(mse.item())
    return psnr


def calculate_ssim(img1, img2, window_size=11, size_average=True):
    """
    Calculate Structural Similarity Index (SSIM)
    
    Args:
        img1: torch.Tensor - First image (B, C, H, W)
        img2: torch.Tensor - Second image (B, C, H, W)
        window_size: int - Size of Gaussian window
        size_average: bool - Average over batch
    
    Returns:
        float - SSIM value in range [0, 1]
    """
    def create_window(window_size, channel):
        """Create Gaussian window"""
        def gaussian(window_size, sigma):
            gauss = torch.Tensor([
                np.exp(-(x - window_size//2)**2/float(2*sigma**2)) 
                for x in range(window_size)
            ])
            return gauss / gauss.sum()
        
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    
    C1 = 0.01**2
    C2 = 0.03**2
    
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def calculate_metrics_batch(sr_batch, hr_batch):
    """
    Calculate PSNR and SSIM for a batch
    
    Args:
        sr_batch: torch.Tensor - Super-resolved images (B, C, H, W)
        hr_batch: torch.Tensor - High-resolution ground truth (B, C, H, W)
    
    Returns:
        dict - {'psnr': float, 'ssim': float}
    """
    # Clamp values to [0, 1]
    sr_batch = torch.clamp(sr_batch, 0, 1)
    hr_batch = torch.clamp(hr_batch, 0, 1)
    
    psnr = calculate_psnr(sr_batch, hr_batch)
    ssim = calculate_ssim(sr_batch, hr_batch)
    
    return {
        'psnr': psnr,
        'ssim': ssim
    }


if __name__ == "__main__":
    # Test metrics
    print("🧪 Testing metrics...")
    
    # Create dummy images
    img1 = torch.rand(2, 3, 256, 256)
    img2 = img1 + torch.randn_like(img1) * 0.01  # Add small noise
    
    psnr = calculate_psnr(img1, img2)
    ssim = calculate_ssim(img1, img2)
    
    print(f"✅ PSNR: {psnr:.2f} dB")
    print(f"✅ SSIM: {ssim:.4f}")
    
    # Test identical images
    psnr_same = calculate_psnr(img1, img1)
    ssim_same = calculate_ssim(img1, img1)
    
    print(f"✅ PSNR (same): {psnr_same} dB")
    print(f"✅ SSIM (same): {ssim_same:.4f}")
