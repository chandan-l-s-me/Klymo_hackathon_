"""
Quick test script to verify SwinIR model and training setup
Run this before starting full training to catch any issues
"""

import sys
import os
sys.path.append('.')

import torch
import yaml
from pathlib import Path

print("=" * 60)
print("🧪 TESTING SWINIR TRAINING SETUP")
print("=" * 60)

# 0. Check dependencies
print("\n0️⃣ Checking dependencies...")
missing_deps = []
try:
    import torch
    print(f"   ✅ PyTorch: {torch.__version__}")
except ImportError:
    missing_deps.append('torch')
    print("   ❌ PyTorch not installed")

try:
    import torchvision
    print(f"   ✅ torchvision: {torchvision.__version__}")
except ImportError:
    missing_deps.append('torchvision')
    print("   ❌ torchvision not installed")

try:
    import timm
    print(f"   ✅ timm: {timm.__version__}")
except ImportError:
    missing_deps.append('timm')
    print("   ❌ timm not installed")

try:
    import yaml
    print(f"   ✅ PyYAML installed")
except ImportError:
    missing_deps.append('pyyaml')
    print("   ❌ PyYAML not installed")

if missing_deps:
    print(f"\n   ⚠️  Missing dependencies: {', '.join(missing_deps)}")
    print(f"   📦 Install with: pip install {' '.join(missing_deps)}")
    print(f"   Or run: pip install timm torchvision pyyaml")
    sys.exit(1)

# 1. Test imports
print("\n1️⃣ Testing imports...")
try:
    from src.models.swinir import SwinIR
    from src.utils.losses import CombinedLoss
    from src.utils.metrics import calculate_psnr, calculate_ssim
    print("   ✅ All modules imported successfully")
except Exception as e:
    print(f"   ❌ Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 2. Load config
print("\n2️⃣ Loading configuration...")
try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print(f"   ✅ Config loaded")
    print(f"      - Scale: {config['model']['upscale']}x")
    print(f"      - Epochs: {config['training']['epochs']}")
    print(f"      - LR: {config['training']['learning_rate']}")
except Exception as e:
    print(f"   ❌ Config error: {e}")
    sys.exit(1)

# 3. Create model
print("\n3️⃣ Creating SwinIR model...")
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    model = SwinIR(
        upscale=config['model']['upscale'],
        in_chans=config['model']['in_chans'],
        img_size=config['model']['img_size'],
        window_size=config['model']['window_size'],
        img_range=config['model']['img_range'],
        depths=config['model']['depths'],
        embed_dim=config['model']['embed_dim'],
        num_heads=config['model']['num_heads'],
        mlp_ratio=config['model']['mlp_ratio'],
        upsampler='pixelshuffle'
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ Model created: {total_params/1e6:.2f}M parameters")
except Exception as e:
    print(f"   ❌ Model creation error: {e}")
    sys.exit(1)

# 4. Test forward pass
print("\n4️⃣ Testing forward pass...")
try:
    test_input = torch.randn(2, 3, 64, 64).to(device)
    with torch.no_grad():
        test_output = model(test_input)
    
    print(f"   ✅ Forward pass successful")
    print(f"      Input: {test_input.shape}")
    print(f"      Output: {test_output.shape}")
    print(f"      Output range: [{test_output.min():.3f}, {test_output.max():.3f}]")
    
    # Verify output shape
    expected_h = test_input.shape[2] * config['model']['upscale']
    expected_w = test_input.shape[3] * config['model']['upscale']
    assert test_output.shape[2] == expected_h, f"Height mismatch: {test_output.shape[2]} != {expected_h}"
    assert test_output.shape[3] == expected_w, f"Width mismatch: {test_output.shape[3]} != {expected_w}"
    print(f"   ✅ Output dimensions correct: {expected_h}×{expected_w}")
except Exception as e:
    print(f"   ❌ Forward pass error: {e}")
    sys.exit(1)

# 5. Test loss functions
print("\n5️⃣ Testing loss functions...")
try:
    criterion = CombinedLoss(
        l1_weight=1.0,
        perceptual_weight=0.1,
        use_perceptual=True
    ).to(device)
    
    pred = torch.randn(2, 3, 256, 256).to(device)
    target = torch.randn(2, 3, 256, 256).to(device)
    
    loss, loss_dict = criterion(pred, target)
    
    print(f"   ✅ Loss computation successful")
    print(f"      Total loss: {loss.item():.6f}")
    print(f"      Pixel loss: {loss_dict['pixel']:.6f}")
    print(f"      Perceptual loss: {loss_dict['perceptual']:.6f}")
except Exception as e:
    print(f"   ❌ Loss computation error: {e}")
    sys.exit(1)

# 6. Test metrics
print("\n6️⃣ Testing metrics...")
try:
    img1 = torch.rand(2, 3, 256, 256).to(device)
    img2 = img1 + torch.randn_like(img1).to(device) * 0.01
    
    psnr = calculate_psnr(img1, img2)
    ssim = calculate_ssim(img1, img2)
    
    print(f"   ✅ Metrics computation successful")
    print(f"      PSNR: {psnr:.2f} dB")
    print(f"      SSIM: {ssim:.4f}")
except Exception as e:
    print(f"   ❌ Metrics error: {e}")
    sys.exit(1)

# 7. Check data directory
print("\n7️⃣ Checking data availability...")
try:
    data_dir = Path('data/processed_test/train')
    lr_dir = data_dir / 'lr'
    hr_dir = data_dir / 'hr'
    
    if not lr_dir.exists():
        print(f"   ⚠️  LR directory not found: {lr_dir}")
        print(f"      Run Day 1 notebook first!")
    else:
        lr_files = list(lr_dir.glob('*.png'))
        hr_files = list(hr_dir.glob('*.png'))
        print(f"   ✅ Data directory exists")
        print(f"      LR images: {len(lr_files)}")
        print(f"      HR images: {len(hr_files)}")
        
        if len(lr_files) == 0:
            print(f"   ⚠️  No training data found!")
            print(f"      Run Day 1 notebook to generate data")
except Exception as e:
    print(f"   ⚠️  Data check error: {e}")

# 8. Test backward pass
print("\n8️⃣ Testing backward pass...")
try:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002)
    
    # Forward
    x = torch.randn(1, 3, 64, 64).to(device)
    y = torch.randn(1, 3, 256, 256).to(device)
    pred = model(x)
    
    # Loss
    loss, _ = criterion(pred, y)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"   ✅ Backward pass successful")
    print(f"      Gradients computed and applied")
except Exception as e:
    print(f"   ❌ Backward pass error: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print("\n🚀 Ready to start training!")
print("\nNext steps:")
print("1. Open notebooks/02_day2_training.ipynb")
print("2. Run all cells to start training")
print("3. Monitor PSNR and SSIM metrics")
print("4. Wait for best_model.pth checkpoint")
print("\nFor quick test (5 minutes):")
print("  - Set num_epochs = 5")
print("  - Set batch_size = 4")
print("\n" + "=" * 60)
