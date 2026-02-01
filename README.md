# 🛰️ Sentinel-2 Super-Resolution using SwinIR

**4x Upscaling** of Sentinel-2 satellite imagery (10m/pixel → 2.5m/pixel) using Transformer-based SwinIR model

---

## 🎯 Project Overview

This project aims to enhance the spatial resolution of Sentinel-2 satellite imagery by 4x using deep learning, revealing finer details of urban structures, roads, and land features without introducing hallucinations.

### Key Features
- ✅ **Zero massive downloads** - GEE API streaming fetches patches on-demand
- ✅ **Immediate start** - No waiting for dataset downloads
- ✅ **Synthetic LR/HR pairs** - 4x downsampling from Sentinel-2 (standard practice)
- ✅ **Transformer architecture** (SwinIR - State-of-the-art)
- ✅ **No hallucinations** (L1 + Perceptual loss, no GAN)
- ✅ **Geospatially accurate** (Preserves real features)
- ✅ **Memory efficient** (On-demand patch loading during training)

### Data Strategy
**Stream, don't download!**
- Fetch 256×256 Sentinel-2 patches via GEE API on-demand
- Create LR (64×64) by 4x downsampling during training
- Keep HR (256×256) as ground truth
- Only cache ~1000 samples if needed (~200MB total)

---

## 📊 Results Preview

| LR Input (Blurry) | HR Output (Sharp) |
|-------------------|-------------------|
| 10m/pixel | 2.5m/pixel |
| *Coming soon* | *Coming soon* |

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/yourusername/projectklymo.git
cd projectklymo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Authenticate Google Earth Engine

```bash
earthengine authenticate
```

Or in Python:
```python
import ee
ee.Authenticate()
ee.Initialize(project='projectklymo')
```

### 3. Run Day 1 Notebook (Data Ingestion)

```bash
jupyter notebook notebooks/01_day1_data_ingestion.ipynb
```

---

## 📁 Project Structure

```
projectklymo/
├── src/
│   ├── data/
│   │   ├── gee_fetcher.py          # Google Earth Engine API
│   │   ├── preprocessing.py         # LR/HR pair creation
│   │   ├── dataset.py               # PyTorch Dataset
│   │   └── augmentation.py          # Data augmentation
│   ├── models/
│   │   └── swinir.py                # SwinIR architecture (Day 2)
│   ├── utils/
│   │   ├── metrics.py               # PSNR, SSIM
│   │   └── visualize.py             # Plotting
│   └── train.py                      # Training script (Day 2)
├── notebooks/
│   ├── 01_day1_data_ingestion.ipynb # Day 1 work
│   ├── 02_day2_training.ipynb       # Day 2 work
│   └── 03_day3_inference.ipynb      # Day 3 work
├── config.yaml                       # Configuration
├── requirements.txt
└── README.md
```

---

## 🗓️ Development Timeline

### **Day 1: Data Ingestion** ✅
- [x] Google Earth Engine setup
- [x] Fetch Sentinel-2 patches via API
- [x] Create synthetic LR/HR pairs
- [x] PyTorch DataLoader with augmentation

### **Day 2: Model Training** 🔄
- [ ] Implement SwinIR architecture
- [ ] Setup training loop
- [ ] Train on Colab T4 GPU (100-150 epochs)
- [ ] Track PSNR/SSIM metrics

### **Day 3: Inference & Demo** 📅
- [ ] Inference on test locations
- [ ] Tile-based processing for large images
- [ ] Streamlit demo UI
- [ ] Video presentation

---

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
data:
  scale_factor: 4              # 4x upscaling
  hr_patch_size: 256           # Patch size
  max_cloud_cover: 10          # Filter clouds
  num_training_samples: 1000   # Dataset size

training:
  epochs: 150
  batch_size: 8
  learning_rate: 0.0002
```

---

## 🧪 Usage

### Fetch Data
```python
from src.data.gee_fetcher import GEEFetcher

fetcher = GEEFetcher('config.yaml')

# Fetch specific location
delhi_patch = fetcher.fetch_specific_location('Delhi')

# Fetch training dataset
fetcher.fetch_dataset(num_samples=1000, save_dir='./data/raw')
```

### Create LR/HR Pairs
```python
from src.data.preprocessing import DataPreprocessor

preprocessor = DataPreprocessor('config.yaml')

# Create synthetic pairs
lr, hr = preprocessor.create_lr_hr_pair(hr_image, scale=4)

# Process entire dataset
preprocessor.process_dataset('./data/raw', './data/processed')
```

### Load Dataset
```python
from src.data.dataset import get_dataloader

train_loader = get_dataloader(
    lr_dir='./data/processed/train/lr',
    hr_dir='./data/processed/train/hr',
    batch_size=8
)
```

---

## 📈 Evaluation Metrics

- **PSNR** (Peak Signal-to-Noise Ratio): Measures pixel-level accuracy
- **SSIM** (Structural Similarity Index): Measures perceptual quality
- **LPIPS** (Learned Perceptual Image Patch Similarity): Deep perceptual metric
- **Visual Inspection**: Manual check for hallucinations

---

## 🛠️ Tech Stack

| Component | Tool |
|-----------|------|
| Data Source | Google Earth Engine API |
| Framework | PyTorch 2.0+ |
| Model | SwinIR (Transformer) |
| Processing | NumPy, OpenCV, Rasterio |
| Augmentation | Albumentations |
| Compute | Google Colab (T4/A100) |
| Demo | Streamlit |

---

## 🎓 References

- **SwinIR Paper**: [Image Restoration Using Swin Transformer](https://arxiv.org/abs/2108.10257)
- **Sentinel-2**: [ESA Copernicus Program](https://sentinel.esa.int/web/sentinel/missions/sentinel-2)
- **Google Earth Engine**: [Developers Guide](https://developers.google.com/earth-engine)

---

## 📝 License

MIT License

---

## 👥 Contributors

- Your Name - [GitHub](https://github.com/yourusername)

---

## 🙏 Acknowledgments

- Klymo Hackathon organizers
- Google Earth Engine team
- SwinIR authors

---

**Status**: 🟢 Day 1 Complete | 🔵 Day 2 In Progress | ⚪ Day 3 Pending

Last Updated: February 1, 2026
