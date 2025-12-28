# U-Net for Cell Segmentation

PyTorch implementation of [U-Net](https://arxiv.org/abs/1505.04597) for biomedical image segmentation, optimized for the ISBI 2012 Cell Segmentation Challenge.

---

## 🚀 Quick Start (Google Colab)

### 1. Setup
```python
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies
!pip install -q scikit-image scipy PyYAML scikit-learn
```

### 2. Download ISBI 2012 Dataset
- Get `train-volume.tif` and `train-labels.tif` from [ISBI Challenge](http://brainiac2.mit.edu/isbi_challenge/)
- Upload to `data` folder


## 📁 Project Structure

```
Unet_implementation/
├── data/
│   ├── train-volume/images/    # 30 training images
│   └── train-labels/labels/    # 30 training labels
├── models/unet.py              # U-Net architecture
├── losses/losses.py            # Loss functions
├── metrics/metrics.py          # Evaluation metrics
├── utils/
│   ├── transforms.py           # Data augmentation
│   └── data_utils.py           # Train/val splitting
├── config.yaml                 # Configuration
├── train.py                    # Training script
└── evaluate.py                 # Evaluation script
```

---

## ⚙️ Key Configuration

Edit `config.yaml`:

```yaml
# Data
data:
  train_val_split: 0.8          # 24 train, 6 validation

# Training
training:
  epochs: 100
  batch_size: 1
  learning_rate: 0.01
  optimizer: "sgd"              # or "adam"
  
  loss:
    type: "combined"            # weighted_ce, dice, or combined
    ce_weight: 1.0
    dice_weight: 1.0

# Augmentation
augmentation:
  elastic_transform:
    enabled: true
    alpha: 250                  # Deformation strength
  horizontal_flip:
    enabled: true
  vertical_flip:
    enabled: true
```

---

## 📊 Expected Results

| Metric | Expected |
|--------|----------|
| Validation Dice | 0.85 - 0.95 |
| Validation IoU | 0.75 - 0.90 |
| Training Time | 2-3 hours (100 epochs, T4 GPU) |

---

## 🎯 Features

 Complete U-Net with skip connections  
 Weighted cross-entropy + Dice loss  
 Border weight maps (emphasizes cell boundaries)  
 Elastic deformation augmentation  
 Automatic train/validation split  
 Best model selection based on validation Dice  
 Prediction visualization  



## 🛠️ Local Installation

```bash
pip install -r requirements.txt
python train.py
python evaluate.py
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+
- 4GB+ GPU (recommended) or CPU

---

## 📈 Training Output

```
Epoch 50/100
--------------------------------------------------
Training:   100%|██████████| 24/24 [00:40<00:00]
Validation: 100%|██████████| 6/6 [00:09<00:00]

Train Loss: 1.98 | Val Loss: 1.99
Train Dice: 0.87 | Val Dice: 0.87
✨ New best Val Dice score: 0.8740
```




## 📝 ISBI 2012 Dataset

- **Images**: 30 slices (512×512 pixels)
- **Type**: Electron microscopy of neuronal structures
- **Split**: 24 training / 6 validation (configurable)
- **Format**: Grayscale images + binary labels


