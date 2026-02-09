# Multi-Images Illumination Adaptive Transformer (IAT)

A transformer-based deep learning model for low-light image enhancement using multiple exposure fusion and adaptive illumination correction with white balance adjustment.

## Features

- **Multi-Exposure Fusion**: Processes multiple exposure images simultaneously for robust enhancement
- **Exposure Correction**: Adaptive handling of underexposed and overexposed regions
- **White Balance Adjustment**: Automatic color temperature and white balance correction
- **Transformer Architecture**: Combines convolutional blocks with Swin Transformer blocks
- **Variable Multi-Exposure Support**: Handles variable number of exposures per set

## Requirements

- Python 3.7
- CUDA 10.2 or higher (recommended for GPU acceleration)
- conda (for environment management)

## Installation

### Step 1: Create Conda Environment with Python 3.7

```bash
# Create new conda environment
conda create -n IAT-env python==3.7.0

# Activate the environment
conda activate IAT-env
```

### Step 2: Install PyTorch and Dependencies

```bash
# Install PyTorch
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
```

### Step 3: Install Additional Dependencies

```bash
# Install from requirements.txt
pip install -r requirements.txt
```

## Project Structure

```
.
├── README.md                               # Project documentation
├── requirements.txt                         # Python dependencies
├── train_enhanced_multi_exposure.py         # Training script
├── inference_enhanced.py                    # Inference script
├── data_loaders/
│   └── multi_exposure_dynamic.py            # Multi-exposure dataset loader
└── model/
    ├── IAT_main_enhanced.py                 # Main model architecture
    ├── blocks.py                            # Transformer and convolution blocks
    ├── exposure_module.py                   # Exposure and white balance module
    ├── global_net.py                        # Global feature extraction network
    ├── losses.py                            # Custom loss functions
    └── multi_exposure_encoder.py            # Multi-exposure fusion encoder
```

## Dataset Structure

Organize your dataset as follows:

```
data/
├── train/
│   ├── set1/
│   │   ├── exp_1.jpg          # First exposure
│   │   ├── exp_2.jpg          # Second exposure
│   │   └── gt.jpg             # Ground truth
│   ├── set2/
│   │   ├── exp_1.jpg
│   │   ├── exp_2.jpg
│   │   └── gt.jpg
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

## Usage

### Training

```bash
python train_enhanced_multi_exposure.py \
    --data_path ./data/train \
    --batch_size 16 \
    --epochs 100 \
    --learning_rate 1e-4 \
    --checkpoint_dir ./checkpoints
```

### Inference

```bash
python inference_enhanced.py \
    --model_path ./checkpoints/model.pth \
    --input_dir ./data/test \
    --output_dir ./results
```

## Main Components

### 1. IAT_Enhanced Model (`model/IAT_main_enhanced.py`)

- Local and global feature extraction with Transformer blocks
- Integrated exposure and white balance modules
- Adaptive illumination correction for blind enhancement

### 2. Multi-Exposure Encoder (`model/multi_exposure_encoder.py`)

- Advanced multi-exposure fusion with cross-attention mechanism
- Residual Dense Blocks (RDB) for powerful feature extraction
- Multi-scale processing with learnable fusion weights
- Supports 1 to N exposure images per set

### 3. Exposure Module (`model/exposure_module.py`)

- ExposureAnalyzer: Analyzes luminance and exposure levels
- AdaptiveGammaCorrection: Corrects image brightness
- WhiteBalanceCorrection: Adjusts color temperature

### 4. Loss Functions (`model/losses.py`)

- **ExposureLoss**: Brightness distribution optimization
- **WhiteBalanceLoss**: Color balance preservation
- **ContrastLoss**: Local contrast enhancement
- **ColorConstancyLoss**: Color consistency maintenance
- **CombinedLoss**: Multi-objective loss integration

### 5. Dataset Loader (`data_loaders/multi_exposure_dynamic.py`)

- Dynamic loading of variable-length multi-exposure sets
- Flexible preprocessing and augmentation
- Support for random and sequential sampling

## Configuration

Key hyperparameters can be modified in the training script:

- `batch_size`: Number of samples per batch (default: 16)
- `learning_rate`: Initial learning rate (default: 1e-4)
- `epochs`: Total training epochs (default: 100)
- `num_exposures`: Number of exposure images (default: 2-3)
- `image_size`: Input image dimensions (default: 512x512)

## Training Tips

1. **Data Preparation**: Ensure consistent image sizes across exposures
2. **Batch Size**: Adjust based on GPU memory (16 for 8GB, 8 for 4GB)
3. **Learning Rate**: Start with 1e-4, reduce if loss plateaus
4. **Mixed Precision**: Use AMP for faster training and lower memory usage
5. **Checkpointing**: Save models every 10 epochs for best results

## Inference

The inference script provides:

- Batch processing of images
- Exposure analysis and metrics
- Multiple export formats (JPG, PNG)
- Detailed enhancement statistics

## Performance Metrics

Supported evaluation metrics:

- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)
- Luminance distribution analysis

## Troubleshooting

### CUDA Out of Memory

- Reduce batch size
- Use smaller image size
- Enable gradient checkpointing

### Poor Enhancement Quality

- Check dataset alignment between exposures
- Verify ground truth image quality
- Increase training epochs
- Adjust loss weights in the training config

### Slow Training

- Enable mixed precision training
- Use CUDA graphs for inference
- Reduce number of validation cycles
