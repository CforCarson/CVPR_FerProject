# Texture-Enhanced Conditional GAN for Facial Expression Recognition

This project implements a multi-stage approach for facial expression recognition using a texture-enhanced conditional GAN and Vision Transformer. The system consists of two main components:

1. **TexPGAN**: A conditional GAN with texture enhancement for generating synthetic facial expression images
2. **Vision Transformer**: A transformer-based model for expression recognition

## Project Structure

```
fer_project/
├── data/
│   ├── fer2013/           # Contains the FER-2013 dataset
│   └── synthetic/         # Stores generated synthetic images
├── models/
│   ├── generator.py       # Original generator architecture
│   ├── discriminator.py   # Original discriminator architecture
│   ├── complex_generator.py     # Enhanced generator architecture
│   ├── complex_discriminator.py # Enhanced discriminator architecture  
│   └── vit.py             # Vision Transformer
├── utils/
│   ├── data_loader.py     # Data loading utilities
│   ├── texture_utils.py   # LBP and texture analysis functions
│   ├── face_validation.py # Face detection validation utilities
│   └── visualization.py   # Plotting and visualization
├── train/
│   ├── train_gan.py       # Original TexPGAN training
│   ├── train_complex_gan.py # Enhanced GAN training
│   └── train_vit.py       # ViT training
├── config.py              # Configuration parameters
└── main.py                # Main execution script
```

## Setup

1. Create a conda environment:
```bash
conda create -n fer_project python=3.8
conda activate fer_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the FER-2013 dataset:
- Download the dataset from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- Place the `fer2013.csv` file in the `./data/fer2013/` directory

## Usage

The project can be run in different stages:

1. **Train the original GAN**:
```bash
python main.py --stage gan --gan_epochs 100
```

2. **Train the improved complex GAN** (recommended for better face quality):
```bash
python main.py --stage complex_gan --gan_epochs 300
```

3. **Train only the ViT** (requires synthetic images to be generated):
```bash
python main.py --stage vit --vit_epochs 50
```

4. **Run the complete pipeline** with improved GAN:
```bash
python main.py --stage all --gan_epochs 300 --vit_epochs 50
```

## Features

### Original TexPGAN
- **Texture Enhancement**: Uses Local Binary Pattern (LBP) features to preserve and enhance texture details in generated images
- **Conditional Generation**: Generates images conditioned on expression labels
- **Dual-Branch Discriminator**: Separates real/fake classification and expression recognition

### Enhanced Complex GAN
- **Deeper Architecture**: Adopts multi-scale convolutions with greater capacity
- **Spectral Normalization**: Improves training stability
- **Self-Attention Mechanism**: Better long-range dependencies in feature maps
- **Multi-scale Texture Enhancement**: Enhanced LBP texture preservation
- **Face Validation**: Automatically filters generated images using multiple face detection methods
- **Improved Training Techniques**: Lower learning rate, feature matching loss, and enhanced schedulers

### Vision Transformer
- Uses transformer architecture for expression recognition
- Conducts comparative experiments on real, synthetic, and mixed datasets

## Results

The system will generate:
- Synthetic facial expression images in `./output/samples/`
- Validated face images in `./output/samples/valid/`
- Trained models in `./output/models/`
- Training curves and evaluation metrics in the project root directory

## Face Detection Validation

The enhanced GAN includes automatic face validation using three detection methods:
1. OpenCV's Haar Cascade (always available)
2. Face Recognition library (if installed)
3. YOLOv8 Face Detection (if installed)

Only images that pass a confidence threshold are saved in the final dataset.
