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
│   ├── generator.py       # Generator architecture
│   ├── discriminator.py   # Discriminator architecture  
│   └── vit.py             # Vision Transformer
├── utils/
│   ├── data_loader.py     # Data loading utilities
│   ├── texture_utils.py   # LBP and texture analysis functions
│   └── visualization.py   # Plotting and visualization
├── train/
│   ├── train_gan.py       # TexPGAN training
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

1. **Train only the GAN**:
```bash
python main.py --stage gan --gan_epochs 100
```

2. **Train only the ViT** (requires synthetic images to be generated):
```bash
python main.py --stage vit --vit_epochs 50
```

3. **Run the complete pipeline**:
```bash
python main.py --stage all --gan_epochs 100 --vit_epochs 50
```

## Features

- **Texture Enhancement**: Uses Local Binary Pattern (LBP) features to preserve and enhance texture details in generated images
- **Conditional Generation**: Generates images conditioned on expression labels
- **Dual-Branch Discriminator**: Separates real/fake classification and expression recognition
- **Vision Transformer**: Uses transformer architecture for expression recognition
- **Comparative Experiments**: Evaluates performance on real, synthetic, and mixed datasets

## Results

The system will generate:
- Synthetic facial expression images in `./output/samples/`
- Trained models in `./output/models/`
- Training curves and evaluation metrics in the project root directory
