import os
import argparse
import torch
import numpy as np
import random
import sys
sys.path.append('./train')
sys.path.append('./utils')

from train.train_gan import train_texPGAN
from train.train_vit import run_comparative_experiments
from utils.data_loader import get_dataloaders
import config

def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_dir_structure():
    """Create necessary directories"""
    os.makedirs('./output/models', exist_ok=True)
    os.makedirs('./output/samples', exist_ok=True)
    os.makedirs('./output/synthetic', exist_ok=True)

def main(args):
    # Set random seed
    set_seed(args.seed)
    
    # Create directory structure
    create_dir_structure()
    
    # Check if data exists
    if not os.path.exists(config.FER2013_DIR):
        print(f"Error: FER2013 dataset not found at {config.FER2013_DIR}")
        print("Please ensure you've downloaded and extracted the dataset correctly.")
        sys.exit(1)
    
    # Get data loaders
    train_loader, test_loader = get_dataloaders(
        config.FER2013_DIR, 
        batch_size=config.BATCH_SIZE
    )
    
    if args.stage == 'gan' or args.stage == 'all':
        print("=== Training TexPGAN for Synthetic Image Generation ===")
        train_texPGAN(
            train_loader=train_loader,
            epochs=args.gan_epochs,
            lr=config.GAN_LR,
            beta1=config.BETA1,
            beta2=config.BETA2,
            lambda_cls=config.LAMBDA_CLS,
            lambda_tex=config.LAMBDA_TEX,
            output_dir=config.OUTPUT_DIR
        )
    
    if args.stage == 'vit' or args.stage == 'all':
        print("=== Training and Evaluating ViT for Expression Recognition ===")
        run_comparative_experiments(output_dir=config.OUTPUT_DIR)
    
    print("=== Project execution completed! ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Texture-Enhanced Conditional GAN for FER")
    parser.add_argument('--stage', type=str, default='all', choices=['gan', 'vit', 'all'],
                        help='Which stage to run: gan, vit, or all')
    parser.add_argument('--gan_epochs', type=int, default=config.GAN_EPOCHS,
                        help='Number of epochs for GAN training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    main(args)