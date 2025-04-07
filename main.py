import os
import argparse
import torch
import numpy as np
import random
from train.train_gan import train_texPGAN
from train.train_vit import run_comparative_experiments
from utils.data_loader import get_dataloaders
import config

def set_seed(seed):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main(args):
    # Set random seed
    set_seed(args.seed)
    
    # Create directories if they don't exist
    os.makedirs('./data/fer2013', exist_ok=True)
    os.makedirs('./data/synthetic', exist_ok=True)
    os.makedirs('./output', exist_ok=True)
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        config.FER2013_CSV_PATH, 
        batch_size=config.BATCH_SIZE
    )
    
    if args.stage == 'gan' or args.stage == 'all':
        print("=== Training TexPGAN for Synthetic Image Generation ===")
        train_texPGAN(
            dataloader=train_loader,
            epochs=args.gan_epochs,
            lr=config.LEARNING_RATE,
            beta1=config.BETA1,
            beta2=config.BETA2,
            lambda_cls=config.LAMBDA_CLS,
            lambda_tex=config.LAMBDA_TEX,
            device=device,
            output_dir=config.OUTPUT_DIR
        )
    
    if args.stage == 'vit' or args.stage == 'all':
        print("=== Training and Evaluating ViT for Expression Recognition ===")
        run_comparative_experiments(device=device)
    
    print("=== Project execution completed! ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Texture-Enhanced Conditional GAN for FER")
    parser.add_argument('--stage', type=str, default='all', choices=['gan', 'vit', 'all'],
                        help='Which stage to run: gan, vit, or all')
    parser.add_argument('--gan_epochs', type=int, default=100, help='Number of epochs for GAN training')
    parser.add_argument('--vit_epochs', type=int, default=50, help='Number of epochs for ViT training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    main(args) 