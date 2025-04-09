import os
import argparse
import torch
import numpy as np
import random
import sys
sys.path.append('./train')
sys.path.append('./utils')

from train.train_gan import train_texPGAN
from train.train_complex_gan import train_complex_GAN
from train.train_vit import run_comparative_experiments
from utils.data_loader import get_dataloaders, balanced_sampling_dataloader
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
    os.makedirs('./output/samples/valid', exist_ok=True)
    os.makedirs('./output/synthetic', exist_ok=True)
    for cls in ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']:
        os.makedirs(f'./output/synthetic/{cls}', exist_ok=True)

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
    print("Initializing data loaders...")
    if args.stage == 'gan' or args.stage == 'complex_gan' or args.stage == 'all':
        if config.USE_AUGMENTED_DATASET:
            print("Using augmented dataset with additional transformations...")
            train_loader, test_loader = get_dataloaders(
                config.FER2013_DIR, 
                batch_size=config.BATCH_SIZE,
                augment=True
            )
            
            # Apply balanced sampling if configured
            if config.USE_BALANCED_SAMPLING:
                print("Applying balanced class sampling...")
                if hasattr(train_loader.dataset, 'original_dataset'):
                    balanced_dataset = train_loader.dataset.original_dataset
                else:
                    balanced_dataset = train_loader.dataset
                
                train_loader = balanced_sampling_dataloader(balanced_dataset, batch_size=config.BATCH_SIZE)
        else:
            train_loader, test_loader = get_dataloaders(
                config.FER2013_DIR, 
                batch_size=config.BATCH_SIZE
            )
    else:
        train_loader, test_loader = get_dataloaders(
            config.FER2013_DIR, 
            batch_size=config.BATCH_SIZE
        )
    
    if args.stage == 'gan':
        print("=== Training Original TexPGAN for Synthetic Image Generation ===")
        print(f"- Texture enhancement weight: {config.LAMBDA_TEX}")
        print(f"- Classification weight: {config.LAMBDA_CLS}")
        print(f"- Training for {args.gan_epochs} epochs")
        print(f"- Will generate {config.NUM_SYNTHETIC_SAMPLES} synthetic images")
        
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
        
        print("\nGAN training completed. Synthetic images generated.")
    
    elif args.stage == 'complex_gan':
        print("=== Training Enhanced Complex GAN for Synthetic Image Generation ===")
        print(f"- Using deeper network architecture with self-attention")
        print(f"- Using spectral normalization and enhanced texture preservation")
        print(f"- Using face detection validation to filter generated images")
        print(f"- Texture enhancement weight: {config.LAMBDA_TEX}")
        print(f"- Classification weight: {config.LAMBDA_CLS}")
        print(f"- Training for {args.gan_epochs} epochs with lower learning rate")
        print(f"- Will generate {config.NUM_SYNTHETIC_SAMPLES} validated synthetic images")
        
        train_complex_GAN(
            train_loader=train_loader,
            epochs=args.gan_epochs,
            lr=config.COMPLEX_GAN_LR,
            beta1=config.BETA1,
            beta2=config.BETA2,
            lambda_cls=config.LAMBDA_CLS,
            lambda_tex=config.LAMBDA_TEX,
            output_dir=config.OUTPUT_DIR
        )
        
        print("\nComplex GAN training completed. Validated synthetic images generated.")
    
    elif args.stage == 'all':
        print("=== Training Enhanced Complex GAN for Synthetic Image Generation ===")
        print(f"- Using deeper network architecture with self-attention")
        print(f"- Using spectral normalization and enhanced texture preservation")
        print(f"- Using face detection validation to filter generated images")
        print(f"- Texture enhancement weight: {config.LAMBDA_TEX}")
        print(f"- Classification weight: {config.LAMBDA_CLS}")
        print(f"- Training for {args.gan_epochs} epochs with lower learning rate")
        print(f"- Will generate {config.NUM_SYNTHETIC_SAMPLES} validated synthetic images")
        
        train_complex_GAN(
            train_loader=train_loader,
            epochs=args.gan_epochs,
            lr=config.COMPLEX_GAN_LR,
            beta1=config.BETA1,
            beta2=config.BETA2,
            lambda_cls=config.LAMBDA_CLS,
            lambda_tex=config.LAMBDA_TEX,
            output_dir=config.OUTPUT_DIR
        )
        
        print("\nComplex GAN training completed. Validated synthetic images generated.")
        
        print("=== Training and Evaluating ViT for Expression Recognition ===")
        print(f"- Using Vision Transformer for classification")
        print(f"- Will conduct comparative experiments on real, synthetic, and mixed datasets")
        print(f"- Training for {args.vit_epochs} epochs for each experiment")
        
        run_comparative_experiments(output_dir=config.OUTPUT_DIR)
    
    elif args.stage == 'vit':
        print("=== Training and Evaluating ViT for Expression Recognition ===")
        print(f"- Using Vision Transformer for classification")
        print(f"- Will conduct comparative experiments on real, synthetic, and mixed datasets")
        print(f"- Training for {args.vit_epochs} epochs for each experiment")
        
        run_comparative_experiments(output_dir=config.OUTPUT_DIR)
    
    print("\n=== Project execution completed! ===")
    print(f"Results saved to {config.OUTPUT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Texture-Enhanced Conditional GAN for FER")
    parser.add_argument('--stage', type=str, default='all', 
                        choices=['gan', 'complex_gan', 'vit', 'all'],
                        help='Which stage to run: gan, complex_gan, vit, or all')
    parser.add_argument('--gan_epochs', type=int, default=config.GAN_EPOCHS,
                        help='Number of epochs for GAN training')
    parser.add_argument('--vit_epochs', type=int, default=config.VIT_EPOCHS,
                        help='Number of epochs for ViT training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    main(args)