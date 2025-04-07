import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append('..')

from models.generator import TextureEnhancedGenerator
from models.discriminator import DualBranchDiscriminator
from utils.data_loader import get_dataloaders, save_synthetic_images
from utils.texture_utils import texture_consistency_loss, create_lbp_visualization
import config

def train_texPGAN(train_loader, epochs=100, lr=0.0002, beta1=0.5, beta2=0.999, 
                 lambda_cls=10.0, lambda_tex=5.0, output_dir='./output'):
    """Train the Texture-Enhanced conditional GAN"""
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'samples'), exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for GAN training: {device}")
    
    # Print CUDA memory information if available
    if device.type == 'cuda':
        print(f"Initial CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Initial CUDA memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    # Initialize models
    generator = TextureEnhancedGenerator(
        latent_dim=config.LATENT_DIM,
        embed_dim=config.GEN_EMBED_DIM,
        num_heads=config.GEN_NUM_HEADS
    ).to(device)
    
    discriminator = DualBranchDiscriminator().to(device)
    
    # Explicitly print model device placement
    print(f"Generator device: {next(generator.parameters()).device}")
    print(f"Discriminator device: {next(discriminator.parameters()).device}")
    
    # Setup optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))
    
    # Loss functions
    adversarial_loss = nn.BCELoss()
    classification_loss = nn.CrossEntropyLoss()
    
    # Create fixed noise and labels for visualization
    fixed_noise = torch.randn(64, config.LATENT_DIM, device=device)
    fixed_labels = torch.tensor([i//8 for i in range(64)], device=device)
    
    # Training loop
    print(f"Starting training for {epochs} epochs...")
    
    # Tracking variables
    G_losses = []
    D_losses = []
    D_x_history = []  # Real images classified as real
    D_G_z_history = []  # Fake images classified as real
    
    for epoch in range(epochs):
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (real_images, labels) in progress_bar:
            batch_size = real_images.size(0)
            
            # Transfer data to device
            real_images = real_images.to(device)
            labels = labels.to(device)
            
            # Ground truths
            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)
            
            # -----------------
            #  Train Discriminator
            # -----------------
            optimizer_D.zero_grad()
            
            # Loss for real images
            real_pred, real_cls, tex_score_real = discriminator(real_images)
            d_real_loss = adversarial_loss(real_pred, valid)
            d_real_cls_loss = classification_loss(real_cls, labels)
            
            # Sample noise and labels
            z = torch.randn(batch_size, config.LATENT_DIM, device=device)
            gen_labels = labels
            
            # Generate a batch of images
            gen_images = generator(z, gen_labels)
            
            # Loss for fake images
            fake_pred, fake_cls, tex_score_fake = discriminator(gen_images.detach())
            d_fake_loss = adversarial_loss(fake_pred, fake)
            
            # Texture consistency loss
            d_tex_loss = torch.mean(torch.abs(tex_score_fake - tex_score_real))
            
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2 + lambda_cls * d_real_cls_loss + lambda_tex * d_tex_loss
            d_loss.backward()
            optimizer_D.step()
            
            # Track discriminator performance
            D_x = real_pred.mean().item()  # Average on real images
            D_G_z1 = fake_pred.mean().item()  # Average on fake images
            D_x_history.append(D_x)
            D_G_z_history.append(D_G_z1)
            
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # Generate new fake images to avoid using detached ones
            gen_images = generator(z, gen_labels)
            
            # Loss measures generator's ability to fool the discriminator
            validity, pred_label, _ = discriminator(gen_images)
            g_loss = adversarial_loss(validity, valid)
            
            # Expression classification loss
            g_cls_loss = classification_loss(pred_label, gen_labels)
            
            # Texture preservation loss
            g_tex_loss = texture_consistency_loss(real_images, gen_images)
            
            # Total generator loss
            g_total_loss = g_loss + lambda_cls * g_cls_loss + lambda_tex * g_tex_loss
            g_total_loss.backward()
            optimizer_G.step()
            
            # Save losses for plotting
            G_losses.append(g_total_loss.item())
            D_losses.append(d_loss.item())
            
            # Update progress bar
            progress_bar.set_description(
                f"[Epoch {epoch+1}/{epochs}] [D loss: {d_loss.item():.4f}] [G loss: {g_total_loss.item():.4f}] [D(x): {D_x:.4f}] [D(G(z)): {D_G_z1:.4f}]"
            )
            
            # Save LBP visualization periodically
            if i % 100 == 0:
                create_lbp_visualization(
                    gen_images[:8], 
                    save_path=os.path.join(output_dir, 'samples', f'lbp_viz_epoch_{epoch}_iter_{i}.png')
                )
                
                # Print GPU memory usage if available
                if device.type == 'cuda':
                    print(f"CUDA memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f}MB allocated, {torch.cuda.memory_reserved(0) / 1024**2:.2f}MB reserved")
        
        # Save generated samples at the end of each epoch
        with torch.no_grad():
            fake_samples = generator(fixed_noise, fixed_labels)
            # Rescale from [-1, 1] to [0, 1]
            fake_samples = (fake_samples + 1) / 2.0
            # Save grid of generated images
            grid = vutils.make_grid(fake_samples, nrow=8, padding=2, normalize=False)
            plt.figure(figsize=(10, 10))
            plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, 'samples', f'generated_epoch_{epoch+1}.png'))
            plt.close()
        
        # Save models periodically
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            torch.save(generator.state_dict(), os.path.join(output_dir, 'models', f'generator_epoch_{epoch+1}.pth'))
            torch.save(discriminator.state_dict(), os.path.join(output_dir, 'models', f'discriminator_epoch_{epoch+1}.pth'))
            
    # Plot training curves
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(G_losses, label='Generator')
    plt.plot(D_losses, label='Discriminator')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(D_x_history, label='D(x)')
    plt.plot(D_G_z_history, label='D(G(z))')
    plt.xlabel('Iterations')
    plt.ylabel('Probability')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()
    
    print(f"Training completed! Models and results saved to {output_dir}")
    
    # Generate and save synthetic images for ViT training
    print("Generating synthetic dataset for ViT training...")
    save_synthetic_images(generator, num_samples=config.NUM_SYNTHETIC_SAMPLES//7, 
                         output_dir=os.path.join(output_dir, 'synthetic'))
    
    return generator, discriminator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TexPGAN")
    parser.add_argument("--epochs", type=int, default=config.GAN_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.GAN_LR, help="Learning rate")
    args = parser.parse_args()
    
    # Load data
    train_loader, _ = get_dataloaders(config.FER2013_DIR, batch_size=args.batch_size)
    
    # Train GAN
    train_texPGAN(
        train_loader=train_loader,
        epochs=args.epochs,
        lr=args.lr,
        beta1=config.BETA1,
        beta2=config.BETA2,
        lambda_cls=config.LAMBDA_CLS,
        lambda_tex=config.LAMBDA_TEX,
        output_dir=config.OUTPUT_DIR
    )