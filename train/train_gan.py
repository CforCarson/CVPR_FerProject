import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import numpy as np
sys.path.append('..')

from models.generator import TextureEnhancedGenerator
from models.discriminator import DualBranchDiscriminator
from utils.data_loader import get_dataloaders, save_synthetic_images, balanced_sampling_dataloader, apply_clahe
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
    print(f"Using device: {device}")
    
    # Initialize models
    generator = TextureEnhancedGenerator(
        latent_dim=config.LATENT_DIM,
        embed_dim=config.GEN_EMBED_DIM,
        num_heads=config.GEN_NUM_HEADS
    ).to(device)
    
    discriminator = DualBranchDiscriminator().to(device)
    
    # Setup optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))
    
    # Learning rate schedulers for better training stability
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=epochs, eta_min=lr/10)
    scheduler_D = optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=epochs, eta_min=lr/10)
    
    # Loss functions
    adversarial_loss = nn.BCELoss()
    classification_loss = nn.CrossEntropyLoss()
    
    # Create fixed noise and labels for visualization
    fixed_noise = torch.randn(64, config.LATENT_DIM, device=device)
    fixed_labels = torch.tensor([i//8 % 7 for i in range(64)], device=device)
    
    # Training loop
    print(f"Starting training for {epochs} epochs...")
    
    # Tracking variables
    G_losses = []
    D_losses = []
    D_x_history = []  # Real images classified as real
    D_G_z_history = []  # Fake images classified as real
    cls_accuracy_history = []  # Track classification accuracy
    texture_loss_history = []  # Track texture loss
    
    # Class distribution counters
    class_counts = {i: 0 for i in range(7)}
    
    for epoch in range(epochs):
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_cls_acc = 0
        epoch_tex_loss = 0
        batches_processed = 0
        
        for i, (real_images, labels) in progress_bar:
            batch_size = real_images.size(0)
            batches_processed += 1
            
            # Track class distribution
            for label in labels:
                class_counts[label.item()] += 1
            
            # Transfer data to device
            real_images = real_images.to(device)
            labels = labels.to(device)
            
            # Apply CLAHE to enhance texture details in real images
            real_images_enhanced = torch.stack([apply_clahe(img.cpu(), 
                                               clip_limit=config.CLAHE_CLIP_LIMIT, 
                                               tile_grid_size=config.CLAHE_GRID_SIZE).to(device) 
                                               for img in real_images])
            
            # Ground truths
            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)
            
            # -----------------
            #  Train Discriminator
            # -----------------
            optimizer_D.zero_grad()
            
            # Loss for real images
            real_pred, real_cls, tex_score_real = discriminator(real_images_enhanced)
            d_real_loss = adversarial_loss(real_pred, valid)
            d_real_cls_loss = classification_loss(real_cls, labels)
            
            # Calculate classification accuracy
            _, predicted_labels = torch.max(real_cls, 1)
            d_cls_accuracy = (predicted_labels == labels).float().mean().item()
            epoch_cls_acc += d_cls_accuracy
            
            # Sample noise and labels
            z = torch.randn(batch_size, config.LATENT_DIM, device=device)
            gen_labels = labels
            
            # Generate a batch of images
            gen_images = generator(z, gen_labels)
            
            # Loss for fake images
            fake_pred, fake_cls, tex_score_fake = discriminator(gen_images.detach())
            d_fake_loss = adversarial_loss(fake_pred, fake)
            
            # Ensure fake and real texture scores have consistent dimensions
            tex_score_fake = F.adaptive_avg_pool2d(tex_score_fake, (1, 1))
            tex_score_real = F.adaptive_avg_pool2d(tex_score_real, (1, 1))

            # Compute texture consistency loss
            d_tex_loss = torch.mean(torch.abs(tex_score_fake - tex_score_real))
            epoch_tex_loss += d_tex_loss.item()
            
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
            
            # Apply CLAHE to enhance texture details in generated images
            gen_images_enhanced = torch.stack([apply_clahe(img.cpu(), 
                                              clip_limit=config.CLAHE_CLIP_LIMIT, 
                                              tile_grid_size=config.CLAHE_GRID_SIZE).to(device) 
                                              for img in gen_images])
            
            # Loss measures generator's ability to fool the discriminator
            validity, pred_label, _ = discriminator(gen_images_enhanced)
            g_loss = adversarial_loss(validity, valid)
            
            # Expression classification loss
            g_cls_loss = classification_loss(pred_label, gen_labels)
            
            # Texture preservation loss
            g_tex_loss = texture_consistency_loss(real_images_enhanced, gen_images_enhanced)
            
            # Feature matching loss
            with torch.no_grad():
                features_real = discriminator.feature_extractor(real_images_enhanced)
            features_fake = discriminator.feature_extractor(gen_images_enhanced)
            feature_matching_loss = nn.MSELoss()(features_fake, features_real.detach())
            
            # Total generator loss
            g_total_loss = g_loss + lambda_cls * g_cls_loss + lambda_tex * g_tex_loss + feature_matching_loss
            g_total_loss.backward()
            optimizer_G.step()
            
            # Save losses for plotting
            G_losses.append(g_total_loss.item())
            D_losses.append(d_loss.item())
            cls_accuracy_history.append(d_cls_accuracy)
            texture_loss_history.append(g_tex_loss.item())
            
            epoch_g_loss += g_total_loss.item()
            epoch_d_loss += d_loss.item()
            
            # Update progress bar
            progress_bar.set_description(
                f"[Epoch {epoch+1}/{epochs}] [D loss: {d_loss.item():.4f}] [G loss: {g_total_loss.item():.4f}] [D(x): {D_x:.4f}] [D(G(z)): {D_G_z1:.4f}] [Cls Acc: {d_cls_accuracy:.4f}]"
            )
            
            # Save LBP visualization periodically
            if i % 100 == 0:
                create_lbp_visualization(
                    gen_images[:8], 
                    save_path=os.path.join(output_dir, 'samples', f'lbp_viz_epoch_{epoch}_iter_{i}.png')
                )
                
                # Also save visualization of the CLAHE enhanced images
                enhanced_grid = vutils.make_grid((gen_images_enhanced[:8] + 1) / 2.0, nrow=4, padding=2, normalize=False)
                plt.figure(figsize=(10, 5))
                plt.imshow(enhanced_grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
                plt.axis('off')
                plt.title("CLAHE Enhanced")
                plt.savefig(os.path.join(output_dir, 'samples', f'clahe_enhanced_epoch_{epoch}_iter_{i}.png'))
                plt.close()
        
        # Calculate average losses and metrics for this epoch
        avg_g_loss = epoch_g_loss / batches_processed
        avg_d_loss = epoch_d_loss / batches_processed
        avg_cls_acc = epoch_cls_acc / batches_processed
        avg_tex_loss = epoch_tex_loss / batches_processed
        
        print(f"Epoch {epoch+1} summary:")
        print(f"  G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}")
        print(f"  Cls Accuracy: {avg_cls_acc:.4f}, Texture Loss: {avg_tex_loss:.4f}")
        
        # Update learning rates
        scheduler_G.step()
        scheduler_D.step()
        current_lr_G = scheduler_G.get_last_lr()[0]
        print(f"  Learning rate: {current_lr_G:.6f}")
        
        # Save generated samples at the end of each epoch
        with torch.no_grad():
            fake_samples = generator(fixed_noise, fixed_labels)
            # Apply CLAHE for better texture
            fake_samples_enhanced = torch.stack([apply_clahe(img.cpu(), 
                                                clip_limit=config.CLAHE_CLIP_LIMIT, 
                                                tile_grid_size=config.CLAHE_GRID_SIZE).to(device) 
                                                for img in fake_samples])
            
            # Rescale from [-1, 1] to [0, 1]
            fake_samples = (fake_samples + 1) / 2.0
            fake_samples_enhanced = (fake_samples_enhanced + 1) / 2.0
            
            # Save grid of regular generated images
            grid = vutils.make_grid(fake_samples, nrow=8, padding=2, normalize=False)
            plt.figure(figsize=(10, 10))
            plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, 'samples', f'generated_epoch_{epoch+1}.png'))
            plt.close()
            
            # Save grid of CLAHE enhanced images
            grid_enhanced = vutils.make_grid(fake_samples_enhanced, nrow=8, padding=2, normalize=False)
            plt.figure(figsize=(10, 10))
            plt.imshow(grid_enhanced.permute(1, 2, 0).cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, 'samples', f'generated_enhanced_epoch_{epoch+1}.png'))
            plt.close()
        
        # Save models periodically
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            torch.save(generator.state_dict(), os.path.join(output_dir, 'models', f'generator_epoch_{epoch+1}.pth'))
            torch.save(discriminator.state_dict(), os.path.join(output_dir, 'models', f'discriminator_epoch_{epoch+1}.pth'))
            
    # Print class distribution summary
    print("\nClass distribution in training:")
    total_samples = sum(class_counts.values())
    for class_idx, count in class_counts.items():
        class_name = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'][class_idx]
        print(f"  {class_name}: {count} ({count/total_samples*100:.2f}%)")
    
    # Plot training curves
    plt.figure(figsize=(20, 10))
    plt.subplot(2, 2, 1)
    plt.plot(G_losses, label='Generator')
    plt.plot(D_losses, label='Discriminator')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Generator and Discriminator Loss')
    
    plt.subplot(2, 2, 2)
    plt.plot(D_x_history, label='D(x)')
    plt.plot(D_G_z_history, label='D(G(z))')
    plt.xlabel('Iterations')
    plt.ylabel('Probability')
    plt.legend()
    plt.title('Discriminator Output Probabilities')
    
    plt.subplot(2, 2, 3)
    plt.plot(cls_accuracy_history)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Classification Accuracy')
    
    plt.subplot(2, 2, 4)
    plt.plot(texture_loss_history)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Texture Consistency Loss')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()
    
    print(f"Training completed! Models and results saved to {output_dir}")
    
    # Generate and save synthetic images for ViT training
    print("Generating synthetic dataset for ViT training...")
    samples_per_class = config.NUM_SYNTHETIC_SAMPLES // 7
    save_synthetic_images(generator, num_samples=samples_per_class, 
                         output_dir=os.path.join(output_dir, 'synthetic'))
    
    return generator, discriminator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TexPGAN")
    parser.add_argument("--epochs", type=int, default=config.GAN_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.GAN_LR, help="Learning rate")
    args = parser.parse_args()
    
    # Load data
    if config.USE_AUGMENTED_DATASET:
        print("Using augmented dataset for training...")
        train_loader, _ = get_dataloaders(config.FER2013_DIR, batch_size=args.batch_size, augment=True)
    else:
        train_loader, _ = get_dataloaders(config.FER2013_DIR, batch_size=args.batch_size)
    
    # Use balanced sampling if configured
    if config.USE_BALANCED_SAMPLING:
        print("Using balanced class sampling...")
        if hasattr(train_loader.dataset, 'original_dataset'):
            # For augmented dataset
            balanced_dataset = train_loader.dataset.original_dataset
        else:
            # For regular dataset
            balanced_dataset = train_loader.dataset
            
        train_loader = balanced_sampling_dataloader(balanced_dataset, batch_size=args.batch_size)
    
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