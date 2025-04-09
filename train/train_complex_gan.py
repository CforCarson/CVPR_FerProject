import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import sys
sys.path.append('..')

from models.complex_generator import ComplexGenerator
from models.complex_discriminator import DualBranchComplexDiscriminator
from utils.data_loader import get_dataloaders, balanced_sampling_dataloader, apply_clahe
from utils.texture_utils import texture_consistency_loss, create_lbp_visualization
from utils.face_validation import validate_batch, filter_and_save_synthetic_images
import config

def train_complex_GAN(train_loader, epochs=300, lr=0.0001, beta1=0.5, beta2=0.999, 
                     lambda_cls=12.0, lambda_tex=8.0, output_dir='./output'):
    """Train the Complex GAN with all the improvements"""
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'samples'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'samples', 'valid'), exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize models with deeper networks
    generator = ComplexGenerator(
        latent_dim=config.LATENT_DIM,
        num_classes=7,
        ngf=64
    ).to(device)
    
    discriminator = DualBranchComplexDiscriminator(
        num_classes=7,
        ndf=64
    ).to(device)
    
    # Setup optimizers with lower learning rate
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))
    
    # Learning rate schedulers with longer decay
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=epochs, eta_min=lr/20)
    scheduler_D = optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=epochs, eta_min=lr/20)
    
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
    face_validity_history = []  # Track face detection validity
    
    best_face_validity = 0.0  # Track best validity score for model saving
    
    for epoch in range(epochs):
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_cls_acc = 0
        epoch_tex_loss = 0
        epoch_face_validity = 0
        batches_processed = 0
        
        generator.train()
        discriminator.train()
        
        for i, (real_images, labels) in progress_bar:
            batch_size = real_images.size(0)
            batches_processed += 1
            
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
            
            # Texture consistency loss
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
            
            # Apply CLAHE for better texture
            gen_images_enhanced = torch.stack([apply_clahe(img.cpu(), 
                                             clip_limit=config.CLAHE_CLIP_LIMIT, 
                                             tile_grid_size=config.CLAHE_GRID_SIZE).to(device) 
                                             for img in gen_images])
            
            # Loss measures generator's ability to fool the discriminator
            validity, pred_label, _ = discriminator(gen_images_enhanced)
            g_loss = adversarial_loss(validity, valid)
            
            # Expression classification loss
            g_cls_loss = classification_loss(pred_label, gen_labels)
            
            # Texture preservation loss (enhanced)
            g_tex_loss = texture_consistency_loss(real_images_enhanced, gen_images_enhanced)
            
            # Feature matching loss with adaptive pooling to ensure same dimensions
            features_real = discriminator.feature_extractor[0:9](real_images_enhanced)
            features_fake = discriminator.feature_extractor[0:9](gen_images_enhanced)
            
            # Ensure dimensions match by using adaptive pooling
            if features_real.shape != features_fake.shape:
                adaptive_pool = nn.AdaptiveAvgPool2d((features_real.shape[2], features_real.shape[3]))
                features_fake = adaptive_pool(features_fake)
                
            feature_matching_loss = nn.MSELoss()(features_fake, features_real.detach())
            
            # Total generator loss
            g_total_loss = g_loss + lambda_cls * g_cls_loss + lambda_tex * g_tex_loss + feature_matching_loss
            g_total_loss.backward()
            optimizer_G.step()
            
            # Track face validity (only occasionally to save computation)
            if i % 10 == 0:
                with torch.no_grad():
                    # Create a small batch for validation
                    valid_mask, confidence = validate_batch(gen_images[:min(16, batch_size)])
                    batch_validity = valid_mask.float().mean().item() if len(valid_mask) > 0 else 0
                    epoch_face_validity += batch_validity
                    face_validity_history.append(batch_validity)
            
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
            
            # Create visualizations periodically
            if i % 100 == 0:
                # Save LBP visualization
                create_lbp_visualization(
                    gen_images[:8], 
                    save_path=os.path.join(output_dir, 'samples', f'lbp_viz_epoch_{epoch}_iter_{i}.png')
                )
                
                # Save grid of generated images
                grid = vutils.make_grid((gen_images[:16] + 1) / 2.0, nrow=4, padding=2, normalize=False)
                plt.figure(figsize=(10, 10))
                plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
                plt.axis('off')
                plt.title(f"Generated Images (Epoch {epoch+1}, Iteration {i})")
                plt.savefig(os.path.join(output_dir, 'samples', f'generated_epoch_{epoch+1}_iter_{i}.png'))
                plt.close()
                
                # Also save visualization of the CLAHE enhanced images
                enhanced_grid = vutils.make_grid((gen_images_enhanced[:16] + 1) / 2.0, nrow=4, padding=2, normalize=False)
                plt.figure(figsize=(10, 10))
                plt.imshow(enhanced_grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
                plt.axis('off')
                plt.title(f"CLAHE Enhanced (Epoch {epoch+1}, Iteration {i})")
                plt.savefig(os.path.join(output_dir, 'samples', f'clahe_enhanced_epoch_{epoch+1}_iter_{i}.png'))
                plt.close()
        
        # Calculate average metrics for this epoch
        avg_g_loss = epoch_g_loss / batches_processed
        avg_d_loss = epoch_d_loss / batches_processed
        avg_cls_acc = epoch_cls_acc / batches_processed
        avg_tex_loss = epoch_tex_loss / batches_processed
        avg_face_validity = epoch_face_validity / (batches_processed / 10 + 1e-8)  # Only checked every 10 batches
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}")
        print(f"  Cls Accuracy: {avg_cls_acc:.4f}, Texture Loss: {avg_tex_loss:.4f}")
        print(f"  Face Validity: {avg_face_validity:.2%}")
        
        # Update learning rates
        scheduler_G.step()
        scheduler_D.step()
        current_lr_G = scheduler_G.get_last_lr()[0]
        print(f"  Learning rate: {current_lr_G:.6f}")
        
        # Save generated samples at the end of each epoch
        with torch.no_grad():
            generator.eval()
            fake_samples = generator(fixed_noise, fixed_labels)
            
            # Apply CLAHE for better texture
            fake_samples_enhanced = torch.stack([apply_clahe(img.cpu(), 
                                               clip_limit=config.CLAHE_CLIP_LIMIT, 
                                               tile_grid_size=config.CLAHE_GRID_SIZE).to(device) 
                                               for img in fake_samples])
            
            # Rescale from [-1, 1] to [0, 1]
            fake_samples = (fake_samples + 1) / 2.0
            fake_samples_enhanced = (fake_samples_enhanced + 1) / 2.0
            
            # Check face validity of samples
            valid_mask, _ = validate_batch(fake_samples * 2 - 1)  # Convert back to [-1, 1] for validation
            validity_score = valid_mask.float().mean().item()
            print(f"  Fixed samples face validity: {validity_score:.2%}")
            
            # Save grid of regular generated images
            grid = vutils.make_grid(fake_samples, nrow=8, padding=2, normalize=False)
            plt.figure(figsize=(10, 10))
            plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.title(f"Generated Images (Epoch {epoch+1})")
            plt.savefig(os.path.join(output_dir, 'samples', f'generated_epoch_{epoch+1}.png'))
            plt.close()
            
            # Save grid of CLAHE enhanced images
            grid_enhanced = vutils.make_grid(fake_samples_enhanced, nrow=8, padding=2, normalize=False)
            plt.figure(figsize=(10, 10))
            plt.imshow(grid_enhanced.permute(1, 2, 0).cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.title(f"CLAHE Enhanced (Epoch {epoch+1})")
            plt.savefig(os.path.join(output_dir, 'samples', f'generated_enhanced_epoch_{epoch+1}.png'))
            plt.close()
            
            # Save valid samples separately
            if validity_score > 0:
                valid_samples = fake_samples[valid_mask]
                valid_grid = vutils.make_grid(valid_samples[:min(64, len(valid_samples))], nrow=8, padding=2, normalize=False)
                plt.figure(figsize=(10, 10))
                plt.imshow(valid_grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
                plt.axis('off')
                plt.title(f"Valid Face Images (Epoch {epoch+1})")
                plt.savefig(os.path.join(output_dir, 'samples', 'valid', f'valid_faces_epoch_{epoch+1}.png'))
                plt.close()
        
        # Save models periodically
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            torch.save(generator.state_dict(), os.path.join(output_dir, 'models', f'generator_epoch_{epoch+1}.pth'))
            torch.save(discriminator.state_dict(), os.path.join(output_dir, 'models', f'discriminator_epoch_{epoch+1}.pth'))
        
        # Save best model based on face validity
        if avg_face_validity > best_face_validity:
            best_face_validity = avg_face_validity
            torch.save(generator.state_dict(), os.path.join(output_dir, 'models', 'generator_best_validity.pth'))
            print(f"  Saved new best model with face validity: {best_face_validity:.2%}")
    
    # Plot training progress
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(G_losses, label='Generator')
    plt.plot(D_losses, label='Discriminator')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(D_x_history, label='D(x)')
    plt.plot(D_G_z_history, label='D(G(z))')
    plt.title('Discriminator Performance')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(cls_accuracy_history)
    plt.title('Classification Accuracy')
    
    plt.subplot(2, 2, 4)
    plt.plot(face_validity_history)
    plt.title('Face Validity Rate')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_progress.png'))
    plt.close()
    
    print("Training completed!")
    
    # Generate and save the final validated synthetic dataset
    print("\nGenerating the final synthetic dataset with face validation...")
    # Load the best model
    generator.load_state_dict(torch.load(os.path.join(output_dir, 'models', 'generator_best_validity.pth')))
    generator.eval()
    
    synthetic_dir = os.path.join(output_dir, 'synthetic')
    filter_and_save_synthetic_images(
        generator=generator,
        output_dir=synthetic_dir,
        samples_per_class=config.NUM_SYNTHETIC_SAMPLES,
        batch_size=32,
        confidence_threshold=0.66
    )
    
    return generator, discriminator 