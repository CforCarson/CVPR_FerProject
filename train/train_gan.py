import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.generator import TextureEnhancedGenerator
from models.discriminator import DualBranchDiscriminator
from utils.data_loader import FER2013Dataset
from utils.texture_utils import texture_consistency_loss
import config

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def save_sample_images(generator, fixed_noise, fixed_labels, epoch, output_dir):
    generator.eval()
    with torch.no_grad():
        fake = generator(fixed_noise, fixed_labels)
        # Rescale from [-1, 1] to [0, 1]
        fake = (fake + 1) / 2.0
        # Save grid of generated images
        grid = vutils.make_grid(fake, nrow=8, padding=2, normalize=False)
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'generated_epoch_{epoch}.png'))
        plt.close()
    generator.train()

def train_texPGAN(dataloader, epochs=100, lr=0.0002, beta1=0.5, beta2=0.999, 
                 lambda_cls=10.0, lambda_tex=5.0, device='cuda', output_dir='./output'):
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'samples'), exist_ok=True)
    
    # Initialize models
    netG = TextureEnhancedGenerator().to(device)
    netD = DualBranchDiscriminator().to(device)
    
    # Apply initial weights
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    # Setup optimizers
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
    
    # Loss functions
    adversarial_loss = nn.BCELoss()
    classification_loss = nn.CrossEntropyLoss()
    
    # Create fixed noise and labels for visualization
    fixed_noise = torch.randn(64, 128, device=device)
    fixed_labels = torch.tensor([i//8 for i in range(64)], device=device) 
    
    # Training loop
    print("Starting training...")
    
    G_losses = []
    D_losses = []
    
    for epoch in range(epochs):
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, (real_images, labels) in progress_bar:
            batch_size = real_images.size(0)
            
            # Transfer data to device
            real_images = real_images.to(device)
            labels = labels.to(device)
            
            # Ground truths
            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)
            
            # -----------------
            #  Train Generator
            # -----------------
            optimizerG.zero_grad()
            
            # Sample noise and labels
            z = torch.randn(batch_size, 128, device=device)
            gen_labels = labels
            
            # Generate a batch of images
            gen_images = netG(z, gen_labels)
            
            # Loss measures generator's ability to fool the discriminator
            validity, pred_label, tex_score_fake = netD(gen_images)
            g_loss = adversarial_loss(validity, valid)
            
            # Expression classification loss
            g_cls_loss = classification_loss(pred_label, gen_labels)
            
            # Texture preservation loss
            g_tex_loss = texture_consistency_loss(real_images, gen_images)
            
            # Total generator loss
            g_total_loss = g_loss + lambda_cls * g_cls_loss + lambda_tex * g_tex_loss
            g_total_loss.backward()
            optimizerG.step()
            
            # -----------------
            #  Train Discriminator
            # -----------------
            optimizerD.zero_grad()
            
            # Loss for real images
            real_pred, real_cls, tex_score_real = netD(real_images)
            d_real_loss = adversarial_loss(real_pred, valid)
            d_real_cls_loss = classification_loss(real_cls, labels)
            
            # Loss for fake images
            fake_pred, fake_cls, _ = netD(gen_images.detach())
            d_fake_loss = adversarial_loss(fake_pred, fake)
            
            # Texture consistency loss
            d_tex_loss = torch.mean(torch.abs(tex_score_fake - tex_score_real))
            
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2 + lambda_cls * d_real_cls_loss + lambda_tex * d_tex_loss
            d_loss.backward()
            optimizerD.step()
            
            # Save losses for plotting
            G_losses.append(g_total_loss.item())
            D_losses.append(d_loss.item())
            
            # Update progress bar
            progress_bar.set_description(
                f"[Epoch {epoch+1}/{epochs}] [D loss: {d_loss.item():.4f}] [G loss: {g_total_loss.item():.4f}]"
            )
        
        # Save generated samples at the end of each epoch
        save_sample_images(netG, fixed_noise, fixed_labels, epoch, os.path.join(output_dir, 'samples'))
        
        # Save models periodically
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            torch.save(netG.state_dict(), os.path.join(output_dir, 'models', f'generator_epoch_{epoch+1}.pth'))
            torch.save(netD.state_dict(), os.path.join(output_dir, 'models', f'discriminator_epoch_{epoch+1}.pth'))
            
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    plt.close()
    
    print(f"Training completed! Models and results saved to {output_dir}")
    return netG, netD

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup data loader
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    train_dataset = FER2013Dataset(config.FER2013_CSV_PATH, transform=transform, mode='train')
    dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    
    # Train the model
    train_texPGAN(
        dataloader=dataloader,
        epochs=config.EPOCHS,
        lr=config.LEARNING_RATE,
        beta1=config.BETA1,
        beta2=config.BETA2,
        lambda_cls=config.LAMBDA_CLS,
        lambda_tex=config.LAMBDA_TEX,
        device=device,
        output_dir=config.OUTPUT_DIR
    ) 