import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, save_path):
    """Plot training and validation curves"""
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'],
                yticklabels=['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()

def save_generated_images(generator, fixed_noise, fixed_labels, epoch, output_dir):
    """Save a grid of generated images"""
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

def plot_texture_comparison(real_image, fake_image, save_path):
    """Plot real and fake images with their LBP features"""
    plt.figure(figsize=(15, 5))
    
    # Plot original images
    plt.subplot(1, 3, 1)
    plt.imshow(real_image, cmap='gray')
    plt.title('Real Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(fake_image, cmap='gray')
    plt.title('Fake Image')
    plt.axis('off')
    
    # Plot LBP features
    from .texture_utils import compute_lbp_image
    
    real_lbp = compute_lbp_image(real_image)
    fake_lbp = compute_lbp_image(fake_image)
    
    plt.subplot(1, 3, 3)
    plt.imshow(np.abs(real_lbp - fake_lbp), cmap='hot')
    plt.title('LBP Difference')
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close() 