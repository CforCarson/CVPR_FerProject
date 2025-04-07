import numpy as np
import torch
from skimage.feature import local_binary_pattern
import cv2

def compute_lbp_features(image, num_points=8, radius=1, method='uniform'):
    """Compute LBP features for a grayscale image"""
    # Ensure image is in grayscale and numpy format
    if isinstance(image, torch.Tensor):
        # If tensor is in range [-1, 1], convert to [0, 255]
        if image.min() < 0:
            image = (image + 1) * 127.5
        image = image.cpu().numpy().squeeze().astype(np.uint8)
    
    # Compute LBP
    lbp = local_binary_pattern(image, num_points, radius, method)
    
    # Compute histogram
    n_bins = num_points + 2 if method == 'uniform' else 2**num_points
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    
    # Normalize histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    
    return hist

def texture_consistency_loss(real_images, fake_images, batch_size=None):
    """Compute texture consistency loss between real and fake images"""
    if batch_size is None:
        batch_size = real_images.size(0)
    
    # Initialize loss
    loss = 0.0
    
    # Process each image in the batch
    for i in range(batch_size):
        real_img = real_images[i].detach().cpu()
        fake_img = fake_images[i].detach().cpu()
        
        # Scale from [-1, 1] to [0, 255]
        real_img = ((real_img + 1) * 127.5).numpy().squeeze().astype(np.uint8)
        fake_img = ((fake_img + 1) * 127.5).numpy().squeeze().astype(np.uint8)
        
        # Compute LBP histograms
        real_hist = compute_lbp_features(real_img)
        fake_hist = compute_lbp_features(fake_img)
        
        # Compute chi-square distance between histograms
        chi_square = np.sum((real_hist - fake_hist)**2 / (real_hist + fake_hist + 1e-10))
        loss += chi_square
    
    return torch.tensor(loss / batch_size, requires_grad=True).to(real_images.device)

def create_lbp_visualization(images, save_path=None):
    """Create visualization of images and their LBP representations"""
    batch_size = min(images.size(0), 8)  # Visualize up to 8 images
    
    # Get the actual image dimensions
    img_sample = ((images[0] + 1) * 127.5).detach().cpu().numpy().squeeze()
    img_height, img_width = img_sample.shape
    
    # Create empty canvas with dynamic dimensions
    canvas = np.zeros((img_height*2, img_width*batch_size), dtype=np.uint8)
    
    for i in range(batch_size):
        # Get image and convert to numpy
        img = ((images[i] + 1) * 127.5).detach().cpu().numpy().squeeze().astype(np.uint8)
        
        # Compute LBP
        lbp = local_binary_pattern(img, 8, 1, 'uniform')
        lbp = ((lbp - lbp.min()) / (lbp.max() - lbp.min() + 1e-10) * 255).astype(np.uint8)
        
        # Place in canvas with dynamic sizing
        canvas[0:img_height, i*img_width:(i+1)*img_width] = img
        canvas[img_height:img_height*2, i*img_width:(i+1)*img_width] = lbp
    
    if save_path:
        cv2.imwrite(save_path, canvas)
        
    return canvas