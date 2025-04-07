import numpy as np
import cv2
from skimage.feature import local_binary_pattern

def compute_lbp_features(image, num_points=8, radius=1, method='uniform'):
    """
    Compute LBP features for a grayscale image
    
    Args:
        image: Input image (numpy array)
        num_points: Number of circularly symmetric neighbor points
        radius: Radius of circle
        method: LBP method ('uniform', 'default', 'ror', or 'var')
        
    Returns:
        LBP histogram features
    """
    # Ensure image is in grayscale and uint8 format
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute LBP
    lbp = local_binary_pattern(image, num_points, radius, method)
    
    # Compute histogram
    n_bins = num_points + 2 if method == 'uniform' else 2**num_points
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    
    # Normalize histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    
    return hist

def compute_lbp_image(image, num_points=8, radius=1, method='uniform'):
    """
    Compute LBP image for visualization or feature extraction
    
    Args:
        image: Input image (numpy array)
        num_points: Number of circularly symmetric neighbor points
        radius: Radius of circle
        method: LBP method
        
    Returns:
        LBP image
    """
    # Ensure image is in grayscale
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute LBP
    lbp = local_binary_pattern(image, num_points, radius, method)
    
    # Normalize to 0-255 for visualization
    lbp_image = ((lbp - lbp.min()) / (lbp.max() - lbp.min()) * 255).astype(np.uint8)
    
    return lbp_image

def texture_consistency_loss(real_images, fake_images, num_points=8, radius=1):
    """
    Compute texture consistency loss between real and fake images
    
    Args:
        real_images: Batch of real images (torch tensor)
        fake_images: Batch of fake images (torch tensor)
        
    Returns:
        Texture consistency loss value
    """
    batch_size = real_images.shape[0]
    loss = 0.0
    
    # Convert tensors to numpy arrays
    real_np = real_images.detach().cpu().numpy()
    fake_np = fake_images.detach().cpu().numpy()
    
    for i in range(batch_size):
        real_img = real_np[i, 0]  # Assuming single channel
        fake_img = fake_np[i, 0]
        
        # Scale to 0-255 and convert to uint8
        real_img = (real_img * 127.5 + 127.5).astype(np.uint8)
        fake_img = (fake_img * 127.5 + 127.5).astype(np.uint8)
        
        # Compute LBP histograms
        real_hist = compute_lbp_features(real_img, num_points, radius)
        fake_hist = compute_lbp_features(fake_img, num_points, radius)
        
        # Compute histogram distance (chi-square distance)
        chi_square_dist = np.sum((real_hist - fake_hist)**2 / (real_hist + fake_hist + 1e-10))
        loss += chi_square_dist
    
    return loss / batch_size 