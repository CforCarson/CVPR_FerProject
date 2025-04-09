import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
from tqdm import tqdm
import sys
import shutil

# Track if we've already displayed import warnings
face_recognition_warning_shown = False
yolo_warning_shown = False

def has_face_opencv(image_tensor, min_size=30):
    """Detect faces using OpenCV's built-in Haar cascade"""
    # Convert tensor to numpy array
    if isinstance(image_tensor, torch.Tensor):
        if image_tensor.min() < 0:  # If normalized to [-1, 1]
            img_np = ((image_tensor.detach() + 1) * 127.5).cpu().numpy().astype(np.uint8)
        else:  # If in [0, 1]
            img_np = (image_tensor.detach() * 255).cpu().numpy().astype(np.uint8)
        
        if len(img_np.shape) == 3 and img_np.shape[0] == 1:  # CHW format
            img_np = img_np.squeeze(0)  # Remove channel dim for grayscale
    else:
        img_np = image_tensor
    
    # Load pre-trained classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(img_np, 1.1, 4, minSize=(min_size, min_size))
    
    return len(faces) > 0

def has_face_dlib(image_tensor):
    """Detect faces using dlib (via face_recognition)"""
    global face_recognition_warning_shown
    
    try:
        import face_recognition
    except ImportError:
        if not face_recognition_warning_shown:
            print("Warning: face_recognition module not installed. Using OpenCV fallback.")
            print("To install: pip install face_recognition")
            face_recognition_warning_shown = True
        return has_face_opencv(image_tensor)
    
    # Convert tensor to numpy array
    if isinstance(image_tensor, torch.Tensor):
        if image_tensor.min() < 0:  # If normalized to [-1, 1] 
            img_np = ((image_tensor.detach() + 1) * 127.5).cpu().numpy().astype(np.uint8)
        else:  # If in [0, 1]
            img_np = (image_tensor.detach() * 255).cpu().numpy().astype(np.uint8)
        
        if len(img_np.shape) == 3 and img_np.shape[0] == 1:  # CHW format
            img_np = np.repeat(img_np.squeeze(0)[..., np.newaxis], 3, axis=2)  # Convert to RGB
        elif len(img_np.shape) == 3 and img_np.shape[0] == 3:  # CHW format with 3 channels
            img_np = np.transpose(img_np, (1, 2, 0))  # CHW -> HWC
    else:
        img_np = image_tensor
        
    # Detect faces
    face_locations = face_recognition.face_locations(img_np)
    
    return len(face_locations) > 0

def has_face_yolo(image_tensor):
    """Detect faces using YOLOv8 model"""
    global yolo_warning_shown
    
    try:
        from ultralytics import YOLO
        from huggingface_hub import hf_hub_download
    except ImportError:
        if not yolo_warning_shown:
            print("Warning: YOLO dependencies not installed. Using OpenCV fallback.")
            print("To install: pip install ultralytics huggingface_hub")
            yolo_warning_shown = True
        return has_face_opencv(image_tensor)
    
    # Convert tensor to PIL image for YOLO
    if isinstance(image_tensor, torch.Tensor):
        if image_tensor.min() < 0:  # If normalized to [-1, 1]
            img_tensor = (image_tensor.detach() + 1) / 2.0  # Normalize to [0, 1]
        else:
            img_tensor = image_tensor.detach()
            
        # Convert grayscale to RGB if needed
        if img_tensor.shape[0] == 1:
            img_tensor = img_tensor.repeat(3, 1, 1)
            
        # Convert to PIL
        to_pil = transforms.ToPILImage()
        image = to_pil(img_tensor.cpu())
    else:
        # Assuming it's already a PIL image or path
        image = image_tensor
    
    # Initialize model (will download if not available)
    try:
        model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return has_face_opencv(image_tensor)
    
    # Run inference
    results = model(image, verbose=False)
    
    # Check if any face detected
    return len(results[0].boxes) > 0

def validate_batch(batch_images, confidence_threshold=0.66):
    """
    Validate a batch of generated face images using multiple detection methods.
    Returns a mask of valid faces and confidence scores.
    
    Args:
        batch_images: Tensor of images [B, C, H, W]
        confidence_threshold: Minimum proportion of detection methods that must detect a face
    
    Returns:
        valid_mask: Boolean tensor of valid images [B]
        confidence: Confidence scores [B]
    """
    batch_size = batch_images.size(0)
    valid_mask = torch.zeros(batch_size, dtype=torch.bool)
    confidence = torch.zeros(batch_size, dtype=torch.float)
    
    for i in range(batch_size):
        img = batch_images[i]
        
        # Try all three detection methods
        methods_passed = 0
        total_methods = 0
        
        # OpenCV (always available)
        total_methods += 1
        if has_face_opencv(img):
            methods_passed += 1
            
        # Face recognition (if available)
        try:
            import face_recognition
            total_methods += 1
            if has_face_dlib(img):
                methods_passed += 1
        except ImportError:
            pass
        
        # YOLO (if available)
        try:
            from ultralytics import YOLO
            total_methods += 1
            if has_face_yolo(img):
                methods_passed += 1
        except ImportError:
            pass
        
        # Calculate confidence
        conf = methods_passed / total_methods
        confidence[i] = conf
        
        # Mark as valid if confidence exceeds threshold
        if conf >= confidence_threshold:
            valid_mask[i] = True
    
    return valid_mask, confidence

def filter_and_save_synthetic_images(generator, output_dir, samples_per_class=1000, batch_size=32, 
                                   confidence_threshold=0.66, classes=None):
    """
    Generate synthetic images, validate them using face detection,
    and save only the valid ones.
    
    Args:
        generator: The trained generator model
        output_dir: Directory to save validated images
        samples_per_class: Number of valid samples to generate per class
        batch_size: Batch size for generation
        confidence_threshold: Minimum confidence for a face to be considered valid
        classes: List of class names (defaults to FER emotions)
    """
    if classes is None:
        classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    device = generator.device
    latent_dim = generator.latent_dim if hasattr(generator, 'latent_dim') else 128
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    for cls in classes:
        cls_dir = os.path.join(output_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
    
    # Track generated count
    generated_counts = {cls: 0 for cls in classes}
    
    # Configure image conversion
    to_pil = transforms.ToPILImage()
    
    generator.eval()
    with torch.no_grad():
        for class_idx, class_name in enumerate(classes):
            pbar = tqdm(total=samples_per_class, desc=f"Generating {class_name}")
            
            # Track validity rate for reporting
            total_generated = 0
            total_valid = 0
            
            while generated_counts[class_name] < samples_per_class:
                # Generate a batch
                batch_size_adjusted = min(batch_size, samples_per_class - generated_counts[class_name])
                z = torch.randn(batch_size_adjusted, latent_dim, device=device)
                labels = torch.full((batch_size_adjusted,), class_idx, device=device)
                
                # Generate images
                fake_images = generator(z, labels)
                total_generated += batch_size_adjusted
                
                # Validate faces
                valid_mask, confidence = validate_batch(fake_images, confidence_threshold)
                valid_images = fake_images[valid_mask]
                valid_confidences = confidence[valid_mask]
                
                # Update validity stats
                batch_valid = valid_mask.sum().item()
                total_valid += batch_valid
                
                # Save valid images
                for i in range(batch_valid):
                    if generated_counts[class_name] >= samples_per_class:
                        break
                        
                    img = valid_images[i]
                    conf = valid_confidences[i].item()
                    
                    # Denormalize from [-1, 1] to [0, 1]
                    img = (img + 1) / 2.0
                    
                    # Convert to PIL and save
                    pil_img = to_pil(img.cpu())
                    img_path = os.path.join(output_dir, class_name, 
                                          f"{class_name}_{generated_counts[class_name]:05d}_conf{conf:.2f}.png")
                    pil_img.save(img_path)
                    
                    generated_counts[class_name] += 1
                    pbar.update(1)
            
            pbar.close()
            
            # Report validity rate
            validity_rate = total_valid / max(1, total_generated) * 100
            print(f"Class {class_name}: Generated {samples_per_class} valid images "
                 f"(validity rate: {validity_rate:.1f}%)")
    
    print(f"Successfully generated and validated {sum(generated_counts.values())} images across {len(classes)} classes")
    return generated_counts 