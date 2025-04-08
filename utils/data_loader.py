import os
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

class FER2013FolderDataset(Dataset):
    """Dataset loader for the pre-organized FER2013 folder structure"""
    def __init__(self, root_dir, mode='train', transform=None):
        """
        Args:
            root_dir: Path to the FER2013 folder
            mode: 'train' or 'test'
            transform: Optional transform to be applied on images
        """
        self.root_dir = os.path.join(root_dir, mode)
        self.transform = transform
        self.classes = sorted(os.listdir(self.root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for target_class in self.classes:
            class_dir = os.path.join(self.root_dir, target_class)
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(class_dir, img_name), self.class_to_idx[target_class]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        if self.transform:
            image = self.transform(image)
            
        return image, target

class SyntheticDataset(Dataset):
    """Dataset for generated synthetic images"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # Assuming same class structure as FER2013
        self.classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        for target_class in self.classes:
            class_dir = os.path.join(root_dir, target_class)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append((os.path.join(class_dir, img_name), self.class_to_idx[target_class]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        if self.transform:
            image = self.transform(image)
            
        return image, target

class AugmentedFER2013Dataset(Dataset):
    """Extended dataset with additional augmentations to increase effective training set size"""
    def __init__(self, root_dir, mode='train', transform=None, num_augmentations=3):
        self.original_dataset = FER2013FolderDataset(root_dir, mode, transform)
        self.transform = transform
        self.num_augmentations = num_augmentations
        
        # Additional augmentation transforms
        self.aug_transforms = [
            transforms.Compose([
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            ]),
            transforms.Compose([
                transforms.RandomAffine(degrees=(-15, 15), translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(p=0.5),
            ]),
            transforms.Compose([
                transforms.RandomRotation(20),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            ])
        ]
    
    def __len__(self):
        return len(self.original_dataset) * (self.num_augmentations + 1)
    
    def __getitem__(self, idx):
        orig_idx = idx % len(self.original_dataset)
        aug_idx = idx // len(self.original_dataset)
        
        image, target = self.original_dataset.samples[orig_idx]
        image = Image.open(image).convert('L')
        
        if aug_idx == 0:
            # Return the original image with standard transforms
            if self.transform:
                image = self.transform(image)
        else:
            # Apply additional augmentation
            aug_idx = (aug_idx - 1) % len(self.aug_transforms)
            image = self.aug_transforms[aug_idx](image)
            if self.transform:
                image = self.transform(image)
        
        return image, target

def get_dataloaders(data_dir, batch_size=64, augment=False):
    """Create dataloaders for training and testing"""
    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create datasets
    if augment:
        # Use augmented dataset for more training samples
        train_dataset = AugmentedFER2013Dataset(data_dir, mode='train', transform=train_transform)
    else:
        train_dataset = FER2013FolderDataset(data_dir, mode='train', transform=train_transform)
        
    test_dataset = FER2013FolderDataset(data_dir, mode='test', transform=test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader

def apply_clahe(image_tensor, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance texture"""
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
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_img = clahe.apply(img_np)
    
    # Convert back to tensor if needed
    if isinstance(image_tensor, torch.Tensor):
        enhanced_tensor = torch.from_numpy(enhanced_img).float() / 255.0
        if image_tensor.min() < 0:  # If normalized to [-1, 1]
            enhanced_tensor = enhanced_tensor * 2 - 1
        
        # Restore original shape
        if len(image_tensor.shape) == 3 and image_tensor.shape[0] == 1:
            enhanced_tensor = enhanced_tensor.unsqueeze(0)
        
        return enhanced_tensor.to(image_tensor.device)
    
    return enhanced_img

def balanced_sampling_dataloader(dataset, batch_size=64):
    """Create a dataloader with balanced sampling across expression classes"""
    # Organize samples by class
    samples_by_class = {}
    for idx, (_, label) in enumerate(dataset.samples):
        if label not in samples_by_class:
            samples_by_class[label] = []
        samples_by_class[label].append(idx)
    
    class_weights = {label: 1.0/len(indices) for label, indices in samples_by_class.items()}
    sample_weights = [class_weights[dataset.samples[idx][1]] for idx in range(len(dataset))]
    
    # Create sampler with replacement to handle imbalanced classes
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )
    
    # Create balanced dataloader
    balanced_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4
    )
    
    return balanced_loader

def save_synthetic_images(generator, num_samples=1000, output_dir='./output/synthetic'):
    """Generate and save synthetic images for later use"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create class directories
    classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    for cls in classes:
        os.makedirs(os.path.join(output_dir, cls), exist_ok=True)
    
    generator.eval()
    with torch.no_grad():
        # Generate samples per class with progress tracking
        print(f"Generating {num_samples} synthetic images per class...")
        for class_idx, class_name in enumerate(classes):
            for i in range(num_samples):
                # Create random noise for diverse samples
                z = torch.randn(1, 128).to(generator.device)
                labels = torch.tensor([class_idx]).to(generator.device)
                
                # Generate image
                fake_img = generator(z, labels)
                
                # Convert to PIL image and save
                img = transforms.ToPILImage()((fake_img[0] + 1) / 2.0)  # Denormalize from [-1,1] to [0,1]
                
                # Apply CLAHE to enhance textures before saving (optional)
                # This helps ensure texture details are well-preserved
                img_np = np.array(img)
                enhanced_img = apply_clahe(img_np)
                img = Image.fromarray(enhanced_img)
                
                img.save(os.path.join(output_dir, class_name, f'synthetic_{i}_{class_name}.png'))
                
            print(f"Generated {num_samples} images for {class_name}")
    
    print(f"Successfully generated {num_samples * len(classes)} synthetic images.")