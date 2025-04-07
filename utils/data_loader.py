import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

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

def get_dataloaders(data_dir, batch_size=64):
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
    train_dataset = FER2013FolderDataset(data_dir, mode='train', transform=train_transform)
    test_dataset = FER2013FolderDataset(data_dir, mode='test', transform=test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader

def save_synthetic_images(generator, num_samples=1000, output_dir='./output/synthetic'):
    """Generate and save synthetic images for later use"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create class directories
    classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    for cls in classes:
        os.makedirs(os.path.join(output_dir, cls), exist_ok=True)
    
    generator.eval()
    with torch.no_grad():
        for i in range(num_samples):
            # Generate roughly equal number of samples per class
            for class_idx, class_name in enumerate(classes):
                z = torch.randn(1, 128).to(generator.device)
                labels = torch.tensor([class_idx]).to(generator.device)
                
                # Generate image
                fake_img = generator(z, labels)
                
                # Convert to PIL image and save
                img = transforms.ToPILImage()((fake_img[0] + 1) / 2.0)  # Denormalize from [-1,1] to [0,1]
                img.save(os.path.join(output_dir, class_name, f'synthetic_{i}_{class_name}.png'))