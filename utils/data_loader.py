import os
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class FER2013Dataset(Dataset):
    def __init__(self, csv_file, transform=None, mode='train'):
        """
        Args:
            csv_file: Path to the FER2013 CSV file
            transform: pytorch transforms for data augmentation
            mode: 'train', 'val', or 'test'
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        
        # Filter data by usage column (Training, PublicTest, PrivateTest)
        if mode == 'train':
            self.data = self.data[self.data['Usage'] == 'Training']
        elif mode == 'val':
            self.data = self.data[self.data['Usage'] == 'PublicTest']
        elif mode == 'test':
            self.data = self.data[self.data['Usage'] == 'PrivateTest']
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        pixels = self.data.iloc[idx, 1].split()
        pixels = np.array([int(pixel) for pixel in pixels], dtype=np.uint8)
        image = pixels.reshape(48, 48)
        
        emotion = self.data.iloc[idx, 0]
        
        if self.transform:
            image = self.transform(image)
            
        return image, emotion

def get_dataloaders(csv_file, batch_size=64):
    """Create dataloaders for training, validation and testing"""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    train_dataset = FER2013Dataset(csv_file, transform=transform, mode='train')
    val_dataset = FER2013Dataset(csv_file, transform=test_transform, mode='val')
    test_dataset = FER2013Dataset(csv_file, transform=test_transform, mode='test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader

class SyntheticDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Directory with generated images
            transform: pytorch transforms
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load synthetic images and their labels
        for label in range(7):  # 7 emotion classes
            class_dir = os.path.join(root_dir, str(label))
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith('.png') or img_name.endswith('.jpg'):
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(label)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label 