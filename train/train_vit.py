import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns

from models.vit import ExpressionViT
from utils.data_loader import FER2013Dataset, SyntheticDataset
import config

def train_vit(model, train_loader, val_loader, epochs=50, lr=0.0001, device='cuda', 
              model_save_path='./vit_model.pth'):
    """Train the Vision Transformer model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (images, labels) in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track loss and accuracy
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            progress_bar.set_description(
                f"[Epoch {epoch+1}/{epochs}] [Train Loss: {loss.item():.4f}]"
            )
        
        # Calculate average training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100. * train_correct / train_total
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate average validation metrics
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100. * val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved with val_loss: {best_val_loss:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('./training_curves.png')
    plt.close()
    
    return model

def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate the model on test data"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Calculate F1 score for each class and average
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Calculate AUROC (one-vs-rest for multiclass)
    auroc_scores = []
    for i in range(7):  # 7 expression classes
        binary_labels = (all_labels == i).astype(int)
        class_probs = all_probs[:, i]
        try:
            auroc = roc_auc_score(binary_labels, class_probs)
            auroc_scores.append(auroc)
        except ValueError:
            # In case a class is not present in the test set
            pass
    
    avg_auroc = np.mean(auroc_scores)
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'],
               yticklabels=['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('./confusion_matrix.png')
    plt.close()
    
    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    print(f"Average AUROC: {avg_auroc:.4f}")
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'auroc': avg_auroc,
        'confusion_matrix': cm
    }

def run_comparative_experiments(device='cuda'):
    """Run comparative experiments with real, synthetic, and mixed data"""
    # Setup dataloaders
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Real data
    real_train_dataset = FER2013Dataset(config.FER2013_CSV_PATH, transform=transform, mode='train')
    real_train_loader = DataLoader(real_train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    
    # Synthetic data
    synthetic_dataset = SyntheticDataset(config.SYNTHETIC_DATA_DIR, transform=transform)
    synthetic_loader = DataLoader(synthetic_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    
    # Mixed data
    mixed_dataset = ConcatDataset([real_train_dataset, synthetic_dataset])
    mixed_loader = DataLoader(mixed_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    
    # Test data
    test_dataset = FER2013Dataset(config.FER2013_CSV_PATH, transform=transform, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Results dictionary
    results = {}
    
    # Train and evaluate on real data
    print("\n===== Training on Real Data =====")
    real_model = ExpressionViT().to(device)
    real_model = train_vit(real_model, real_train_loader, test_loader, 
                          epochs=config.VIT_EPOCHS, lr=config.VIT_LR, 
                          model_save_path='./real_vit_model.pth')
    real_model.load_state_dict(torch.load('./real_vit_model.pth'))
    results['real'] = evaluate_model(real_model, test_loader, device)
    
    # Train and evaluate on synthetic data
    print("\n===== Training on Synthetic Data =====")
    synthetic_model = ExpressionViT().to(device)
    synthetic_model = train_vit(synthetic_model, synthetic_loader, test_loader,
                               epochs=config.VIT_EPOCHS, lr=config.VIT_LR,
                               model_save_path='./synthetic_vit_model.pth')
    synthetic_model.load_state_dict(torch.load('./synthetic_vit_model.pth'))
    results['synthetic'] = evaluate_model(synthetic_model, test_loader, device)
    
    # Train and evaluate on mixed data
    print("\n===== Training on Mixed Data =====")
    mixed_model = ExpressionViT().to(device)
    mixed_model = train_vit(mixed_model, mixed_loader, test_loader,
                           epochs=config.VIT_EPOCHS, lr=config.VIT_LR,
                           model_save_path='./mixed_vit_model.pth')
    mixed_model.load_state_dict(torch.load('./mixed_vit_model.pth'))
    results['mixed'] = evaluate_model(mixed_model, test_loader, device)
    
    # Print comparative results
    print("\n===== Comparative Results =====")
    print(f"{'Dataset':<10} {'Accuracy':<10} {'F1 Score':<10} {'AUROC':<10}")
    print("-" * 42)
    for dataset, metrics in results.items():
        print(f"{dataset:<10} {metrics['accuracy']:<10.4f} {metrics['f1_score']:<10.4f} {metrics['auroc']:<10.4f}")
    
    return results

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Run comparative experiments
    run_comparative_experiments(device) 