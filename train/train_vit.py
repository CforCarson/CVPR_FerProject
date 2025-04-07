import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import sys
sys.path.append('..')

from models.vit import ExpressionViT
from utils.data_loader import FER2013FolderDataset, SyntheticDataset
import config

def train_vit(model, train_loader, test_loader, epochs=50, lr=0.0001, 
              model_name="vit_model", output_dir='./output'):
    """Train the Vision Transformer model"""
    # Create output directory
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training tracking
    best_acc = 0.0
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    
    print(f"Starting training for {epochs} epochs...")
    
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
            
            # Track statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            progress_bar.set_description(
                f"[Epoch {epoch+1}/{epochs}] [Train Loss: {loss.item():.4f}]"
            )
        
        # Calculate training metrics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100. * train_correct / train_total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        
        # Evaluation
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Track statistics
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        # Calculate test metrics
        epoch_test_loss = test_loss / len(test_loader)
        epoch_test_acc = 100. * test_correct / test_total
        test_losses.append(epoch_test_loss)
        test_accs.append(epoch_test_acc)
        
        # Update learning rate
        scheduler.step(epoch_test_loss)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%, "
              f"Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_acc:.2f}%")
        
        # Save best model
        if epoch_test_acc > best_acc:
            best_acc = epoch_test_acc
            torch.save(model.state_dict(), os.path.join(output_dir, 'models', f'{model_name}_best.pth'))
            print(f"Saved best model with test accuracy: {best_acc:.2f}%")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(test_losses, label='Test')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(test_accs, label='Test')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_training_curves.png'))
    plt.close()
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, 'models', f'{model_name}_final.pth'))
    
    return model, best_acc

def evaluate_model(model, test_loader, output_dir='./output', model_name="vit_model"):
    """Evaluate model performance with detailed metrics"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # For storing predictions and labels
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
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
    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names,
               yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, f'{model_name}_confusion_matrix.png'))
    plt.close()
    
    # Per-class metrics
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        true_pos = np.sum((all_preds == i) & (all_labels == i))
        false_pos = np.sum((all_preds == i) & (all_labels != i))
        false_neg = np.sum((all_preds != i) & (all_labels == i))
        
        precision = true_pos / (true_pos + false_pos + 1e-10)
        recall = true_pos / (true_pos + false_neg + 1e-10)
        class_f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        per_class_metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': class_f1
        }
    
    # Print results
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    print(f"Average AUROC: {avg_auroc:.4f}")
    print("\nPer-class metrics:")
    for class_name, metrics in per_class_metrics.items():
        print(f"{class_name}: Precision: {metrics['precision']:.4f}, "
              f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'auroc': avg_auroc,
        'confusion_matrix': cm,
        'per_class_metrics': per_class_metrics
    }

def run_comparative_experiments(output_dir='./output'):
    """Run experiments with real, synthetic, and mixed datasets"""
    from torchvision import transforms
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create transform for all datasets
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])
    
    # Create datasets
    real_dataset = FER2013FolderDataset(root_dir=config.FER2013_DIR, mode='train', transform=transform)
    synthetic_dataset = SyntheticDataset(root_dir=os.path.join(output_dir, 'synthetic'), transform=transform)
    test_dataset = FER2013FolderDataset(root_dir=config.FER2013_DIR, mode='test', transform=transform)
    
    # Create mixed dataset
    mixed_dataset = ConcatDataset([real_dataset, synthetic_dataset])
    
    # Create dataloaders
    real_loader = DataLoader(real_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    synthetic_loader = DataLoader(synthetic_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    mixed_loader = DataLoader(mixed_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Store results
    results = {}
    
    # Train on real data
    print("\n===== Training on Real Data =====")
    real_model = ExpressionViT().to(device)
    real_model, real_acc = train_vit(
        real_model, real_loader, test_loader, 
        epochs=config.VIT_EPOCHS, lr=config.VIT_LR,
        model_name="real_vit", output_dir=output_dir
    )
    real_model.load_state_dict(torch.load(os.path.join(output_dir, 'models', 'real_vit_best.pth')))
    results['real'] = evaluate_model(real_model, test_loader, output_dir, "real_vit")
    
    # Train on synthetic data
    print("\n===== Training on Synthetic Data =====")
    synthetic_model = ExpressionViT().to(device)
    synthetic_model, synth_acc = train_vit(
        synthetic_model, synthetic_loader, test_loader,
        epochs=config.VIT_EPOCHS, lr=config.VIT_LR,
        model_name="synthetic_vit", output_dir=output_dir
    )
    synthetic_model.load_state_dict(torch.load(os.path.join(output_dir, 'models', 'synthetic_vit_best.pth')))
    results['synthetic'] = evaluate_model(synthetic_model, test_loader, output_dir, "synthetic_vit")
    
    # Train on mixed data
    print("\n===== Training on Mixed Data =====")
    mixed_model = ExpressionViT().to(device)
    mixed_model, mixed_acc = train_vit(
        mixed_model, mixed_loader, test_loader,
        epochs=config.VIT_EPOCHS, lr=config.VIT_LR,
        model_name="mixed_vit", output_dir=output_dir
    )
    mixed_model.load_state_dict(torch.load(os.path.join(output_dir, 'models', 'mixed_vit_best.pth')))
    results['mixed'] = evaluate_model(mixed_model, test_loader, output_dir, "mixed_vit")
    
    # Comparative summary
    print("\n===== Comparative Results =====")
    print(f"{'Dataset':<10} {'Accuracy':<10} {'F1 Score':<10} {'AUROC':<10}")
    print("-" * 45)
    for dataset, metrics in results.items():
        print(f"{dataset:<10} {metrics['accuracy']:.4f}      {metrics['f1_score']:.4f}      {metrics['auroc']:.4f}")
    
    # Create comparative visualization
    plt.figure(figsize=(12, 6))
    datasets = list(results.keys())
    accuracies = [results[d]['accuracy'] for d in datasets]
    f1_scores = [results[d]['f1_score'] for d in datasets]
    aurocs = [results[d]['auroc'] for d in datasets]
    
    x = np.arange(len(datasets))
    width = 0.25
    
    plt.bar(x - width, accuracies, width, label='Accuracy')
    plt.bar(x, f1_scores, width, label='F1 Score')
    plt.bar(x + width, aurocs, width, label='AUROC')
    
    plt.xlabel('Training Dataset')
    plt.ylabel('Score')
    plt.title('Comparison of Model Performance')
    plt.xticks(x, datasets)
    plt.legend()
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparative_results.png'))
    plt.close()
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate ViT")
    parser.add_argument("--epochs", type=int, default=config.VIT_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.VIT_LR, help="Learning rate")
    args = parser.parse_args()
    
    run_comparative_experiments(output_dir=config.OUTPUT_DIR)