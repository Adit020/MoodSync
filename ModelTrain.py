import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import os
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Custom Dataset Class for AffectNet
class AffectNetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        AffectNet Dataset with 8 emotion categories
        Expected folder structure:
        root_dir/
        ├── neutral/
        ├── happiness/
        ├── sadness/
        ├── surprise/
        ├── fear/
        ├── disgust/
        ├── anger/
        └── contempt/
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Define emotion labels (AffectNet 8 categories)
        self.emotions = ['neutral', 'happiness', 'sadness', 'surprise', 
                        'fear', 'disgust', 'anger', 'contempt']
        self.emotion_to_idx = {emotion: idx for idx, emotion in enumerate(self.emotions)}
        
        # Load all image paths and labels
        self.samples = []
        for emotion in self.emotions:
            emotion_path = os.path.join(root_dir, emotion)
            if os.path.exists(emotion_path):
                for img_name in os.listdir(emotion_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(emotion_path, img_name)
                        self.samples.append((img_path, self.emotion_to_idx[emotion]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load and process image
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image if loading fails
            if self.transform:
                return self.transform(Image.new('RGB', (224, 224))), label
            return Image.new('RGB', (224, 224)), label

# Emotion Recognition Model
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=8, pretrained=True):
        super(EmotionCNN, self).__init__()
        
        # Use ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Replace final layer for 8 emotions
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# Training Configuration
class TrainingConfig:
    def __init__(self):
        self.batch_size = 32
        self.learning_rate = 0.001
        self.num_epochs = 50
        self.weight_decay = 1e-4
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = 8

# Data Transformations
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

# Training Function
def train_model(model, train_loader, val_loader, config):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, 
                          weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    
    train_losses = []
    val_accuracies = []
    best_acc = 0.0
    
    for epoch in range(config.num_epochs):
        # Training Phase
        model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(config.device), target.to(config.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}/{config.num_epochs}, '
                      f'Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        # Validation Phase
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(config.device), target.to(config.device)
                output = model(data)
                val_loss += criterion(output, target).item()
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        val_acc = 100 * correct / total
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch+1}/{config.num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_emotion_model.pth')
            print(f'New best model saved with accuracy: {best_acc:.2f}%')
        
        scheduler.step()
        print('-' * 60)
    
    return train_losses, val_accuracies

# Evaluation Function
def evaluate_model(model, test_loader, config):
    model.eval()
    all_predictions = []
    all_targets = []
    
    emotions = ['neutral', 'happiness', 'sadness', 'surprise', 
               'fear', 'disgust', 'anger', 'contempt']
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(config.device), target.to(config.device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(all_targets, all_predictions, target_names=emotions))
    
    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=emotions, yticklabels=emotions)
    plt.title('Emotion Recognition Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

# Debug function to check dataset structure
def check_dataset_structure(root_dir):
    """Debug function to check if dataset structure is correct"""
    print(f"\nChecking dataset structure in: {root_dir}")
    
    if not os.path.exists(root_dir):
        print(f"❌ Directory {root_dir} does not exist!")
        return False
    
    emotions = ['Neutral', 'Happy', 'Sad', 'Surprise', 
                'Fear', 'Disgust', 'Anger', 'Contempt']
    
    total_images = 0
    for emotion in emotions:
        emotion_path = os.path.join(root_dir, emotion)
        if os.path.exists(emotion_path):
            images = [f for f in os.listdir(emotion_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"✓ {emotion}: {len(images)} images")
            total_images += len(images)
        else:
            print(f"❌ {emotion} folder missing")
    
    print(f"Total images found: {total_images}")
    return total_images > 0

# Main Training Script
def main():
    # Configuration
    config = TrainingConfig()
    print(f"Using device: {config.device}")
    
    # Data paths (modify these paths according to your dataset location)
    train_dir = r'C:\Project\Final Mood\Back\AffectNet\train'  # Path to training data
    val_dir = r'C:\Project\Final Mood\Back\AffectNet\val'      # Path to validation data
    test_dir = r'C:\Project\Final Mood\Back\AffectNet\test'    # Path to test data
    
    # Debug: Check if your data structure is correct
    print("=" * 60)
    print("DATASET STRUCTURE CHECK")
    print("=" * 60)
    
    train_exists = check_dataset_structure(train_dir)
    val_exists = check_dataset_structure(val_dir)
    test_exists = check_dataset_structure(test_dir)
    
    if not train_exists:
        print("\n❌ TRAINING DATA NOT FOUND!")
        print("Please ensure your data is organized as follows:")
        print("data/")
        print("├── train/")
        print("│   ├── neutral/")
        print("│   ├── happiness/")
        print("│   ├── sadness/")
        print("│   ├── surprise/")
        print("│   ├── fear/")
        print("│   ├── disgust/")
        print("│   ├── anger/")
        print("│   └── contempt/")
        print("├── val/")
        print("│   └── (same structure)")
        print("└── test/")
        print("    └── (same structure)")
        print("\nCurrent working directory:", os.getcwd())
        return
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    train_dataset = AffectNetDataset(train_dir, transform=train_transform)
    val_dataset = AffectNetDataset(val_dir, transform=val_transform)
    test_dataset = AffectNetDataset(test_dir, transform=val_transform)
    
    print(f"\nDataset Summary:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    # Only proceed if we have training data
    if len(train_dataset) == 0:
        print("❌ No training data found. Please check your dataset structure.")
        return
    
    # Create data loaders only if datasets have samples
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                             shuffle=True, num_workers=4)
    
    if len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
                               shuffle=False, num_workers=4)
    else:
        print("⚠️  No validation data found. Using training data for validation.")
        val_loader = train_loader
    
    if len(test_dataset) > 0:
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, 
                                shuffle=False, num_workers=4)
    else:
        print("⚠️  No test data found. Will skip final evaluation.")
        test_loader = None
    
    # Initialize model
    model = EmotionCNN(num_classes=config.num_classes, pretrained=True)
    model = model.to(config.device)
    
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    print("Starting training...")
    train_losses, val_accuracies = train_model(model, train_loader, val_loader, config)
    
    # Load best model for evaluation
    model.load_state_dict(torch.load('best_emotion_model.pth'))
    print("Loaded best model for evaluation")
    
    # Evaluate on test set (if available)
    if test_loader is not None:
        print("Evaluating on test set...")
        evaluate_model(model, test_loader, config)
    else:
        print("No test data available for final evaluation.")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='orange')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

# Inference Function for MoodSync
def predict_emotion(model, image_path, transform, device):
    """
    Predict emotion from a single image
    """
    emotions = ['neutral', 'happiness', 'sadness', 'surprise', 
               'fear', 'disgust', 'anger', 'contempt']
    
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_idx].item()
    
    return emotions[predicted_idx], confidence

if __name__ == "__main__":
    main()