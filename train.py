import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models, datasets
from sklearn.model_selection import train_test_split
from PIL import Image
import time
import os
from tqdm import tqdm

class StratifiedImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def get_dataloaders(data_dir, batch_size, image_size, num_workers, val_split=0.2):

    train_transform = transforms.Compose([
        transforms.TrivialAugmentWide(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=data_dir)
    num_classes = len(full_dataset.classes)

    class_counts = {class_name: 0 for class_name in full_dataset.classes}
    for _, label_idx in full_dataset.samples:
        class_name = full_dataset.classes[label_idx]
        class_counts[class_name] += 1

    for class_name, count in class_counts.items():
        print(f"- {class_name}: {count} images")
    print("-" * 30)

    image_paths = [item[0] for item in full_dataset.samples]
    labels = [item[1] for item in full_dataset.samples]

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, 
        labels, 
        test_size=val_split, 
        stratify=labels,
        random_state=42
    )
    
    print(f" Data loaded and split successfully.")
    print(f"Classes found: {full_dataset.classes}")
    print(f"Total images: {len(image_paths)}")
    print(f"Training images: {len(train_paths)}")
    print(f"Validation images: {len(val_paths)}")

    train_dataset = StratifiedImageDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = StratifiedImageDataset(val_paths, val_labels, transform=val_transform)

    return train_dataset, val_dataset, num_classes

def create_model(num_classes):
    model = models.efficientnet_b2(weights='IMAGENET1K_V1')
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    print(f"Using EfficientNet-B2. Final layer updated for {num_classes} classes.")
    return model

def train_epoch(model, device, loader, optimizer, loss_fn, scaler, scheduler, epoch_num, total_epochs):
    model.train()
    progress_bar = tqdm(loader, desc=f"Epoch {epoch_num}/{total_epochs}", leave=True)
    
    for data, target in progress_bar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            output = model(data)
            loss = loss_fn(output, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

def validate_epoch(model, device, loader, loss_fn):
    model.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                output = model(data)
                val_loss += loss_fn(output, target).item() * data.size(0)
            
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    val_loss /= len(loader.dataset)
    accuracy = 100. * correct / len(loader.dataset)
    print(f"Validation set: Avg loss: {val_loss:.4f}, Accuracy: {correct}/{len(loader.dataset)} ({accuracy:.2f}%)")
    return accuracy

if __name__ == '__main__':
    NUM_WORKERS = os.cpu_count()
    IMAGE_SIZE = 260
    EPOCHS = 15
    BATCH_SIZE = 64
    
    DATA_DIRECTORY = "./data_sorted/"
    SAVE_DIRECTORY = "./"
    
    MODEL_SAVE_PATH = os.path.join(SAVE_DIRECTORY, "waste_classifier.pth")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds, val_ds, num_classes = get_dataloaders(DATA_DIRECTORY, BATCH_SIZE, IMAGE_SIZE, NUM_WORKERS)
    
    train_labels = train_ds.labels
    
    class_counts = torch.bincount(torch.tensor(train_labels))
    class_weights = 1. / class_counts.float()
    
    sample_weights = torch.tensor([class_weights[t] for t in train_labels])
    
    # Create the WeightedRandomSampler
    sampler = WeightedRandomSampler(weights=sample_weights, 
                                    num_samples=len(sample_weights), 
                                    replacement=True)
                                                     
    print(" WeightedRandomSampler configured for training loader to handle class imbalance.")
    
    # Create the training DataLoader with the sampler.
    train_loader = DataLoader(train_ds, BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True, sampler=sampler)
    
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    # --- Model, Loss, and Optimizer ---
    model = create_model(num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, epochs=EPOCHS, steps_per_epoch=len(train_loader))
    
    scaler = torch.amp.GradScaler()
    best_accuracy = 0.0

    # --- Training Loop ---
    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        
        train_epoch(model, device, train_loader, optimizer, loss_fn, scaler, scheduler, epoch, EPOCHS)
        current_accuracy = validate_epoch(model, device, val_loader, loss_fn)
        
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f" New best model saved to '{MODEL_SAVE_PATH}' with accuracy: {best_accuracy:.2f}%")
            
        print(f"Time for epoch {epoch}: {time.time() - start_time:.2f} seconds\n")
            
    print(f"--- Training finished ---\n Best validation accuracy: {best_accuracy:.2f}%")