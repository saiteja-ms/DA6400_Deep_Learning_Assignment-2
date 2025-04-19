# partB/finetune_resnet.py

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.data import Dataset
import wandb # Optional: For logging experiments
from tqdm import tqdm # Import tqdm for progress bars
import numpy as np # Needed for random seed setting

# --- Helper Class for Applying Transforms to Subsets ---
class TransformedSubset(Dataset):
    """
    A Dataset wrapper that applies a transform to a Subset.
    Necessary because Subsets themselves don't handle transforms.
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        # Ensure input is a PIL Image before applying transforms if needed
        # (ImageFolder usually returns PIL Images, so this is generally fine)
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune ResNet50 on iNaturalist subset')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing the iNaturalist dataset (train/test subfolders)')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of classes in the dataset')
    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training and evaluation')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for the optimizer')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Fraction of training data to use for validation')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save the best model weights')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='inaturalist-finetune-resnet50',
                        help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='WandB entity (username or team name)') # Optional: if needed

    return parser.parse_args()

# --- Main Fine-tuning Function ---
def main(args):
    # --- Reproducibility ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        # Performance can suffer slightly with deterministic settings
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        # Set benchmark True for potential speedup if input sizes don't vary
        torch.backends.cudnn.benchmark = True

    # --- Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Initialize WandB (Optional) ---
    if args.use_wandb:
        try:
            wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args)
            print("WandB initialized.")
        except Exception as e:
            print(f"Could not initialize WandB: {e}. Continuing without WandB logging.")
            args.use_wandb = False # Disable wandb if initialization fails

    # --- Create Output Directory ---
    os.makedirs(args.output_dir, exist_ok=True)
    best_model_path = os.path.join(args.output_dir, 'best_finetuned_resnet50.pth')

    # --- Load Model (ResNet50) ---
    print("Loading pre-trained ResNet50 model...")
    # Use the recommended weights parameter
    weights = models.ResNet50_Weights.IMAGENET1K_V1 # Or IMAGENET1K_V2 for potentially better weights
    model = models.resnet50(weights=weights)

    # --- Freeze Pre-trained Layers ---
    print("Freezing pre-trained layers...")
    for param in model.parameters():
        param.requires_grad = False

    # --- Replace Final Classifier Layer ---
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, args.num_classes)
    print(f"Replaced final layer for {args.num_classes} classes. New layer requires grad: {model.fc.weight.requires_grad}")

    model = model.to(device)
    if args.use_wandb:
        wandb.watch(model, log='gradients', log_freq=100) # Watch model gradients

    # --- Data Preprocessing ---
    # Get transforms recommended by the pre-trained model weights - use these for validation/test
    # val_test_transform = weights.transforms() # This provides a standard eval transform pipeline

    # Define explicit transforms for clarity and potential customization
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), # Slightly less aggressive crop
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10), # Add slight rotation
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1), # Add color jitter
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- Load Data ---
    print("Loading dataset...")
    train_val_dataset_path = os.path.join(args.data_dir, 'train')
    test_dataset_path = os.path.join(args.data_dir, 'test')

    if not os.path.isdir(train_val_dataset_path) or not os.path.isdir(test_dataset_path):
        raise FileNotFoundError(f"Dataset not found at {args.data_dir}. Expected 'train' and 'test' subfolders.")

    # Load the full training dataset *without* transforms initially to allow splitting
    # We pass a minimal transform (ToTensor) here to make splitting work easily,
    # the actual train/val transforms will be applied by the TransformedSubset wrapper.
    # Alternatively, load with no transform and ensure TransformedSubset handles PIL images.
    full_train_dataset = ImageFolder(root=train_val_dataset_path) # Loads PIL images

    # Split train into train/validation sets
    num_train_val = len(full_train_dataset)
    num_val = int(num_train_val * args.val_split)
    num_train = num_train_val - num_val

    # Use random_split to get Subset objects
    # Consider using StratifiedShuffleSplit from sklearn for better class balance if needed
    train_subset, val_subset = random_split(full_train_dataset, [num_train, num_val],
                                           generator=torch.Generator().manual_seed(args.seed)) # Ensure split is reproducible

    # Apply appropriate transforms using the wrapper class
    train_dataset = TransformedSubset(train_subset, transform=train_transform)
    val_dataset = TransformedSubset(val_subset, transform=val_test_transform)

    # Load test dataset with test transforms
    test_dataset = ImageFolder(root=test_dataset_path, transform=val_test_transform)

    print(f"Dataset sizes: Train={len(train_dataset)}, Validation={len(val_dataset)}, Test={len(test_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # --- Loss Function and Optimizer ---
    criterion = nn.CrossEntropyLoss()

    # Optimize only the parameters of the final layer (the ones requiring gradients)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    # Optional: Use a learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)

    print("Optimizer configured to train only the final layer.")
    print(f"Using LR scheduler: {scheduler.__class__.__name__}")

    # --- Training and Validation Loop ---
    best_val_acc = 0.0
    start_time = time.time()

    print("\nStarting Training...")
    for epoch in range(args.epochs):
        epoch_start_time = time.time()

        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Wrap train_loader with tqdm for a progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=False)
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # Update tqdm description with current batch loss
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        # Wrap val_loader with tqdm for a progress bar
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", leave=False)
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                # Update tqdm description with current batch loss (optional for val)
                val_pbar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val

        epoch_duration = time.time() - epoch_start_time
        print(f'Epoch {epoch+1}/{args.epochs} [{epoch_duration:.2f}s] -> '
              f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | '
              f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}')

        # Step the scheduler based on validation accuracy
        scheduler.step(epoch_val_acc)
        current_lr = optimizer.param_groups[0]['lr'] # Get current LR

        # --- Log Metrics (Optional: WandB) ---
        if args.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": epoch_train_loss,
                "train_accuracy": epoch_train_acc,
                "val_loss": epoch_val_loss,
                "val_accuracy": epoch_val_acc,
                "learning_rate": current_lr,
                "epoch_duration_sec": epoch_duration
            })

        # --- Save Best Model ---
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"    -> New best model saved to {best_model_path} (Val Acc: {best_val_acc:.4f})")
            if args.use_wandb:
                 # Save as summary metric in WandB for easy access
                 wandb.summary["best_val_accuracy"] = best_val_acc
                 wandb.summary["best_epoch"] = epoch + 1

    total_training_time = time.time() - start_time
    print(f"\nTraining Finished. Total time: {total_training_time/60:.2f} minutes")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")

    # --- Final Test Evaluation ---
    print("\nEvaluating on Test Set using the best model...")
    # Load best model weights
    try:
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model weights from {best_model_path}")
    except FileNotFoundError:
        print(f"Warning: Best model file not found at {best_model_path}. Evaluating with the last epoch model.")
        # Optional: Add logic here if you want to explicitly handle this case differently

    model.eval()
    correct_test = 0
    total_test = 0
    test_loss = 0.0

    # Wrap test_loader with tqdm
    test_pbar = tqdm(test_loader, desc="Testing", leave=False)
    with torch.no_grad():
        for inputs, labels in test_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    final_test_loss = test_loss / total_test
    final_test_acc = correct_test / total_test
    print(f'Final Test Loss: {final_test_loss:.4f}, Final Test Accuracy: {final_test_acc:.4f}')

    # --- Log Final Test Metrics (Optional: WandB) ---
    if args.use_wandb:
        # Update summary metrics which appear prominently in the WandB UI
        wandb.summary["final_test_loss"] = final_test_loss
        wandb.summary["final_test_accuracy"] = final_test_acc
        wandb.summary["total_training_time_min"] = total_training_time / 60

        # Log the best model file as an artifact
        try:
            model_artifact_name = f'resnet50-finetuned-feature-extractor-{wandb.run.id}'
            artifact = wandb.Artifact(model_artifact_name, type='model',
                                    description="ResNet50 fine-tuned (feature extractor) on iNaturalist subset",
                                    metadata={"best_val_acc": best_val_acc,
                                              "test_acc": final_test_acc,
                                              "epochs_trained": args.epochs,
                                              "learning_rate": args.lr})
            artifact.add_file(best_model_path)
            wandb.log_artifact(artifact)
            print(f"Logged model artifact '{model_artifact_name}' to WandB.")
        except FileNotFoundError:
            print(f"Could not log model artifact: File not found at {best_model_path}")
        except Exception as e:
             print(f"Error logging artifact to WandB: {e}")

        wandb.finish()
        print("WandB run finished.")

if __name__ == '__main__':
    args = parse_args()
    main(args)