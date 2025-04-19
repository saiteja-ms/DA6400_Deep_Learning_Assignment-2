# dataset.py
import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import numpy as np
import logging
import random # Needed for shuffling within classes

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class INaturalistDataset(Dataset):
    """
    Custom Dataset class for the iNaturalist subset.
    Loads images from a directory structure where each sub-directory
    represents a class.
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the class subdirectories.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Root directory not found: {root_dir}")
        self.root_dir = root_dir
        self.transform = transform
        try:
            # Ensure we only list directories
            self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
            if not self.classes:
                raise ValueError(f"No class subdirectories found in {root_dir}")
        except OSError as e:
            raise OSError(f"Error reading directory {root_dir}: {e}")

        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}
        logging.info(f"Found {len(self.classes)} classes: {list(self.idx_to_class.values())}")

        self.images = []
        self.labels = []

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            try:
                img_names = os.listdir(class_dir)
                if not img_names:
                    logging.warning(f"No images found in class directory: {class_dir}")
                    continue # Skip empty class directories
                for img_name in img_names:
                    img_path = os.path.join(class_dir, img_name)
                    # Basic check for common image file extensions
                    if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(img_path)
                        self.labels.append(self.class_to_idx[class_name])
                    else:
                         logging.debug(f"Skipping non-image file: {img_path}")
            except OSError as e:
                logging.error(f"Could not read images from {class_dir}: {e}")

        if not self.images:
             raise RuntimeError(f"No valid image files found in any class subdirectories under {root_dir}")
        logging.info(f"Loaded {len(self.images)} image paths in total.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        try:
            # Ensure image is opened and converted robustly
            with Image.open(img_path) as img:
                image = img.convert("RGB")
        except Exception as e:
            logging.error(f"Error opening or converting image {img_path}: {e}")
            # Return a placeholder or raise error. Raising is safer during development.
            raise IOError(f"Could not process image {img_path}") from e

        label = self.labels[idx]

        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                logging.error(f"Error applying transform to image {img_path}: {e}")
                raise RuntimeError(f"Transform failed for image {img_path}") from e

        return image, label

def get_data_loaders(data_dir, batch_size=32, val_split=0.2, augment=True, num_workers=4, img_size=224, seed=42):
    """
    Creates training, validation, and test DataLoaders without using sklearn.

    Args:
        data_dir (str): Path to the dataset directory (contains 'train', 'test').
        batch_size (int): Batch size.
        val_split (float): Fraction of training data for validation (must be > 0 and < 1).
        augment (bool): Apply data augmentation to training set.
        num_workers (int): Number of DataLoader workers.
        img_size (int): Target image size (height and width).
        seed (int): Random seed for shuffling during split.

    Returns:
        tuple: (train_loader, val_loader, test_loader, class_names)
    """
    logging.info(f"Setting up data loaders: batch_size={batch_size}, augment={augment}, val_split={val_split}, img_size={img_size}")
    random.seed(seed)
    np.random.seed(seed)

    # Standard ImageNet mean and std
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    target_size = (img_size, img_size)

    # Define transformations (same as before)
    if augment:
        train_transform_list = [
            transforms.Resize(target_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            normalize
        ]
        logging.info("Using data augmentation for training.")
    else:
        train_transform_list = [
            transforms.Resize(target_size),
            transforms.ToTensor(),
            normalize
        ]
        logging.info("Data augmentation for training is disabled.")
    train_transform = transforms.Compose(train_transform_list)
    val_test_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        normalize
    ])

    # --- Dataset Instantiation ---
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    if not os.path.isdir(train_dir): raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not os.path.isdir(test_dir): raise FileNotFoundError(f"Test directory not found: {test_dir}")

    full_dataset = INaturalistDataset(root_dir=train_dir, transform=train_transform)
    test_dataset = INaturalistDataset(root_dir=test_dir, transform=val_test_transform)
    num_classes = len(full_dataset.classes)
    if num_classes == 0: raise ValueError("No classes found in the training directory.")
    logging.info(f"Number of classes: {num_classes}")

    # --- Stratified Train/Validation Split (Manual Implementation) ---
    train_indices = []
    val_indices = []
    targets = np.array(full_dataset.labels)
    dataset_size = len(targets)

    if val_split <= 0 or val_split >= 1:
        logging.warning("val_split is not within (0, 1). Skipping validation split.")
        train_indices = list(range(dataset_size))
        val_loader = None
    else:
        indices_by_class = {}
        for idx, label in enumerate(targets):
            if label not in indices_by_class:
                indices_by_class[label] = []
            indices_by_class[label].append(idx)

        for label, indices in indices_by_class.items():
            n_class_samples = len(indices)
            if n_class_samples == 0:
                continue # Should not happen if dataset loading is correct

            np.random.shuffle(indices) # Shuffle indices within the class

            n_val = int(np.floor(val_split * n_class_samples))

            # Ensure at least one sample for training if possible
            if n_class_samples > 1 and n_val == n_class_samples:
                n_val = n_class_samples - 1 # Keep at least one for training
            elif n_class_samples <= 1:
                 n_val = 0 # Cannot split single sample, keep for training

            val_indices.extend(indices[:n_val])
            train_indices.extend(indices[n_val:])

        logging.info(f"Manual stratified split: {len(train_indices)} train, {len(val_indices)} validation samples.")
        # Shuffle the combined train indices again for better batch randomness
        random.shuffle(train_indices)

        # Create validation subset (needs the val_test_transform)
        original_transform = full_dataset.transform
        full_dataset.transform = val_test_transform
        val_subset = Subset(full_dataset, val_indices)
        full_dataset.transform = original_transform # Restore train transform

        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=torch.cuda.is_available()
        )

    # Create training subset and loader
    train_subset = Subset(full_dataset, train_indices)
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=True
    )

    # Create test loader
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )

    logging.info("Data loaders created successfully.")
    return train_loader, val_loader, test_loader, full_dataset.classes