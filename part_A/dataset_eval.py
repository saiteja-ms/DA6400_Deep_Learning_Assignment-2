# dataset_eval.py
# Data loading utilities specifically for evaluation (no generator needed)

import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import numpy as np
import logging
import random

# --- Import INaturalistDataset ---
# Assuming it's defined in a file accessible relative to this one
# If dataset.py is in the same directory:
try:
    from dataset import INaturalistDataset
except ImportError:
    # If evaluate.py is in part_A and dataset.py is too
    try:
        from .dataset import INaturalistDataset # Use relative import
    except ImportError as e:
         print(f"Error importing INaturalistDataset in dataset_eval.py: {e}")
         print("Ensure dataset.py exists and is importable.")
         sys.exit(1)


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


def get_data_loaders_eval(data_dir, batch_size=32, num_workers=2, img_size=224, seed=42):
    """
    Creates DataLoaders for evaluation ONLY (train, val, test).
    Does NOT expect or use a torch.Generator object.
    Does NOT perform validation split (assumes val_split=0).
    """
    logging.info(f"DataLoaders Eval: batch={batch_size}, workers={num_workers}, img_size={img_size}")
    random.seed(seed); np.random.seed(seed) # Seed basic operations if needed

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    target_size = (img_size, img_size)

    # Use evaluation transforms for all splits when just evaluating
    eval_transform = transforms.Compose([
        transforms.Resize(target_size), # Simple resize for eval usually ok
        # Or use Resize(256)/CenterCrop(224) if preferred
        transforms.ToTensor(),
        normalize
    ])

    train_dir, test_dir = os.path.join(data_dir, 'train'), os.path.join(data_dir, 'test')
    if not os.path.isdir(train_dir): raise FileNotFoundError(f"Training dir not found: {train_dir}")
    if not os.path.isdir(test_dir): raise FileNotFoundError(f"Test dir not found: {test_dir}")

    try:
        # Load datasets with the evaluation transform
        train_dataset = INaturalistDataset(root_dir=train_dir, transform=eval_transform)
        test_dataset = INaturalistDataset(root_dir=test_dir, transform=eval_transform)
    except Exception as e: logging.error(f"Error loading dataset: {e}", exc_info=True); raise e

    # No validation split, so val_loader is None
    val_loader = None
    # Train loader might be needed if e.g. visualizing filters requires a sample
    # Set shuffle=False for eval purposes if using train_loader samples
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=torch.cuda.is_available())

    logging.info("Evaluation data loaders created.")
    # Return None for val_loader consistently
    return train_loader, None, test_loader, train_dataset.classes # Get classes from train dataset