# utils.py
import random
import numpy as np
import torch
import os
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_seed(seed=42):
    """Set seed for reproducibility across relevant libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic algorithms are used for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False # Disable benchmark for determinism
        logging.info("CUDA available. Setting deterministic CUDA operations.")
    else:
        logging.info("CUDA not available. Setting CPU seeds only.")
    os.environ['PYTHONHASHSEED'] = str(seed)
    logging.info(f"Seed set to {seed}")

def str2bool(v):
    """Helper function to parse boolean command-line arguments."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Parameter count is now a method within the model class
# FLOPs calculation is now a method within the model class