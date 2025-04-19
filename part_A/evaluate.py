# evaluate.py
# Evaluates a pre-trained CustomCNN model on the test set for Part A Q4.
# Generates test accuracy and prediction grid.

import torch
import torch.nn as nn
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import logging
import wandb
import time
from tqdm import tqdm # Use standard tqdm
import sys

# --- Try importing local modules ---
# Assuming running from parent directory 'Assignment_2'
try:
    from part_A.utils import set_seed, str2bool
    from part_A.model import CustomCNN, _ACTIVATIONS
    # Import get_data_loaders WITHOUT generator argument expected, and INaturalistDataset
    from part_A.dataset_eval import get_data_loaders_eval, INaturalistDataset
    print("Imported modules using package path 'part_A'.")
except ImportError:
    # Fallback if running from 'part_A' directory
    try:
        from utils import set_seed, str2bool
        from model import CustomCNN, _ACTIVATIONS
        # Import get_data_loaders WITHOUT generator argument expected, and INaturalistDataset
        from dataset_eval import get_data_loaders_eval, INaturalistDataset
        print("Imported modules assuming script is in part_A.")
    except ImportError as e:
        print(f"Error importing local modules: {e}")
        print("Ensure evaluate.py and dependent files (utils, model, dataset_eval) exist and are importable.")
        sys.exit(1)

from wandb.sdk.wandb_settings import Settings

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', stream=sys.stdout)

# --- Visualization Helpers ---
_MEAN = np.array([0.485, 0.456, 0.406])
_STD = np.array([0.229, 0.224, 0.225])
def denormalize(tensor):
    try:
        tensor = tensor.clone().cpu().numpy(); tensor = np.transpose(tensor, (1, 2, 0))
        tensor = _STD * tensor + _MEAN; tensor = np.clip(tensor, 0, 1)
        return tensor
    except Exception as e: logging.error(f"Dnorm err: {e}"); return np.zeros((224, 224, 3))

# --- Main Evaluation Function ---
def evaluate_model(args):
    """ Loads a trained model and evaluates it on the test set. """
    run = None
    try:
        # --- W&B Init ---
        run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            config=vars(args), # Log command-line args used for evaluation
            job_type="evaluation",
            name=f"eval-{os.path.basename(args.model_path).split('.')[0]}"[:128], # Run name based on model file
            settings=Settings(init_timeout=300)
        )
        if not run: raise Exception("wandb.init failed")
        logging.info(f"--- Evaluating Model: {args.model_path} ---")

        # --- Setup ---
        set_seed(args.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        logging.info(f"Using device: {device}")

        # --- Get Data Loaders & Class Info (Using Eval Version) ---
        # This version of get_data_loaders should NOT expect a 'generator' argument
        _, _, test_loader, classes = get_data_loaders_eval(
            data_dir=args.data_dir, batch_size=args.batch_size,
            num_workers=args.num_workers, img_size=args.img_size
        )
        num_classes = len(classes)
        idx_to_class = {i: name for i, name in enumerate(classes)}
        logging.info(f"Test data loader ready. Num classes: {num_classes}")

        # --- Load Model ---
        logging.info("Attempting to load model architecture and weights...")
        if not os.path.isfile(args.model_path):
            raise FileNotFoundError(f"Model checkpoint file not found: {args.model_path}")

        checkpoint = torch.load(args.model_path, map_location=device)

        # --- Determine Architecture ---
        if 'config' in checkpoint:
            model_config = checkpoint['config']
            logging.info(f"Loading architecture from checkpoint config: {model_config}")
            # Update args ONLY for model creation, don't overwrite args logged to W&B config
            img_size_model = model_config.get('img_size', args.img_size)
            num_filters_model = model_config['num_filters']
            if isinstance(num_filters_model, int): num_filters_model = [num_filters_model] * 5
            filter_sizes_model = model_config['filter_sizes']
            activation_model = model_config['activation_name']
            dense_neurons_model = model_config['dense_neurons']
            dropout_rate_model = model_config['dropout_rate']
            batch_norm_model = model_config['batch_norm']
            # filter_organization not strictly needed if num_filters_list is present
        else:
            logging.warning("Model config not found in checkpoint. Using CLI args for architecture.")
            # Use the specific CLI args for architecture if config is missing
            img_size_model = args.img_size
            num_filters_model = args.num_filters_base # Use base if list not provided
            filter_sizes_model = [args.filter_size_base] * 5
            activation_model = args.activation
            dense_neurons_model = args.dense_neurons
            dropout_rate_model = args.dropout_rate
            batch_norm_model = args.batch_norm
            # Need filter_organization if num_filters_base is used
            filter_organization_model = args.filter_organization

        # Recreate model architecture
        model = CustomCNN(
            num_classes=num_classes,
            num_filters=num_filters_model, # Use inferred/provided arch params
            filter_organization=filter_organization_model if 'filter_organization_model' in locals() else 'same', # Provide if using base filters
            filter_sizes=filter_sizes_model,
            activation_name=activation_model,
            dense_neurons=dense_neurons_model,
            dropout_rate=dropout_rate_model,
            batch_norm=batch_norm_model,
            img_size=img_size_model
        ).to(device)

        # Load weights
        state_dict = checkpoint.get('model_state_dict', checkpoint) # Handle both formats
        model.load_state_dict(state_dict)
        logging.info(f"Successfully loaded model weights from {args.model_path}")

        # --- Evaluation on Test Set ---
        model.eval(); correct_test = 0; total_test = 0; test_loss_accum = 0.0
        all_test_images = []; all_test_labels = []; all_test_preds = []
        eval_start_time = time.time()
        logging.info("Starting evaluation on the test set...")
        test_pbar = tqdm(test_loader, desc="Testing", leave=False, file=sys.stdout)
        criterion = nn.CrossEntropyLoss() # Need loss for optional logging

        with torch.no_grad():
            for inputs, labels in test_pbar:
                if torch.any(labels < 0): continue
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss_accum += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0); correct_test += (predicted == labels).sum().item()
                if len(all_test_images) < args.grid_rows * args.grid_cols * 2:
                     all_test_images.append(inputs.cpu()); all_test_labels.append(labels.cpu()); all_test_preds.append(predicted.cpu())
        test_pbar.close()

        eval_duration = time.time() - eval_start_time
        final_test_loss = test_loss_accum / total_test if total_test > 0 else float('nan')
        final_test_acc = 100. * correct_test / total_test if total_test > 0 else 0
        logging.info(f'Evaluation finished. Time: {eval_duration:.2f}s')
        logging.info(f'Test Loss: {final_test_loss:.4f}, Test Accuracy: {final_test_acc:.2f}%')

        # Log Final Test Metrics to W&B Summary
        wandb.run.summary["final_test_loss"] = final_test_loss
        wandb.run.summary["final_test_accuracy"] = final_test_acc
        wandb.run.summary["total_test_samples"] = total_test
        wandb.run.summary["eval_duration_sec"] = eval_duration

        # --- Generate and Log Prediction Grid ---
        logging.info("Generating prediction grid...")
        try:
            if all_test_images:
                all_test_images=torch.cat(all_test_images); all_test_labels=torch.cat(all_test_labels); all_test_preds=torch.cat(all_test_preds)
                num_classes_plot=min(num_classes, args.grid_rows); samples_per_class=args.grid_cols
                plt.figure(figsize=(samples_per_class*3, num_classes_plot*3.5)); plotted_count=0
                for i in range(num_classes):
                    if plotted_count >= num_classes_plot*samples_per_class: break
                    class_indices = (all_test_labels == i).nonzero(as_tuple=True)[0]
                    if len(class_indices)==0: continue
                    # Use torch.randperm for reproducibility if seed is set
                    plot_indices=class_indices[torch.randperm(len(class_indices), generator=torch.Generator().manual_seed(args.seed))[:samples_per_class]]
                    for j, idx in enumerate(plot_indices):
                        if plotted_count >= num_classes_plot*samples_per_class: break
                        img=all_test_images[idx]; true_lbl_idx=all_test_labels[idx].item(); pred_lbl_idx=all_test_preds[idx].item()
                        img_denorm=denormalize(img)
                        ax=plt.subplot(num_classes_plot, samples_per_class, plotted_count + 1)
                        ax.imshow(img_denorm); title_color = 'green' if true_lbl_idx==pred_lbl_idx else 'red'
                        ax.set_title(f'T:{idx_to_class[true_lbl_idx]}\nP:{idx_to_class[pred_lbl_idx]}',color=title_color, fontsize=9); ax.axis('off')
                        plotted_count += 1
                plt.tight_layout(); grid_save_path = os.path.join(args.output_dir, 'prediction_grid.png')
                os.makedirs(args.output_dir, exist_ok=True); plt.savefig(grid_save_path, dpi=150); plt.close()
                wandb.log({"prediction_grid": wandb.Image(grid_save_path)}, commit=True)
                logging.info(f"Prediction grid saved to {grid_save_path} and logged.")
            else: logging.warning("No images collected for grid.")
        except Exception as e: logging.error(f"Failed grid gen: {e}", exc_info=True); plt.close()

    except Exception as e:
         logging.error(f"Error during evaluation: {e}", exc_info=True)
         if run: wandb.finish(exit_code=1) # Mark run as failed
         sys.exit(1) # Exit script on error

    finally:
        # Ensure finish is called
        if run and wandb.run is not None and wandb.run.id == run.id:
             if hasattr(wandb.run, 'finished') and not wandb.run.finished:
                 try: wandb.finish()
                 except Exception as fe: logging.error(f"Error finishing W&B run: {fe}")

# --- Argument Parser ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a saved Custom CNN model.")
    # Required
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model checkpoint (.pth)")
    parser.add_argument("--output_dir", type=str, default="./output_evaluation", help="Directory to save prediction grid")
    # W&B
    parser.add_argument("--wandb_project", type=str, default="CNN-iNaturalist-Scratch", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity")
    # Setup
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--img_size", type=int, default=224, help="Image size model was trained with")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    # Grid
    parser.add_argument("--grid_rows", type=int, default=10, help="Rows in prediction grid")
    parser.add_argument("--grid_cols", type=int, default=3, help="Cols in prediction grid")

    # --- Model Architecture Arguments (Fallback if config not in checkpoint) ---
    parser.add_argument("--num_filters_base", type=int, default=None, help="Base num filters (if config missing)")
    parser.add_argument("--filter_size_base", type=int, default=None, help="Filter size (if config missing)")
    parser.add_argument("--filter_organization", type=str, default=None, help="Filter organization (if config missing)")
    parser.add_argument("--activation", type=str, default=None, help="Activation function (if config missing)")
    parser.add_argument("--dense_neurons", type=int, default=None, help="Dense neurons (if config missing)")
    parser.add_argument("--dropout_rate", type=float, default=None, help="Dropout rate (if config missing)")
    parser.add_argument("--batch_norm", type=str2bool, default=None, help="Batch norm used (if config missing)")

    args = parser.parse_args()

    if not os.path.isdir(args.data_dir): logging.error(f"Data dir not found: {args.data_dir}"); sys.exit(1)
    if not os.path.isfile(args.model_path): logging.error(f"Model path not found: {args.model_path}"); sys.exit(1)

    evaluate_model(args)

    print("--- Evaluation Script Finished ---")