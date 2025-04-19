# sweep.py (Refactored)
import wandb
import argparse
import os
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm
import numpy as np 

# --- Import necessary components directly ---
# Assuming utils.py, model.py, dataset.py are accessible
# Adjust path if needed:
# script_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(script_dir) # Or adjust as needed
# sys.path.append(project_root)
from .utils import set_seed, str2bool
from .model import CustomCNN, _ACTIVATIONS
from .dataset import get_data_loaders
from wandb.sdk.wandb_settings import Settings # For timeout setting

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ===============================================
#  Core Training Logic (Moved from train.py)
# ===============================================
def train_sweep_trial(args): # Takes the global args object now
    """
    The training function called by wandb.agent for each trial.
    Uses wandb.config for hyperparameters.
    """
    run = None # Initialize run to None for error handling
    try:
        # --- Initialize W&B for this trial ---
        # 'config' will be automatically populated by the agent
        run = wandb.init(settings=Settings(init_timeout=300)) # Increased timeout
        config = wandb.config # Access hyperparameters

        # Create a meaningful name based on hyperparameters
        run_name = f"fl_{config.num_filters}_fo_{config.filter_organization}_fs_{config.filter_size}_bs_{config.batch_size}_ac_{config.activation}" # Example name
        wandb.run.name = run_name
        # wandb.run.save() # Usually not needed, name updates automatically

        logging.info(f"Starting W&B Run: {run.name} with ID: {run.id}")
        logging.info(f"Run config: {dict(config)}") # Log the actual config for this run

        # Set seed for reproducibility within the trial
        set_seed(args.seed)

        # --- Device Setup ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")

        # --- Get data loaders ---
        # Note: batch_size, data_augmentation now come from wandb.config
        train_loader, val_loader, _, classes = get_data_loaders(
            data_dir=args.data_dir,
            batch_size=config.batch_size, # Use W&B config
            augment=config.data_augmentation, # Use W&B config
            num_workers=args.num_workers,
            img_size=args.img_size,
            val_split=args.val_split,
            seed=args.seed
        )
        num_classes = len(classes)
        logging.info(f"Data loaders ready. Num classes: {num_classes}")
        if val_loader is None:
             logging.warning("Validation loader is None. Skipping validation.")

        # --- Create model using wandb.config ---
        model = CustomCNN(
            num_classes=num_classes,
            num_filters=config.num_filters,
            filter_organization=config.filter_organization,
            filter_sizes=[config.filter_size] * 5,
            activation_name=config.activation,
            dense_neurons=config.dense_neurons,
            dropout_rate=config.dropout_rate,
            batch_norm=config.batch_norm,
            img_size=args.img_size
        ).to(device)
        logging.info("Model created.")

        # --- Log model info ---
        wandb.watch(model, log="all", log_freq=100)
        total_params = model.count_parameters()
        total_flops_estimate = model.calculate_computations()
        wandb.log({
            "total_parameters": total_params,
            "total_flops_estimate (MACs*2)": total_flops_estimate
        }, commit=False)
        logging.info(f"Model Parameters: {total_params:,}")
        logging.info(f"Estimated FLOPs (MACs*2): {total_flops_estimate:,.0f}")

        # --- Loss, Optimizer, Scheduler ---
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.2, verbose=False)

        # --- Training Loop ---
        best_val_acc = 0.0
        best_epoch = 0
        logging.info(f"Starting training for {config.epochs} epochs...")
        for epoch in range(config.epochs):
            torch.cuda.empty_cache()
            epoch_start_time = time.time()
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]", leave=False)

            for i, (inputs, labels) in enumerate(train_pbar):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total_train += labels.size(0)
                correct_train += predicted.eq(labels).sum().item()
                if i % 10 == 0:
                    train_pbar.set_postfix({
                        'Loss': f'{running_loss / (i+1):.4f}',
                        'Acc': f'{100. * correct_train / total_train:.2f}%'
                    })
            train_pbar.close()
            train_loss = running_loss / len(train_loader)
            train_accuracy = 100. * correct_train / total_train

            # --- Validation ---
            val_loss = float('nan')
            val_accuracy = float('nan')
            if val_loader:
                model.eval()
                val_correct = 0
                val_total = 0
                running_val_loss = 0.0
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Val]", leave=False)
                with torch.no_grad():
                    for i_val, (inputs, labels) in enumerate(val_pbar):
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        running_val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        val_total += labels.size(0)
                        val_correct += predicted.eq(labels).sum().item()
                        if i_val % 5 == 0:
                             val_pbar.set_postfix({
                                 'Loss': f'{running_val_loss / (i_val+1):.4f}',
                                 'Acc': f'{100. * val_correct / val_total:.2f}%'
                             })
                val_pbar.close()
                if len(val_loader) > 0:
                    val_loss = running_val_loss / len(val_loader)
                    val_accuracy = 100. * val_correct / val_total
                    scheduler.step(val_loss)

            # --- Logging ---
            epoch_duration = time.time() - epoch_start_time
            log_dict = {
                'epoch': epoch + 1, 'train_loss': train_loss, 'train_accuracy': train_accuracy,
                'epoch_duration_sec': epoch_duration, 'learning_rate': optimizer.param_groups[0]['lr']
            }
            if not np.isnan(val_loss): log_dict['val_loss'] = val_loss
            if not np.isnan(val_accuracy): log_dict['val_accuracy'] = val_accuracy
            wandb.log(log_dict)
            logging.info(
                f'Ep {epoch+1}/{config.epochs} | Tr Loss: {train_loss:.4f}, Tr Acc: {train_accuracy:.2f}% | ' +
                (f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}% | ' if val_loader and not np.isnan(val_accuracy) else 'Val: N/A | ') +
                f'LR: {optimizer.param_groups[0]["lr"]:.6f} | Time: {epoch_duration:.2f}s'
             )

            # --- Save Best Model ---
            if val_loader and not np.isnan(val_accuracy) and val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                best_epoch = epoch + 1
                # Saving locally within the agent run isn't strictly necessary if relying on W&B artifacts
                # but can be useful for debugging. We can simplify this or remove if desired.
                # model_filename = f"best_model_ep{best_epoch}_acc{best_val_acc:.2f}.pth"
                # model_path = os.path.join(wandb.run.dir, model_filename)
                # torch.save({'model_state_dict': model.state_dict(), ...}, model_path)
                # logging.info(f"Checkpoint saved for best val_acc: {best_val_acc:.2f}%")

        # --- End of Training Logging ---
        wandb.run.summary["best_val_accuracy"] = best_val_acc if best_epoch > 0 else None
        wandb.run.summary["best_epoch"] = best_epoch if best_epoch > 0 else None
        logging.info(f"Run {run.name} finished. Best val_acc: {best_val_acc:.2f}% at epoch {best_epoch}")

    except Exception as e:
         logging.error(f"Error during training trial {run.id if run else 'unknown'}: {e}", exc_info=True)
         if run: run.finish(exit_code=1) # Mark run as failed if exception occurs
         # Re-raise if needed, or allow agent to continue
         # raise e

    finally:
        # Ensure W&B run is finished properly even if errors occurred before the end
        if run is not None and wandb.run is not None and wandb.run.id == run.id:
             # Double-check if it's already finished before calling again (optional, but cleaner)
             if hasattr(wandb.run, 'finished') and wandb.run.finished is False:
                 try:
                     wandb.finish()
                     logging.debug(f"W&B run {run.id} finished in finally block.")
                 except Exception as final_finish_e:
                     logging.error(f"Error during wandb.finish in finally block: {final_finish_e}")
             else:
                 logging.debug(f"W&B run {run.id} was already finished.")
        else:
            # This case might happen if wandb.init() failed entirely in the try block
            logging.debug("Wandb run was not initialized or context mismatch in finally block.")

# ===============================================
# Sweep Setup (Similar to before, maybe simplified)
# ===============================================
def setup_sweep(args):
    """Defines and initializes the W&B sweep configuration."""
    sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
        'program': 'part_A/train.py', # Path to the training script
        'parameters': {
            # Define sweep parameters here (same as before)
            'num_filters': {'values': [32, 64, 128]},
            'filter_organization': {'values': ['same', 'half']}, # 'double'
            'filter_size': {'values': [3, 5]},
            'activation': {'values': ['relu', 'gelu', 'silu', 'mish']},
            'dense_neurons': {'values': [128, 256, 512]},
            'dropout_rate': {'distribution': 'uniform', 'min': 0.1, 'max': 0.5},
            'batch_norm': {'values': [True, False]},
            'data_augmentation': {'values': [True, False]},
            'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 1e-2},
            'batch_size': {'values': [16, 32, 64]},
            'epochs': {'value': args.epochs} # Use epochs from args for sweep duration
        }
    }
    # Optional: Load from YAML...
    # if args.sweep_config and ... etc ...

    sweep_id = wandb.sweep(sweep_config, project=args.wandb_project, entity=args.wandb_entity)
    logging.info(f"Sweep initialized with ID: {sweep_id}")
    return sweep_id

# ===============================================
# Main Execution Block
# ===============================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run W&B Hyperparameter sweep for Custom CNN model")
    # Args needed for sweep setup and passed down to train_sweep_trial
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--count", type=int, default=20, help="Number of sweep runs")
    parser.add_argument("--wandb_project", type=str, default="CNN-iNaturalist-Scratch-Sweep", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity (username or team)") # Often inferred if logged in
    # parser.add_argument("--sweep_config", type=str, default=None, help="Optional YAML sweep config path")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument("--img_size", type=int, default=224, help="Target image size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split fraction")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs PER SWEEP TRIAL") # Control trial length

    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        logging.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)

    sweep_id = setup_sweep(args)

    # --- Run the agent, calling train_sweep_trial directly ---
    # Need to pass 'args' to train_sweep_trial, use a lambda
    wandb.agent(sweep_id, function=lambda: train_sweep_trial(args), count=args.count)

    logging.info("Sweep agent finished.")