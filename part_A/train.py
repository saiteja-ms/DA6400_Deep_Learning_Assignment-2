# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import argparse
import os
import logging
import time
import numpy as np
from tqdm import tqdm  # Import tqdm
from model import CustomCNN, _ACTIVATIONS
from dataset import get_data_loaders
from utils import set_seed, str2bool
from wandb.sdk.wandb_settings import Settings


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(args):
    """Trains the CustomCNN model with hyperparameters specified in args."""
    set_seed(args.seed)

    # --- Initialize W&B ---
    run = wandb.init(
        project=args.wandb_project,
        config=vars(args),
        id=args.wandb_run_id,
        resume="allow",
        settings = Settings(init_timeout=300)
    )
    logging.info(f"Wandb run initialized. ID: {run.id}, Name: {run.name}")
    logging.info(f"Run config: {wandb.config}")

    # --- Device Setup ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache() # Clear cache before initializing anything else
        
    else:
        device = torch.device("cpu")
        logging.info("CUDA not available. Using CPU.")

    # --- Get data loaders ---
    try:
        train_loader, val_loader, _, classes = get_data_loaders(
            data_dir=args.data_dir,
            batch_size=wandb.config.batch_size, # Use W&B config here now
            augment=wandb.config.data_augmentation, # Use W&B config
            num_workers=args.num_workers,
            img_size=args.img_size,
            val_split=args.val_split,
            seed=args.seed # Pass seed to dataloader split
        )
        num_classes = len(classes)
        logging.info(f"Data loaders ready. Num classes: {num_classes}")
        if val_loader is None:
            logging.warning("Validation loader is None. Skipping validation.")
    except Exception as e:
        logging.error(f"Data loading failed: {e}", exc_info=True)
        wandb.finish(exit_code=1)
        exit(1)

    # --- Create model ---
    try:
        if wandb.config.activation not in _ACTIVATIONS:
             raise ValueError(f"Invalid activation function: {wandb.config.activation}")

        model = CustomCNN(
            num_classes=num_classes,
            num_filters=wandb.config.num_filters,
            filter_organization=wandb.config.filter_organization,
            filter_sizes=[wandb.config.filter_size] * 5,
            activation_name=wandb.config.activation,
            dense_neurons=wandb.config.dense_neurons,
            dropout_rate=wandb.config.dropout_rate,
            batch_norm=wandb.config.batch_norm,
            img_size=args.img_size
        ).to(device)
        logging.info("Model created and moved to device.")
    except Exception as e:
        logging.error(f"Failed to create model: {e}", exc_info=True)
        wandb.finish(exit_code=1)
        exit(1)

    # --- Log model info to W&B ---
    wandb.watch(model, log="all", log_freq=100)
    total_params = model.count_parameters()
    total_flops_estimate = model.calculate_computations()
    wandb.log({
        "total_parameters": total_params,
        "total_flops_estimate (MACs*2)": total_flops_estimate
    }, commit=False)
    logging.info(f"Model Parameters: {total_params:,}")
    logging.info(f"Estimated FLOPs (MACs*2): {total_flops_estimate:,.0f}")

    # --- Define loss function, optimizer, and scheduler ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.2, verbose=False) # verbose=False cleans up tqdm output

    # --- Training Loop ---
    best_val_acc = 0.0
    best_epoch = 0
    start_time = time.time()
    logging.info(f"Starting training for {wandb.config.epochs} epochs...")

    for epoch in range(wandb.config.epochs):
        torch.cuda.empty_cache()
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # --- Wrap train_loader with tqdm ---
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{wandb.config.epochs} [Train]", leave=False)
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

            # Update tqdm postfix with running metrics
            if i % 10 == 0: # Update less frequently to avoid overhead
                train_pbar.set_postfix({
                    'Loss': f'{running_loss / (i+1):.4f}',
                    'Acc': f'{100. * correct_train / total_train:.2f}%'
                })
        train_pbar.close() # Close the inner progress bar

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100. * correct_train / total_train

        # --- Validation ---
        val_loss = 0.0
        val_accuracy = 0.0
        if val_loader:
            model.eval()
            val_correct = 0
            val_total = 0
            # --- Wrap val_loader with tqdm ---
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{wandb.config.epochs} [Val]", leave=False)
            with torch.no_grad():
                for inputs, labels in val_pbar:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

                    # Update validation postfix
                    if len(val_pbar) > 0 and i % 5 == 0: # Update less frequently
                         val_pbar.set_postfix({
                             'Loss': f'{val_loss / (val_pbar.n + 1):.4f}', # Use pbar.n for current count
                             'Acc': f'{100. * val_correct / val_total:.2f}%'
                         })
            val_pbar.close() # Close the validation progress bar

            if len(val_loader) > 0:
                 val_loss /= len(val_loader)
                 val_accuracy = 100. * val_correct / val_total
                 scheduler.step(val_loss)
            else:
                 val_loss = float('nan') # Indicate no validation was done
                 val_accuracy = float('nan')

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        else:
             # Handle case where no validation loader exists
             val_loss = float('nan')
             val_accuracy = float('nan')
             # scheduler.step(train_loss) # Optional: step based on train loss


        epoch_duration = time.time() - epoch_start_time
        # --- Log metrics to W&B ---
        log_dict = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'epoch_duration_sec': epoch_duration,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        # Only log val metrics if they were calculated
        if not np.isnan(val_loss): log_dict['val_loss'] = val_loss
        if not np.isnan(val_accuracy): log_dict['val_accuracy'] = val_accuracy

        wandb.log(log_dict)

        # Print summary line after tqdm bars are closed
        logging.info(
            f'Epoch {epoch+1}/{wandb.config.epochs} | '
            f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | ' +
            (f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}% | ' if val_loader else 'Val: N/A | ') +
            f'LR: {optimizer.param_groups[0]["lr"]:.6f} | Time: {epoch_duration:.2f}s'
         )

        # --- Save the best model ---
        if val_loader and not np.isnan(val_accuracy) and val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_epoch = epoch + 1
            model_filename = f"best_model_ep{best_epoch}_acc{best_val_acc:.2f}.pth"
            model_path = os.path.join(wandb.run.dir, model_filename)
            try:
                torch.save({
                    'epoch': best_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_accuracy': best_val_acc,
                    'config': model.config
                    }, model_path)
                logging.info(f"New best model saved: {model_filename}")
            except Exception as e:
                logging.error(f"Error saving model checkpoint: {e}")

    # --- End of Training ---
    total_training_time = time.time() - start_time
    logging.info(f"Training finished. Total time: {total_training_time:.2f}s")

    # --- Log best metrics to W&B summary ---
    wandb.run.summary["best_val_accuracy"] = best_val_acc if not np.isnan(val_accuracy) else None
    wandb.run.summary["best_epoch"] = best_epoch if best_epoch > 0 else None
    wandb.run.summary["total_training_time_sec"] = total_training_time
    if best_epoch > 0:
        logging.info(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
    else:
        logging.info("No best model saved based on validation accuracy.")

    # --- Save Final Best Model as W&B Artifact ---
    # (Code for finding and logging the artifact remains the same)
    best_model_files = [f for f in os.listdir(wandb.run.dir) if f.startswith("best_model_") and f.endswith(".pth")]
    if best_model_files:
        # A simple way to find the 'best' based on filename accuracy/epoch might work
        # Or track the best_model_path variable directly in the loop
        best_model_path = os.path.join(wandb.run.dir, max(best_model_files)) # Simplistic sort
        try:
            logging.info(f"Logging best model artifact: {os.path.basename(best_model_path)}")
            best_model_artifact = wandb.Artifact(f"model-{run.id}", type="model",
                                                 description=f"Best model from run {run.id}, acc {best_val_acc:.2f}",
                                                 metadata=dict(wandb.config)) # Log hyperparams with artifact
            best_model_artifact.add_file(best_model_path, name="best_model.pth") # Standardize artifact filename
            run.log_artifact(best_model_artifact)
        except Exception as e:
             logging.error(f"Failed to log model artifact: {e}")

    wandb.finish()
    logging.info("Wandb run finished.")

# --- Argument Parser (remains the same) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Custom CNN model")
    # ... (keep all arguments from the previous version) ...
    # --- Setup Arguments ---
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory (containing train/test)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--img_size", type=int, default=224, help="Target image size (e.g., 224 for 224x224)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument("--val_split", type=float, default=0.2, help="Fraction of training data for validation")

    # --- W&B Arguments ---
    parser.add_argument("--wandb_project", type=str, default="CNN-iNaturalist-Scratch", help="Wandb project name")
    parser.add_argument("--wandb_run_id", type=str, default=None, help="Wandb run ID to resume (used by sweep agent)")

    # --- Hyperparameters (Define defaults, sweeps will override) ---
    parser.add_argument("--num_filters", type=int, default=32, help="Base number of filters for conv layers")
    parser.add_argument("--filter_size", type=int, default=3, help="Kernel size for conv layers")
    parser.add_argument("--filter_organization", type=str, default="same", choices=["same", "double", "half"], help="Filter organization strategy")
    parser.add_argument("--activation", type=str, default="relu", choices=_ACTIVATIONS.keys(), help="Activation function")
    parser.add_argument("--dense_neurons", type=int, default=128, help="Neurons in the hidden dense layer")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--batch_norm", type=str2bool, default='True', help="Use batch normalization (True/False)")
    parser.add_argument("--data_augmentation", type=str2bool, default='True', help="Use data augmentation (True/False)")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")

    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        logging.error(f"Data directory not found: {args.data_dir}")
        exit(1)
        

    train_model(args)