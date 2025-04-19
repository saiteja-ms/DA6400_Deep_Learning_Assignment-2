# visualize.py
import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
import wandb
from model import CustomCNN, _ACTIVATIONS
from dataset import get_data_loaders
from utils import set_seed, str2bool
from evaluate import denormalize # Import denormalize function

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def visualize_filters(model, layer_index=0, save_path='first_layer_filters.png'):
    """Visualize the filters of a specified convolutional layer."""
    target_layer = None
    conv_count = 0
    # Iterate through the model's defined layers
    if not hasattr(model, 'layers') or not isinstance(model.layers, nn.ModuleList):
        logging.error("Model does not have a 'layers' attribute of type ModuleList.")
        return None

    for i, layer in enumerate(model.layers):
        if isinstance(layer, nn.Conv2d):
            if conv_count == layer_index:
                target_layer = layer
                break
            conv_count += 1

    if target_layer is None:
        logging.warning(f"Could not find Conv2D layer at index {layer_index}.")
        return None

    logging.info(f"Visualizing filters from layer {layer_index} (type: {type(target_layer)}).")
    try:
        filters = target_layer.weight.data.cpu().clone() # Get weights on CPU
    except Exception as e:
        logging.error(f"Error accessing weights for layer {layer_index}: {e}")
        return None

    if filters.dim() != 4: # Should be (out_channels, in_channels, H, W)
        logging.error(f"Filter weights have unexpected dimensions: {filters.shape}")
        return None

    # Normalize filters for visualization: [0, 1] range per filter (across all channels)
    n_filters, n_channels, fh, fw = filters.shape
    filters_norm = filters.view(n_filters, -1) # Flatten channels, H, W
    f_min = filters_norm.min(dim=1, keepdim=True)[0]
    f_max = filters_norm.max(dim=1, keepdim=True)[0]
    filters_norm = (filters_norm - f_min) / (f_max - f_min + 1e-8)
    filters_norm = filters_norm.view(n_filters, n_channels, fh, fw) # Reshape back

    # Plot the filters (average over input channels)
    rows = int(np.sqrt(n_filters))
    cols = int(np.ceil(n_filters / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(max(8, cols * 1.5), max(6, rows * 1.5)))
    axes = axes.flatten() if n_filters > 1 else [axes] # Handle single filter case

    for i in range(n_filters):
        # Average over input channels for visualization
        f_display = filters_norm[i].mean(dim=0)
        ax = axes[i]
        ax.imshow(f_display, cmap='viridis')
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.set_title(f'Filter {i}', fontsize=8) # Optional title

    # Turn off axes for any remaining subplots
    for j in range(n_filters, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f'Filters from Conv Layer {layer_index} ({n_filters} filters)', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout

    try:
        plt.savefig(save_path, dpi=150)
        logging.info(f"Filter visualization saved to {save_path}")
        plt.close(fig)
        return save_path
    except Exception as e:
        logging.error(f"Failed to save filter visualization: {e}")
        plt.close(fig)
        return None


def simplified_gradient_visualization(model, input_image, target_conv_layer_idx=4, target_neuron_idx=0):
    """
    Visualize gradient w.r.t input for a specific neuron's activation sum.
    NOTE: This is NOT true Guided Backpropagation. It shows input influence.
    """
    model.eval()
    input_image = input_image.clone().requires_grad_(True)
    if input_image.grad is not None:
        input_image.grad.zero_() # Zero gradients if they exist

    activation_output = None
    hooks = []

    # --- Find Target Layer & Register Hook ---
    target_layer = None
    conv_count = 0
    module_list = list(model.layers.children())
    target_module_index = -1

    for i, layer in enumerate(module_list):
        if isinstance(layer, nn.Conv2d):
            if conv_count == target_conv_layer_idx:
                target_layer = layer
                target_module_index = i
                break
            conv_count += 1

    if target_layer is None:
        logging.error(f"Could not find target Conv layer index {target_conv_layer_idx}.")
        return None

    # Hook function to store the output of the target layer
    def hook_fn(module, input, output):
        nonlocal activation_output
        activation_output = output

    hook = target_layer.register_forward_hook(hook_fn)
    hooks.append(hook)

    # --- Forward Pass to Target Layer ---
    # Manually pass through layers up to and including the target conv layer
    x = input_image
    for i, layer in enumerate(module_list):
         x = layer(x)
         if i == target_module_index:
             break # Stop after the target layer's forward pass

    # Check if hook captured the activation
    if activation_output is None:
        logging.error(f"Hook failed to capture activation for layer {target_conv_layer_idx}.")
        for h in hooks: h.remove()
        return None

    # --- Backward Pass ---
    model.zero_grad() # Zero all model gradients

    # Check neuron index validity
    num_filters_in_layer = activation_output.shape[1]
    if not (0 <= target_neuron_idx < num_filters_in_layer):
         logging.error(f"Target neuron index {target_neuron_idx} out of range for layer {target_conv_layer_idx} ({num_filters_in_layer} filters).")
         for h in hooks: h.remove()
         return None

    # Target the sum of the spatial dimensions for the specific neuron (filter map)
    target = activation_output[0, target_neuron_idx].sum()

    try:
        target.backward() # Compute gradients back to the input image
    except Exception as e:
        logging.error(f"Backward pass failed during gradient visualization: {e}")
        for h in hooks: h.remove()
        return None

    # Remove hooks
    for h in hooks: h.remove()

    # --- Process Gradients ---
    if input_image.grad is None:
        logging.error("Input image gradient is None after backward pass.")
        return None

    gradients = input_image.grad.data.cpu().squeeze(0) # Remove batch dim, move to CPU

    # Visualize: absolute value, max over channels, normalize
    gradients = gradients.abs()
    gradients = gradients.max(dim=0)[0] # Max across color channels -> (H, W)
    g_min, g_max = gradients.min(), gradients.max()
    gradients_norm = (gradients - g_min) / (g_max - g_min + 1e-8) # Normalize to [0, 1]

    return gradients_norm.numpy()


def run_visualizations(args):
    """Loads model and runs selected visualizations."""
    set_seed(args.seed)

    # --- W&B Init ---
    run = wandb.init(project=f"{args.wandb_project}-Viz", config=vars(args), job_type="visualization")
    logging.info("Initialized Wandb for Visualization.")

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Get Data (need one sample image) ---
    try:
        _, _, test_loader, classes = get_data_loaders(
            data_dir=args.data_dir, batch_size=1, augment=False,
            num_workers=args.num_workers, img_size=args.img_size, val_split=0
        )
        num_classes = len(classes)
        sample_batch = next(iter(test_loader))
        sample_image_tensor = sample_batch[0].to(device) # Get image tensor on device
        logging.info("Sample image obtained.")
    except StopIteration:
        logging.error("Test loader is empty.")
        wandb.finish(exit_code=1); return
    except Exception as e:
        logging.error(f"Failed to get data/sample: {e}", exc_info=True)
        wandb.finish(exit_code=1); return

    # --- Load Model ---
    logging.warning("Ensure architecture args passed match the saved model!")
    try:
        if not os.path.isfile(args.model_path):
            raise FileNotFoundError(f"Model weights file not found: {args.model_path}")

        checkpoint = torch.load(args.model_path, map_location=device)
        model_config = checkpoint.get('config')

        if model_config:
            logging.info("Loading architecture from checkpoint config.")
            # Override CLI args if config exists
            args.img_size = model_config.get('img_size', args.img_size)
            # Handle potential list/int for num_filters
            loaded_num_filters = model_config['num_filters']
            args.num_filters = loaded_num_filters if isinstance(loaded_num_filters, list) else model_config.get('num_filters_base', args.num_filters_base)
            args.filter_sizes = model_config['filter_sizes']
            args.activation = model_config['activation_name']
            args.dense_neurons = model_config['dense_neurons']
            args.dropout_rate = model_config['dropout_rate']
            args.batch_norm = model_config['batch_norm']
        else:
            logging.warning("Model config not found in checkpoint. Using CLI args.")
            # Prepare args for model creation from CLI defaults/inputs
            args.num_filters = args.num_filters_base # Use base filter count
            args.filter_sizes = [args.filter_size_base] * 5


        model = CustomCNN(
            num_classes=num_classes,
            num_filters=args.num_filters, # Use list or base int
            filter_organization=args.filter_organization, # May not be needed if list is used
            filter_sizes=args.filter_sizes,
            activation_name=args.activation,
            dense_neurons=args.dense_neurons,
            dropout_rate=args.dropout_rate,
            batch_norm=args.batch_norm,
            img_size=args.img_size
        ).to(device)

        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        logging.info(f"Loaded model weights from {args.model_path}")
        model.eval()

    except Exception as e:
        logging.error(f"Failed to load model: {e}", exc_info=True)
        wandb.finish(exit_code=1); return

    # --- Visualize First Layer Filters ---
    logging.info("Visualizing first layer filters...")
    filter_save_path = 'first_layer_filters_viz.png'
    try:
        saved_path = visualize_filters(model, layer_index=0, save_path=filter_save_path)
        if saved_path and os.path.exists(saved_path):
            wandb.log({"first_layer_filters": wandb.Image(saved_path)})
    except Exception as e:
        logging.error(f"Filter visualization failed: {e}", exc_info=True)

    # --- Visualize Neuron Activations (Simplified Gradient - Optional Q4) ---
    logging.info("Visualizing neuron activation gradients (simplified)...")
    num_neurons_to_viz = 10
    target_conv_idx = 4 # CONV5 (0-indexed)

    # Check actual number of filters in the target layer
    conv_layers = [l for l in model.layers if isinstance(l, nn.Conv2d)]
    if target_conv_idx >= len(conv_layers):
        logging.warning(f"Target Conv layer index {target_conv_idx} out of bounds.")
    else:
        filters_in_target = conv_layers[target_conv_idx].out_channels
        num_neurons_to_viz = min(num_neurons_to_viz, filters_in_target)
        logging.info(f"Targeting {num_neurons_to_viz} neurons in CONV layer {target_conv_idx}.")

        ncols = 5
        nrows = (num_neurons_to_viz + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3.5))
        axes = axes.flatten() if num_neurons_to_viz > 1 else [axes]

        for i in range(num_neurons_to_viz):
            try:
                gradients = simplified_gradient_visualization(
                    model, sample_image_tensor,
                    target_conv_layer_idx=target_conv_idx, target_neuron_idx=i
                )
                ax = axes[i]
                if gradients is not None:
                    im = ax.imshow(gradients, cmap='hot')
                    ax.set_title(f'Neuron {i}')
                else:
                    ax.set_title(f'Neuron {i}\n(Error)')
                ax.set_xticks([])
                ax.set_yticks([])
            except Exception as e:
                 logging.error(f"Error visualizing neuron {i}: {e}", exc_info=True)
                 axes[i].set_title(f'Neuron {i}\n(Viz Error)')
                 axes[i].axis('off')

        for j in range(num_neurons_to_viz, len(axes)): axes[j].axis('off')
        fig.suptitle(f'Input Gradients for CONV Layer {target_conv_idx} Neurons', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        neuron_save_path = 'neuron_activation_gradients.png'
        try:
            plt.savefig(neuron_save_path, dpi=150)
            wandb.log({"neuron_activation_gradients": wandb.Image(neuron_save_path)})
            logging.info(f"Neuron visualization saved to {neuron_save_path}")
        except Exception as e:
            logging.error(f"Failed to save/log neuron viz: {e}")
        plt.close(fig)

    # --- Finish ---
    wandb.finish()
    logging.info("Wandb visualization run finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize CNN model features")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model checkpoint (.pth)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--img_size", type=int, default=224, help="Image size (must match model if config not saved)")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--wandb_project", type=str, default="CNN-iNaturalist-Scratch", help="Wandb project")

    # --- Model Arch Args (Defaults needed if config missing from checkpoint) ---
    parser.add_argument("--num_filters_base", type=int, default=32, help="Base number of filters")
    parser.add_argument("--filter_size_base", type=int, default=3, help="Filter kernel size")
    parser.add_argument("--filter_organization", type=str, default="same", help="Filter organization")
    parser.add_argument("--activation", type=str, default="relu", help="Activation function")
    parser.add_argument("--dense_neurons", type=int, default=128, help="Dense neurons")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--batch_norm", type=str2bool, default='True', help="Batch normalization used")

    args = parser.parse_args()
    run_visualizations(args)