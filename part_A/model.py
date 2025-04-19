# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Activation function mapping
_ACTIVATIONS = {
    'relu': nn.ReLU,
    'gelu': nn.GELU,
    'silu': nn.SiLU,
    'mish': nn.Mish,
}

class CustomCNN(nn.Module):
    """
    A customizable CNN model with 5 convolutional blocks.
    Each block: Conv2d -> [BatchNorm2d] -> Activation -> MaxPool2d -> [Dropout].
    Followed by fully connected layers.
    """
    def __init__(self, input_channels=3, num_classes=10,
                 filter_sizes=None, # Expect list of 5 sizes
                 num_filters=None,  # Expect list/int for filter counts
                 filter_organization="same", # 'same', 'double', 'half'
                 activation_name='relu', # Name of activation function
                 dense_neurons=128,
                 dropout_rate=0.3,
                 batch_norm=True,
                 img_size=224): # Input image size
        """
        Args see train.py args help. Uses img_size for dynamic calculation.
        """
        super(CustomCNN, self).__init__()

        # --- Parameter Validation & Setup ---
        if filter_sizes is None:
            filter_sizes = [3] * 5
        elif len(filter_sizes) != 5:
            raise ValueError("filter_sizes must be a list of length 5")

        if num_filters is None:
            num_filters_list = [32] * 5 # Default base number
            logging.warning("num_filters not provided, defaulting base to 32")
        elif isinstance(num_filters, int): # Allow passing a single int
             base_filters = num_filters
             if filter_organization == "double":
                num_filters_list = [base_filters * (2**i) for i in range(5)]
             elif filter_organization == "half":
                num_filters_list = [max(1, base_filters // (2**i)) for i in range(5)]
             else: # 'same'
                num_filters_list = [base_filters] * 5
        elif isinstance(num_filters, list) and len(num_filters) == 5:
             num_filters_list = num_filters # Use the provided list directly
        else:
             raise ValueError("num_filters must be an int or a list of length 5")

        if activation_name not in _ACTIVATIONS:
            raise ValueError(f"Unsupported activation: {activation_name}. Choose from {list(_ACTIVATIONS.keys())}")
        activation = _ACTIVATIONS[activation_name]

        logging.info(f"Building CustomCNN: filters={num_filters_list}, sizes={filter_sizes}, "
                     f"activation={activation_name}, dense={dense_neurons}, dropout={dropout_rate}, bn={batch_norm}")

        self.layers = nn.ModuleList()
        in_channels = input_channels
        current_dim = img_size

        # --- Convolutional Blocks ---
        for i in range(5):
            k_size = filter_sizes[i]
            out_channels = num_filters_list[i]
            padding = k_size // 2 # 'same' padding for odd kernels

            conv = nn.Conv2d(in_channels, out_channels, kernel_size=k_size, padding=padding, bias=not batch_norm)
            self.layers.append(conv)

            if batch_norm:
                self.layers.append(nn.BatchNorm2d(out_channels))

            self.layers.append(activation())
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            current_dim = current_dim // 2 # Dimension reduction after pooling

            if dropout_rate > 0:
                self.layers.append(nn.Dropout(dropout_rate)) # Dropout after pooling

            in_channels = out_channels

        # --- Dense Layers ---
        self.final_conv_output_channels = num_filters_list[-1]
        self.final_feature_map_dim = current_dim
        self.flattened_size = self.final_conv_output_channels * self.final_feature_map_dim * self.final_feature_map_dim

        if self.flattened_size == 0:
             raise ValueError(f"Flattened size is zero. Input img_size {img_size} might be too small for 5 pooling layers.")

        self.fc1 = nn.Linear(self.flattened_size, dense_neurons)
        self.dropout_fc = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(dense_neurons, num_classes)

        # Store config for calculations
        self.config = {
            'input_channels': input_channels, 'num_classes': num_classes,
            'filter_sizes': filter_sizes, 'num_filters': num_filters_list,
            'activation_name': activation_name, 'dense_neurons': dense_neurons,
            'dropout_rate': dropout_rate, 'batch_norm': batch_norm, 'img_size': img_size
        }

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = x.view(x.size(0), -1) # Flatten

        x = F.relu(self.fc1(x)) # Keep ReLU common here, or use self.config['activation_name']
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x

    def count_parameters(self):
        """Count the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def calculate_computations(self):
        """
        Estimate FLOPs (Multiply-Accumulate operations * 2). Simplified.
        Q1: What is the total number of computations? (symbolic answer needed from this logic)
        Assume m filters (out_channels) in each layer of size k*k, n neurons (dense_neurons).
        """
        total_macs = 0
        h, w = self.config['img_size'], self.config['img_size']
        in_channels = self.config['input_channels']
        num_filters = self.config['num_filters'] # This is the list
        filter_sizes = self.config['filter_sizes']

        # Conv layers MACs: K*K * Cin * Cout * Hout * Wout
        for i in range(5):
            k = filter_sizes[i]
            cout = num_filters[i]
            # Output size after pool
            h_out, w_out = h // 2, w // 2
            macs_conv = (k * k * in_channels * cout * h_out * w_out)
            total_macs += macs_conv
            in_channels = cout
            h, w = h_out, w_out
            # Add BatchNorm MACs (approx 2*Cout*Hout*Wout per element * 2 ops = 4*...) if BN enabled
            if self.config['batch_norm']:
                total_macs += 2 * cout * h_out * w_out # Approximation for scale and shift

        # Dense layers MACs: Cin * Cout
        total_macs += self.flattened_size * self.config['dense_neurons']
        total_macs += self.config['dense_neurons'] * self.config['num_classes']

        # FLOPs ≈ 2 * MACs
        total_flops = 2 * total_macs
        return total_flops