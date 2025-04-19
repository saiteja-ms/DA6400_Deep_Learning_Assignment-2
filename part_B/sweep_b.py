# sweep_b.py
# Script to create the W&B sweep configuration for Part B

import wandb
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def create_part_b_sweep(project_name, entity_name):
    """Defines and initializes the W&B sweep for fine-tuning."""

    sweep_config_part_b = {
        'method': 'bayes',
        'metric': {
            'name': 'best_val_accuracy', # Target summary metric
            'goal': 'maximize'
        },
        'parameters': {
            # --- Fine-tuning Strategy ---
            'finetune_strategy': {
                'values': ['feature_extract', 'unfreeze_last_k', 'finetune_all']
            },
            'unfreeze_layers': { # K for unfreeze_last_k
                'values': [1, 2, 3]
            },
            # --- Optimizer ---
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 5e-3
            },
            'weight_decay': {
                'values': [0, 0.0001, 0.0005]
            },
            # --- Training ---
            'batch_size': {
                'values': [16, 32] # Adjust based on GPU memory
            },
            'epochs': {
                'value': 15 # Fixed epochs per trial
            },
            # --- Fixed seed for all trials (can be useful) ---
            'seed': {
                'value': 42
            }
            # Note: data_dir, img_size etc. are passed via agent script, not swept
        }
        # Optional: Early termination
        # 'early_terminate': { 'type': 'hyperband', 'min_iter': 5 }
    }

    logging.info("Sweep configuration defined:")
    # print(sweep_config_part_b) # Optionally print the config

    try:
        sweep_id = wandb.sweep(sweep_config_part_b, project=project_name, entity=entity_name)
        logging.info(f"Sweep created successfully!")
        print(f"\n--- Sweep ID: {sweep_id} ---")
        print(f"--- Sweep URL: https://wandb.ai/{entity_name or '[entity]'}/{project_name}/sweeps/{sweep_id} ---")
        print(f"\nTo run the agent (replace placeholders):")
        print(f"wandb agent {entity_name or '[entity]'}/{project_name}/{sweep_id}")
        print("\nAlternatively, use the agent script/notebook with this Sweep ID.")
        return sweep_id
    except Exception as e:
        logging.error(f"Failed to create wandb sweep: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create W&B Sweep for Part B Fine-tuning")
    # Use the same project as Part A to see all runs together
    parser.add_argument('--project', type=str, default='CNN-iNaturalist-Scratch', help='W&B Project Name')
    parser.add_argument('--entity', type=str, default=None, help='W&B Entity (username or team). Often inferred.')
    args = parser.parse_args()

    create_part_b_sweep(args.project, args.entity)