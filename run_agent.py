# run_agent.py
import wandb
import argparse
import sys
import os
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Import the training function ---
# Adjust path if your structure is different.
# This assumes run_agent.py is in Assignment_2 and sweep.py is in Assignment_2/part_A/
try:
    from part_A.sweep import train_sweep_trial
except ImportError as e:
    logging.error(f"Failed to import train_sweep_trial from part_A.sweep: {e}")
    logging.error("Make sure run_agent.py is in the correct directory (e.g., Assignment_2)")
    logging.error("and the path 'part_A.sweep' is correct.")
    sys.exit(1)

# --- Parse arguments needed by BOTH the agent and the training function ---
def parse_args():
    parser = argparse.ArgumentParser(description="Run W&B agent for a specific sweep ID.")
    parser.add_argument("sweep_id", help="The full W&B sweep ID (e.g., entity/project/sweep_id)")
    parser.add_argument("--count", type=int, default=None, help="Number of runs for THIS agent instance to execute (optional)")

    # --- Include NON-SWEEP arguments required by train_sweep_trial ---
    # These are passed down via the args object within the lambda
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument("--img_size", type=int, default=224, help="Target image size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split fraction")
    # We don't need wandb_project/entity here, they are part of sweep_id
    # We don't need hyperparameter args here (like --epochs, --batch_size),
    # as those will come from wandb.config inside train_sweep_trial

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # --- Basic Validation ---
    if not os.path.isdir(args.data_dir):
        logging.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)
    if len(args.sweep_id.split('/')) != 3:
        logging.error(f"Invalid sweep_id format: {args.sweep_id}. Expected 'entity/project/sweep_id'.")
        sys.exit(1)

    logging.info(f"Starting W&B agent for sweep: {args.sweep_id}")
    logging.info(f"Agent will run max {args.count if args.count else 'unlimited'} trials.")

    try:
        # Call wandb.agent, passing the imported training function.
        # The lambda allows us to pass the parsed 'args' object to train_sweep_trial.
        wandb.agent(args.sweep_id,
                    function=lambda: train_sweep_trial(args),
                    count=args.count) # Pass count if provided
    except Exception as e:
        logging.error(f"W&B Agent execution failed: {e}", exc_info=True)
        sys.exit(1)

    logging.info("W&B agent finished.")