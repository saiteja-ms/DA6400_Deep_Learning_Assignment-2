# M S SAI TEJA | ME21B171

DA6401 Assignment - 2
*(Note: Data directory like `inaturalist_12K` is assumed to be present locally but not committed to Git)*

## Link to the wandb report: [![W&B Report](https://img.shields.io/badge/Report-W%26B-blue?logo=weightsandbiases)](https://wandb.ai/teja_sai-indian-institute-of-technology-madras/CNN_FROM_SCRATCH_SWEEP/reports/DA6400-Assignment-2-Report--VmlldzoxMjM2NjA5Mw?accessToken=ine6bzgu1w7bhdv67yxdgshyjery80su5p9jt9n17s2pucb7opfa7j0ralxw9pye)
## Link to the github repo: [GitHub Repo](https://github.com/saiteja-ms/DA6400_Deep_Learning_Assignment-2)
---

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/[your-user-id]/DA6400_Deep_Learning_Assignment-2.git
    cd DA6400_Deep_Learning_Assignment-2
    ```

2.  **Create a Python Environment:** (Using Conda is recommended)
    ```bash
    conda create -n DLA2 python=3.11 # Or your preferred Python version
    conda activate DLA2
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    # Ensure PyTorch+CUDA is installed correctly for your system
    # Refer to: https://pytorch.org/get-started/locally/
    # Example (check website for command matching your CUDA version):
    # conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    ```

4.  **Download Dataset:**
    *   Download the iNaturalist subset provided for the assignment.
    *   Extract it into the root of this repository (or update paths in the scripts). Ensure you have the following structure:
        ```
        ./inaturalist_12K/
            ├── train/
            │   ├── class_A/
            │   └── ...
            └── test/
                ├── class_A/
                └── ...
        ```

5.  **Login to Weights & Biases:**
    ```bash
    wandb login
    ```
    Follow the prompts to enter your API key.

---

## Running the Code

**Important:** Activate your conda environment (`conda activate DLA2`) before running any scripts. All commands should be run from the root directory (`DA6400_Deep_Learning_Assignment-2`).

### Part A: Training from Scratch

1.  **Run Hyperparameter Sweep:**
    *   This script (`part_A/sweep.py`) defines the sweep config, creates the sweep on W&B, and runs the agent to execute training trials.
    ```bash
    python part_A/sweep.py --data_dir ./inaturalist_12K --count 50 --wandb_project "YourProject_PartA_Sweep"
    ```
    *   `--count`: Number of trials for the agent to run.
    *   `--wandb_project`: Name for the W&B project.
    *   Monitor the sweep progress on the W&B dashboard link provided.

2.  **Evaluate Best Model:**
    *   Identify the best run from the sweep in the W&B UI.
    *   Download its `best_model.pth` artifact or find the local path.
    *   Run the evaluation script:
    ```bash
    python part_A/evaluate.py --data_dir ./inaturalist_12K --model_path path/to/best_partA_model.pth --wandb_project "YourProject_PartA_Eval"
    ```
    *   *(Note: Add necessary architecture args if the model checkpoint doesn't contain the config).*
    *   This logs test accuracy and a prediction grid to W&B.

3.  **Optional Visualizations:**
    ```bash
    python part_A/visualize.py --data_dir ./inaturalist_12K --model_path path/to/best_partA_model.pth --wandb_project "YourProject_PartA_Viz"
    ```
    *   *(Note: Add architecture args if needed).*
    *   Logs filter/neuron visualizations to W&B.

### Part B: Fine-tuning ResNet50

1.  **Create & Run Fine-tuning Sweep:**
    *   This script (`part_B/sweep_b.py`) defines the fine-tuning sweep config (strategies, LRs, etc.), creates the sweep on W&B, and runs the agent.
    ```bash
    python part_B/sweep_b.py --data_dir ./inaturalist_12K --count 20 --wandb_project "YourProject_PartB_Sweep"
    ```
    *   `--count`: Number of fine-tuning trials for the agent to run.
    *   `--wandb_project`: Use the *same project* as Part A if you want to compare runs easily, or a different one.
    *   Monitor the sweep on W&B. It will explore different `finetune_strategy` values.

2.  **Evaluate Best Fine-tuned Model (Optional but Recommended):**
    *   Identify the best run from the Part B sweep.
    *   Download its best model weights (you might need to modify `train_b.py` to save the best model state like in Part A, or retrieve it via W&B artifacts if logged).
    *   Create an `evaluate_b.py` script (similar structure to `part_A/evaluate.py` but loading `ResNet50`, replacing the head, and loading the fine-tuned weights) or add an evaluation mode to `train_b.py`.
    *   Run evaluation:
    ```bash
    # Example command if you create evaluate_b.py
    # python part_B/evaluate_b.py --data_dir ./inaturalist_12K --model_path path/to/best_partB_model.pth --num_classes 10 ...
    ```

---

## Results and Analysis

Detailed results, observations, sweep plots (Parameter Importance, Parallel Coordinates, etc.), and comparisons between training from scratch (Part A) and fine-tuning (Part B) can be found in the associated Weights & Biases Report:


Key findings include:
*   The best validation accuracy obtained by CNN model trained from scratch is approx. 40% and its test accuracy is 
*   The best validation accuracy obtained by finetuning the pretrained ResNet model is approx. 78.5% and its test accuracy is 


---

## Dependencies

See `requirements.txt`. Key libraries include:
*   PyTorch & Torchvision
*   NumPy
*   Weights & Biases (`wandb`)
*   Pillow (PIL)
*   tqdm
*   PyYAML (if using YAML for sweeps)

