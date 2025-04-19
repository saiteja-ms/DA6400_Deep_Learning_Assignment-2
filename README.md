# M S SAI TEJA | ME21B171

# DA6401 Assignment - 2: CNN Training & Fine-Tuning
This repository contains the code and results for Assignment 2 of the DA6400 Deep Learning course, focusing on image classification using a subset of the iNaturalist dataset.

## Link to the wandb report: [![W&B Report](https://img.shields.io/badge/Report-W%26B-blue?logo=weightsandbiases)](https://wandb.ai/teja_sai-indian-institute-of-technology-madras/CNN_FROM_SCRATCH_SWEEP/reports/DA6401-Assignment-2-Report--VmlldzoxMjM2NjA5Mw?accessToken=ine6bzgu1w7bhdv67yxdgshyjery80su5p9jt9n17s2pucb7opfa7j0ralxw9pye)
## Link to the github repo: [GitHub Repo](https://github.com/saiteja-ms/DA6401_Deep_Learning_Assignment-2)
---

## Overview

This assignment explores two key aspects of Convolutional Neural Networks (CNNs) for image classification:

1.  **Part A: Training from Scratch:** Building, training, and performing hyperparameter optimization (using Weights & Biases Sweeps) for a custom CNN model directly on the iNaturalist subset.
2.  **Part B: Fine-tuning a Pre-trained Model:** Leveraging a pre-trained ResNet50 model (trained on ImageNet) and fine-tuning it for the iNaturalist subset task using different layer freezing strategies, also explored via W&B Sweeps(but sweep is not expected from the question, but truly our interest) .

*(Note: Data directory like `inaturalist_12K` is assumed to be present locally but not committed to Git)*

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/saiteja-ms/DA6401_Deep_Learning_Assignment-2.git
    cd DA6401_Deep_Learning_Assignment-2
    ```

2.  **Create a Python Environment:** (Using Conda is recommended)
    ```bash
    conda create -n DLA2 python=3.9 # Or your preferred Python version
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
    python part_A/sweep.py --data_dir ./inaturalist_12K --count 50 --wandb_project "CNN_FROM_SCRATCH_SWEEP" --entity "teja_sai-indian-institute-of-technology-madras" --epochs 10
    ```
    *   `--count`: Number of trials for the agent to run.
    *   `--wandb_project` / `--wandb_entity`: Your W&B details.
    *   `--epochs`: Sets the fixed number of epochs per sweep trial.
    *   Monitor the sweep progress on the W&B dashboard link provided (e.g., `https://wandb.ai/teja_sai-indian-institute-of-technology-madras/CNN_FROM_SCRATCH_SWEEP/sweeps/6hw3q9fq`). *You can also run agents using the Kaggle notebook provided in the repo.*

2.  **Train & Evaluate Best Model (Q4):**
    *   **First:** Identify the best hyperparameter configuration from your completed sweep (from the W&B UI).
    *   **Second:** Open train-evaluate-py.ipynb notebook and run it with the best hyperparameter that you have obtained
    *   This notebook trains the model with the hardcoded `BEST_CONFIG`, evaluates it on the test set, saves the best model weights locally and as a W&B artifact, and generates the required 10x3 prediction grid, logging results to the specified W&B project.
    *   I have also given the option of running the above as a script using part_A/evaluatepy, but I have primarily done via train-evaluate-py.ipynb notebook(kaggle was running faster than my laptop gpu).

### Part B: Fine-tuning ResNet50

1.  ### Part B: Fine-tuning ResNet50
    * I have chosen the strategy of freezing all layers except the final classification layer, and so created a script named "part_B/finetune_resnet.py"
    ```bash
    python part_B/finetune_resnet.py --data_dir ./inaturalist_12K  
---
    * It saves the best model parameters based on the best validation accuracy and reports the test accuracy of the best model obtained based on the validation accuracy. 
    You can also convert the same script as a notebook(it has been coded flexibly in that way), also I have provided the notebook as well in the name "finetune_resnet.ipynb".
---

## Results and Analysis

Detailed results, observations, sweep plots (Parameter Importance, Parallel Coordinates, etc.), and comparisons between training from scratch (Part A) and fine-tuning (Part B) can be found in the associated Weights & Biases Report:

Key findings include:
*   The best validation accuracy obtained by CNN model trained from scratch is nearly 40%(around 39%) and its test accuracy is also around 40%.
*   The best validation accuracy obtained by finetuning the pretrained ResNet model is approx. 78.99% and its test accuracy is 79.35%
*   Finetuning the pretrained ResNet model is giving better result when compared to other models.

---

## Dependencies

See `requirements.txt`. Key libraries include:
*   PyTorch & Torchvision
*   NumPy
*   Weights & Biases (`wandb`)
*   Pillow (PIL)
*   tqdm
*   PyYAML (if using YAML for sweeps)
*   matplotlib

