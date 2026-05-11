
# VGG16 Pruning and Genetic Algorithm Optimization

This repository contains a research pipeline for training, pruning, and optimizing a VGG16 model on the CIFAR-10 dataset using traditional pruning methods and Genetic Algorithms (GA).

## 🚀 Execution Pipeline

### 01. Training & Evaluation
* **`01_train_basemodel.py`**: Trains a VGG16 model from scratch on CIFAR-10.
    * **Hyperparameters**:
        * **Optimizer**: SGD (Momentum: 0.9)
        * **Learning Rate**: 0.01 (with StepLR scheduler)
        * **Batch Size**: 64
        * **Weight Decay**: 5e-4
        * **Precision**: Mixed Precision (FP16) via `torch.cuda.amp`
* **`01_test_model.py`**: Loads a `.pth` model checkpoint and evaluates it on the full CIFAR-10 test set. It also generates and saves a **Confusion Matrix** to visualize classification performance.

### 02. Traditional Pruning
* **`02_prune_basemodel.py`**: Prunes the convolutional layers of the loaded `.pth` model using L1-norm and random pruning techniques.
    * **Sparsity Levels**: Specifically tests the model at 25%, 50%, and 75% sparsity to observe accuracy degradation.

### 03. Evolutionary Optimization
* **`03_genetic_algorithm.py`**: Runs a Genetic Algorithm (GA) to find the optimal pruning mask (chromosome) for the pretrained model. 
    * It evolves a population of masks to find a configuration that maximizes accuracy while maintaining desired sparsity.
    * Supports checkpointing via `checkpoint.pkl` to resume long-running experiments.

### 04. Recovery & Fine-tuning
* **`04_finetune.py`**: Applies the final optimal mask obtained from the GA's convergence and fine-tunes the remaining weights to recover accuracy.
* **`04_rewind.py`**: Implements weight rewinding by applying the GA-derived mask and resetting unpruned weights to their values from an earlier training state before retraining.
