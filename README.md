# VGG16 Pruning and Genetic Algorithm Optimization

This repository contains a research pipeline for training, pruning, and optimizing a VGG16 model on the CIFAR-10 dataset using traditional magnitude-based methods and Genetic Algorithms (GA).

---

## 💻 Development Environment & Resources

All experiments conducted in this work were performed using the hardware and software configuration detailed below:

### Hardware Specifications
* **Processor (CPU):** Intel Core i7-10750H (2.6 GHz base frequency)
* **Memory (RAM):** 16 GB RAM
* **Graphics Processing Unit (GPU):** NVIDIA GeForce RTX 2070 (8 GB VRAM)

> **Note on VRAM & Batch Size:** The 8 GB of VRAM provided by the RTX 2070 was a critical factor in determining the maximum batch sizes used during model training, particularly for memory-intensive architectures like VGG-16. Larger batch sizes accelerate training by enabling more parallelization, but must be balanced with the available VRAM to prevent out-of-memory errors during the forward and backward passes of training. Despite the RTX 2070 not being the latest GPU model at the time of conducting this research, it made efficient use of CUDA cores for parallel processing during training and inference to maintain modest computation times.

### Software Environment & Libraries
* **Operating System (OS):** Windows 11
* **Integrated Development Environment (IDE):** PyCharm
* **Python Version:** 3.8
* **Environment Management:** Anaconda

### Core Deep Learning Framework
**PyTorch** was selected over TensorFlow primarily due to its robust `torch.nn.utils.prune` module. This toolkit supports both structured and unstructured pruning methods and allows customization of pruning techniques without requiring low-level code implementation—a critical advantage for generating our custom pruning masks.

The primary PyTorch modules utilized are:
1. `torchvision`: Computer vision functionality and datasets.
2. `torch.nn`: Neural network components and layers.
3. `torch.optim`: Optimization algorithms.
4. `torch.cuda.amp`: Mixed precision training with `GradScaler` and `autocast` to optimize VRAM consumption.
5. `torch.nn.utils.prune`: Network pruning operations.

### Supporting Libraries
* **NumPy:** Essential for numerical computing and efficient array operations.
* **Scikit-Learn:** Used for calculating accuracy metrics, evaluating model performance, and generating matrix data.
* **tqdm:** Provides clean progress bar visualization for training loops to closely monitor long training sessions.
* **Pickle:** Enables object serialization, which is key for model saving and creating training checkpoints.

---

## 🚀 Execution Pipeline

### 01. Training & Evaluation
* **`01_train_basemodel.py`**: Trains a VGG16 model from scratch on the CIFAR-10 dataset.
    * **Hyperparameters**:
        * **Optimizer**: SGD (Momentum: 0.9)
        * **Learning Rate**: 0.01 (managed with a `StepLR` scheduler)
        * **Batch Size**: 64
        * **Weight Decay**: 5e-4
        * **Precision**: Mixed Precision (FP16) via `torch.cuda.amp`
* **`01_test_model.py`**: Loads a saved `.pth` model checkpoint and evaluates it on the full CIFAR-10 test set. It also generates and saves a **Confusion Matrix** to map out and visualize classification performance across all 10 classes.

### 02. Traditional Pruning
* **`02_prune_basemodel.py`**: Loads the baseline `.pth` model and applies unstructured pruning to its convolutional layers.
    * **Methods**: L1-norm pruning and Random pruning.
    * **Sparsity Levels**: Specifically evaluates the network at **25%**, **50%**, and **75%** sparsity targets to observe accuracy degradation curves.

### 03. Evolutionary Optimization
* **`03_genetic_algorithm.py`**: The core optimization script. It runs a Genetic Algorithm (GA) to discover the optimal pruning mask (represented as a chromosome) on the pretrained model. 
    * It evolves a population of masks over multiple generations to find a sparse filter structure that maximizes validation accuracy.
    * Includes automated checkpointing via `checkpoint.pkl` to safely save progress and resume long-running evolutionary steps.

### 04. Recovery & Fine-tuning
* **`04_finetune.py`**: Takes the optimal pruning mask obtained from the GA's final convergence, applies it structurally to the VGG16 model, and fine-tunes the remaining weights to recover lost performance.
* **`04_rewind.py`**: Implements **Weight Rewinding**. Instead of standard fine-tuning, it applies the GA-derived mask and resets the remaining unpruned weights back to their exact values from an earlier training epoch before retraining them.

---

## 📁 Utility Scripts
* `utils.py`: Contains general helper functions for data formatting, model validation, and tensor mask extraction.
* `ga_utils.py`: Contains evolutionary logic components including crossover rules, mutations, and filter-importance calculation formulas.

---

## 📊 Experimental Results

Below are the benchmark data tables compiled directly from experiments evaluating traditional methods (L1-Norm, Random) against the custom Genetic Algorithm under multiple operational constraints.

### 1. Fine-tuned Pruning Results
In this strategy, the remaining weights are fine-tuned after applying the pruning filters to recover baseline accuracy definitions.

| Pruning Method | Sparsity (%) | Accuracy (%) | NumOps |
| :--- | :---: | :---: | :---: |
| **L1-Norm** | 25% | 89.82% | 11,032,848 |
| | 50% | 86.70% | 7,355,232 |
| | 75% | 85.28% | 3,677,616 |
| **Random** | 25% | 86.52% | 11,032,848 |
| | 50% | 86.48% | 7,355,232 |
| | 75% | 10.00% *(Collapsed)* | 3,677,616 |
| **Genetic Algorithm** | 25% | **90.56%** | 11,700,540 |
| | 50% | **88.31%** | **7,145,838** |

* **Observation:** Random pruning completely collapses to a baseline noise value (10.00%) at high compression (75%), demonstrating that performance at extreme sparsity is structural. The Genetic Algorithm consistently outperforms both standard heuristics in fine-tuned environments, while discovering a matrix with significantly reduced operational counts (`NumOps`) at 50% sparsity.

### 2. Reinitialized Pruning Results
In this strategy, the structural topology generated by the pruning mask is isolated, and the remaining weights are completely reinitialized and trained from scratch for 100 epochs.

| Pruning Method | Sparsity (%) | Accuracy (%) | NumOps |
| :--- | :---: | :---: | :---: |
| **L1-Norm** | 25% | **94.70%** | 11,032,848 |
| | 50% | **92.52%** | 7,355,232 |
| | 75% | **91.50%** | 3,677,616 |
| **Random** | 25% | 92.36% | 11,032,848 |
| | 50% | 92.25% | 7,355,232 |
| | 75% | 10.00% *(Collapsed)* | 3,677,616 |
| **Genetic Algorithm** | 25% | 92.02% | 11,700,540 |
| | 50% | 88.17% | **7,145,838** |

* **Observation:** Reinitialization allows the architecture to escape local minima inherited from pre-training. While L1-Norm provides a strong structural baseline in the scratch setting, the GA remains highly competitive while aggressively mapping configurations to limit operations dynamically.

### 3. GA Structural Progression Optimization
A comparative look showing how the Genetic Algorithm explicitly optimizes masks over training generations.

* **25% Target Global Sparsity**
  * *Generation 10 Best:* 91.71% Accuracy | 11,843,280 NumOps (Reinitialized)
  * *Final Convergence Best:* **92.02% Accuracy** | **11,700,540 NumOps** (Reinitialized)
* **50% Target Global Sparsity**
  * *Generation 10 Best:* 88.22% Accuracy | 7,265,538 NumOps (Reinitialized)
  * *Final Convergence Best:* 88.17% Accuracy | **7,145,838 NumOps** (Reinitialized)
