import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import time
import torch.backends.cudnn as cudnn
import random
import time
from ga_utils import *
from utils import *
import torch.nn.utils.prune as prune
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import vgg16
from torch.cuda.amp import GradScaler, autocast
from  utils import create_tensor_from_mask
import torch.nn.utils.prune as prune
# Function to visualize the filters as grids
from torch.utils.data import Subset, DataLoader

# Function to save the state
def save_state(file_path, generation, population, bestAmongAllFitness, bestAmongAllFitnessGene):
    state = {
        "generation": generation,
        "population": population,
        "bestAmongAllFitness": bestAmongAllFitness,
        "bestAmongAllFitnessGene": bestAmongAllFitnessGene
    }
    with open(file_path, "wb") as f:
        pickle.dump(state, f)
    print(f"State saved at generation {generation} to {file_path}")

# Function to load the state
def load_state(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            state = pickle.load(f)
        print(f"State loaded from {file_path}")
        return state
    else:
        print(f"No checkpoint found at {file_path}. Starting from scratch.")
        return None

# Function to load the model
def load_model(path='vgg16_cifar10_scratch.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.vgg16()
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)
    model.load_state_dict(torch.load(path))
    model = model.to(device)
    print(f'Model loaded from {path}')
    return model

# Function to create balanced subsets
def create_balanced_subset(dataset, subset_fraction):
    targets = np.array(dataset.targets)  # Get class labels
    indices = np.arange(len(targets))  # All indices
    _, balanced_indices = train_test_split(
        indices, test_size=subset_fraction, stratify=targets, random_state=42
    )
    return Subset(dataset, balanced_indices)

# Training function
def train(epoch, model, trainloader, device, optimizer, criterion, scaler):
    start_time = time.time()
    model.train()
    running_loss = 0.0
    train_loader_tqdm = tqdm(trainloader, desc=f"Epoch {epoch} Training", leave=True)

    for batch_idx, (inputs, targets) in enumerate(train_loader_tqdm):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # Mixed precision block
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # Validate loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Invalid loss value: {loss}. Skipping batch.")
            continue

        # Scale the loss and backpropagate
        scaler.scale(loss).backward()

        # Unscale gradients and check if they exist
        scaler.unscale_(optimizer)
        gradients_exist = any(param.grad is not None and torch.any(param.grad != 0) for param in model.parameters())

        if gradients_exist:
            scaler.step(optimizer)  # Update model parameters
        else:
            print("Skipping optimizer step due to no gradients.")

        # Update scaler
        scaler.update()

        running_loss += loss.item()
        avg_loss = running_loss / (batch_idx + 1)
        train_loader_tqdm.set_postfix(loss=avg_loss)

    end_time = time.time()
    epoch_time = end_time - start_time
    print(f'Epoch {epoch + 1} completed in {epoch_time:.2f} seconds')

# Testing function
def test(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    test_loader_tqdm = tqdm(testloader, desc="Testing", leave=True)  # Add progress bar for testing
    with torch.no_grad():
        for inputs, targets in test_loader_tqdm:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            accuracy = 100. * correct / total
            test_loader_tqdm.set_postfix(accuracy=accuracy)  # Update progress bar with accuracy
    return accuracy

if __name__ == '__main__':
    # Define device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Define transformations for the CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224 as VGG16 expects
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Create balanced subsets (25% of the original)
    train_subset = create_balanced_subset(trainset, 0.25)
    test_subset = create_balanced_subset(testset, 0.25)

    # Create DataLoaders for the subsets
    trainloader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=4)
    testloader = DataLoader(test_subset, batch_size=64, shuffle=False, num_workers=4)

    # Initialize a VGG16 model
    model = load_model('vgg16_cifar10_scratch.pth')

    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

        # Base_line_accuracy = test_model(model, testloader,device)
    Base_line_accuracy = 91.05
    print("Base_line_accuracy")
    print(Base_line_accuracy)
    Base_line_numOps = 0
    for layer in model.features:
        if isinstance(layer, torch.nn.Conv2d):
            filters = layer.out_channels
            kernel_size = layer.kernel_size
            input_channels = layer.in_channels
            output_height = layer.output_padding[0] if layer.output_padding[0] != 0 else layer.stride[0]
            output_width = layer.output_padding[1] if layer.output_padding[1] != 0 else layer.stride[1]

            # Number of operations
            filter_ops = (kernel_size[0] * kernel_size[1] * input_channels) * (output_height * output_width)
            Base_line_numOps += filter_ops * filters
    print("Base_line_numOps")
    print(Base_line_numOps)

    # Define genetic algorithm parameters
    gene_arch = architecture_genome(model)
    populationSize = 20
    mutationRate = 0.01
    maxGeneration = 40
    sparseCriteria = 0.5

    filename = f"./Best_Gene_sparsity{int(sparseCriteria * 100)}.txt"

    # Load or initialize the algorithm's state
    state = load_state("checkpoint.pkl")
    if state:
        generation = state["generation"]
        population = state["population"]
        bestAmongAllFitness = state["bestAmongAllFitness"]
        bestAmongAllFitnessGene = state["bestAmongAllFitnessGene"]
    else:
        bestGeneFile = open(filename, "a")
        bestGeneFile.truncate(0)
        bestGeneFile.close()
        generation = 1
        population = generateInitialPopulation(populationSize, gene_arch, sparseCriteria, model)
        bestAmongAllFitness = 0
        bestAmongAllFitnessGene = []



    # Main loop for the genetic algorithm
    while generation <= maxGeneration:
        print(f"Starting Generation: {generation}")
        bestFitness = 0
        avgFitness = 0
        candidate_number = 1
        bestGeneFile = open(filename, "a")
        for candidate in population:
            print("Candidate : ", candidate_number)
            i = 0
            numOps = 0
            pruned_filters = 0
            # Reload the model for each candidate
            model = load_model('vgg16_cifar10_scratch.pth')
            if device == 'cuda':
                model = torch.nn.DataParallel(model)
                cudnn.benchmark = True

            for layer in model.features:
                if isinstance(layer, torch.nn.Conv2d):
                    # Custom_mask = torch.randint(0, 2, size=layer.weight.shape)
                    layer_mask = torch.tensor(candidate.chromosome[i]).to(device)
                    # Get the shape of the current layer's weights
                    #                  print(layer_mask)

                    # Create the tensor based on the mask
                    masked_tensor = create_tensor_from_mask(layer_mask, layer).to(device)
                    # calcujlate nops
                    filters = layer.out_channels
                    kernel_size = layer.kernel_size
                    input_channels = layer.in_channels
                    output_height = layer.output_padding[0] if layer.output_padding[0] != 0 else layer.stride[0]
                    output_width = layer.output_padding[1] if layer.output_padding[1] != 0 else layer.stride[1]

                    # Number of operations
                    count_of_ones = candidate.chromosome[i].count(1)
                    filter_ops = (kernel_size[0] * kernel_size[1] * input_channels) * (output_height * output_width)
                    numOps += filter_ops * count_of_ones

                    # Calculate the sparsity based on the binary chromosome
                    # Number of pruned filters is the count of zeros in the chromosome
                    pruned_filters = candidate.chromosome[i].count(0) + pruned_filters

                    # Total filters is simply the length of the chromosome
                    #  total_filters = len(candidate.chromosome[i])

                    # Calculate sparsity as the fraction of pruned filters
                    #  sparsity = pruned_filters / total_filters

                    # Print the sparsity for the current layer
                    #    print(f"Sparsity for Layer {i + 1} ({layer}): {sparsity:.4f}")

                    # Apply the mask to the weights
                    with torch.no_grad():
                        prune.custom_from_mask(layer, name='weight', mask=masked_tensor)
                        prune.custom_from_mask(layer, name='bias', mask=layer_mask)

                    i = i + 1

            total_filter_num = sum(len(layer) for layer in candidate.chromosome)
            total_sparsity = pruned_filters / total_filter_num
            print("NumOps: ", numOps)
            print("Total Sparsity:", total_sparsity)

                # Main training loop
                # Loss and optimizer

            # Train and evaluate the model
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)
            scaler = GradScaler()

            for epoch in range(3):
                train(epoch, model, trainloader, device, optimizer, criterion, scaler)
                model_acc = test(model, testloader, device)
                #model_acc = random.randint(0, 100)
                #time.sleep(0.1)
            print("Model Accuracy :", model_acc)
            candidate.fitness = (0.4 * model_acc/Base_line_accuracy) - (0.6 * numOps/Base_line_numOps) + 0.6

            print("Fitness : ", candidate.fitness)

            if (candidate.fitness > bestFitness):
                bestFitness = candidate.fitness
                bestFitnessGene = candidate.chromosome.copy()
                if (bestFitness > bestAmongAllFitness):
                    bestAmongAllFitness = bestFitness
                    bestAmongAllFitnessGene = candidate.chromosome.copy()
            avgFitness += candidate.fitness

            candidate_number = candidate_number + 1

        # Log the generation results
        bestGeneFile.write(
            f"Generation: {generation}\tBest Fitness: {bestFitness}\tAverage Fitness: {avgFitness / len(population)}\nBest Gene: {bestFitnessGene}\n\n")
        print(
            f"Generation: {generation}\tBest Fitness: {bestFitness}\tAverage Fitness: {avgFitness / len(population)}")

        # GENERATE NEW_POPULATION
        if generation > 1:
            # FILTER POPULATION
            ranked_population = sorted(population, key=lambda candidate: candidate.fitness, reverse=True)
            # Select the top `populationSize` individuals
            population = ranked_population[:populationSize]

        # SELECTION
        print("Generating New Population")
        population = generateNewPopulation(population, mutationRate, gene_arch, sparseCriteria, model)
        generation += 1

        print("Saving state")
        save_state("checkpoint.pkl", generation, population,bestAmongAllFitness, bestAmongAllFitnessGene)

    bestGeneFile = open(filename, "a")
    bestGeneFile.write(f"Best In all Fitness: {bestAmongAllFitness}\nBest In all Gene: {bestAmongAllFitnessGene}")
    bestGeneFile.close()
    print("Algorithm completed.")
