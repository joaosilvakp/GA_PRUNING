import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import math


def visualize_filters(filter_list):
    fig, axes = plt.subplots(len(filter_list), 1, figsize=(15, 12))

    # If there's only one layer, wrap `axes` in a list to handle it uniformly
    if len(filter_list) == 1:
        axes = [axes]

    for layer_idx, (layer_filters, ax) in enumerate(zip(filter_list, axes)):
        num_filters = len(layer_filters)

        # Calculate the sparsity of the layer
        sparsity_percentage = (np.sum(np.array(layer_filters) == 0) / num_filters) * 100

        # Calculate the necessary grid dimensions (rows and columns)
        rows = int(math.ceil(math.sqrt(num_filters)))
        cols = int(math.ceil(num_filters / rows))

        # Reshape the layer's filter array to match the grid size
        grid = np.zeros((rows, cols))  # Initialize grid with zeros (white)
        grid.flat[:num_filters] = layer_filters  # Assign filter values (0 or 1) to the grid

        # Display the grid as a matrix of black and white squares
        ax.imshow(grid, cmap='gray', interpolation='nearest')

        # Add title with sparsity percentage
        ax.set_title(f'Layer {layer_idx + 1} - {num_filters} Filters\nSparsity: {sparsity_percentage:.2f}%')

        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])

    # Adjust layout for better display
    plt.tight_layout()
    plt.show()


def architecture_genome(model):  # function to create the genome that will be used in the GA.
    # List to store the arrays
    filter_arrays = []

    # Iterate over the model layers
    for layer in model.features:
        if isinstance(layer, torch.nn.Conv2d):
            # Create an array with bits (1s and 0s) for the number of filters
            num_filters = layer.out_channels
            filter_array = [1] * num_filters
            filter_arrays.append(filter_array)

    return filter_arrays


def crossover(parent1, parent2):
    child = DNA(len(parent1.chromosome))
    index = int(len(parent1.chromosome) / 2)
    if (random.uniform(0, 1) < 0.5):
        child.chromosome = parent1.chromosome[:index] + parent2.chromosome[index:]
    else:

        child.chromosome = parent2.chromosome[:index] + parent1.chromosome[index:]
    return child


def layer_wise_crossover(child1, child2, parent1, parent2):
    num_layers = len(parent1.chromosome)

    # Iterate through each layer (chromosome) of the VGG model
    for i in range(num_layers):
        length_layer = len(parent1.chromosome[i])
        layer1 = parent1.chromosome[i]  # Layer from parent 1
        layer2 = parent2.chromosome[i]  # Layer from parent 2

        middle = length_layer // 2  # The middle point of the layers

        # Generate a crossover point using Gaussian distribution around the middle
        crossover_point = int(random.gauss(middle, length_layer * 0.1))  # Adjust sigma (spread) as needed

        # Ensure the crossover point is within the valid range
        crossover_point = max(0, min(crossover_point, length_layer - 1))

        # Create two new layers for the children by swapping parts of the layers
        child1_layer = layer1[:crossover_point] + layer2[
            crossover_point:]  # First child gets part from parent1, rest from parent2
        child2_layer = layer2[:crossover_point] + layer1[crossover_point:]

        # Assign the new mixed layer to the child's chromosome
        child1.chromosome[i] = child1_layer
        child2.chromosome[i] = child2_layer

    return child1, child2


def mutation(child, mutationRate):
    for i in range(len(child.chromosome)):
        for j in range(len(child.chromosome[i])):
            if random.uniform(0, 1) < mutationRate:
                if child.chromosome[i][j] == 0:
                    child.chromosome[i][j] = 1
                else:
                    child.chromosome[i][j] = 0


def selection(population):
    while True:
        index = random.randint(0, len(population) - 1)
        if (random.uniform(0, 0.4) < population[
            index].fitness):  # the higher the fitness the greater the chance to get select for next gen
            return population[index]


# CREATE POPULATION
def generateInitialPopulation(populationSize, gene_arch, sparseCriteria, model):
    population = []
    for i in range(populationSize):
        population.append(DNA(gene_arch, sparseCriteria, model))
    return population


def roulette_wheel_selection(population, exclude=None):
    """
    Select an individual from the population using roulette wheel selection.
    Optionally exclude a specific individual.
    """
    total_fitness = sum(individual.fitness for individual in population)

    pick = random.uniform(0, total_fitness)
    current = 0

    for individual in population:
        # Skip the excluded individual if provided
        if exclude and individual == exclude:
            continue
        current += individual.fitness
        if current >= pick:
            return individual


def tournament_selection(population, tournament_size=3):
    """
    Select an individual from the population using tournament selection.
    A subset of the population is randomly chosen, and the best individual is selected.
    """
    # Randomly pick a subset of individuals from the population
    tournament_contestants = random.sample(population, tournament_size)

    # Sort by fitness in descending order to pick the best
    tournament_contestants.sort(key=lambda individual: individual.fitness, reverse=True)

    # Return the best individual with a high probability but allow less fit to win occasionally
    return tournament_contestants[0]  # Or you can introduce some randomness here if you want a probabilistic element


def generateNewPopulation(population, mutationRate, gene_arch, sparseCriteria, model, tournament_size=3):
    newPool = []
    selected_pairs = []  # List to store selected parent pairs

    for _ in range(len(population) // 2):
        children = [DNA(gene_arch, sparseCriteria, model) for _ in range(4)]

        # Select parent1 using tournament selection
        parent1 = tournament_selection(population, tournament_size)

        # Select parent2 using tournament selection, ensuring it's different from parent1
        parent2 = tournament_selection(population, tournament_size)

        # Ensure parents are distinct
        while parent1 == parent2:
            parent2 = tournament_selection(population, tournament_size)

        # Check if this pair has already been selected
        pair = (parent1, parent2)
        pair_reverse = (parent2, parent1)  # Also consider reversed pairs

        # If the pair has been selected, choose new parents
        while pair in selected_pairs or pair_reverse in selected_pairs:
            parent1 = tournament_selection(population, tournament_size)
            parent2 = tournament_selection(population, tournament_size)

            # Ensure parents are distinct
            while parent1 == parent2:
                parent2 = tournament_selection(population, tournament_size)

            # Re-check if the pair has been selected before
            pair = (parent1, parent2)
            pair_reverse = (parent2, parent1)

        # Add the new pair to the selected pairs list
        selected_pairs.append(pair)

        # Generate children using crossover
        children[0], children[1] = layer_wise_crossover(children[0], children[1], parent1, parent2)
        children[2], children[3] = layer_wise_crossover(children[2], children[3], parent1, parent2)

        # Apply mutations and sparsity checks
        for child in children:
            mutation(child, mutationRate)
            child.checkSparse(model)
            newPool.append(child)

    return newPool


class DNA:
    def __init__(self, gene_arch, sparsecriteria, model):
        self.chromosome = self.generateChromosome(gene_arch)
        self.fitness = 0
        self.sparseCriteria = sparsecriteria
        if sparsecriteria != 0:  # 0.50 MEANS 50% SO HALF
            self.checkSparse(model)

    #  visualize_filters(self.chromosome)

    def generateChromosome(self, gene_arch):
        chromosome = []
        for layer in gene_arch:
            # Randomize the values for each filter in the layer
            randomized_layer = [random.randint(0, 1) for _ in layer]
            chromosome.append(randomized_layer)
        return chromosome

    def checkSparse(self, model):
        total_filter_num = sum(len(layer) for layer in self.chromosome)
        active_filter_num = sum(layer.count(1) for layer in self.chromosome)
        sparsity = active_filter_num / total_filter_num
        target_active_filter_num = int(total_filter_num * (1 - self.sparseCriteria))

        if active_filter_num == target_active_filter_num:
            return  # Sparsity is already correct; no adjustment needed

        # Step 2: Calculate filter importance for each layer based on L1 norm
        filter_importances = []

        # Iterate over the convolutional layers in model.features
        # Separate index to track convolutional layers only
        conv_layer_index = 0

        for layer in model.features:
            if isinstance(layer, torch.nn.Conv2d):
                # Only increment the conv_layer_index when it's a Conv2d layer
                # Ensure that the current layer in model corresponds to a layer in the chromosome
                chromosome_layer = self.chromosome[conv_layer_index]

                # Compute L1 norm for each filter (weight) in this Conv2d layer
                filter_norms = layer.weight.data.abs().sum(dim=(1, 2, 3)).cpu().numpy()
                filter_importances.append((filter_norms, conv_layer_index, chromosome_layer))

                # Increment the conv_layer_index after processing a Conv2d layer
                conv_layer_index += 1

        # Step 3: Flatten importance values and indices for sorting
        importance_indices = []
        for filter_norms, layer_idx, chromosome_layer in filter_importances:
            for filter_idx, importance in enumerate(filter_norms):
                importance_indices.append((importance, layer_idx, filter_idx))

        # Step 4: Sort filters by importance (highest to lowest)
        importance_indices.sort(reverse=True, key=lambda x: x[0])  # Sort by importance (descending)

        # Separate active and inactive filters after sorting
        active_filters = [(imp, layer, idx) for imp, layer, idx in importance_indices if
                          self.chromosome[layer][idx] == 1]
        inactive_filters = [(imp, layer, idx) for imp, layer, idx in importance_indices if
                            self.chromosome[layer][idx] == 0]

        if target_active_filter_num > active_filter_num:
            # Need to activate more filters: choose the most important inactive filters
            filters_to_activate = inactive_filters[:target_active_filter_num - active_filter_num]
            for _, layer, idx in filters_to_activate:
                self.chromosome[layer][idx] = 1
        else:
            # Need to deactivate filters: choose the least important active filters
            filters_to_deactivate = active_filters[-(active_filter_num - target_active_filter_num):]
            for _, layer, idx in filters_to_deactivate:
                self.chromosome[layer][idx] = 0
