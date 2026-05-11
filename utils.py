import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import re
import time
import torch.backends.cudnn as cudnn
# Apply Mask to Dataset


def random_prune(X, Y, geneticCandidate):
    mask = []
    # print(len(geneticCandidate.chromosome))
    for r in range(len(geneticCandidate.chromosome)):
        if geneticCandidate.chromosome[r] == 0:
            mask.append(r)
    X_new = X.drop(X.columns[mask], axis=1)
    Y_new = Y.drop(Y.columns[mask], axis=1)

    return X_new, Y_new


# TRAIN and TEST MODEL
def train(X, Y, geneticCandidate, train_labels, test_labels):
    global model

    model.fit(X.values, train_labels)
    predictions = model.predict(Y)
    score = accuracy_score(test_labels, predictions)
    print(score)
    geneticCandidate.fitness = score
    print(geneticCandidate.fitness)


def check_if_best(geneticCandidate, model_filename, Best_Model):
    global model
    if geneticCandidate.fitness > Best_Model:
        pickle.dump(model, open(model_filename, 'wb'))


def evaluatePopulation(population):
    for geneticCandidate in population:
        # print(geneticCandidate.chromosome)
        X, Y = train_features, test_features
        X, Y = random_prune(X, Y, geneticCandidate)

        print("Training model and testing candidate")
        train(X, Y, geneticCandidate, train_labels, test_labels)
        check_if_best(geneticCandidate, model_filename,Best_Model)


# Function to test the model on the test dataset
def test_model(model, testloader,device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(total=len(testloader), desc='Testing', unit='batch') as pbar:
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                pbar.update(1)

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the 10000 test images: {accuracy:.2f}%')
    return accuracy

# Function to save the model
def save_model(model, path='vgg16_cifar10.pth'):
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')

# Function to load the model
def load_model(path='vgg16_cifar10.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.vgg16()
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)
    model.load_state_dict(torch.load(path))
    model = model.to(device)
    print(f'Model loaded from {path}')
    return model


# Function to expand the binary mask to the shape (out_channels, in_channels, kernel_size, kernel_size)
def expand_mask(mask, out_channels, in_channels, kernel_size):
    # Ensure the mask is the correct length
    assert len(mask) == out_channels, "Mask length must match the number of output channels"

    # Expand the mask to (out_channels, in_channels, kernel_size, kernel_size)
    if isinstance(kernel_size, tuple):
        k_h, k_w = kernel_size
    else:
        k_h = k_w = kernel_size

    expanded_mask = mask.view(out_channels, 1, 1, 1).expand(out_channels, in_channels, k_h, k_w)
    return expanded_mask


# Function to create the tensor for the Conv2D layer based on the mask
def create_tensor_from_mask(mask, layer):
    # Extract the shape of the weights from the Conv2d layer
    out_channels = layer.out_channels
    in_channels = layer.in_channels
    kernel_size = layer.kernel_size  # could be an int or a tuple

    # Create the expanded mask
    expanded_mask = expand_mask(mask, out_channels, in_channels, kernel_size)

    return expanded_mask


def load_mask_from_file(file_path):
    arrays_list = []  # List to store arrays
    parsing = False  # Flag to track if we are in the parsing state

    # Open the file and iterate over each line
    with open(file_path, 'r') as file:
        for line in file:
            # Check if we found the start of the arrays
            if "Best In all Gene: [" in line:
                parsing = True  # Start parsing
                # Extract everything after "Best In all Gene: ["
                line = line.split("Best In all Gene: [", 1)[1]

            # If we're in the parsing state, look for arrays
            if parsing:
                # Remove unnecessary characters and split the line into arrays using regex
                arrays = re.findall(r'\[([01, ]+)\]', line)

                for array_str in arrays:
                    # Convert the string into an actual list of integers
                    array = [int(num) for num in array_str.split(',')]
                    arrays_list.append(array)

                # Check if the line contains the end of arrays "]]"
                if "]]" in line:
                    break  # Stop parsing

    return arrays_list


