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
from GA import *
from utils import *
import torch.nn.utils.prune as prune
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns


# Function to generate and display a confusion matrix
def generate_confusion_matrix(model, testloader, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in tqdm(testloader, desc='Collecting predictions', unit='batch'):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix for CIFAR-10')
    plt.tight_layout()
    plt.savefig('cifar10_confusion_matrix.png')
    plt.show()

    return cm

# Function to test the model on the test dataset
def test_model(model, testloader):
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
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)

    # Class names for CIFAR-10
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Initialize a VGG16 model
    model = load_model(path='vgg16_cifar10_scratch.pth')

    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    # Test the model and get accuracy
    test_accuracy = test_model(model, testloader)

    # Generate and display confusion matrix
    confusion_mat = generate_confusion_matrix(model, testloader, class_names)

    print("Testing complete! Confusion matrix saved as 'cifar10_confusion_matrix.png'")

    #test_model(model, testloader) #accuracy =79.32%