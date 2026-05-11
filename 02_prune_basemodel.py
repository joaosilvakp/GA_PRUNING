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
from ga_utils import *
from utils import *
import torch.nn.utils.prune as prune
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
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

# LOAD PRE-TRAINED MODEL
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


def train(epoch, model, trainloader, device, optimizer, criterion, scaler):
    model.train()
    running_loss = 0.0
    train_loader_tqdm = tqdm(trainloader, desc=f"Epoch {epoch} Training", leave=True)  # Add progress bar for training
    for batch_idx, (inputs, targets) in enumerate(train_loader_tqdm):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        with autocast():  # Enable mixed precision
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        avg_loss = running_loss / (batch_idx + 1)
        train_loader_tqdm.set_postfix(loss=avg_loss)  # Update progress bar with average loss

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

# Function to prune Conv2d layers using L1 structured pruning
def prune_vgg16_conv_layers(model, amount):
    pruning_mask = []  # To store the pruning status for each layer
    sparsity_report = {}  # To store sparsity percentages for each layer

    for layer_idx, layer in enumerate(model.features):
        if isinstance(layer, nn.Conv2d):
            # Use torch's built-in pruning for structured pruning by filter
            prune.ln_structured(layer, name="weight", amount=amount, n=1, dim=0)

            # Generate a pruning mask for this layer
            mask = (layer.weight_mask.sum(dim=(1, 2, 3)) > 0).int().tolist()  # 1 for retained, 0 for pruned
            pruning_mask.append(mask)

            # Calculate sparsity
            num_filters = len(mask)
            num_pruned = mask.count(0)
            sparsity = 100 * num_pruned / num_filters
            sparsity_report[f"Layer {layer_idx}"] = f"{sparsity:.2f}%"

            # Apply pruning permanently
            prune.remove(layer, "weight")

    # Print sparsity report
    print("Sparsity Report:")
    for layer, sparsity in sparsity_report.items():
        print(f"{layer}: {sparsity}")

    return model, pruning_mask


print("25 % ")

# Training and testing process in a loop
model = load_model('vgg16_cifar10_scratch.pth')
if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
#model_acc = test_model(model, testloader, device)
#print(model_acc)
# Initialize a VGG16 model

model, pruning_mask = prune_vgg16_conv_layers(model, amount=0.25)
print(pruning_mask)
# Test the model after pruning
print("Testing model after pruning")
model_acc = test_model(model, testloader, device)
print(model_acc)

# Fine-tune the pruned model
# Define loss function and optimizer
path = f'vgg16_cifar10_prune_iter_L1_25.pth'
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
scaler = GradScaler()  # For mixed precision

# Main training loop
num_epochs = 50
best_acc = 0.0
for epoch in range(num_epochs):
    train(epoch, model, trainloader, device, optimizer, criterion, scaler)
    acc = test(model, testloader, device)
    scheduler.step()
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), path)
        print(f"Model saved with accuracy: {best_acc:.2f}%")

##############
print("50 % ")
# Training and testing process in a loop
model = load_model('vgg16_cifar10_scratch.pth')
if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

model_acc = test_model(model, testloader, device)
print(model_acc)
# Initialize a VGG16 model



model, pruning_mask = prune_vgg16_conv_layers(model, amount=0.5)
print(pruning_mask)
# Test the model after pruning
print("Testing model after pruning")
test_model(model, testloader, device)

# Fine-tune the pruned model
# Define loss function and optimizer
path = f'vgg16_cifar10_prune_iter_L1_50.pth'
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
scaler = GradScaler()  # For mixed precision

# Main training loop
num_epochs = 50
best_acc = 0.0
for epoch in range(num_epochs):
    train(epoch, model, trainloader, device, optimizer, criterion, scaler)
    acc = test(model, testloader, device)
    scheduler.step()
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), path)
        print(f"Model saved with accuracy: {best_acc:.2f}%")

##############
print("75 % ")
# Training and testing process in a loop
model = load_model('vgg16_cifar10_scratch.pth')  # Iterate the process three times
if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
model_acc = test_model(model, testloader, device)
print(model_acc)
# Initialize a VGG16 model



model, pruning_mask = prune_vgg16_conv_layers(model, amount=0.75)
print(pruning_mask)
# Test the model after pruning
print("Testing model after pruning")
test_model(model, testloader, device)

# Fine-tune the pruned model
# Define loss function and optimizer
path = f'vgg16_cifar10_prune_iter_L1_75.pth'
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
scaler = GradScaler()  # For mixed precision

# Main training loop
num_epochs = 50
best_acc = 0.0
for epoch in range(num_epochs):
    train(epoch, model, trainloader, device, optimizer, criterion, scaler)
    acc = test(model, testloader, device)
    scheduler.step()
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), path)
        print(f"Model saved with accuracy: {best_acc:.2f}%")



