from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import vgg16
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import numpy as np
import os


def train(epoch, model, trainloader, device, optimizer, criterion, scaler):
    model.train()
    running_loss = 0.0
    total_loss = 0.0
    train_loader_tqdm = tqdm(trainloader, desc=f"Epoch {epoch} Training", leave=True)
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
        total_loss += loss.item()
        avg_loss = running_loss / (batch_idx + 1)
        train_loader_tqdm.set_postfix(loss=avg_loss)  # Update progress bar with average loss

    return total_loss / len(trainloader)


def validate(model, validloader, device, criterion):
    model.eval()
    running_loss = 0.0
    valid_loader_tqdm = tqdm(validloader, desc="Validation", leave=True)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valid_loader_tqdm):
            inputs, targets = inputs.to(device), targets.to(device)

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            running_loss += loss.item()
            avg_loss = running_loss / (batch_idx + 1)
            valid_loader_tqdm.set_postfix(loss=avg_loss)

    return running_loss / len(validloader)


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


def plot_loss_curves(train_losses, valid_losses, save_path="loss_curves.png"):
    """Plot training and validation loss curves and save the figure."""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, valid_losses, 'r-', label='Validation Loss')

    plt.title('Training and Validation Loss Curves', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    # Adding annotations for minimum points
    min_train_epoch = np.argmin(train_losses) + 1
    min_train_loss = min(train_losses)
    min_valid_epoch = np.argmin(valid_losses) + 1
    min_valid_loss = min(valid_losses)

    plt.annotate(f'Min: {min_train_loss:.4f}',
                 xy=(min_train_epoch, min_train_loss),
                 xytext=(min_train_epoch + 5, min_train_loss + 0.1),
                 arrowprops=dict(facecolor='blue', shrink=0.05, alpha=0.7))

    plt.annotate(f'Min: {min_valid_loss:.4f}',
                 xy=(min_valid_epoch, min_valid_loss),
                 xytext=(min_valid_epoch + 5, min_valid_loss - 0.2),
                 arrowprops=dict(facecolor='red', shrink=0.05, alpha=0.7))

    # Save the figure in high resolution
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Loss curves saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory for plots
    output_dir = "training_results"
    os.makedirs(output_dir, exist_ok=True)

    # Define transformations for the CIFAR-10 dataset
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

    # Split training data into train and validation sets (80/20 split)
    train_size = int(0.8 * len(trainset))
    valid_size = len(trainset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(trainset, [train_size, valid_size])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    validloader = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=4)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)

    # Load pre-trained VGG16 model and modify the classifier for CIFAR-10
    model = vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(4096, 10)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    scaler = GradScaler()  # For mixed precision

    # Lists to store loss values
    train_losses = []
    valid_losses = []

    # Main training loop
    num_epochs = 100
    best_acc = 0.0
    for epoch in range(num_epochs):
        # Train and get average loss
        train_loss = train(epoch, model, trainloader, device, optimizer, criterion, scaler)
        train_losses.append(train_loss)

        # Validate and get average loss
        valid_loss = validate(model, validloader, device, criterion)
        valid_losses.append(valid_loss)

        # Test accuracy
        acc = test(model, testloader, device)

        print(
            f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Test Acc: {acc:.2f}%")

        # Save model if it has the best accuracy
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(output_dir, "vgg16_cifar10_best_test.pth"))
            print(f"Model saved with accuracy: {best_acc:.2f}%")

        # Update learning rate
        scheduler.step()

        # Plot and save the loss curves after each epoch
        plot_loss_curves(train_losses, valid_losses, save_path=os.path.join(output_dir, "loss_curves.png"))

        # Save intermediate loss data to CSV
        loss_data = np.column_stack((np.arange(1, epoch + 2), train_losses, valid_losses))
        np.savetxt(os.path.join(output_dir, "loss_data.csv"),
                   loss_data,
                   delimiter=',',
                   header="epoch,train_loss,valid_loss",
                   comments='')

    # Final plot after training completes
    plot_loss_curves(train_losses, valid_losses, save_path=os.path.join(output_dir, "final_loss_curves.png"))

    # Plot the entire training process with different visualization
    plt.figure(figsize=(12, 8))
    epochs = range(1, len(train_losses) + 1)

    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, valid_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss Over Time', fontsize=16)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    plt.subplot(2, 1, 2)
    # Calculate moving average to smooth the curves
    window_size = min(5, len(train_losses))
    train_losses_smooth = np.convolve(train_losses, np.ones(window_size) / window_size, mode='valid')
    valid_losses_smooth = np.convolve(valid_losses, np.ones(window_size) / window_size, mode='valid')
    smooth_epochs = range(window_size, len(train_losses) + 1)

    plt.plot(smooth_epochs, train_losses_smooth, 'b-', label='Training Loss (Smoothed)')
    plt.plot(smooth_epochs, valid_losses_smooth, 'r-', label='Validation Loss (Smoothed)')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss (Smoothed)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comprehensive_loss_analysis.png"), dpi=300, bbox_inches='tight')
    print(f"Comprehensive loss analysis saved to {os.path.join(output_dir, 'comprehensive_loss_analysis.png')}")