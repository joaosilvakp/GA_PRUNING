import torch
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# Custom imports
from utils import load_model, test_model


def generate_confusion_matrix(model, testloader, class_names, device):
    """Generates and saves a confusion matrix plot."""
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

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix for CIFAR-10')
    plt.savefig('cifar10_confusion_matrix.png')

    return cm


if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load dataset
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # Load pre-trained model
    print("Loading model...")
    model = load_model(path='vgg16_cifar10_scratch.pth')
    model = model.to(device)

    if device.type == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    # Run evaluation
    print("Starting evaluation...")
    test_accuracy = test_model(model, testloader, device)
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    # Generate Confusion Matrix
    generate_confusion_matrix(model, testloader, class_names, device)
    print("Testing complete! Confusion matrix saved as 'cifar10_confusion_matrix.png'")