import torch
from dataset import AISDataset, transform
from torch.utils.data import DataLoader
import torchvision.models as models
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
test_dataset = AISDataset(root_dir="spectrograms/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load trained model
resnet = models.resnet18(pretrained=True)
num_classes = len(os.listdir("spectrograms/test"))
resnet.fc = torch.nn.Linear(resnet.fc.in_features, num_classes)
resnet.load_state_dict(torch.load("models/resnet_mmsi.pth"))
resnet = resnet.to(device)
resnet.eval()

# Evaluate accuracy
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = resnet(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
