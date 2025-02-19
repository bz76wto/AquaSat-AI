import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from dataset import AISDataset, transform
from torch.utils.data import DataLoader
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
train_dataset = AISDataset(root_dir="spectrograms/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Load Pretrained ResNet
resnet = models.resnet18(pretrained=True)
num_classes = len(os.listdir("spectrograms/train"))
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
resnet = resnet.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr=0.001)

# Train model
num_epochs = 10
for epoch in range(num_epochs):
    resnet.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = resnet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

# Save trained model
torch.save(resnet.state_dict(), "models/resnet_mmsi.pth")
