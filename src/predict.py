import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
resnet = models.resnet18(pretrained=True)
num_classes = 100  # Adjust based on dataset
resnet.fc = torch.nn.Linear(resnet.fc.in_features, num_classes)
resnet.load_state_dict(torch.load("models/resnet_mmsi.pth"))
resnet = resnet.to(device)
resnet.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def predict_mmsi(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = resnet(image)
        _, predicted = torch.max(output, 1)

    return predicted.item()

# Predict MMSI
image_path = "spectrograms/test_sample.png"
print(f"Predicted MMSI: {predict_mmsi(image_path)}")
