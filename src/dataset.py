import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

# Dataset class for AIS spectrograms
class AISDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_map = {mmsi: i for i, mmsi in enumerate(os.listdir(root_dir))}

        for mmsi in os.listdir(root_dir):
            mmsi_dir = os.path.join(root_dir, mmsi)
            for img_name in os.listdir(mmsi_dir):
                self.image_paths.append(os.path.join(mmsi_dir, img_name))
                self.labels.append(self.label_map[mmsi])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Image transformation for ResNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

