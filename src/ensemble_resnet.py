import torch
import torch.nn as nn
import torchvision.models as models

class ResNetEnsemble(nn.Module):
    def __init__(self, num_classes):
        super(ResNetEnsemble, self).__init__()

        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet34 = models.resnet34(pretrained=True)
        self.resnet50 = models.resnet50(pretrained=True)

        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)
        self.resnet34.fc = nn.Linear(self.resnet34.fc.in_features, num_classes)
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, num_classes)

    def forward(self, x):
        out1 = self.resnet18(x)
        out2 = self.resnet34(x)
        out3 = self.resnet50(x)

        out = (out1 + out2 + out3) / 3  # Averaging predictions
        return out
