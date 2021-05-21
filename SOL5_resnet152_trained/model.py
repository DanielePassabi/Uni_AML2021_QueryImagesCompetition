import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models

class ResNet(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(ResNet, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Dropout(0.5),
                                      nn.Linear(num_ftrs, num_classes))

    def forward(self, x):
        return self.model(x)
