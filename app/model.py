# app/model.py

import torch
import torch.nn as nn
from torchvision import models

class EfficientNetModel(nn.Module):
    def __init__(self, num_classes=9):
        super(EfficientNetModel, self).__init__()
        self.base_model = models.efficientnet_b0(pretrained=False)
        self.base_model.classifier = nn.Sequential(
            nn.Linear(self.base_model.classifier[1].in_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)
