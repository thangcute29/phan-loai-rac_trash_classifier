import torch.nn as nn
from torchvision import models

def get_model(num_classes):
    """
    ResNet18 pretrained và fine-tune lớp cuối
    """
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model
