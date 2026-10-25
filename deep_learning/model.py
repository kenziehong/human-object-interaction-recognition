"""Torchvision image classifiers and backbone freeze controls."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models

from config import CLASS_NAMES


def build_model(backbone: str, pretrained: bool = True) -> nn.Module:
    """Create a supported ImageNet model with a six-class linear head."""
    if backbone == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    elif backbone == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
    else:
        raise ValueError(f"Backbone chưa hỗ trợ: {backbone}")
    return model


def set_backbone_trainable(model, backbone: str, trainable: bool) -> None:
    """Freeze or unfreeze feature layers while always training the classifier."""
    for parameter in model.parameters():
        parameter.requires_grad = trainable
    classifier = model.fc if backbone == "resnet50" else model.classifier
    for parameter in classifier.parameters():
        parameter.requires_grad = True

