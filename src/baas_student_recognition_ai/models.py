import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    mobilenet_v3_small,
)


class PretrainedMobileNetV3StudentEncoder(nn.Module):
    """ImageNet-pretrained torchvision backbone with a small embedding head."""

    def __init__(self, embedding_size: int = 128):
        super().__init__()
        backbone = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1,
        )
        self.features = backbone.features
        feature_size = backbone.classifier[0].in_features
        self.projection = nn.Linear(feature_size, embedding_size)

    def forward(self, image):
        feature = self.features(image)
        feature = F.adaptive_avg_pool2d(feature, 1).flatten(1)
        return F.normalize(self.projection(feature), dim=1)


class StudentEncoderTrainer(nn.Module):
    def __init__(self, class_count: int, embedding_size: int = 128, encoder=None):
        super().__init__()
        self.encoder = encoder or PretrainedMobileNetV3StudentEncoder(embedding_size)
        self.classifier = nn.Linear(embedding_size, class_count)

    def forward(self, image):
        embedding = self.encoder(image)
        return embedding, self.classifier(embedding) * 12.0
