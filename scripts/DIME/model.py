# DIME/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class EfficientNetWithClinical(nn.Module):
    """
    Combined EfficientNet and clinical features model with attention mechanism.

    Args:
        num_clinical_features (int): Number of clinical features
        dropout_rate (float): Dropout rate for regularization
    """
    def __init__(self, num_clinical_features=3, dropout_rate=0.3):
        super().__init__()

        # Initialize and customize EfficientNet backbone
        self.backbone = models.efficientnet_b0(pretrained=True)
        original_weight = self.backbone.features[0][0].weight.clone()
        self.backbone.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.backbone.features[0][0].weight.data = original_weight.mean(dim=1, keepdim=True)

        # Set up feature extraction
        for param in self.backbone.parameters():
            param.requires_grad = False
        for idx in [6, 7]:
            for param in self.backbone.features[idx].parameters():
                param.requires_grad = True

        # Add dimension reduction layers
        self.dim_reduce = nn.Sequential(
            nn.Conv2d(1280, 320, 1),
            nn.BatchNorm2d(320),
            nn.ReLU(inplace=True)
        )

        # Clinical features processing
        self.clinical_net = nn.Sequential(
            nn.Linear(num_clinical_features, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        # Cross-attention mechanism
        self.clinical_attention = nn.Sequential(
            nn.Linear(16, 128),
            nn.GELU(),
            nn.Linear(128, 320),
            nn.Sigmoid()
        )

        # Final prediction layers
        self.final_layers = nn.Sequential(
            nn.Linear(320 + 16, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, image, clinical):
        """Forward pass combining image and clinical features"""
        x = self.backbone.features(image)
        x = self.dim_reduce(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        image_features = torch.flatten(x, 1)

        clinical_features = self.clinical_net(clinical)
        attention_weights = self.clinical_attention(clinical_features)
        attended_image_features = image_features * attention_weights

        combined = torch.cat((attended_image_features, clinical_features), dim=1)
        return self.final_layers(combined)