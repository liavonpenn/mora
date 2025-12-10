import torch
import torch.nn as nn

from mantis.architecture import Mantis8M
from mantis.trainer import MantisTrainer

import torch.nn.functional as F

class MantisBackbone(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.network = Mantis8M()
        self.network = self.network.from_pretrained("/home/nsccgz/liopank/RAG4MTS/baselines/checkpoints/mantis_model/Mantis-8M")

        self.model = MantisTrainer(device=device, network=self.network)

        # for param in self.model.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        """Encode time series data"""
        features = self.model(x)
        return features

class MantisHead(nn.Module):
    def __init__(self, num_classes, hidden_dim=6*256):
        super().__init__()

        # self.fc1 = nn.Sequential(
        #     nn.LayerNorm(hidden_dim),
        #     nn.Linear(hidden_dim, hidden_dim)
        # )
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, features):
        return self.fc(features)



class MantisModel(nn.Module):
    def __init__(self, num_classes, device, feature_dim=6*256, threshold=0.5):
        super().__init__()
        self.backbone = MantisBackbone(device)
        self.head = MantisHead(num_classes)
        self.upsampler = nn.Upsample(size=224, mode='linear', align_corners=True)

        self.num_classes = num_classes


    def forward(self, x, return_features=False):

        # x, x_trend = self.dft_series_decomp(x)
        imu_data = x.permute(0, 2, 1)
        imu_data = self.upsampler(imu_data)
        embedding = self.backbone.model.transform(imu_data)

        if return_features:
            return embedding

        logits = self.head(embedding)  # [B, C]
        return embedding, logits 

    def pretrain(self, x):
        """Pre-training method"""
        z = self.backbone(x)
        return z