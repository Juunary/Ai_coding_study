# point2cad/learned_models.py

import torch
import torch.nn as nn

class MLPPlanePredictor(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=256, output_dim=4):
        """
        MLP for predicting plane parameters from point cloud segments.
        Input: point cloud of shape [B, 1000, 3]
        Output: predicted plane parameters [nx, ny, nz, d] of shape [B, 4]
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 1000, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        x: [B, 1000, 3]
        return: [B, 4]
        """
        x = x.view(x.size(0), -1)  # Flatten to [B, 3000]
        return self.mlp(x)
