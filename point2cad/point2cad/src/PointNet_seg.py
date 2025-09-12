import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetSeg(nn.Module):
    """
    (체크포인트 호환 버전)
      - stem: 3 → 64 → 128 → 1024
      - skip concat: 64 + 1024 = 1088
    """
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv1d(3,   64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128,1024,1)
        self.bn1, self.bn2, self.bn3 = (
            nn.BatchNorm1d(64),
            nn.BatchNorm1d(128),
            nn.BatchNorm1d(1024),
        )
        # 64 + 1024 = 1088
        self.conv4 = nn.Conv1d(1088, 512, 1)
        self.conv5 = nn.Conv1d(512, 256, 1)
        self.conv6 = nn.Conv1d(256, 128, 1)
        self.conv7 = nn.Conv1d(128, num_classes, 1)
        self.bn4, self.bn5, self.bn6 = (
            nn.BatchNorm1d(512),
            nn.BatchNorm1d(256),
            nn.BatchNorm1d(128),
        )

    def forward(self, x):               # x:(B,3,N)
        B, _, N = x.size()
        x1 = F.relu(self.bn1(self.conv1(x)))     # (B, 64,  N)
        x2 = F.relu(self.bn2(self.conv2(x1)))    # (B, 128, N)
        x3 = F.relu(self.bn3(self.conv3(x2)))    # (B,1024, N)
        g  = torch.max(x3, 2, keepdim=True)[0].repeat(1,1,N)  # (B,1024,N)
        x  = torch.cat([x1, g], 1)               # (B,1088,N)
        x  = F.relu(self.bn4(self.conv4(x)))     # (B,512,N)
        x  = F.relu(self.bn5(self.conv5(x)))     # (B,256,N)
        x  = F.relu(self.bn6(self.conv6(x)))     # (B,128,N)
        return self.conv7(x)                     # (B,num_classes,N)
