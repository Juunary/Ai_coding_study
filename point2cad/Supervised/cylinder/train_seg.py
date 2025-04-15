#!/usr/bin/env python3
"""
train_seg.py
------------
라벨 0·1·2 로 준비된 .xyzc 데이터(Ground‑Truth)를 이용해
세그멘테이션 네트워크(출력 3채널)를 지도학습하고,
학습된 가중치를 seg_3cls.pth 로 저장합니다.

사용 예:
    python train_seg.py \
        --train_list data/train_files.txt \
        --epochs 100 \
        --batch 4 \
        --save seg_3cls.pth
"""
import os, argparse, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from tqdm import tqdm
root = "./" 
# ────────────────────────────────────────────────
# 1)  Dataset (.xyzc → Tensor)
# ────────────────────────────────────────────────
class XYzcDataset(Dataset):
    def __init__(self, file_list, n_points=4096):
        self.files = file_list
        self.n_points = n_points
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        xyzc = np.loadtxt(self.files[idx])
        xyz, lab = xyzc[:, :3], xyzc[:, 3].astype(np.int64)
        sel = np.random.choice(len(xyz), self.n_points,
                               replace=len(xyz) < self.n_points)
        xyz, lab = xyz[sel], lab[sel]
        xyz -= xyz.mean(0, keepdims=True)
        xyz /= np.max(np.linalg.norm(xyz, axis=1))
        return (torch.from_numpy(xyz).float(),
                torch.from_numpy(lab))

# ────────────────────────────────────────────────
# 2)  네트워크 (PointNetSeg 예시, 3‑클래스)
#     ▸ generate_segmentation.py 에서 복사해 오세요
# ────────────────────────────────────────────────
class PointNetSeg(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        import torch.nn.functional as F
        self.F = F
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1, self.bn2, self.bn3 = nn.BatchNorm1d(64), nn.BatchNorm1d(128), nn.BatchNorm1d(1024)
        self.conv4 = nn.Conv1d(1088, 512, 1)
        self.conv5 = nn.Conv1d(512, 256, 1)
        self.conv6 = nn.Conv1d(256, 128, 1)
        self.conv7 = nn.Conv1d(128, num_classes, 1)
        self.bn4, self.bn5, self.bn6 = nn.BatchNorm1d(512), nn.BatchNorm1d(256), nn.BatchNorm1d(128)
    def forward(self, x):          # x:(B,3,N)
        B, _, N = x.size()
        F = self.F
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2)))
        g  = torch.max(x3, 2, keepdim=True)[0].repeat(1,1,N)
        x  = torch.cat([x1, g], 1)
        x  = F.relu(self.bn4(self.conv4(x)))
        x  = F.relu(self.bn5(self.conv5(x)))
        x  = F.relu(self.bn6(self.conv6(x)))
        return self.conv7(x)       # (B,3,N)

# ────────────────────────────────────────────────
# 3)  학습 루프
# ────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_list', required=True,
                    help='txt 파일: 각 줄에 .xyzc 경로')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch',  type=int, default=4)
    ap.add_argument('--save',   default='seg_3cls.pth')
    ap.add_argument('--npts',   type=int, default=4096)
    args = ap.parse_args()

    files = [l.strip() for l in open(args.train_list) if l.strip()]
    assert len(files) > 0, "train_list 가 비어 있습니다."

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds  = XYzcDataset(files, n_points=args.npts)
    dl  = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True)

    model = PointNetSeg(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(args.epochs):
        model.train(); running = 0
        for xyz, lab in tqdm(dl, desc=f'Epoch {ep}', leave=False):
            xyz, lab = xyz.to(device), lab.to(device)
            loss = criterion(model(xyz.permute(0,2,1)), lab)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()
        print(f'Epoch {ep:3d}  loss={running/len(dl):.4f}')

    torch.save(model.state_dict(), args.save)
    print('✅  학습 완료 →', args.save)

if __name__ == '__main__':
    main()
