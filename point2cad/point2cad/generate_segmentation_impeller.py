#!/usr/bin/env python3
# 3번 서버의 point2cad/generate_segmentation_impeller0804_2.py 파일을 기반으로 작성되었습니다.

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from src.PointNet_seg import PointNetSeg

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, targets):
        logpt = -self.ce(logits, targets)
        pt = torch.exp(logpt)
        focal_loss = -((1 - pt) ** self.gamma) * logpt
        return focal_loss.mean()


def train_impeller(xyzc_path, model_out, lr, stop_loss, device, sample_size):
    print(f"[Train] Using device: {device}")
    data = np.loadtxt(xyzc_path).astype(np.float32)
    points = data[:, :3]
    orig_labels = data[:, 3].astype(np.int64)

    unique_labels = np.unique(orig_labels)
    label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    labels = np.array([label_to_idx[l] for l in orig_labels], dtype=np.int64)

    num_classes = len(unique_labels)
    model = PointNetSeg(num_classes=num_classes).to(device)

    counts = np.bincount(labels, minlength=num_classes)
    inv_freq = 1.0 / (counts + 1e-6)
    weights = inv_freq / np.sum(inv_freq) * num_classes
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion = FocalLoss(gamma=2.0, weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)

    best_loss = float('inf')
    best_state = None
    full_loss_history = []

    N = points.shape[0]
    steps_per_epoch = max(1, N // sample_size)

    model.train()
    for epoch in range(1, 201):
        cycle_loss = 0.0
        for _ in range(steps_per_epoch):
            idx = np.random.choice(N, sample_size, replace=False)
            x_batch = points[idx]; y_batch = labels[idx]
            x_tensor = torch.from_numpy(x_batch.T).unsqueeze(0).to(device)
            y_tensor = torch.from_numpy(y_batch).to(device)

            optimizer.zero_grad()
            logits = model(x_tensor).squeeze(0).permute(1,0)
            loss = criterion(logits, y_tensor)
            loss.backward(); optimizer.step()
            cycle_loss += loss.item()
        avg_loss = cycle_loss / steps_per_epoch
        scheduler.step()
        print(f"Epoch {epoch}/200  BatchLoss: {avg_loss:.6f}")

        # full dataset evaluation
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                x_full = torch.from_numpy(points.T).unsqueeze(0).to(device)
                y_full = torch.from_numpy(labels).to(device)
                logits_full = model(x_full).squeeze(0).permute(1,0)
                full_loss = criterion(logits_full, y_full).item()
            model.train()

            full_loss_history.append(full_loss)
            print(f"  FullLoss: {full_loss:.6f}")
            # save best
            if full_loss < best_loss:
                best_loss = full_loss
                best_state = model.state_dict().copy()
                torch.save({'state_dict': best_state,'unique_labels':unique_labels.tolist()}, 
                           os.path.join(os.path.dirname(model_out), 'best_'+os.path.basename(model_out)))
                print(f"  [Best] FullLoss improved → {best_loss:.6f}")
            # early stop
            if len(full_loss_history)>=5 and all(l <= stop_loss for l in full_loss_history[-5:]):
                print(f"  [EarlyStop] last 5 FullLoss ≤ {stop_loss}")
                break

    # final save
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    torch.save({'state_dict': best_state,'unique_labels':unique_labels.tolist()}, model_out)
    print(f"[Saved] Model → {model_out}  (best_loss={best_loss:.6f})")

    # plot
    plt.figure(); plt.plot(full_loss_history, marker='o')
    plt.title('Full Loss History'); plt.xlabel('Eval (x10 epochs)'); plt.ylabel('Loss'); plt.grid(True)
    plt.savefig('full_loss_curve.png'); print("Loss curve → full_loss_curve.png")


def infer_impeller(xyz_path, model_path, device):
    print(f"[Infer] Using {device}")
    data = np.loadtxt(xyz_path).astype(np.float32)
    points = data[:, :3]
    chk = torch.load(model_path, map_location=device)
    inv_map = {i:l for i,l in enumerate(chk['unique_labels'])}
    num_classes = len(chk['unique_labels'])
    model = PointNetSeg(num_classes=num_classes).to(device)
    model.load_state_dict(chk['state_dict']); model.eval()

    with torch.no_grad():
        logits = model(torch.from_numpy(points.T).unsqueeze(0).to(device))
        preds = logits.squeeze(0).argmax(dim=0).cpu().numpy()
    unique, counts = np.unique(preds, return_counts=True)
    print('Predicted classes:', unique)
    print('Counts:', counts)

    out = np.hstack([points, np.array([inv_map[i] for i in preds])[:,None]])
    out_path = xyz_path.replace('.xyz','_impeller_pred.xyzc')
    np.savetxt(out_path, out, fmt='%.6f'); print(f"Saved → {out_path}")

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['train','infer'])
    p.add_argument('--xyzc_path',type=str); p.add_argument('--xyz_path',type=str)
    p.add_argument('--model_out',type=str,default='./models/impeller_seg.pth')
    p.add_argument('--model_path',type=str)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--stop_loss',type=float,default=0.1)
    p.add_argument('--batch_size',type=int,default=2000)
    args=p.parse_args()
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"GPUs: {torch.cuda.device_count()}, Using: {dev}")
    if args.mode=='train':
        assert args.xyzc_path, 'need --xyzc_path'
        train_impeller(args.xyzc_path,args.model_out,args.lr,args.stop_loss,dev,args.batch_size)
    else:
        assert args.model_path and args.xyz_path, 'need --model_path/--xyz_path'
        infer_impeller(args.xyz_path,args.model_path,dev)
