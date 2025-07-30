#!/usr/bin/env python3
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.PointNet_seg import PointNetSeg

def train_impeller(xyzc_path, model_out, epochs, lr, device):
    # 단일 임펠러 .xyzc 파일 로딩
    data = np.loadtxt(xyzc_path).astype(np.float32)  # (N, 4)
    points = data[:, :3]
    orig_labels = data[:, 3].astype(np.int64)

    # 원본 레이블 값을 0~C-1로 매핑
    unique_labels = np.unique(orig_labels)
    label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    labels = np.array([label_to_idx[l] for l in orig_labels], dtype=np.int64)

    num_classes = len(unique_labels)
    model = PointNetSeg(num_classes=num_classes).to(device)

    # 클래스 빈도 반비례 가중치
    counts = np.bincount(labels, minlength=num_classes)
    inv_freq = 1.0 / (counts + 1e-6)
    weights = inv_freq / np.sum(inv_freq) * num_classes
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    N = points.shape[0]
    sample_size = min(1000, N)
    steps_per_epoch = max(1, N // sample_size)

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for _ in range(steps_per_epoch):
            # 무작위 서브샘플링
            idx = np.random.choice(N, sample_size, replace=False)
            x_batch = points[idx]
            y_batch = labels[idx]

            x_tensor = torch.from_numpy(x_batch.T).unsqueeze(0).to(device)  # (1,3,S)
            y_tensor = torch.from_numpy(y_batch).to(device)               # (S,)

            optimizer.zero_grad()
            logits = model(x_tensor)                 # (1, C, S)
            logits_flat = logits.squeeze(0).permute(1, 0)  # (S, C)
            loss = criterion(logits_flat, y_tensor)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / steps_per_epoch
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}")

    # 간단 검증: 전체 포인트 예측 분포 확인
    model.eval()
    with torch.no_grad():
        full_logits = model(torch.from_numpy(points.T).unsqueeze(0).to(device))
        preds_full = full_logits.squeeze(0).argmax(dim=0).cpu().numpy()
    uniq_idx = np.unique(preds_full)
    mapped = [float(unique_labels[i]) for i in uniq_idx]
    print(f"Predicted indices: {uniq_idx.tolist()}")
    print(f"Mapped original labels: {mapped}")

    # 모델 저장
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    torch.save({
        'state_dict': model.state_dict(),
        'unique_labels': unique_labels.tolist()
    }, model_out)
    print(f"Model saved to {model_out} with {num_classes} classes")


def infer_impeller(xyz_path, model_path, device):
    data = np.loadtxt(xyz_path).astype(np.float32)
    points = data[:, :3]

    checkpoint = torch.load(model_path, map_location=device)
    unique_labels = checkpoint.get('unique_labels')
    if unique_labels is None:
        raise RuntimeError('Checkpoint에 unique_labels 정보가 없습니다.')
    inv_map = {idx: lbl for idx, lbl in enumerate(unique_labels)}
    num_classes = len(unique_labels)

    model = PointNetSeg(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    x_tensor = torch.from_numpy(points.T).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x_tensor)                 # (1, C, N)
        preds_idx = logits.squeeze(0).argmax(dim=0).cpu().numpy()  # (N,)

    preds_orig = np.array([inv_map[i] for i in preds_idx], dtype=np.float32)
    out = np.hstack([points, preds_orig[:, None]])

    out_path = xyz_path.replace('.xyz', '_impeller_pred.xyzc')
    np.savetxt(out_path, out, fmt='%.6f')
    print(f"Prediction saved to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Impeller 전용 과적합 세그멘테이션")
    parser.add_argument('mode', choices=['train', 'infer'], help='train or infer')
    parser.add_argument('--xyzc_path', type=str, help='학습용 .xyzc 파일 경로')
    parser.add_argument('--xyz_path', type=str, help='추론용 .xyz 파일 경로')
    parser.add_argument('--model_out', type=str, default='./models/impeller_seg.pth', help='모델 저장 경로')
    parser.add_argument('--model_path', type=str, help='추론용 모델 경로')
    parser.add_argument('--epochs', type=int, default=200, help='학습 epoch 수')
    parser.add_argument('--lr', type=float, default=1e-3, help='학습률')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.mode == 'train':
        if not args.xyzc_path:
            parser.error('train 모드에서는 --xyzc_path 필요')
        train_impeller(args.xyzc_path, args.model_out, args.epochs, args.lr, device)
    else:
        if not args.model_path or not args.xyz_path:
            parser.error('infer 모드에서는 --model_path 와 --xyz_path 필요')
        infer_impeller(args.xyz_path, args.model_path, device)
