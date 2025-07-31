#!/usr/bin/env python3
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.PointNet_seg import PointNetSeg

# python generate_segmentation_impeller.py train --xyzc_path "..\assets\xyz\merged_imp_gpu_label.xyzc" --model_out ".\models\impeller_seg.pth" --lr 1e-4 --stop_loss 0.01

def train_impeller(xyzc_path, model_out, lr, stop_loss, device):
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
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  # ✅ weight_decay 추가
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    consecutive = 0
    epoch = 0
    N = points.shape[0]
    sample_size = min(1000, N)
    steps_per_cycle = max(1, N // sample_size)

    full_loss_history = []
    full_loss_check_cycle = 10  # ✅ 주기적 평가 간격

    model.train()
    while True:
        epoch += 1
        cycle_loss = 0.0
        for _ in range(steps_per_cycle):
            idx = np.random.choice(N, sample_size, replace=False)
            x_batch = points[idx]
            y_batch = labels[idx]

            x_tensor = torch.from_numpy(x_batch.T).unsqueeze(0).to(device)
            y_tensor = torch.from_numpy(y_batch).to(device)

            optimizer.zero_grad()
            logits = model(x_tensor)
            logits_flat = logits.squeeze(0).permute(1, 0)
            loss = criterion(logits_flat, y_tensor)
            loss.backward()
            optimizer.step()
            cycle_loss += loss.item()

        avg_loss = cycle_loss / steps_per_cycle
        scheduler.step(avg_loss)
        print(f"Cycle {epoch}, Loss: {avg_loss:.6f}")

        # ✅ 전체 데이터셋 평가 (10주기마다)
        if epoch % full_loss_check_cycle == 0:
            with torch.no_grad():
                x_full = torch.from_numpy(points.T).unsqueeze(0).to(device)
                y_full = torch.from_numpy(labels).to(device)
                logits = model(x_full)
                logits_flat = logits.squeeze(0).permute(1, 0)
                full_loss = criterion(logits_flat, y_full).item()
                print(f"[Full Eval @Cycle {epoch}] Loss: {full_loss:.6f}")

                full_loss_history.append(full_loss)
                if len(full_loss_history) >= 10 and all(l <= stop_loss for l in full_loss_history[-10:]):
                    print(f"[Early Stop] 최근 10회 full loss 모두 {stop_loss} 이하 → 학습 종료")
                    break

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    torch.save({
        'state_dict': model.state_dict(),
        'unique_labels': unique_labels.tolist()
    }, model_out)
    print(f"Model saved to {model_out} with {num_classes} classes after {epoch} cycles")


def infer_impeller(xyz_path, model_path, device):
    print(f"[Infer] Using device: {device}")
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
        logits = model(x_tensor)
        preds_idx = logits.squeeze(0).argmax(dim=0).cpu().numpy()

    # 예측 분포 확인
    unique_preds, counts = np.unique(preds_idx, return_counts=True)
    print("Predicted class indices:", unique_preds)
    print("Counts per class:", counts)

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
    parser.add_argument('--lr', type=float, default=1e-3, help='학습률')
    parser.add_argument('--stop_loss', type=float, default=0.01, help='연속 종료 손실 임계값')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Available GPUs: {torch.cuda.device_count()}, Using Device: {device}")

    if args.mode == 'train':
        if not args.xyzc_path:
            parser.error('train 모드에서는 --xyzc_path 필요')
        train_impeller(args.xyzc_path, args.model_out, args.lr, args.stop_loss, device)
    else:
        if not args.model_path or not args.xyz_path:
            parser.error('infer 모드에서는 --model_path 와 --xyz_path 필요')
        infer_impeller(args.xyz_path, args.model_path, device)
