#!/usr/bin/env python3  # 파이썬 실행 환경 지정
import argparse  # 명령행 인자 파싱용 모듈
import os  # 파일/디렉토리 관련 모듈
import numpy as np  # 수치 연산용 라이브러리
import torch  # PyTorch 딥러닝 프레임워크
import torch.nn as nn  # 신경망 관련 모듈
import torch.optim as optim  # 최적화 알고리즘 모듈
from src.PointNet_seg import PointNetSeg  # PointNet 기반 세그멘테이션 모델 임포트

# 학습 실행 예시 주석
# python generate_segmentation_impeller.py train --xyzc_path "..\assets\xyz\merged_imp_gpu_label.xyzc" --model_out ".\models\impeller_seg.pth" --lr 1e-4 --stop_loss 0.01

def train_impeller(xyzc_path, model_out, lr, stop_loss, device):
    print(f"[Train] Using device: {device}")  # 사용 디바이스 출력
    data = np.loadtxt(xyzc_path).astype(np.float32)  # xyzc 파일 로드 및 float32 변환
    points = data[:, :3]  # 포인트 좌표 추출
    orig_labels = data[:, 3].astype(np.int64)  # 원본 라벨 추출 및 int64 변환

    unique_labels = np.unique(orig_labels)  # 고유 라벨 추출
    label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}  # 라벨 인덱스 매핑 생성
    labels = np.array([label_to_idx[l] for l in orig_labels], dtype=np.int64)  # 라벨을 인덱스로 변환

    num_classes = len(unique_labels)  # 클래스 개수 계산
    model = PointNetSeg(num_classes=num_classes).to(device)  # 모델 생성 및 디바이스 할당

    counts = np.bincount(labels, minlength=num_classes)  # 각 클래스별 샘플 개수 계산
    inv_freq = 1.0 / (counts + 1e-4)  # 클래스별 역빈도 계산
    weights = inv_freq / np.sum(inv_freq) * num_classes  # 클래스 가중치 정규화
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)  # 텐서로 변환 및 디바이스 할당
    criterion = nn.CrossEntropyLoss(weight=class_weights)  # 가중치 적용된 크로스엔트로피 손실 함수
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  # Adam 옵티마이저 생성
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)  # 러닝레이트 스케줄러

    consecutive = 0  # 연속 종료 카운터 (사용 안함)
    epoch = 0  # 에폭 카운터
    N = points.shape[0]  # 전체 샘플 개수
    sample_size = N  # 배치 크기 (전체 데이터 사용)
    steps_per_cycle = max(1, N // sample_size)  # 한 사이클 내 스텝 수 (1)

    full_loss_history = []  # 전체 데이터셋 손실 기록 리스트
    full_loss_check_cycle = 10  # 전체 평가 주기

    model.train()  # 모델을 학습 모드로 설정
    while True:  # 무한 반복 (조기 종료 조건까지)
        epoch += 1  # 에폭 증가
        cycle_loss = 0.0  # 사이클 손실 초기화
        for _ in range(steps_per_cycle):  # 사이클 내 반복 (여기선 1회)
            idx = np.random.choice(N, sample_size, replace=False)  # 랜덤 샘플 인덱스 선택
            x_batch = points[idx]  # 배치 포인트 추출
            y_batch = labels[idx]  # 배치 라벨 추출

            x_tensor = torch.from_numpy(x_batch.T).unsqueeze(0).to(device)  # 포인트 텐서 변환 및 디바이스 할당
            y_tensor = torch.from_numpy(y_batch).to(device)  # 라벨 텐서 변환 및 디바이스 할당

            optimizer.zero_grad()  # 옵티마이저 그래디언트 초기화
            logits = model(x_tensor)  # 모델 추론
            logits_flat = logits.squeeze(0).permute(1, 0)  # 출력 텐서 형태 변환
            loss = criterion(logits_flat, y_tensor)  # 손실 계산
            loss.backward()  # 그래디언트 역전파
            optimizer.step()  # 파라미터 업데이트
            cycle_loss += loss.item()  # 손실 누적

        avg_loss = cycle_loss / steps_per_cycle  # 평균 손실 계산
        scheduler.step(avg_loss)  # 러닝레이트 스케줄러 업데이트
        print(f"Cycle {epoch}, Loss: {avg_loss:.6f}")  # 현재 사이클 손실 출력

        # 전체 데이터셋 평가 (10주기마다)
        if epoch % full_loss_check_cycle == 0:
            with torch.no_grad():  # 그래디언트 계산 비활성화
                x_full = torch.from_numpy(points.T).unsqueeze(0).to(device)  # 전체 포인트 텐서 변환
                y_full = torch.from_numpy(labels).to(device)  # 전체 라벨 텐서 변환
                logits = model(x_full)  # 전체 데이터 추론
                logits_flat = logits.squeeze(0).permute(1, 0)  # 출력 텐서 형태 변환
                full_loss = criterion(logits_flat, y_full).item()  # 전체 손실 계산
                print(f"[Full Eval @Cycle {epoch}] Loss: {full_loss:.6f}")  # 전체 손실 출력

                full_loss_history.append(full_loss)  # 손실 기록 추가
                # 최근 10회 손실이 모두 stop_loss 이하이면 조기 종료
                if len(full_loss_history) >= 10 and all(l <= stop_loss for l in full_loss_history[-10:]):
                    print(f"[Early Stop] 최근 10회 full loss 모두 {stop_loss} 이하 → 학습 종료")
                    break  # 학습 종료

    os.makedirs(os.path.dirname(model_out), exist_ok=True)  # 모델 저장 디렉토리 생성
    torch.save({
        'state_dict': model.state_dict(),  # 모델 파라미터 저장
        'unique_labels': unique_labels.tolist()  # 고유 라벨 정보 저장
    }, model_out)  # 모델 파일 저장
    print(f"Model saved to {model_out} with {num_classes} classes after {epoch} cycles")  # 저장 완료 출력


def infer_impeller(xyz_path, model_path, device):
    print(f"[Infer] Using device: {device}")  # 사용 디바이스 출력
    data = np.loadtxt(xyz_path).astype(np.float32)  # xyz 파일 로드 및 float32 변환
    points = data[:, :3]  # 포인트 좌표 추출

    checkpoint = torch.load(model_path, map_location=device)  # 모델 체크포인트 로드
    unique_labels = checkpoint.get('unique_labels')  # 고유 라벨 정보 추출
    if unique_labels is None:  # 라벨 정보 없으면 에러 발생
        raise RuntimeError('Checkpoint에 unique_labels 정보가 없습니다.')
    inv_map = {idx: lbl for idx, lbl in enumerate(unique_labels)}  # 인덱스→원본라벨 매핑 생성
    num_classes = len(unique_labels)  # 클래스 개수 계산

    model = PointNetSeg(num_classes=num_classes).to(device)  # 모델 생성 및 디바이스 할당
    model.load_state_dict(checkpoint['state_dict'])  # 모델 파라미터 로드
    model.eval()  # 모델을 평가 모드로 설정

    x_tensor = torch.from_numpy(points.T).unsqueeze(0).to(device)  # 포인트 텐서 변환 및 디바이스 할당
    with torch.no_grad():  # 그래디언트 계산 비활성화
        logits = model(x_tensor)  # 모델 추론
        preds_idx = logits.squeeze(0).argmax(dim=0).cpu().numpy()  # 예측 클래스 인덱스 추출

    # 예측 분포 확인
    unique_preds, counts = np.unique(preds_idx, return_counts=True)  # 예측 클래스별 개수 집계
    print("Predicted class indices:", unique_preds)  # 예측 클래스 인덱스 출력
    print("Counts per class:", counts)  # 클래스별 개수 출력

    preds_orig = np.array([inv_map[i] for i in preds_idx], dtype=np.float32)  # 인덱스를 원본 라벨로 변환
    out = np.hstack([points, preds_orig[:, None]])  # 좌표와 예측 라벨 합치기

    out_path = xyz_path.replace('.xyz', '_impeller_pred.xyzc')  # 결과 파일 경로 생성
    np.savetxt(out_path, out, fmt='%.6f')  # 결과 파일 저장
    print(f"Prediction saved to {out_path}")  # 저장 완료 출력

if __name__ == '__main__':  # 메인 실행부
    parser = argparse.ArgumentParser(description="Impeller 전용 과적합 세그멘테이션")  # 인자 파서 생성
    parser.add_argument('mode', choices=['train', 'infer'], help='train or infer')  # 모드 인자 추가
    parser.add_argument('--xyzc_path', type=str, help='학습용 .xyzc 파일 경로')  # 학습 데이터 경로 인자
    parser.add_argument('--xyz_path', type=str, help='추론용 .xyz 파일 경로')  # 추론 데이터 경로 인자
    parser.add_argument('--model_out', type=str, default='./models/impeller_seg.pth', help='모델 저장 경로')  # 모델 저장 경로 인자
    parser.add_argument('--model_path', type=str, help='추론용 모델 경로')  # 추론 모델 경로 인자
    parser.add_argument('--lr', type=float, default=1e-3, help='학습률')  # 학습률 인자
    parser.add_argument('--stop_loss', type=float, default=0.01, help='연속 종료 손실 임계값')  # 조기 종료 손실 임계값 인자
    args = parser.parse_args()  # 인자 파싱

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 사용 디바이스 결정
    print(f"Available GPUs: {torch.cuda.device_count()}, Using Device: {device}")  # GPU 정보 출력

    if args.mode == 'train':  # 학습 모드
        if not args.xyzc_path:  # 학습 데이터 경로 없으면 에러
            parser.error('train 모드에서는 --xyzc_path 필요')
        train_impeller(args.xyzc_path, args.model_out, args.lr, args.stop_loss, device)  # 학습 함수 실행
    else:  # 추론 모드
        if not args.model_path or not args.xyz_path:  # 모델/데이터 경로 없으면 에러
            parser.error('infer 모드에서는 --model_path 와 --xyz_path 필요')
        infer_impeller(args.xyz_path, args.model_path, device)  # 추론 함수 실행
