# Ai_coding_study

## 📌 프로젝트 설명
건국대학교 통계적 인공지능 연구실에서 진행한 프로젝트들을 정리한 저장소입니다.  
딥러닝 및 3D 모델링 관련 기술들을 공부하고 구현한 내용을 기록하고 있습니다.  
현재 `Point2CAD` 기술 구현을 진행 중입니다.

## 🛠 주요 기능 및 핵심 특징
- 그동안 공부했던 딥러닝 및 컴퓨터 비전 프로젝트 기록
- 주요 내용
  - **Pytorch** (Last month)
  - **YOLO** (Last month)
  - **YOLOv2** (3 weeks ago)
  - **Deep Learning** (Last month)
  - **Point2CAD** (현재 진행 중)
- 최신 진행 사항은 `Point2CAD` 프로젝트로, 3D Point Cloud를 CAD 모델로 변환하는 기술 구현에 집중하고 있습니다.

## ⚙ 설치 및 실행 방법
본 프로젝트의 `Point2CAD` 실행 환경은 다음과 같습니다.

### ✅ Conda 환경 구성 예시
```yaml
name: point2cad
channels:
  - pytorch
  - nvidia
  - defaults
dependencies:
  - python=3.9.21
  - pytorch-cuda=11.8
  - torch=2.2.2+cu121
  - torchvision=0.17.2+cu121
  - numpy=1.23.5
  - matplotlib=3.3.4
  - open3d=0.19.0
  - trimesh=4.6.5
  - scikit-learn=1.2.2
  - scipy=1.13.1
  - geomdl=5.3.1
  - 기타 필수 패키지들 (상세 내용은 `environment.yml` 참조)

```
### ✅ CUDA / GPU
CUDA 12.0 이상 지원

GPU 환경에서 학습 및 추론 가능

### 💻 기술 스택
```
Language: Python 3.9.21

Framework & Library:

PyTorch 2.2.2 + cu121

Torchvision 0.17.2 + cu121

Open3D

Trimesh

GeomDL

Scikit-learn

Matplotlib

CUDA: 12.0 이상 지원
```
