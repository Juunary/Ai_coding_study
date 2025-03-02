import cv2
import os
import numpy as np
from albumentations import Compose, HueSaturationValue, RandomBrightnessContrast, MotionBlur, GaussianBlur, GaussNoise, CoarseDropout, Normalize
from albumentations.pytorch import ToTensorV2

# 폴더 경로 설정
input_dir = './datasets/images/train'
labels_dir = './datasets/labels/train'
output_dir = './augmented_images'
os.makedirs(output_dir, exist_ok=True)

# Albumentations 증강 파이프라인 설정
transform = Compose([
    HueSaturationValue(p=1.0),
    RandomBrightnessContrast(p=1.0),
    GaussianBlur(blur_limit=(1, 3), p=0.5),
    MotionBlur(blur_limit=5, p=0.5),
    GaussNoise(var_limit=(0.001, 0.005), p=0.5),
    CoarseDropout(max_holes=8, max_height=16, max_width=16, min_holes=1, p=0.5),
    Normalize(mean=(0, 0, 0), std=(1, 1, 1), p=1),
    ToTensorV2()
], bbox_params={'format': 'yolo', 'label_fields': ['labels']})

# 데이터 증강 수행
startIndex, endIndex, genIndex, augNum = 1278, 1280, 1281, 50
for i in range(startIndex, endIndex):
    image_path = os.path.join(input_dir, f"data{i}.png")
    label_path = os.path.join(labels_dir, f"data{i}.txt")

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 바운딩 박스 로드
    with open(label_path, 'r') as f:
        bboxes, labels = [], []
        for line in f.readlines():
            parts = line.strip().split()
            labels.append(int(parts[0]))
            bboxes.append(list(map(float, parts[1:])))

    # 데이터 증강 반복 수행
    for j in range(augNum):
        try:
            augmented = transform(image=image, bboxes=bboxes, labels=labels)
            augmented_image = augmented['image'].permute(1, 2, 0).numpy()
            augmented_image = (augmented_image * 255).astype(np.uint8)
            augmented_image_bgr = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)

            # 증강 이미지 저장
            cv2.imwrite(os.path.join(output_dir, f"data{genIndex}.png"), augmented_image_bgr)
            with open(os.path.join(output_dir, f"data{genIndex}.txt"), 'w') as f:
                for bbox in augmented['bboxes']:
                    f.write(f"0 {' '.join(map(str, bbox))}\n")

            genIndex += 1
            print(f"Image {i}, Iteration {j+1}: Success, Saved as data{genIndex-1}.png")
        except Exception as e:
            print(f"Augmentation failed for Image {i}, Iteration {j}: {e}")
