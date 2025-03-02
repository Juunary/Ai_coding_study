from ultralytics import YOLO

def train_model():
    # YOLOv8 모델 로드
    model = YOLO('yolov8n_transfer_learning4.pt')

    # 백본 일부 동결
    for param in model.model.model[:6].parameters():
        param.requires_grad = False
    for param in model.model.model[6:].parameters():
        param.requires_grad = True

    # 모델 학습
    model.train(
        data='./path.yaml',
        epochs=20,
        imgsz=1080,
        batch=8,
        name='yolov8s_search_box_detection',
        save=True,
        save_period=5,
        save_dir='runs/train',
        augment=True,
        lr0=0.001,
        lrf=0.01,
    )

    model.save('yolov8n_transfer_learning5.pt')

if __name__ == "__main__":
    train_model()
