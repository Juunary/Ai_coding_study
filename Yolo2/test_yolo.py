from ultralytics import YOLO
import cv2

def test_model(image_path):
    # 학습된 모델 로드
    model = YOLO('yolov8n_transfer_learning5.pt')
    
    # 이미지 로드 및 객체 검출
    image = cv2.imread(image_path)
    results = model(image)

    # 바운딩 박스 시각화
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            class_name = result.names[int(box.cls[0])]
            confidence = round(float(box.conf[0]), 2)
            label = f'{class_name} {confidence}'
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_model('./test1.png')
