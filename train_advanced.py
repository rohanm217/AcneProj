from ultralytics import YOLO

def train_professional_model():
    model = YOLO('yolov8m.pt')

    model.train(
        data=r'C:\Users\malig\OneDrive\Desktop\acneproj\data.yaml',
        epochs=175,
        imgsz=640,
        batch=16,
        patience=50,
        optimizer='AdamW',
        device=0,
        name='skin_pro_medium_v1'
    )

if __name__ == '__main__':
    train_professional_model()