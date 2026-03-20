from ultralytics import YOLO
import torch

def main():
    model = YOLO("yolov8n.pt")
    
    results = model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        device=0,
        workers=8,
        batch=16
    )

if __name__ == "__main__":
    main()