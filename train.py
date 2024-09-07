import torch
from ultralytics import YOLO

# Check available GPUs
print("Number of GPUs available:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Load a pretrained YOLO model
model = YOLO("yolov8n.pt")  # load a pretrained model

# Train the model using the dedicated GPU (e.g., GPU 0)
results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device="0")  # Change "0" to the index of the dedicated GPU
