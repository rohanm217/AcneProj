from ultralytics import YOLO

# Load a model
model = YOLO(r"runs\detect\train4\weights\best.pt")  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model(["datasets/trainingdata/images/train/9e9bde8e-imagesoffruit2.jpg", "datasets/trainingdata/images/train/36d50ad3-appleimage3.jpg"])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk