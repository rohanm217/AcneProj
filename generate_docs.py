from ultralytics import YOLO
import os

# Load your custom-trained Medium model
model_path = r'C:\Users\malig\OneDrive\Desktop\acneproj\runs\detect\skin_pro_medium_v1\weights\best.pt'
model = YOLO(model_path)

# Pointing to the VALIDATION folder for more images
source_path = r'C:\Users\malig\OneDrive\Desktop\acneproj\advanced_data\valid\images'

results = model.predict(
    source=source_path,
    save=True,          # Saves the image with bounding boxes
    conf=0.25,          # Only show detections the AI is reasonably sure about
    save_txt=False,     # We don't need raw text files for documentation
    project='runs/detect',
    name='val_evidence_gallery',
    exist_ok=True       # Overwrites the folder instead of creating 'gallery2', 'gallery3'
)

print(f"Done! Evidence gallery created in: runs/detect/val_evidence_gallery")