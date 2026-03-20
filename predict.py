from ultralytics import YOLO
import os

model = YOLO(r'C:\Users\malig\OneDrive\Desktop\acneproj\runs\detect\train3\weights\best.pt')

source_img = r'C:\Users\malig\OneDrive\Desktop\acneproj\test_skin_acne.jpg'

results = model.predict(
    source=source_img,
    conf=0.25,
    save=True,
    project='runs/detect',
    name='predictions'
)

for result in results:
    print(f"Found {len(result.boxes)} skin issues!")
    for box in result.boxes:
        class_id = int(box.cls[0])
        label = model.names[class_id]
        confidence = float(box.conf[0])
        print(f"- {label}: {confidence:.2f}")

print(f"Results saved to: {results[0].save_dir}")   