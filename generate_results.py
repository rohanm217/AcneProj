from ultralytics import YOLO
import os

def generate_project_gallery():
    model = YOLO(r'C:\Users\malig\OneDrive\Desktop\acneproj\runs\detect\skin_pro_medium_v1\weights\best.pt')
    
    # Run prediction on the entire test folder
    results = model.predict(
        source=r'C:\Users\malig\OneDrive\Desktop\acneproj\advanced_data\test\images',
        conf=0.25,
        save=True,      # This saves the images with boxes drawn on them
        line_width=2,
        project='runs/detect',
        name='final_test_gallery'
    )
    
    print("Gallery generated in: runs/detect/final_test_gallery")

if __name__ == '__main__':
    generate_project_gallery()