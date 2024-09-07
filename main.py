from fastapi.staticfiles import StaticFiles
import uvicorn
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
import base64
import io
from PIL import Image
from ultralytics import YOLO
import numpy as np

app = FastAPI()
app.mount("/static", StaticFiles(directory="static_assets"), name="static")

model = YOLO("yolov8n.pt")


class ImageData(BaseModel):
    image_base64: str


def swap_channels(image_np):
    # Swap the color channels from BGR to RGB
    return image_np[..., ::-1]  # Reverse the last dimension


@app.post("/image")
def image(data: ImageData):
    try:
        # Decode the base64 image
        img_data = base64.b64decode(data.image_base64.split(",")[1])
        image = Image.open(io.BytesIO(img_data)).convert('RGB')  # Ensure RGB format

        # Process the image with YOLO
        results = model(image)

        # Convert result to RGB manually if needed
        result_image_np = results[0].plot()  # This returns a numpy array

        if result_image_np.shape[-1] == 3:  # Ensure it has 3 color channels
            # Swap BGR to RGB if necessary
            result_image_np = swap_channels(result_image_np)

        result_image = Image.fromarray(result_image_np, 'RGB')  # Ensure RGB format

        buffered = io.BytesIO()
        result_image.save(buffered, format="PNG")  # Save as PNG for better color fidelity
        result_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Extract detected class names
        detected_classes = results[0].names  # Get class names from the results
        detected_class_indices = results[0].boxes.cls.numpy()  # Get class indices from the results

        # Map indices to names
        detected_classes_list = [detected_classes[int(cls_index)] for cls_index in detected_class_indices]

        return {
            "image_base64": result_image_base64,
            "classes": detected_classes_list
        }
    except Exception as e:
        print(f"Error processing image: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

