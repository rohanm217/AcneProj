from pathlib import Path
from fastapi.staticfiles import StaticFiles
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import base64
import io
from PIL import Image
from ultralytics import YOLO

app = FastAPI()
app.mount("/static", StaticFiles(directory="static_assets"), name="static")

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "runs" / "detect" / "skin_pro_medium_v1" / "weights" / "best.pt"
model = YOLO(str(MODEL_PATH) if MODEL_PATH.exists() else "yolov8n.pt")

RECOMMENDATIONS = {
    'Pimples':     {'AM': 'Salicylic Acid Cleanser',              'PM': 'Benzoyl Peroxide 5% Spot Treatment'},
    'blackhead':   {'AM': 'BHA Liquid Exfoliant',                 'PM': 'Double Cleanse (Oil + Water)'},
    'conglobata':  {'AM': 'Gentle Hydrating Cleanser',            'PM': 'URGENT: Requires Prescription (Isotretinoin)'},
    'crystanlline':{'AM': 'Centella Asiatica Soothing Mist',      'PM': 'Ceramide Barrier Repair Cream'},
    'cystic':      {'AM': 'Anti-inflammatory Serum (Niacinamide)','PM': 'Adapalene 0.1% Gel + Deep Hydration'},
    'folliculitis':{'AM': 'Antibacterial Wash (Benzoyl Peroxide)', 'PM': 'Warm Compress + Mupirocin (if prescribed)'},
    'keloid':      {'AM': 'Silicone Gel / Sheet',                 'PM': 'Consult Professional for Corticosteroid info'},
    'milium':      {'AM': 'Mild Lactic Acid Exfoliation',         'PM': 'Do Not Squeeze - Professional Extraction Only'},
    'papular':     {'AM': 'Azelaic Acid 10% Suspension',          'PM': 'Zinc-based Soothing Cream'},
    'purulent':    {'AM': 'Hydrocolloid Pimple Patch',            'PM': 'Gentle Non-foaming Cleanser (Avoid physical scrubs)'},
}

SEVERITY_WEIGHTS = {
    'milium': 1, 'blackhead': 1, 'crystanlline': 1,
    'Pimples': 2, 'papular': 2,
    'purulent': 3, 'folliculitis': 3,
    'cystic': 4, 'keloid': 4,
    'conglobata': 5,
}


def compute_severity(unique_classes: list[str]) -> dict:
    score = sum(SEVERITY_WEIGHTS.get(c, 1) for c in unique_classes)
    if score == 0:
        label = "Clear"
    elif score <= 2:
        label = "Mild"
    elif score <= 6:
        label = "Moderate"
    elif score <= 12:
        label = "Severe"
    else:
        label = "Critical"
    return {"score": score, "label": label}


class ImageData(BaseModel):
    image_base64: str


@app.post("/image")
def image(data: ImageData):
    try:
        raw = data.image_base64
        img_bytes = base64.b64decode(raw.split(",")[1] if "," in raw else raw)
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        results = model(pil_image)

        result_np = results[0].plot()
        result_np = result_np[..., ::-1]  # BGR -> RGB
        result_image = Image.fromarray(result_np, "RGB")

        buffered = io.BytesIO()
        result_image.save(buffered, format="PNG")
        result_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        names = results[0].names
        detections = [
            {"class": names[int(box.cls[0])], "confidence": round(float(box.conf[0]), 3)}
            for box in results[0].boxes
        ]

        unique_classes = list({d["class"] for d in detections})
        recommendations = {cls: RECOMMENDATIONS.get(cls, {"AM": "Gentle Care", "PM": "Consult Professional"})
                           for cls in unique_classes}

        return {
            "image_base64": result_b64,
            "detections": detections,
            "recommendations": recommendations,
            "severity": compute_severity(unique_classes),
        }
    except Exception as e:
        print(f"Error processing image: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
