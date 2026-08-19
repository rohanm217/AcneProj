from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import base64
import io
from PIL import Image
from ultralytics import YOLO
import uvicorn

from config import MODEL_PATH, RECOMMENDATIONS, compute_severity, check_conflicts

BASE_DIR = Path(__file__).parent
app = FastAPI(title="SkinScan AI")

static_dir = BASE_DIR / "static_assets"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

model = YOLO(str(MODEL_PATH) if MODEL_PATH.exists() else "yolov8n.pt")
model_loaded = MODEL_PATH.exists()


class ImageData(BaseModel):
    image_base64: str
    conf: float = 0.35


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "model_path": str(MODEL_PATH),
        "classes": list(model.names.values()) if model_loaded else [],
    }


@app.get("/")
def index():
    html_path = BASE_DIR / "static_assets" / "index.html"
    if html_path.exists():
        return FileResponse(str(html_path))
    return {"message": "SkinScan AI API is running. POST /image to analyse skin."}


@app.post("/image")
def analyze_image(data: ImageData):
    try:
        raw = data.image_base64
        img_bytes = base64.b64decode(raw.split(",")[1] if "," in raw else raw)
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        results = model.predict(pil_image, conf=data.conf, verbose=False)

        result_np = results[0].plot()[..., ::-1]  # BGR -> RGB
        buf = io.BytesIO()
        Image.fromarray(result_np, "RGB").save(buf, format="PNG")
        result_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        names = results[0].names
        detections = [
            {"class": names[int(box.cls[0])], "confidence": round(float(box.conf[0]), 3)}
            for box in results[0].boxes
        ]

        unique_classes = list({d["class"] for d in detections})
        score, label = compute_severity(unique_classes)

        return {
            "image_base64": result_b64,
            "detections": detections,
            "recommendations": {
                cls: RECOMMENDATIONS.get(cls, {"AM": "Gentle Care", "PM": "Consult Professional"})
                for cls in unique_classes
            },
            "severity": {"score": score, "label": label},
            "conflicts": check_conflicts(unique_classes),
        }
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)