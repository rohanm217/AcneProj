SkinScan AI

AI-powered skin condition detector that identifies acne-related conditions from photos and recommends AM/PM skincare routines.

Built with YOLOv8, FastAPI, and CustomTkinter.

Features
10-class detection — pimples, blackheads, cystic acne, folliculitis, keloids, and more
Skincare recommendations — condition-specific AM/PM product suggestions
Severity scoring — weighted score from Clear → Critical
Ingredient conflict warnings — flags risky product combinations
Desktop app — upload, webcam, scan history, before/after compare, PDF export
Web app — drag-and-drop UI served by FastAPI
Quick Start
1. Install dependencies
bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
2. Get model weights

Weights are not committed to git. Either use your existing trained weights or train a new model:

bash
python train_advanced.py

This saves weights to runs/detect/skin_pro_medium_v1/weights/best.pt.

3. Run the app

Desktop (full features):

bash
python main_gui.py

Web API + browser UI:

bash
python main.py

Then open http://localhost:8000.

Project Structure
acneproj/
├── main_gui.py          # Desktop app (history, compare, PDF)
├── main.py              # FastAPI web server
├── config.py            # Paths, recommendations, severity logic
├── train_advanced.py    # Train YOLOv8m model (canonical)
├── metrics.py           # Evaluate model on validation set
├── predict.py           # Run inference on a single image
├── test_model.py        # Smoke test (model loads + predicts)
├── generate_docs.py     # Generate validation evidence gallery
├── generate_results.py  # Generate test-set prediction gallery
├── data.yaml            # Dataset config for YOLO
├── advanced_data/       # Train/val/test images + labels
├── static_assets/       # Web UI (index.html)
└── runs/detect/         # Training outputs (gitignored)
Model
Property	Value
Architecture	YOLOv8m
Epochs	175
Classes	10
Train images	~4,300
Val images	~308

Final training metrics (epoch 175), read directly from the trained checkpoint:

Metric	Value
mAP@50	0.3457
mAP@50-95	0.1758
Precision	0.4430
Recall	0.3418

Run a fresh evaluation:

bash
python metrics.py
Results

Real output from generate_results.py running the trained model on unseen test images — boxes and labels are the model's own predictions, not hand-annotated.

<p align="center"> <img src="docs/results/example_1.jpg" width="45%" alt="SkinScan AI detecting Pimples, papular, and purulent conditions on a test image"> <img src="docs/results/example_2.jpg" width="45%" alt="SkinScan AI detecting multiple papular and purulent conditions on a dense acne cluster"> </p>

The model correctly separates multiple condition types in the same photo (Pimples, papular, purulent) with confidence scores in the 0.35–0.88 range. On denser clusters (right image), it produces overlapping/duplicate boxes on the same lesion — a known limitation at the current NMS/confidence settings, and a reasonable next step for improving precision.

API
Endpoint	Method	Description
/	GET	Web UI
/health	GET	Server and model status
/image	POST	Analyze a base64-encoded image

Example request body:

json
{
  "image_base64": "<base64 string>",
  "conf": 0.35
}
Training
bash
python train_advanced.py

Requires a CUDA GPU (device=0 in the script). To train on CPU, change device=0 to device='cpu' in train_advanced.py.

Utility Scripts
bash
python predict.py path/to/photo.jpg          # Single-image inference
python test_model.py                         # Verify model loads and runs
python generate_docs.py                      # Validation gallery → runs/detect/val_evidence_gallery
python generate_results.py                   # Test gallery → runs/detect/final_test_gallery
Disclaimer

SkinScan AI is for informational purposes only and is not medical advice. Detection accuracy is limited by the training data and model performance. Always consult a licensed dermatologist for diagnosis and treatment.

Known Limitations
Class label crystanlline is a dataset typo (crystalline acne) baked into the trained model
Test set is very small (2 images) — use the validation set for meaningful evaluation
Model weights (.pt) and training runs are gitignored; clone + train to get started fresh
Tech Stack

Python · YOLOv8 (Ultralytics) · PyTorch · FastAPI · CustomTkinter · SQLite · OpenCV · fpdf2