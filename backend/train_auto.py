"""
Fine-tune YOLO11n on auto-rickshaw dataset.
"""

from pathlib import Path
from ultralytics import YOLO
import shutil
import os

data_yaml = Path("c:/Users/Dhruv/Downloads/SIH Prototype/sarthi-vision/backend/data/yolo_auto/data.yaml")
model_save_dir = Path("c:/Users/Dhruv/Downloads/SIH Prototype/sarthi-vision/backend/app/models")
model_save_dir.mkdir(parents=True, exist_ok=True)

# Load pretrained base model
model = YOLO("yolo11n.pt")

print("Starting YOLO11n training on auto-rickshaw dataset...")
results = model.train(
    data=str(data_yaml),
    epochs=5,
    imgsz=416,
    batch=8,
    device="cpu",
    workers=2,
    project="c:/Users/Dhruv/Downloads/SIH Prototype/sarthi-vision/backend/runs",
    name="auto_rickshaw_train",
    exist_ok=True,
    verbose=True
)

# Copy best model to app/models/yolo11n-autorickshaw.pt
best_pt = Path("c:/Users/Dhruv/Downloads/SIH Prototype/sarthi-vision/backend/runs/auto_rickshaw_train/weights/best.pt")
target_pt = model_save_dir / "yolo11n-autorickshaw.pt"

if best_pt.exists():
    shutil.copy2(best_pt, target_pt)
    print(f"Successfully saved fine-tuned auto-rickshaw model to {target_pt}")
else:
    last_pt = Path("c:/Users/Dhruv/Downloads/SIH Prototype/sarthi-vision/backend/runs/auto_rickshaw_train/weights/last.pt")
    if last_pt.exists():
        shutil.copy2(last_pt, target_pt)
        print(f"Saved last checkpoint to {target_pt}")
    else:
        print("Training completed. Checking weights directory...")
