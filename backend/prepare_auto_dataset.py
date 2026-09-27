"""
Convert FGVD annotations with auto-rickshaw to YOLO format and train YOLO11.
"""

import glob
import json
import os
import shutil
from pathlib import Path

# Paths
BASE_DIR = Path("c:/Users/Dhruv/Downloads/SIH Prototype")
FGVD_DIR = BASE_DIR / "fgvd_extracted"
OUTPUT_DIR = BASE_DIR / "sarthi-vision/backend/data/yolo_auto"

os.makedirs(OUTPUT_DIR / "images/train", exist_ok=True)
os.makedirs(OUTPUT_DIR / "images/val", exist_ok=True)
os.makedirs(OUTPUT_DIR / "labels/train", exist_ok=True)
os.makedirs(OUTPUT_DIR / "labels/val", exist_ok=True)

# Find all annotations with autorickshaw
ann_files = glob.glob(str(FGVD_DIR / "**/ann/*.json"), recursive=True)
valid_samples = []

for a_path in ann_files:
    with open(a_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check if autorickshaw is present
    objects = data.get("objects", [])
    auto_objs = [o for o in objects if o.get("classTitle") == "autorickshaw"]

    if not auto_objs:
        continue

    # Find corresponding image
    base_name = Path(a_path).name.replace(".json", "")
    parent = Path(a_path).parent.parent
    img_path = parent / "img" / base_name

    if not img_path.exists():
        continue

    width = data["size"]["width"]
    height = data["size"]["height"]

    # Convert boxes to YOLO format (normalized x_center, y_center, w, h)
    yolo_lines = []
    for o in auto_objs:
        ext = o["points"]["exterior"]
        x1 = min(ext[0][0], ext[1][0])
        y1 = min(ext[0][1], ext[1][1])
        x2 = max(ext[0][0], ext[1][0])
        y2 = max(ext[0][1], ext[1][1])

        # Clamp
        x1 = max(0, min(width, x1))
        y1 = max(0, min(height, y1))
        x2 = max(0, min(width, x2))
        y2 = max(0, min(height, y2))

        box_w = x2 - x1
        box_h = y2 - y1

        if box_w < 5 or box_h < 5:
            continue

        x_center = (x1 + box_w / 2.0) / width
        y_center = (y1 + box_h / 2.0) / height
        norm_w = box_w / width
        norm_h = box_h / height

        # Class 0: auto-rickshaw
        yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

    if yolo_lines:
        valid_samples.append((img_path, base_name, yolo_lines))

print(f"Total valid samples with autorickshaws: {len(valid_samples)}")

# Split train / val (80 / 20)
split_idx = int(len(valid_samples) * 0.8)
train_samples = valid_samples[:split_idx]
val_samples = valid_samples[split_idx:]

for img_p, base, lines in train_samples:
    dest_img = OUTPUT_DIR / "images/train" / base
    dest_lbl = OUTPUT_DIR / "labels/train" / (Path(base).stem + ".txt")
    shutil.copy2(img_p, dest_img)
    with open(dest_lbl, "w") as f:
        f.write("\n".join(lines))

for img_p, base, lines in val_samples:
    dest_img = OUTPUT_DIR / "images/val" / base
    dest_lbl = OUTPUT_DIR / "labels/val" / (Path(base).stem + ".txt")
    shutil.copy2(img_p, dest_img)
    with open(dest_lbl, "w") as f:
        f.write("\n".join(lines))

# Create data.yaml
yaml_content = f"""path: {OUTPUT_DIR.resolve().as_posix()}
train: images/train
val: images/val

names:
  0: auto-rickshaw
"""

yaml_path = OUTPUT_DIR / "data.yaml"
with open(yaml_path, "w") as f:
    f.write(yaml_content)

print(f"Created data.yaml at {yaml_path}")
print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")
