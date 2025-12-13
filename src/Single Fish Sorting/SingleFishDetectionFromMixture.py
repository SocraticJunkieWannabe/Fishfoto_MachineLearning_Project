import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO


# -----------------------
# CONFIG
# -----------------------
MODEL_PATH = "yolo11x-seg.pt"
CONF = 0.15
IOU = 0.5
IMGSZ = 1280
MAX_DET = 300


def find_project_root(start_dir: Path) -> Path:
    for p in [start_dir] + list(start_dir.parents):
        if (p / "data").exists():
            return p
    raise RuntimeError("Could not find project root containing a 'data' folder.")


def main():
    base_dir = Path(__file__).resolve().parent
    project_root = find_project_root(base_dir)

    mixture_dir = project_root / "data" / "Labeled Images" / "mixture"

    candidates = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        candidates += list(mixture_dir.glob(ext))

    print("MIXTURE DIR:", str(mixture_dir))
    print("FOUND:", len(candidates))

    if not candidates:
        raise RuntimeError(f"No images found in: {mixture_dir}")

    image_path = random.choice(candidates)
    print("USING IMAGE:", str(image_path))

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    model = YOLO(MODEL_PATH)
    results = model.predict(
        source=str(image_path),
        conf=CONF,
        iou=IOU,
        imgsz=IMGSZ,
        max_det=MAX_DET,
        verbose=False
    )

    r = results[0]

    # Get an annotated image with boxes drawn by Ultralytics
    annotated_bgr = r.plot() 

    # Print detections
    if r.boxes is None or len(r.boxes) == 0:
        print("No detections.")
    else:
        names = model.names
        print("Detections:", len(r.boxes))
        for i, b in enumerate(r.boxes):
            cls_id = int(b.cls[0].item()) if b.cls is not None else -1
            conf = float(b.conf[0].item()) if b.conf is not None else 0.0
            label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
            xyxy = b.xyxy[0].cpu().numpy().astype(int).tolist()
            print(f"  #{i:03d}  {label}  conf={conf:.2f}  box={xyxy}")

    # Display with matplotlib (no cv2.imshow needed)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_rgb)
    plt.title(f"YOLO detections: {image_path.name}")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()