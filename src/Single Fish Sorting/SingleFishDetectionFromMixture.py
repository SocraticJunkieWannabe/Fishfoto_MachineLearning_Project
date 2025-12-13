import os
import random
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Optional display (works without OpenCV GUI)
SHOW_WITH_MATPLOTLIB = False
if SHOW_WITH_MATPLOTLIB:
    import matplotlib.pyplot as plt


# -----------------------
# CONFIG
# -----------------------
MODEL_PATH = "yolo11x-seg.pt"   # replace with your fish-trained model if you have one
CONF = 0.15
IOU = 0.5
IMGSZ = 1280
MAX_DET = 300

SAVE_ANNOTATED = True
SAVE_CROPS = True
CROP_MODE = "bbox"             # "bbox" or "mask" (mask requires seg model)
PAD = 8


# -----------------------
# HELPERS
# -----------------------
def clip(v, lo, hi):
    return max(lo, min(hi, v))


def find_project_root(start_dir: Path) -> Path:
    for p in [start_dir] + list(start_dir.parents):
        if (p / "data").exists():
            return p
    raise RuntimeError("Could not find project root containing a 'data' folder.")


def pick_random_mixture_image(project_root: Path) -> str:
    input_folder = project_root / "data" / "Labeled Images" / "mixture"
    candidates = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        candidates += list(input_folder.glob(ext))

    print("INPUT FOLDER:", str(input_folder))
    print("IMAGES FOUND:", len(candidates))

    if not candidates:
        raise RuntimeError(f"No images found in: {input_folder}")

    chosen = random.choice(candidates)
    print("USING RANDOM IMAGE:", str(chosen))
    return str(chosen)


def ensure_output_dir(project_root: Path) -> Path:
    out_dir = project_root / "data" / "SingleFishDetection_Test"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("OUTPUT DIR:", str(out_dir))
    return out_dir


# -----------------------
# MAIN
# -----------------------
def main():
    base_dir = Path(__file__).resolve().parent
    project_root = find_project_root(base_dir)

    out_dir = ensure_output_dir(project_root)
    image_path = pick_random_mixture_image(project_root)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    H, W = img.shape[:2]
    stem = os.path.splitext(os.path.basename(image_path))[0]

    model = YOLO(MODEL_PATH)

    results = model.predict(
        source=image_path,
        conf=CONF,
        iou=IOU,
        imgsz=IMGSZ,
        max_det=MAX_DET,
        verbose=False
    )

    r = results[0]
    names = model.names

    # Annotated image (BGR)
    annotated = r.plot()

    # Print detections summary
    if r.boxes is None or len(r.boxes) == 0:
        print("No detections -> crops will not be saved.")
    else:
        print(f"Detections: {len(r.boxes)}")
        for i, b in enumerate(r.boxes):
            cls_id = int(b.cls[0].item()) if b.cls is not None else -1
            conf = float(b.conf[0].item()) if b.conf is not None else 0.0
            label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
            xyxy = b.xyxy[0].cpu().numpy().astype(int).tolist()
            print(f"  #{i:03d}  class={label}  conf={conf:.2f}  box={xyxy}")

    # Save annotated
    if SAVE_ANNOTATED:
        out_annot = out_dir / f"{stem}_annotated.png"
        cv2.imwrite(str(out_annot), annotated, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print("Saved annotated image:", str(out_annot))

    # Save crops
    if SAVE_CROPS and r.boxes is not None and len(r.boxes) > 0:
        crops_dir = out_dir / f"{stem}_crops"
        crops_dir.mkdir(parents=True, exist_ok=True)

        masks = None
        if getattr(r, "masks", None) is not None:
            masks = r.masks.data.cpu().numpy()

        for i, b in enumerate(r.boxes):
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int).tolist()

            x1 = clip(x1 - PAD, 0, W - 1)
            y1 = clip(y1 - PAD, 0, H - 1)
            x2 = clip(x2 + PAD, 0, W - 1)
            y2 = clip(y2 + PAD, 0, H - 1)

            crop = img[y1:y2, x1:x2].copy()
            if crop.size == 0:
                continue

            cls_id = int(b.cls[0].item()) if b.cls is not None else -1
            conf = float(b.conf[0].item()) if b.conf is not None else 0.0
            label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]

            if CROP_MODE == "mask" and masks is not None and i < len(masks):
                m = masks[i]  # HxW
                m_crop = m[y1:y2, x1:x2]
                crop = crop * (m_crop[..., None] > 0.5)

            out_crop = crops_dir / f"{stem}_fish_{i:03d}_{label}_c{conf:.2f}.png"
            cv2.imwrite(str(out_crop), crop, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        print("Saved crops in:", str(crops_dir))

    # Optional display via matplotlib (no OpenCV GUI needed)
    if SHOW_WITH_MATPLOTLIB:
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 7))
        plt.imshow(rgb)
        plt.title("YOLO detections")
        plt.axis("off")
        plt.show()

    # Optional: open the annotated image with Windows default viewer
    # (comment out if you don't want it)
    try:
        if SAVE_ANNOTATED:
            os.startfile(str(out_annot))
    except Exception:
        pass


if __name__ == "__main__":
    main()
