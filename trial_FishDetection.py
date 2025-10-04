import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os

folderPath = "./Images/"
image_path = f"{folderPath}T18 kilu 20241019_120112897.jpg"

# -----------------------------
# 1. Load image
# -----------------------------
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# -----------------------------
# 2. Load YOLOv5 model
# -----------------------------
model = YOLO('yolov5s.pt')

# -----------------------------
# 3. Run detection
# -----------------------------
results_list = model.predict(image_path)  # returns a list
results = results_list[0]  # get the first (and only) result

# -----------------------------
# 4. Create output folder
# -----------------------------
output_dir = "detected_fishes"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# 5. Process detections
# -----------------------------
for i, box in enumerate(results.boxes):
    xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
    confidence = float(box.conf[0])
    label = model.names[int(box.cls[0])]

    # Draw rectangle & label
    cv2.rectangle(image_rgb, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    cv2.putText(image_rgb, label, (xmin, ymin-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

    # Crop and save individual fish
    fish_crop = image_rgb[ymin:ymax, xmin:xmax]
    cv2.imwrite(os.path.join(output_dir, f"{label}_{i}.jpg"), cv2.cvtColor(fish_crop, cv2.COLOR_RGB2BGR))

# -----------------------------
# 6. Show final image
# -----------------------------
plt.figure(figsize=(12, 8))
plt.imshow(image_rgb)
plt.axis("off")
plt.show()

print(f"Individual fishes saved in '{output_dir}' folder.")
