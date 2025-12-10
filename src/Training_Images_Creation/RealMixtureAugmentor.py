import glob
import os
import random
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm


# -----------------------
# PATHS
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data")

INPUT_FOLDER = os.path.join(DATA_PATH, "Labeled Images", "mixture")
OUTPUT_BASE = os.path.join(DATA_PATH, "Real_Mixtures_Augmented")

TRAIN_FOLDER = os.path.join(OUTPUT_BASE, "training")
TEST_FOLDER  = os.path.join(OUTPUT_BASE, "testing")

os.makedirs(TRAIN_FOLDER, exist_ok=True)
os.makedirs(TEST_FOLDER, exist_ok=True)

# Load images (png + jpg + jpeg)
image_paths = (
    glob.glob(os.path.join(INPUT_FOLDER, "*.png")) +
    glob.glob(os.path.join(INPUT_FOLDER, "*.jpg")) +
    glob.glob(os.path.join(INPUT_FOLDER, "*.jpeg"))
)

print("INPUT FOLDER:", INPUT_FOLDER)
print("TOTAL IMAGES FOUND:", len(image_paths))
print("TRAIN FOLDER:", TRAIN_FOLDER)
print("TEST FOLDER :", TEST_FOLDER)


# -----------------------
# TRAIN / TEST SPLIT (30% TEST)
# -----------------------
random.seed(42)
random.shuffle(image_paths)

test_size  = int(0.3 * len(image_paths))
test_images  = image_paths[:test_size]
train_images = image_paths[test_size:]

print("TRAIN IMAGES:", len(train_images))
print("TEST IMAGES :", len(test_images))


# -----------------------
# AUGMENTATION CONFIG
# -----------------------
AUG_PER_IMAGE = 5  # only applied to TRAINING images


# -----------------------
# CROP
# -----------------------
def crop_to_inner_box_fast(image, margin_ratio=0.03):
    H, W = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    edges = cv2.Canny(blur, 40, 120)

    kernel = np.ones((25, 25), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image

    main = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(main)

    if w < 0.4 * W or h < 0.4 * H:
        return image

    mx = int(w * margin_ratio)
    my = int(h * margin_ratio)

    x1 = max(0, x + mx)
    y1 = max(0, y + my)
    x2 = min(W, x + w - mx)
    y2 = min(H, y + h - my)

    cropped = image[y1:y2, x1:x2]

    if cropped.size == 0:
        return image

    return cropped


# -----------------------
# AUGMENTATION
# -----------------------
augmenter = A.Compose([
    A.Affine(
        translate_percent=(0.02, 0.02),
        scale=(0.95, 1.05),
        rotate=0,
        mode=cv2.BORDER_CONSTANT,
        cval=(0, 0, 0),
        p=1.0
    ),
    A.RandomBrightnessContrast(
        brightness_limit=0.03,
        contrast_limit=0.03,
        p=0.3
    )
])


def rotate_discrete(image):
    angle = np.random.choice([0, 90, 180, 270])
    if angle == 0:
        return image
    elif angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    else:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


# -----------------------
# PROCESS TRAINING IMAGES (W/ AUGMENT)
# -----------------------
print("\nPROCESSING TRAINING IMAGES...")

for img_path in tqdm(train_images):
    filename = os.path.basename(img_path)
    base_name = os.path.splitext(filename)[0]

    image = cv2.imread(img_path)
    if image is None:
        continue

    cropped = crop_to_inner_box_fast(image)

    for i in range(AUG_PER_IMAGE):
        rotated = rotate_discrete(cropped)
        augmented = augmenter(image=rotated)["image"]

        out_path = os.path.join(
            TRAIN_FOLDER,
            f"{base_name}_aug_{i}.png"
        )

        cv2.imwrite(out_path, augmented)


# -----------------------
# PROCESS TEST IMAGES (W/o AUGMENT)
# -----------------------
print("\nPROCESSING TEST IMAGES (W/o AUGMENT)...")

for img_path in tqdm(test_images):
    filename = os.path.basename(img_path)
    base_name = os.path.splitext(filename)[0]

    image = cv2.imread(img_path)
    if image is None:
        continue

    cropped = crop_to_inner_box_fast(image)

    out_path = os.path.join(
        TEST_FOLDER,
        f"{base_name}.png"
    )

    cv2.imwrite(out_path, cropped)


print("\nREAL MIXTURE DATASET READY!")
print("Training images  ->", TRAIN_FOLDER)
print("Testing images   ->", TEST_FOLDER)