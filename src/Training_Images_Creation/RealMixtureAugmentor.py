import glob
import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm


# -----------------------
# PATHS
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "data"))

real_mixture_path = os.path.join(data_path, "Labeled Images", "mixture")
augmented_mixtures_path = os.path.join(data_path, "Real_Mixtures_Augmented")

os.makedirs(augmented_mixtures_path, exist_ok=True)

mixture_paths = (
    glob.glob(os.path.join(real_mixture_path, "*.png")) +
    glob.glob(os.path.join(real_mixture_path, "*.jpg")) +
    glob.glob(os.path.join(real_mixture_path, "*.jpeg"))
)

print("INPUT:", real_mixture_path)
print("OUTPUT:", augmented_mixtures_path)
print("IMAGES FOUND:", len(mixture_paths))


# -----------------------
# AUGMENTATION CONFIG
# -----------------------
AUG_PER_IMAGE = 5


# -----------------------
# GEOMETRY HELPERS
# -----------------------
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left

    return rect


def crop_to_inner_box_with_perspective(image, margin_ratio=0.03):
    """
    Detects the main box using contours.
    Applies perspective correction only if a large quadrilateral is found.
    Otherwise falls back to a bounding-box crop.
    Always crops slightly inside to avoid box edges and floor.
    """

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
    area = cv2.contourArea(main)

    if area < 0.4 * H * W:
        return image

    peri = cv2.arcLength(main, True)
    approx = cv2.approxPolyDP(main, 0.02 * peri, True)

    if len(approx) == 4:
        pts = approx.reshape(4, 2).astype("float32")
        rect = order_points(pts)

        (tl, tr, br, bl) = rect

        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))

        if maxWidth > 0.5 * W and maxHeight > 0.5 * H:
            dst = np.array(
                [
                    [0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1],
                ],
                dtype="float32",
            )

            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(
                image,
                M,
                (maxWidth, maxHeight),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )

            h, w = warped.shape[:2]
            mx = int(w * margin_ratio)
            my = int(h * margin_ratio)

            warped = warped[my:h - my, mx:w - mx]

            if warped.shape[0] > 0.4 * H and warped.shape[1] > 0.4 * W:
                return warped

    x, y, w, h = cv2.boundingRect(main)

    mx = int(w * margin_ratio)
    my = int(h * margin_ratio)

    x1 = max(0, x + mx)
    y1 = max(0, y + my)
    x2 = min(W, x + w - mx)
    y2 = min(H, y + h - my)

    cropped = image[y1:y2, x1:x2]

    if cropped.shape[0] < 0.4 * H or cropped.shape[1] < 0.4 * W:
        return image

    return cropped


# -----------------------
# PURE PHOTO-LIKE AUGMENTATION
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
# MAIN LOOP
# -----------------------
for img_path in tqdm(mixture_paths):
    filename = os.path.basename(img_path)
    base_name = os.path.splitext(filename)[0]

    image = cv2.imread(img_path)
    if image is None:
        print("FAILED TO LOAD:", img_path)
        continue

    cropped = crop_to_inner_box_with_perspective(image, margin_ratio=0.03)

    for i in range(AUG_PER_IMAGE):
        rotated = rotate_discrete(cropped)
        augmented = augmenter(image=rotated)["image"]

        out_path = os.path.join(
            augmented_mixtures_path,
            f"{base_name}_aug_{i}.png"
        )

        cv2.imwrite(out_path, augmented)

print("Augmentation completed.")