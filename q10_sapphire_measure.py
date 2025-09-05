
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# -------------------- Config --------------------
IMAGE_PATH = "./assets/sapphire.jpg"
OUT_DIR = "outputs_q10"
os.makedirs(OUT_DIR, exist_ok=True)

# Camera / geometry
F_MM = 8.0
Z_MM = 480.0
PIXEL_PITCH_UM = 1.12

# Min component size (px) to keep (avoid tiny noise)
MIN_AREA_PX = 200

# -------------------- Helpers --------------------


def fill_holes(binary_u8: np.ndarray) -> np.ndarray:
    h, w = binary_u8.shape
    inv = cv2.bitwise_not(binary_u8)
    flood = inv.copy()
    fmask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(flood, fmask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(binary_u8, holes)
    return filled


def connected_components(mask_u8: np.ndarray, min_area=MIN_AREA_PX):
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (mask_u8 > 0).astype(np.uint8), connectivity=8
    )
    areas, bboxes, ids = [], [], []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            areas.append(area)
            bboxes.append(stats[i, :4])
            ids.append(i)
    return areas, bboxes, ids, labels


def split_into_two(mask_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask_u8 > 0)
    coords = np.stack([xs, ys], axis=1).astype(np.float32)
    if coords.shape[0] < 2:
        return mask_u8.copy(), np.zeros_like(mask_u8)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    K = 2
    _, labels_k, _ = cv2.kmeans(
        coords, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    labels_k = labels_k.flatten()

    mask1 = np.zeros_like(mask_u8, dtype=np.uint8)
    mask2 = np.zeros_like(mask_u8, dtype=np.uint8)
    mask1[ys[labels_k == 0], xs[labels_k == 0]] = 255
    mask2[ys[labels_k == 1], xs[labels_k == 1]] = 255
    return mask1, mask2


def px_area_to_mm2(area_px: int, f_mm: float, z_mm: float, pixel_pitch_um: float) -> float:
    p_mm = pixel_pitch_um / 1000.0
    scale = (z_mm / f_mm) ** 2
    return area_px * scale * (p_mm ** 2)


# -------------------- 1) Load image --------------------
assert os.path.exists(IMAGE_PATH), f"Image not found: {IMAGE_PATH}"
bgr = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
assert bgr is not None, "Failed to read image with OpenCV."
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

# -------------------- 2) Segmentation (color + morphology) --------------------
lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
L, a, b = cv2.split(lab)
b_inv = 255 - b
_, mask0 = cv2.threshold(b_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Morphology
ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
mask_clean = cv2.morphologyEx(mask0, cv2.MORPH_OPEN, ker, iterations=1)
mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, ker, iterations=2)

# (b) Fill holes
mask_filled = fill_holes(mask_clean)

# -------------------- 3) Connected components (try straight; else split) --------------------
areas_px, bboxes, ids, labels = connected_components(mask_filled, MIN_AREA_PX)

# If we didn’t get 2 components, split by spatial k-means
if len(areas_px) != 2:
    m1, m2 = split_into_two(mask_filled)
    # Recompute stats on each split mask
    areas1, _, _, _ = connected_components(m1, MIN_AREA_PX)
    areas2, _, _, _ = connected_components(m2, MIN_AREA_PX)
    # If either split is empty, fall back to raw counts
    if len(areas1) == 0 or len(areas2) == 0:
        # keep original (even if 1 blob)
        masks_final = [mask_filled]
        areas_final_px = areas_px if areas_px else [
            int((mask_filled > 0).sum())]
    else:
        masks_final = [m1, m2]
        areas_final_px = [int((m1 > 0).sum()), int((m2 > 0).sum())]
else:
    # Already two components
    masks_final = []
    for comp_id in ids:
        masks_final.append((labels == comp_id).astype(np.uint8) * 255)
    areas_final_px = areas_px

# Sort by area (desc) for stable reporting
order = np.argsort(areas_final_px)[::-1]
areas_final_px = [areas_final_px[i] for i in order]
masks_final = [masks_final[i] for i in order]

# -------------------- 4) Convert px areas -> mm^2 --------------------
areas_mm2 = [px_area_to_mm2(a, F_MM, Z_MM, PIXEL_PITCH_UM)
             for a in areas_final_px]

# -------------------- 5) Visualizations & outputs --------------------
# Pipeline figure
fig = plt.figure(figsize=(12, 6))
plt.subplot(231)
plt.imshow(rgb)
plt.title("Original")
plt.axis("off")
plt.subplot(232)
plt.imshow(b_inv, cmap="gray")
plt.title("Lab b (inverted)")
plt.axis("off")
plt.subplot(233)
plt.imshow(mask0, cmap="gray")
plt.title("Otsu threshold")
plt.axis("off")
plt.subplot(234)
plt.imshow(mask_clean, cmap="gray")
plt.title("Morph clean")
plt.axis("off")
plt.subplot(235)
plt.imshow(mask_filled, cmap="gray")
plt.title("Holes filled")
plt.axis("off")
# Overlay boxes/contours for final masks
overlay = rgb.copy()
for mk in masks_final:
    cnts, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, cnts, -1, (255, 0, 0), 2)
plt.subplot(236)
plt.imshow(overlay)
plt.title("Final regions")
plt.axis("off")

pipeline_path = os.path.join(OUT_DIR, "sapphire_segmentation_pipeline.png")
fig.savefig(pipeline_path, bbox_inches="tight", dpi=160)
plt.close(fig)

# Save each final mask (optional)
for i, mk in enumerate(masks_final, 1):
    cv2.imwrite(os.path.join(OUT_DIR, f"sapphire_mask_{i}.png"), mk)

# Print results
print("=== Q10 Results ===")
print(
    f"Pixel pitch: {PIXEL_PITCH_UM:.2f} µm   |   f = {F_MM} mm,  Z = {Z_MM} mm")
for i, (a_px, a_mm2) in enumerate(zip(areas_final_px, areas_mm2), 1):
    print(
        f"Sapphire {i}:  area_px = {a_px:,} px   ->   area = {a_mm2:.2f} mm^2")

print(f"\nSaved pipeline figure: {pipeline_path}")
for i in range(len(masks_final)):
    print(
        f"Saved mask {i+1}: {os.path.join(OUT_DIR, f'sapphire_mask_{i+1}.png')}")
