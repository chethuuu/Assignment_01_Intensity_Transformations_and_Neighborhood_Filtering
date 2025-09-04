import os
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

try:
    import cv2
    has_cv2 = True
except Exception:
    has_cv2 = False

# ---- paths ----
BASE_DIR = os.path.dirname(__file__)
IMG_PATH = os.path.join(BASE_DIR, "assets", "daisy.jpg")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

assert os.path.exists(IMG_PATH), f"Image not found: {IMG_PATH}"

# ---- load image ----
if has_cv2:
    img_bgr = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError("Failed to load image with OpenCV.")
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
else:
    img = np.array(Image.open(IMG_PATH).convert("RGB"))

h, w = img.shape[:2]

def savefig(fig, name):
    out = os.path.join(OUT_DIR, f"{name}.png")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out

# ---- segmentation ----
if has_cv2:
    mask = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    margin_y, margin_x = int(0.14*h), int(0.14*w)
    rect = (margin_x, margin_y, w - 2*margin_x, h - 2*margin_y)

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask_bin = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
else:
    # fallback k-means (very rough)
    arr = img.reshape(-1, 3).astype(np.float32)
    np.random.seed(0)
    centers = arr[np.random.choice(len(arr), 2, replace=False)]
    for _ in range(10):
        dists = np.sqrt(((arr[:, None, :] - centers[None, :, :])**2).sum(-1))
        labels = dists.argmin(1)
        new_centers = np.vstack([arr[labels==k].mean(0) if np.any(labels==k) else centers[k] for k in range(2)])
        if np.allclose(new_centers, centers, atol=1e-3):
            break
        centers = new_centers
    counts = np.bincount(labels, minlength=2)
    fg_label = np.argmin(counts)
    mask_bin = (labels.reshape(h, w) == fg_label).astype(np.uint8)

fg = (img * mask_bin[..., None]).astype(np.uint8)
bg = (img * (1 - mask_bin[..., None])).astype(np.uint8)
mask_vis = (mask_bin * 255).astype(np.uint8)

# ---- blur background & composite ----
if has_cv2:
    blurred = cv2.GaussianBlur(img, (51, 51), 0)
else:
    blurred = np.array(Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=8)))

enhanced = (fg + (blurred * (1 - mask_bin[..., None]))).astype(np.uint8)

# ---- plots ----
fig1 = plt.figure(figsize=(10, 3))
plt.subplot(1,3,1); plt.imshow(img); plt.title("Original"); plt.axis("off")
plt.subplot(1,3,2); plt.imshow(mask_vis, cmap="gray"); plt.title("Segmentation Mask"); plt.axis("off")
plt.subplot(1,3,3); plt.imshow(fg); plt.title("Foreground"); plt.axis("off")
p1 = savefig(fig1, "grabcut_mask_and_fg")

fig2 = plt.figure(figsize=(10, 3))
plt.subplot(1,3,1); plt.imshow(bg); plt.title("Background Only"); plt.axis("off")
plt.subplot(1,3,2); plt.imshow(blurred); plt.title("Blurred Image"); plt.axis("off")
plt.subplot(1,3,3); plt.imshow(enhanced); plt.title("Enhanced (Blurred Background)"); plt.axis("off")
p2 = savefig(fig2, "bg_blur_and_composite")

print("Saved:", p1)
print("Saved:", p2)
