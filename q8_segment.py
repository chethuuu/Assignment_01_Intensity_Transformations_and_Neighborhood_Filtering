import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

def save_rgb(bgr, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(path)
    print("Saved:", path)

def save_gray(gray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(gray).save(path)
    print("Saved:", path)

def add_label(bgr, text):
    out = bgr.copy()
    cv2.putText(out, text, (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 2, cv2.LINE_AA)
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="assets/flower.png")
    parser.add_argument("--outdir", default="outputs/q8")
    args = parser.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(args.input)
    h, w = img.shape[:2]

    # (a) GrabCut segmentation (init with rectangle)
    mask = np.zeros((h, w), np.uint8)
    rect = (10, 10, w - 20, h - 20)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # binary mask: 1=foreground, 0=background
    mask_bin = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype(np.uint8)
    save_gray((mask_bin * 255).astype(np.uint8), outdir / "mask.png")

    fg = img * mask_bin[:, :, None]
    bg = img * (1 - mask_bin)[:, :, None]
    save_rgb(img, outdir / "original.png")
    save_rgb(fg,  outdir / "foreground.png")
    save_rgb(bg,  outdir / "background.png")

    # (b) Blur background only
    blurred = cv2.GaussianBlur(img, (21, 21), 0)
    enhanced = fg + blurred * (1 - mask_bin)[:, :, None]
    save_rgb(enhanced, outdir / "enhanced.png")

    # ---- All-in-one 2x3 panel ----
    mask_vis_bgr = cv2.cvtColor((mask_bin * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    row1 = np.hstack([
        add_label(img,       "Original"),
        add_label(mask_vis_bgr, "Mask"),
        add_label(fg,        "Foreground"),
    ])
    row2 = np.hstack([
        add_label(bg,        "Background"),
        add_label(enhanced,  "Enhanced (blurred BG)"),
        add_label(blurred,   "Blurred (full)"),
    ])
    panel = np.vstack([row1, row2])
    save_rgb(panel, outdir / "q8_all_in_one.png")

if __name__ == "__main__":
    main()
