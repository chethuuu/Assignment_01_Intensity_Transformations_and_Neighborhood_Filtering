# q5_foreground_hist_eq_all_in_one.py
import argparse
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt


def save_gray(a: np.ndarray, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a).save(p)
    print("Saved:", p)


def save_rgb_from_bgr(a_bgr: np.ndarray, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(p)
    print("Saved:", p)


def make_all_in_one_panel(orig_bgr, H, S, V, mask, result_bgr, out_path: Path):
    """Save a 2x3 collage: Original | H | S / V | Mask | Result"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    orig_rgb   = cv2.cvtColor(orig_bgr,   cv2.COLOR_BGR2RGB)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 7))

    ax = plt.subplot(2, 3, 1); ax.imshow(orig_rgb); ax.set_title("Original"); ax.axis("off")
    ax = plt.subplot(2, 3, 2); ax.imshow(H, cmap="gray"); ax.set_title("Hue (H)"); ax.axis("off")
    ax = plt.subplot(2, 3, 3); ax.imshow(S, cmap="gray"); ax.set_title("Saturation (S)"); ax.axis("off")
    ax = plt.subplot(2, 3, 4); ax.imshow(V, cmap="gray"); ax.set_title("Value (V)"); ax.axis("off")
    ax = plt.subplot(2, 3, 5); ax.imshow(mask, cmap="gray"); ax.set_title("Mask"); ax.axis("off")
    ax = plt.subplot(2, 3, 6); ax.imshow(result_rgb); ax.set_title("Result (FG equalized)"); ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved:", out_path)


def hist_eq_foreground(img_bgr: np.ndarray, outdir: Path):
    # (a) HSV split
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    save_gray(H, outdir / "H.png")
    save_gray(S, outdir / "S.png")
    save_gray(V, outdir / "V.png")

    # (b) Otsu on S → binary mask (foreground = white)
    ret, mask = cv2.threshold(S, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    save_gray(mask, outdir / "mask.png")

    # (c) Foreground-only histogram on V
    fg_V = cv2.bitwise_and(V, V, mask=mask)
    fg_vals = fg_V[mask == 255]
    hist = np.bincount(fg_vals, minlength=256).astype(np.float64)

    # (d) CDF
    cdf = np.cumsum(hist)
    if cdf[-1] == 0:
        # no foreground detected; return original
        return img_bgr, H, S, V, mask
    cdf /= cdf[-1]

    # (e) LUT from CDF → equalize only foreground pixels
    lut = np.round(255.0 * cdf).astype(np.uint8)
    V_eq = V.copy()
    V_eq[mask == 255] = lut[V[mask == 255]]

    # (f) Recombine
    hsv_eq = cv2.merge([H, S, V_eq])
    bgr_eq = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
    return bgr_eq, H, S, V, mask


def main():
    ap = argparse.ArgumentParser(description="Q5: Foreground-only hist. equalization (all-in-one panel).")
    ap.add_argument("--input",  default="assets/jeniffer.jpg")
    ap.add_argument("--outdir", default="outputs/q5")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {args.input}")

    save_rgb_from_bgr(img, outdir / "original.png")
    result, H, S, V, mask = hist_eq_foreground(img, outdir)
    save_rgb_from_bgr(result, outdir / "result_eq_foreground.png")

    # ▶ All-in-one panel
    make_all_in_one_panel(img, H, S, V, mask, result, outdir / "q5_all_in_one.png")


if __name__ == "__main__":
    main()
