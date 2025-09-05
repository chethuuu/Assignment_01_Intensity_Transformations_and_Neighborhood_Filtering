# q9_rice.py
import argparse
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

def save_u8(a, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a).save(p)
    print("Saved:", p)

def add_gaussian(img, sigma=15.0):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def add_salt_pepper(img, prob=0.02):
    rnd = np.random.rand(*img.shape)
    out = img.copy()
    out[rnd < prob/2] = 255
    out[rnd > 1 - prob/2] = 0
    return out

def denoise(img, noise_type):
    if noise_type == "gaussian":
        return cv2.GaussianBlur(img, (5, 5), 0)
    if noise_type == "sp":
        return cv2.medianBlur(img, 3)
    return img

def otsu_segment(img):
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw

def morph_clean(bw):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k, iterations=2)
    return closed

def count_components(bw, min_area=30):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    keep = np.zeros_like(bw, dtype=np.uint8)
    count = 0
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 255
            count += 1
    return keep, count

def main():
    ap = argparse.ArgumentParser(description="Q9: denoise → Otsu → morphology → count.")
    ap.add_argument("--input", default="assets/q9/rice.png")
    ap.add_argument("--outdir", default="outputs/q9")
    ap.add_argument("--make-noisy", choices=["none", "gaussian", "sp"], default="none")
    ap.add_argument("--gauss-sigma", type=float, default=15.0)
    ap.add_argument("--sp-prob", type=float, default=0.02)
    ap.add_argument("--min-area", type=int, default=30)
    args = ap.parse_args()

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {args.input}")
    save_u8(img, out / "0_rice_original.png")

    if args.make_noisy == "gaussian":
        noisy = add_gaussian(img, sigma=args.gauss_sigma)
        save_u8(noisy, out / "1_rice_gaussian.png")
        den = denoise(noisy, "gaussian")
    elif args.make_noisy == "sp":
        noisy = add_salt_pepper(img, prob=args.sp_prob)
        save_u8(noisy, out / "1_rice_saltpepper.png")
        den = denoise(noisy, "sp")
    else:
        noisy = img
        den = denoise(noisy, "none")

    save_u8(den, out / "2_denoised.png")

    bw = otsu_segment(den)
    save_u8(bw, out / "3_otsu_binary.png")

    bw_clean = morph_clean(bw)
    save_u8(bw_clean, out / "4_morph_clean.png")

    kept, count = count_components(bw_clean, min_area=args.min_area)
    save_u8(kept, out / "5_components_kept.png")
    with open(out / "count.txt", "w") as f:
        f.write(f"Estimated rice grain count: {count}\n")
    print(f"Estimated rice grain count: {count}")

if __name__ == "__main__":
    main()
