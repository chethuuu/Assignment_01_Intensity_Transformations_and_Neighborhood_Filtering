import argparse
from pathlib import Path
import numpy as np
import cv2
from PIL import Image


def save_gray(a, p):
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a).save(p)


def save_rgb(a, p):
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cv2.cvtColor(a, cv2.COLOR_BGR2RGB)).save(p)


def hist_eq_foreground(img_bgr, outdir: Path):
    # (a) Convert to HSV and split into H, S, V planes
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    save_gray(H, outdir / "H.png")
    save_gray(S, outdir / "S.png")
    save_gray(V, outdir / "V.png")

    # (b) Threshold S channel to create a binary mask for the foreground
    _, thr = cv2.threshold(S, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (S > thr).astype(np.uint8) * 255
    save_gray(mask, outdir / "mask.png")

    # (c) Obtain foreground values of V using the mask and compute histogram
    fg_vals = V[mask == 255]
    hist = np.bincount(fg_vals, minlength=256).astype(np.float64)

    # (d) Compute cumulative histogram (CDF)
    cdf = np.cumsum(hist)
    cdf /= cdf[-1]

    # (e) Build LUT from CDF and histogram-equalize the foreground
    lut = np.round(255.0 * cdf).astype(np.uint8)
    V_eq = V.copy()
    V_eq[mask == 255] = lut[V[mask == 255]]

    # (f) Recombine with background and save final result
    hsv_eq = cv2.merge([H, S, V_eq])
    bgr_eq = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
    return bgr_eq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/assests/jeniffer.jpg")
    ap.add_argument("--outdir", default="outputs/q5")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    img = cv2.imread(args.input, cv2.IMREAD_COLOR)

    save_rgb(img, outdir / "original.png")
    result = hist_eq_foreground(img, outdir)
    save_rgb(result, outdir / "result_eq_foreground.png")


if __name__ == "__main__":
    main()
