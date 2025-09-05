import argparse
from pathlib import Path
import numpy as np
import cv2 as cv

# ---- tiny utils ----
def load_gray(p):
    img = cv.imread(str(p), cv.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Input not found: {p}")
    return img

def ensure_dir(p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)

def save(img, path):
    ensure_dir(path)
    cv.imwrite(str(path), img)
    print("Saved:", path)

def norm255(a):
    a = np.abs(a).astype(np.float64)
    m = a.max()
    return (a / m * 255.0).astype(np.uint8) if m > 0 else np.zeros_like(a, dtype=np.uint8)

def mag255(dx, dy):
    m = np.hypot(dx, dy)
    mm = m.max()
    return (m / mm * 255.0).astype(np.uint8) if mm > 0 else np.zeros_like(m, dtype=np.uint8)

# ---- (a) Sobel via filter2D ----
def sobel_filter2d(gray):
    kx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]], dtype=np.float64)
    ky = np.array([[ 1,  2,  1],
                   [ 0,  0,  0],
                   [-1, -2, -1]], dtype=np.float64)
    dx = cv.filter2D(gray, cv.CV_64F, kx, borderType=cv.BORDER_REPLICATE)
    dy = cv.filter2D(gray, cv.CV_64F, ky, borderType=cv.BORDER_REPLICATE)
    return dx, dy, mag255(dx, dy)

# ---- (b) Sobel via manual conv ----
def conv2d_naive(img, k):
    h, w = img.shape
    kh, kw = k.shape
    py, px = kh // 2, kw // 2
    pad = np.pad(img.astype(np.float64), ((py, py), (px, px)), mode="edge")
    out = np.zeros((h, w), dtype=np.float64)
    kf = np.flipud(np.fliplr(k.astype(np.float64)))  # conv = flip kernel
    for y in range(h):
        for x in range(w):
            out[y, x] = np.sum(pad[y:y+kh, x:x+kw] * kf)
    return out

def sobel_manual(gray):
    kx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]], dtype=np.float64)
    ky = np.array([[ 1,  2,  1],
                   [ 0,  0,  0],
                   [-1, -2, -1]], dtype=np.float64)
    dx = conv2d_naive(gray, kx)
    dy = conv2d_naive(gray, ky)
    return dx, dy, mag255(dx, dy)

# ---- (c) Sobel via separable filters ----
# Kx = g^T * d,  Ky = d^T * g  with g=[1,2,1], d=[1,0,-1]
def sobel_separable(gray):
    g = np.array([1, 2, 1], dtype=np.float64)
    d = np.array([1, 0, -1], dtype=np.float64)
    dx = cv.sepFilter2D(gray, cv.CV_64F, d, g, borderType=cv.BORDER_REPLICATE)  # horiz d, vert g
    dy = cv.sepFilter2D(gray, cv.CV_64F, g, d, borderType=cv.BORDER_REPLICATE)  # horiz g, vert d
    return dx, dy, mag255(dx, dy)

def main():
    p = argparse.ArgumentParser(description="Q6 – Sobel: filter2D, manual, separable")
    p.add_argument("--input",  default="assets/einstein.png")
    p.add_argument("--outdir", default="outputs/q6")
    args = p.parse_args()

    outdir = Path(args.outdir)
    I = load_gray(args.input)
    save(I, outdir / "q6_input_gray.png")

    # (a) filter2D
    dx_a, dy_a, mag_a = sobel_filter2d(I)
    save(norm255(dx_a), outdir / "q6a_dx_filter2d.png")
    save(norm255(dy_a), outdir / "q6a_dy_filter2d.png")
    save(mag_a,         outdir / "q6a_mag_filter2d.png")

    # (b) manual
    dx_b, dy_b, mag_b = sobel_manual(I)
    save(norm255(dx_b), outdir / "q6b_dx_manual.png")
    save(norm255(dy_b), outdir / "q6b_dy_manual.png")
    save(mag_b,         outdir / "q6b_mag_manual.png")

    # (c) separable
    dx_c, dy_c, mag_c = sobel_separable(I)
    save(norm255(dx_c), outdir / "q6c_dx_separable.png")
    save(norm255(dy_c), outdir / "q6c_dy_separable.png")
    save(mag_c,         outdir / "q6c_mag_separable.png")

    # quick numeric sanity check
    print("max |Δ(filter2D - manual)|     dx:", float(np.max(np.abs(dx_a - dx_b))),
          "dy:", float(np.max(np.abs(dy_a - dy_b))))
    print("max |Δ(filter2D - separable)|  dx:", float(np.max(np.abs(dx_a - dx_c))),
          "dy:", float(np.max(np.abs(dy_a - dy_c))))
    print("Done. Results in:", outdir.resolve())

if __name__ == "__main__":
    main()
