import argparse
from pathlib import Path
import numpy as np
import cv2 as cv
from PIL import Image

# ---------- utils ----------
def ensure_gray(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.uint8)

def to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.abs(arr).astype(np.float64)
    m = arr.max()
    if m > 0:
        arr = arr / m * 255.0
    return arr.astype(np.uint8)

def grad_mag(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    mag = np.hypot(dx, dy)
    m = mag.max()
    if m > 0:
        mag = mag / m * 255.0
    return mag.astype(np.uint8)

def save(img: np.ndarray, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(p)
    print(f"Saved: {p}")

# ---------- (a) Sobel via filter2D ----------
def sobel_filter2d(gray: np.ndarray):
    kx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]], dtype=np.float32)
    ky = np.array([[ 1,  2,  1],
                   [ 0,  0,  0],
                   [-1, -2, -1]], dtype=np.float32)
    dx = cv.filter2D(gray, cv.CV_64F, kx, borderType=cv.BORDER_REPLICATE)
    dy = cv.filter2D(gray, cv.CV_64F, ky, borderType=cv.BORDER_REPLICATE)
    return dx, dy, grad_mag(dx, dy)

# ---------- (b) Sobel via my own convolution ----------
def conv2d_naive(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    h, w = img.shape
    kh, kw = k.shape
    py, px = kh // 2, kw // 2
    pad = np.pad(img, ((py, py), (px, px)), mode="edge").astype(np.float64)
    out = np.zeros((h, w), dtype=np.float64)
    kf = np.flipud(np.fliplr(k.astype(np.float64)))  # convolution flips the kernel
    for y in range(h):
        for x in range(w):
            out[y, x] = np.sum(pad[y:y+kh, x:x+kw] * kf)
    return out

def sobel_manual(gray: np.ndarray):
    kx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]], dtype=np.float64)
    ky = np.array([[ 1,  2,  1],
                   [ 0,  0,  0],
                   [-1, -2, -1]], dtype=np.float64)
    dx = conv2d_naive(gray, kx)
    dy = conv2d_naive(gray, ky)
    return dx, dy, grad_mag(dx, dy)

# ---------- (c) Sobel via separable property ----------
# [ [1,0,-1],[2,0,-2],[1,0,-1] ] = [1,2,1]^T * [1,0,-1]
def sobel_separable(gray: np.ndarray):
    g = np.array([1, 2, 1], dtype=np.float32)   # smoothing
    d = np.array([1, 0, -1], dtype=np.float32)  # derivative

    # Gx: vertical smoothing then horizontal derivative
    tmp_x = cv.sepFilter2D(gray, cv.CV_64F, kernelX=np.array([1], dtype=np.float32), kernelY=g)
    dx    = cv.sepFilter2D(tmp_x, cv.CV_64F, kernelX=d, kernelY=np.array([1], dtype=np.float32))

    # Gy: vertical derivative then horizontal smoothing
    tmp_y = cv.sepFilter2D(gray, cv.CV_64F, kernelX=np.array([1], dtype=np.float32), kernelY=d)
    dy    = cv.sepFilter2D(tmp_y, cv.CV_64F, kernelX=g, kernelY=np.array([1], dtype=np.float32))

    return dx, dy, grad_mag(dx, dy)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Q6 – Sobel filtering: filter2D, manual, separable")
    ap.add_argument("--input", default="assets/einstein.png", help="Path to Einstein image")
    ap.add_argument("--outdir", default="outputs/q6", help="Directory for results")
    args = ap.parse_args()

    img_path = Path(args.input)
    if not img_path.exists():
        raise FileNotFoundError(f"Input not found: {img_path}")

    I = ensure_gray(img_path)
    outdir = Path(args.outdir)
    save(I, outdir / "q6_input_gray.png")

    # (a)
    dx_a, dy_a, mag_a = sobel_filter2d(I)
    save(to_uint8(dx_a), outdir / "q6a_dx_filter2d.png")
    save(to_uint8(dy_a), outdir / "q6a_dy_filter2d.png")
    save(mag_a,            outdir / "q6a_mag_filter2d.png")

    # (b)
    dx_b, dy_b, mag_b = sobel_manual(I)
    save(to_uint8(dx_b), outdir / "q6b_dx_manual.png")
    save(to_uint8(dy_b), outdir / "q6b_dy_manual.png")
    save(mag_b,            outdir / "q6b_mag_manual.png")

    # (c)
    dx_c, dy_c, mag_c = sobel_separable(I)
    save(to_uint8(dx_c), outdir / "q6c_dx_separable.png")
    save(to_uint8(dy_c), outdir / "q6c_dy_separable.png")
    save(mag_c,            outdir / "q6c_mag_separable.png")

    # small numeric sanity check
    print("max |Δ(filter2D - manual)|     dx:", np.max(np.abs(dx_a - dx_b)),
          "dy:", np.max(np.abs(dy_a - dy_b)))
    print("max |Δ(filter2D - separable)|  dx:", np.max(np.abs(dx_a - dx_c)),
          "dy:", np.max(np.abs(dy_a - dy_c)))
    print(f"✅ Done. Results saved in {outdir.resolve()}")

if __name__ == "__main__":
    main()
