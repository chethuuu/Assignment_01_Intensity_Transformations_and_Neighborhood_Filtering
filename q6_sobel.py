import argparse
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

def load_gray_u8(p):
    return np.array(Image.open(p).convert("L"), dtype=np.uint8)

def save_u8(a, p):
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a).save(p)

def to_u8_vis(x):
    x = np.abs(x).astype(np.float32)
    m = x.max() if x.max() > 0 else 1.0
    return np.clip(255.0 * (x / m), 0, 255).astype(np.uint8)

def conv2d(img, k):
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    pad = np.pad(img.astype(np.float32), ((ph, ph), (pw, pw)), mode="edge")
    out = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j] = np.sum(pad[i:i+kh, j:j+kw] * k)
    return out

def conv1d_sep(img, kv, kh):
    # vertical then horizontal
    kv = kv.reshape(-1, 1).astype(np.float32)
    kh = kh.reshape(1, -1).astype(np.float32)
    t = conv2d(img, kv)
    return conv2d(t, kh)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="assets/einstein.png")
    ap.add_argument("--outdir", default="outputs/q6")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    img = load_gray_u8(args.input)

    # Sobel kernels
    KX = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=np.float32)
    KY = KX.T

    # (a) cv2.filter2D
    gx_a = cv2.filter2D(img.astype(np.float32), -1, KX)
    gy_a = cv2.filter2D(img.astype(np.float32), -1, KY)
    mag_a = np.hypot(gx_a, gy_a)
    save_u8(to_u8_vis(gx_a), outdir / "a_gx.png")
    save_u8(to_u8_vis(gy_a), outdir / "a_gy.png")
    save_u8(to_u8_vis(mag_a), outdir / "a_mag.png")

    # (b) custom 2D convolution
    gx_b = conv2d(img, KX)
    gy_b = conv2d(img, KY)
    mag_b = np.hypot(gx_b, gy_b)
    save_u8(to_u8_vis(gx_b), outdir / "b_gx.png")
    save_u8(to_u8_vis(gy_b), outdir / "b_gy.png")
    save_u8(to_u8_vis(mag_b), outdir / "b_mag.png")

    # (c) separable property: [[1],[2],[1]] * [1,0,-1]
    kv = np.array([1,2,1], dtype=np.float32)
    kh = np.array([1,0,-1], dtype=np.float32)
    gx_c = conv1d_sep(img, kv, kh)        # Gx
    gy_c = conv1d_sep(img, kh, kv)        # Gy (transpose order)
    mag_c = np.hypot(gx_c, gy_c)
    save_u8(to_u8_vis(gx_c), outdir / "c_gx.png")
    save_u8(to_u8_vis(gy_c), outdir / "c_gy.png")
    save_u8(to_u8_vis(mag_c), outdir / "c_mag.png")

if __name__ == "__main__":
    main()
