from pathlib import Path
import argparse
import glob
import numpy as np
from PIL import Image

# ---------------- basics ----------------


def to_u8(img: Image.Image) -> np.ndarray:
    a = np.array(img)
    if a.ndim == 3 and a.shape[2] > 3:
        a = a[:, :, :3]
    return a.astype(np.uint8)


def as_3d(a: np.ndarray) -> np.ndarray:
    return a[:, :, None] if a.ndim == 2 else a


def save(arr: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)
    print("Saved:", path)

# ---------------- zoom methods ----------------


def zoom_nearest(src: np.ndarray, s: float) -> np.ndarray:
    if not (0 < s <= 10):
        raise ValueError("s in (0,10].")
    src3 = as_3d(src)
    H, W, C = src3.shape
    Ho, Wo = max(1, int(round(H * s))), max(1, int(round(W * s)))
    out = np.empty((Ho, Wo, C), dtype=np.uint8)
    for i in range(Ho):
        x = min(H - 1, int(round(i / s)))
        for j in range(Wo):
            y = min(W - 1, int(round(j / s)))
            out[i, j] = src3[x, y]
    return out if C > 1 else out[:, :, 0]


def zoom_bilinear(src: np.ndarray, s: float) -> np.ndarray:
    if not (0 < s <= 10):
        raise ValueError("s in (0,10].")
    src3 = as_3d(src).astype(np.float32)
    H, W, C = src3.shape
    Ho, Wo = max(1, int(round(H * s))), max(1, int(round(W * s)))
    out = np.empty((Ho, Wo, C), dtype=np.float32)
    for i in range(Ho):
        x = i / s
        x0, x1 = int(np.floor(x)), min(int(np.floor(x)) + 1, H - 1)
        ax = x - x0
        for j in range(Wo):
            y = j / s
            y0, y1 = int(np.floor(y)), min(int(np.floor(y)) + 1, W - 1)
            ay = y - y0
            p00, p01 = src3[x0, y0], src3[x0, y1]
            p10, p11 = src3[x1, y0], src3[x1, y1]
            top = (1 - ay) * p00 + ay * p01
            bot = (1 - ay) * p10 + ay * p11
            out[i, j] = (1 - ax) * top + ax * bot
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out if C > 1 else out[:, :, 0]

# ---------------- metrics ----------------


def normalized_ssd(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError("Shape mismatch.")
    a, b = a.astype(np.float32), b.astype(np.float32)
    return float(np.mean((a - b) ** 2) / (255.0 ** 2))

# ---------------- helpers ----------------


def crop_center(x: np.ndarray, H: int, W: int) -> np.ndarray:
    h, w = x.shape[:2]
    y0, x0 = (h - H) // 2, (w - W) // 2
    return x[y0:y0+H, x0:x0+W]


def find_pairs():
    exts = ("png", "jpg", "jpeg", "bmp")
    smalls = []
    for e in exts:
        smalls += glob.glob(f"*_*small*.{e}") + glob.glob(f"*_small.{e}")
    # map {base: path}
    larges = {}
    for e in exts:
        for p in glob.glob(f"*_*large*.{e}") + glob.glob(f"*_large.{e}"):
            base = Path(p).stem.replace("_large", "")
            larges[base] = p
    pairs = []
    for s in smalls:
        base = Path(s).stem.replace("_small", "")
        if base in larges:
            pairs.append((s, larges[base], base))
    return pairs

# ---------------- CLI ----------------


def main():
    ap = argparse.ArgumentParser(
        description="Image zoom (nearest/bilinear) + 4x SSD test")
    ap.add_argument("--in", dest="inp", help="Input image path")
    ap.add_argument("--scale", type=float, default=2.0, help="s in (0,10]")
    ap.add_argument(
        "--method", choices=["nearest", "bilinear"], default="bilinear")
    ap.add_argument("--out", dest="outp", help="Output path")
    ap.add_argument("--run-ssd-test", action="store_true",
                    help="Compare *_small.* upscaled x4 vs *_large.* originals")
    args = ap.parse_args()

    if args.run_ssd_test:
        pairs = find_pairs()
        if not pairs:
            print("No *_small.* / *_large.* pairs found in CWD.")
            return
        print("Image\t\tSSD(nearest)\tSSD(bilinear)")
        outdir = Path("out_q7")
        for sp, lp, base in pairs:
            small = to_u8(Image.open(sp))
            large = to_u8(Image.open(lp))
            zn = zoom_nearest(small, 4.0)
            zb = zoom_bilinear(small, 4.0)
            H, W = min(large.shape[0], zn.shape[0], zb.shape[0]), \
                min(large.shape[1], zn.shape[1], zb.shape[1])
            Lc, Nc, Bc = crop_center(large, H, W), crop_center(
                zn, H, W), crop_center(zb, H, W)
            ssd_n, ssd_b = normalized_ssd(Nc, Lc), normalized_ssd(Bc, Lc)
            print(f"{base}\t\t{ssd_n:.6f}\t{ssd_b:.6f}")
            save(zn, outdir / f"{base}_x4_nearest.png")
            save(zb, outdir / f"{base}_x4_bilinear.png")
        print("Note: lower SSD = better (closer to original).")
        return

    if not args.inp:
        print("Provide --in <image> or use --run-ssd-test")
        return

    img = to_u8(Image.open(args.inp))
    out = zoom_nearest(
        img, args.scale) if args.method == "nearest" else zoom_bilinear(img, args.scale)
    out_path = Path(args.outp) if args.outp else Path(
        f"zoom_{args.method}_s{args.scale:.2f}.png")
    save(out, out_path)


if __name__ == "__main__":
    main()
