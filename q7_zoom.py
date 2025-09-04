from pathlib import Path
import argparse
import glob
import numpy as np
from PIL import Image


def _to_numpy_uint8(img: Image.Image) -> np.ndarray:
    """Convert a PIL image to np.uint8 array (H,W[,C])."""
    arr = np.array(img)
    if arr.ndim == 2:
        return arr.astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] > 3:  
        arr = arr[:, :, :3]
    return arr.astype(np.uint8)


def _ensure_3d(arr: np.ndarray) -> np.ndarray:
    """Make array 3D (H,W,1) for grayscale to unify math."""
    if arr.ndim == 2:
        return arr[:, :, None]
    return arr


def zoom_nearest(src: np.ndarray, s: float) -> np.ndarray:
    """Nearest-neighbor zoom. src: uint8 (H,W[,C]). s in (0,10]."""
    if s <= 0 or s > 10:
        raise ValueError("Scale s must be in (0, 10].")
    src = _ensure_3d(src)
    H, W, C = src.shape
    Ho = max(1, int(round(H * s)))
    Wo = max(1, int(round(W * s)))
    out = np.zeros((Ho, Wo, C), dtype=np.uint8)

    for i in range(Ho):
        x = i / s
        xi = int(round(x))
        xi = min(max(xi, 0), H - 1)
        for j in range(Wo):
            y = j / s
            yj = int(round(y))
            yj = min(max(yj, 0), W - 1)
            out[i, j] = src[xi, yj]
    return out if src.shape[2] > 1 else out[:, :, 0]


def zoom_bilinear(src: np.ndarray, s: float) -> np.ndarray:
    """Bilinear interpolation zoom. src: uint8 (H,W[,C]). s in (0,10]."""
    if s <= 0 or s > 10:
        raise ValueError("Scale s must be in (0, 10].")
    src = _ensure_3d(src)
    H, W, C = src.shape
    Ho = max(1, int(round(H * s)))
    Wo = max(1, int(round(W * s)))
    out = np.zeros((Ho, Wo, C), dtype=np.float32)

    src_f = src.astype(np.float32)

    for i in range(Ho):
        x = i / s
        x0 = int(np.floor(x))
        x1 = min(x0 + 1, H - 1)
        ax = x - x0  # weight in x
        for j in range(Wo):
            y = j / s
            y0 = int(np.floor(y))
            y1 = min(y0 + 1, W - 1)
            ay = y - y0  # weight in y

            # four neighbors
            p00 = src_f[x0, y0]
            p01 = src_f[x0, y1]
            p10 = src_f[x1, y0]
            p11 = src_f[x1, y1]

            # bilinear blend
            top = (1 - ay) * p00 + ay * p01
            bot = (1 - ay) * p10 + ay * p11
            out[i, j] = (1 - ax) * top + ax * bot

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out if src.shape[2] > 1 else out[:, :, 0]


def normalized_ssd(a: np.ndarray, b: np.ndarray) -> float:
    """
    Normalized sum of squared differences:
    mean((a - b)^2) / 255^2  -> range ~ [0,1].
    Inputs must be same shape, uint8 or castable.
    """
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    a_f = a.astype(np.float32)
    b_f = b.astype(np.float32)
    ssd = np.mean((a_f - b_f) ** 2) / (255.0 ** 2)
    return float(ssd)


def save_image(arr: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def run_ssd_test():
    """
    Finds every '*_small.*' image, 4x upscales it with both methods,
    compares to matching '*_large.*' image, and prints normalized SSD.
    """
    print("== SSD test (4x upscaling) ==")
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    smalls = []
    for e in exts:
        smalls.extend(glob.glob(f"*__small.{e.split('.')[-1]}"))
        smalls.extend(glob.glob(f"*_{'small'}{e[1:]}"))  # generic pattern
    # Also catch names like foo_small.png with a single underscore
    for e in exts:
        smalls.extend(glob.glob(f"*_*small*{e[1:]}"))
    # Deduplicate while preserving order
    seen, smalls_unique = set(), []
    for s in smalls:
        if s not in seen and "_small" in s:
            smalls_unique.append(s)
            seen.add(s)

    if not smalls_unique:
        # Fallback: simple pattern
        for e in exts:
            smalls_unique.extend(glob.glob(f"*_*small{e[1:]}"))
    if not smalls_unique:
        print("No '*_small.*' files found in the current folder.")
        return

    out_dir = Path("out_q7")
    results = []
    for spath in smalls_unique:
        sp = Path(spath)
        stem = sp.stem.replace("_small", "")
        # try to find the corresponding large file by extension probing
        large_path = None
        for e in (".png", ".jpg", ".jpeg", ".bmp"):
            cand = Path(f"{stem}_large{e}")
            if cand.exists():
                large_path = cand
                break
        if large_path is None:
            print(f"Skipping '{sp.name}': matching '*_large.*' not found.")
            continue

        small_img = _to_numpy_uint8(Image.open(sp))
        large_img = _to_numpy_uint8(Image.open(large_path))

        nn_up = zoom_nearest(small_img, 4.0)
        bl_up = zoom_bilinear(small_img, 4.0)

        # If shapes don’t match exactly (rounding), center-crop to min size
        Ht, Wt = large_img.shape[:2]
        Hn, Wn = nn_up.shape[:2]
        Hb, Wb = bl_up.shape[:2]
        Hmin, Wmin = min(Ht, Hn, Hb), min(Wt, Wn, Wb)

        def center_crop(x):
            H, W = x.shape[:2]
            y0 = (H - Hmin) // 2
            x0 = (W - Wmin) // 2
            return x[y0:y0+Hmin, x0:x0+Wmin]

        large_c = center_crop(large_img)
        nn_c = center_crop(nn_up)
        bl_c = center_crop(bl_up)

        ssd_nn = normalized_ssd(nn_c, large_c)
        ssd_bl = normalized_ssd(bl_c, large_c)
        results.append((stem, ssd_nn, ssd_bl))

        # save outputs for visual check
        save_image(nn_up, out_dir / f"{stem}_x4_nearest.png")
        save_image(bl_up, out_dir / f"{stem}_x4_bilinear.png")

    if results:
        print("Image\t\tSSD (nearest)\tSSD (bilinear)")
        for name, s1, s2 in results:
            print(f"{name}\t\t{s1:.6f}\t{s2:.6f}")
        print("\nNote: lower SSD means closer to the original.")
    else:
        print("No pairs processed.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", type=str, help="Input image path")
    p.add_argument("--scale", type=float, default=2.0,
                   help="Scale factor s (0,10]")
    p.add_argument(
        "--method", choices=["nearest", "bilinear"], default="bilinear")
    p.add_argument("--out", dest="out_path", type=str,
                   help="Output path (optional)")
    p.add_argument("--run-ssd-test", action="store_true",
                   help="Run 4x SSD test on *_small/*_large pairs in CWD")
    args = p.parse_args()

    if args.run_ssd_test:
        run_ssd_test()
        return

    if not args.in_path:
        print("Provide --in <image> or use --run-ssd-test")
        return

    img = _to_numpy_uint8(Image.open(args.in_path))
    s = args.scale
    if args.method == "nearest":
        out = zoom_nearest(img, s)
    else:
        out = zoom_bilinear(img, s)

    out_path = Path(args.out_path) if args.out_path else Path(
        f"zoom_{args.method}_s{s:.2f}.png")
    save_image(out, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
