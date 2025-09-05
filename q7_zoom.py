import argparse
from pathlib import Path
import numpy as np
from PIL import Image


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

# part (a): nearest neighbor interpolation
def zoom_nearest(src: np.ndarray, s: float) -> np.ndarray:
    if not (0 < s <= 10):
        raise ValueError("s in (0,10].")
    src3 = as_3d(src)
    H, W, C = src3.shape
    Ho, Wo = max(1, int(round(H * s))), max(1, int(round(W * s)))
    ys = np.clip((np.arange(Ho) / s).round().astype(int), 0, H - 1)
    xs = np.clip((np.arange(Wo) / s).round().astype(int), 0, W - 1)
    out = src3[ys[:, None], xs[None, :], :]
    out = out if src.ndim == 3 else out[:, :, 0]
    return out.astype(np.uint8)

# part (b): bilinear interpolation
def zoom_bilinear(src: np.ndarray, s: float) -> np.ndarray:
    if not (0 < s <= 10):
        raise ValueError("s in (0,10].")
    src3 = as_3d(src).astype(np.float32)
    H, W, C = src3.shape
    Ho, Wo = max(1, int(round(H * s))), max(1, int(round(W * s)))

    y = np.arange(Ho, dtype=np.float32) / s
    x = np.arange(Wo, dtype=np.float32) / s
    y0 = np.floor(y).astype(int); x0 = np.floor(x).astype(int)
    y1 = np.clip(y0 + 1, 0, H - 1); x1 = np.clip(x0 + 1, 0, W - 1)
    y0 = np.clip(y0, 0, H - 1);     x0 = np.clip(x0, 0, W - 1)

    wy = (y - y0).reshape(-1, 1, 1)
    wx = (x - x0).reshape(1, -1, 1)

    Ia = src3[y0[:, None], x0[None, :], :]
    Ib = src3[y0[:, None], x1[None, :], :]
    Ic = src3[y1[:, None], x0[None, :], :]
    Id = src3[y1[:, None], x1[None, :], :]

    top = Ia * (1 - wx) + Ib * wx
    bot = Ic * (1 - wx) + Id * wx
    out = top * (1 - wy) + bot * wy
    out = out if src.ndim == 3 else out[:, :, 0]
    return np.clip(out, 0, 255).astype(np.uint8)


def nssd(a: np.ndarray, b: np.ndarray) -> float:
    a3, b3 = as_3d(a.astype(np.float32)), as_3d(b.astype(np.float32))
    if a3.shape != b3.shape:
        raise ValueError("Shapes must match for SSD.")
    diff2 = (a3 - b3) ** 2
    return float(diff2.mean() / (255.0 ** 2))


def run_zoom(input_path: Path, s: float, method: str, out_path: Path):
    img = to_u8(Image.open(input_path))
    if method == "nearest":
        out = zoom_nearest(img, s)
    elif method == "bilinear":
        out = zoom_bilinear(img, s)
    else:
        raise ValueError("method must be 'nearest' or 'bilinear'")
    save(out, out_path)


def run_ssd_tests(root: Path):
    pairs = [
        (root / "im01small.png", root / "im01.png"),
        (root / "im02small.png", root / "im02.png"),
        (root / "im03small.png", root / "im03.png"),
        (root / "taylor_small.jpg", root / "taylor.jpg"),
    ]
    for small, large in pairs:
        if not (small.exists() and large.exists()):
            continue
        sm = to_u8(Image.open(small))
        lg = to_u8(Image.open(large))
        up_near = zoom_nearest(sm, 4.0)
        up_blin = zoom_bilinear(sm, 4.0)
        if up_near.shape != lg.shape:
            # if originals differ slightly, center-crop to match
            H = min(up_near.shape[0], lg.shape[0])
            W = min(up_near.shape[1], lg.shape[1])
            up_near = up_near[:H, :W, ...] if up_near.ndim == 3 else up_near[:H, :W]
            up_blin = up_blin[:H, :W, ...] if up_blin.ndim == 3 else up_blin[:H, :W]
            lg = lg[:H, :W, ...] if lg.ndim == 3 else lg[:H, :W]
        ssd_near = nssd(up_near, lg)
        ssd_blin = nssd(up_blin, lg)
        print(f"{small.name} → {large.name}: NSSD(nearest)={ssd_near:.6f}, NSSD(bilinear)={ssd_blin:.6f}")


def main():
    p = argparse.ArgumentParser(description="Q7: image zoom (nearest/bilinear) + SSD tests.")
    p.add_argument("--input", type=str, help="Path to input image")
    p.add_argument("--scale", type=float, default=2.0, help="Scale s in (0,10]")
    p.add_argument("--method", choices=["nearest", "bilinear"], default="nearest")
    p.add_argument("--output", type=str, default="outputs/q7/out.png")
    p.add_argument("--assets_dir", type=str, default="assets", help="Folder containing im01small.png, etc. for SSD test")
    p.add_argument("--run-ssd-test", action="store_true", help="Evaluate NSSD at x4 on provided small/original pairs")
    args = p.parse_args()

    if args.input:
        run_zoom(Path(args.input), args.scale, args.method, Path(args.output))

    if args.run_ssd_test:
        run_ssd_tests(Path(args.assets_dir))


if __name__ == "__main__":
    main()
