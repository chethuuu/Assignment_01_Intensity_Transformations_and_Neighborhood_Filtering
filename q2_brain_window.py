import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def window_lut(low: float, high: float) -> np.ndarray:
    """256-value LUT mapping [low, high] to [0, 255]."""
    if high <= low:
        high = low + 1.0
    x = np.arange(256, dtype=np.float32)
    y = (x - low) * (255.0 / (high - low))
    y = np.clip(y, 0, 255)
    return y.astype(np.uint8)

def apply_lut(img_gray: np.ndarray, lut: np.ndarray) -> np.ndarray:
    return lut[img_gray]

def save_image(a: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a).save(path)
    print(f"Saved: {path}")

def save_curve(lut: np.ndarray, title: str, path: Path):
    plt.figure()
    plt.plot(np.arange(256), lut)
    plt.xlabel("Input intensity")
    plt.ylabel("Output intensity")
    plt.title(title)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")

def save_histogram(img_gray: np.ndarray, path: Path):
    plt.figure()
    plt.hist(img_gray.ravel(), bins=256, range=(0, 255))
    plt.xlabel("Intensity")
    plt.ylabel("Count")
    plt.title("Brain slice histogram")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")

def main():
    ap = argparse.ArgumentParser(description="WM/GM windowing with transform plots")
    ap.add_argument("--input", default="assets/brain.png", help="Path to grayscale brain image")
    ap.add_argument("--outdir", default="outputs", help="Directory for results")
    ap.add_argument("--wm", nargs=2, type=float, metavar=("LOW", "HIGH"),
                    help="Window for white matter")
    ap.add_argument("--gm", nargs=2, type=float, metavar=("LOW", "HIGH"),
                    help="Window for gray matter")
    args = ap.parse_args()

    img = Image.open(args.input).convert("L")
    I = np.array(img, dtype=np.uint8)

    outdir = Path(args.outdir)

    wm_low, wm_high = (135.0, 165.0) if not args.wm else tuple(args.wm)
    gm_low, gm_high = (170.0, 200.0) if not args.gm else tuple(args.gm)

    wm_lut = window_lut(wm_low, wm_high)
    gm_lut = window_lut(gm_low, gm_high)

    wm_img = apply_lut(I, wm_lut)
    gm_img = apply_lut(I, gm_lut)
    save_image(wm_img, outdir / "q2_wm_accentuated.png")
    save_image(gm_img, outdir / "q2_gm_accentuated.png")

    save_curve(wm_lut, f"WM window (low={wm_low}, high={wm_high})", outdir / "q2_wm_curve.png")
    save_curve(gm_lut, f"GM window (low={gm_low}, high={gm_high})", outdir / "q2_gm_curve.png")

    save_histogram(I, outdir / "q2_histogram.png")

if __name__ == "__main__":
    main()
