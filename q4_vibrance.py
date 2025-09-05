import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def save_rgb(rgb: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)
    print("Saved:", path)


def vibrance_lut(a: float = 0.7, sigma: float = 70.0) -> np.ndarray:
    """
    Build 256-value LUT for:
        f(x) = min( x + a*128*exp(-((x-128)^2)/(2*sigma^2)), 255 )
    x in [0,255], returns uint8.
    """
    x = np.arange(256, dtype=np.float32)
    boost = a * 128.0 * np.exp(-((x - 128.0) ** 2) / (2.0 * sigma ** 2))
    y = np.minimum(x + boost, 255.0)
    return y.astype(np.uint8)


def plot_curve(lut: np.ndarray, title: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(256)
    plt.figure()
    plt.plot(x, lut)
    plt.xlim(0, 255)
    plt.ylim(0, 255)
    plt.xlabel("Input saturation")
    plt.ylabel("Output saturation")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved:", out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Q4: Vibrance enhancement on S channel (HSV).")
    parser.add_argument("--input", default="assets/spiderman.png")
    parser.add_argument("--outdir", default="outputs/q4")
    parser.add_argument("--a", type=float, default=0.7,
                        help="Strength parameter a in [0,1].")
    parser.add_argument("--sigma", type=float, default=70.0,
                        help="Sigma for Gaussian boost (default 70).")
    args = parser.parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load RGB → HSV
    rgb = load_rgb(in_path)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)

    # Build LUT and apply on Saturation
    lut = vibrance_lut(a=args.a, sigma=args.sigma)
    S_new = lut[S]

    # Merge and convert back to RGB
    hsv_new = cv2.merge([H, S_new, V])
    rgb_new = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2RGB)

    # Save outputs
    save_rgb(rgb, outdir / "original.png")
    save_rgb(rgb_new, outdir / "vibrance_a{:.2f}.png".format(args.a))
    plot_curve(
        lut, f"Vibrance transform (a={args.a}, σ={args.sigma})", outdir / "vibrance_curve.png")


if __name__ == "__main__":
    main()
