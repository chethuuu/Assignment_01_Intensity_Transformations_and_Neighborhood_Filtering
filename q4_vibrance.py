import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def save_rgb(img: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)
    print("Saved:", path)


def save_gray(img_u8: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_u8).save(path)
    print("Saved:", path)

# Vibrance LUT
def vibrance_lut(a: float, sigma: float) -> np.ndarray:
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


def plot_hist_s(S: np.ndarray, S_new: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.hist(S.ravel(), bins=256, range=(0, 255), alpha=0.6, label="S (orig)")
    plt.hist(S_new.ravel(), bins=256, range=(
        0, 255), alpha=0.6, label="S (enhanced)")
    plt.xlim(0, 255)
    plt.xlabel("S value")
    plt.ylabel("Count")
    plt.title("Saturation histograms (before/after)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved:", out_path)


def side_by_side(orig_rgb: np.ndarray, enhanced_rgb: np.ndarray, a: float, sigma: float, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(orig_rgb)
    plt.axis("off")
    plt.title("Original")
    plt.subplot(1, 2, 2)
    plt.imshow(enhanced_rgb)
    plt.axis("off")
    plt.title(f"Vibrance (a={a:.2f}, σ={sigma:.0f})")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved:", out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Q4: Vibrance enhancement on HSV Saturation.")
    parser.add_argument("--input",  default="assets/spiderman.png")
    parser.add_argument("--outdir", default="outputs/q4")
    parser.add_argument("--a",      type=float, default=0.7,
                        help="Strength a in [0,1]")
    parser.add_argument("--sigma",  type=float, default=70.0,
                        help="Gaussian sigma (default 70)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # (a) Split image to H, S, V
    rgb = load_rgb(Path(args.input))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)  # H:[0,179], S,V:[0,255]
    H, S, V = cv2.split(hsv)
    save_gray(H, outdir / "H.png")
    save_gray(S, outdir / "S.png")
    save_gray(V, outdir / "V.png")

    # (b) Apply the given transform to S using LUT
    lut = vibrance_lut(a=args.a, sigma=args.sigma)
    S_new = lut[S]

    # (c) Parameter a is adjustable; print/report chosen value
    print(f"Chosen a = {args.a:.2f}, sigma = {args.sigma:.0f}")

    # (d) Recombine and convert back to RGB
    hsv_new = cv2.merge([H, S_new, V])
    rgb_new = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2RGB)

    # (e) Save required outputs
    save_rgb(rgb,     outdir / "original.png")
    save_rgb(rgb_new, outdir / f"vibrance_a{args.a:.2f}.png")
    plot_curve(
        lut, f"Intensity transform on S (a={args.a:.2f}, σ={args.sigma:.0f})", outdir / "vibrance_curve.png")
    plot_hist_s(S, S_new, outdir / "saturation_hist_before_after.png")
    side_by_side(rgb, rgb_new, args.a, args.sigma,
                 outdir / "comparison_side_by_side.png")


if __name__ == "__main__":
    main()
