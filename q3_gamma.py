import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt


def to_bgr_u8(img_path: Path) -> np.ndarray:
    """Load RGB with PIL and return BGR uint8 for OpenCV."""
    rgb = np.array(Image.open(img_path).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def save_rgb(arr_rgb: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr_rgb).save(path)
    print("Saved:", path)


def gamma_lut(gamma: float) -> np.ndarray:
    """256-value LUT: y = (x/255)^gamma * 255"""
    x = np.arange(256, dtype=np.float32) / 255.0
    y = np.power(x, gamma) * 255.0
    return np.clip(y, 0, 255).astype(np.uint8)


def plot_hist_L(L: np.ndarray, title: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.hist(L.ravel(), bins=256, range=(0, 255))
    plt.title(title)
    plt.xlabel("L channel (0–255)")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved:", path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="assets/color-space.jpg")
    parser.add_argument("--output", default="outputs/q3/gamma_lab.png")
    parser.add_argument("--outdir", default="outputs/q3")
    parser.add_argument("--gamma", type=float, default=0.8,
                        help="Gamma for L channel")
    args = parser.parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir)
    out_img = Path(args.output)

    # 1) Load → BGR → L*a*b*
    bgr = to_bgr_u8(in_path)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)  # L in [0,255] in OpenCV's LAB

    # 2) Hist of original L
    plot_hist_L(L, "Original L-channel Histogram",
                outdir / "hist_L_original.png")

    # 3) Gamma on L using LUT
    lut = gamma_lut(args.gamma)
    L_gamma = lut[L]

    # 4) Recombine and convert back to RGB
    lab_gamma = cv2.merge([L_gamma, a, b])
    bgr_gamma = cv2.cvtColor(lab_gamma, cv2.COLOR_LAB2BGR)
    rgb_gamma = cv2.cvtColor(bgr_gamma, cv2.COLOR_BGR2RGB)
    outdir.mkdir(parents=True, exist_ok=True)
    save_rgb(rgb_gamma, out_img)

    # 5) Hist of corrected L
    plot_hist_L(
        L_gamma, f"Gamma-corrected L (γ={args.gamma}) Histogram", outdir / "hist_L_gamma.png")

    # 6) (Optional) Save the LUT curve used
    x = np.arange(256)
    plt.figure()
    plt.plot(x, lut)
    plt.xlim(0, 255)
    plt.ylim(0, 255)
    plt.xlabel("Input L")
    plt.ylabel("Output L")
    plt.title(f"LUT for Gamma (γ={args.gamma})")
    plt.grid(True, alpha=0.3)
    plt.savefig(outdir / "lut_gamma.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved:", outdir / "lut_gamma.png")


if __name__ == "__main__":
    main()
