import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def window_lut(low: int, high: int) -> np.ndarray:
    if high <= low:
        high = low + 1
    x = np.arange(256, dtype=np.float32)
    y = (x - low) * (255.0 / (high - low))
    y = np.clip(y, 0, 255)
    return y.astype(np.uint8)


def apply_lut(img_gray: np.ndarray, lut: np.ndarray) -> np.ndarray:
    return lut[img_gray]


def save_image(arr: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)
    print("Saved:", path)


def save_lut_plot(lut: np.ndarray, title: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(256)
    plt.figure()
    plt.plot(x, lut)
    plt.xlim(0, 255)
    plt.ylim(0, 255)
    plt.xlabel("Input intensity")
    plt.ylabel("Output intensity")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved:", path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="assets/brain.png")
    parser.add_argument("--outdir", default="outputs/q2")
    args = parser.parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir)

    # Load brain image as grayscale
    img = Image.open(in_path).convert("L")
    a = np.array(img, dtype=np.uint8)

    # ---- (a) Accentuate white matter (brighter tissues) ----
    # Window around high intensities (tuned for this image)
    lut_white = window_lut(low=160, high=230)
    white_img = apply_lut(a, lut_white)
    save_image(white_img, outdir / "brain_white.png")
    save_lut_plot(
        lut_white, "Q2(a): LUT for white matter (window 160–230)", outdir / "lut_white.png")

    # ---- (b) Accentuate gray matter (mid tones) ----
    # Window around mid intensities (tuned for this image)
    lut_gray = window_lut(low=110, high=180)
    gray_img = apply_lut(a, lut_gray)
    save_image(gray_img, outdir / "brain_gray.png")
    save_lut_plot(
        lut_gray, "Q2(b): LUT for gray matter (window 110–180)", outdir / "lut_gray.png")


if __name__ == "__main__":
    main()
