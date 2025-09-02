import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, img_as_float, img_as_ubyte


def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def save_image_rgb01(rgb01: np.ndarray, path: Path):
    ensure_dir(path)
    io.imsave(str(path), img_as_ubyte(np.clip(rgb01, 0, 1)))
    print(f"Saved: {path}")


def save_histogram_L(L_norm: np.ndarray, title: str, path: Path):
    plt.figure()
    plt.hist(L_norm.ravel(), bins=256, range=(0, 1))
    plt.xlabel("L* (normalized 0–1)")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    ensure_dir(path)
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def save_gamma_curve(gamma: float, path: Path):
    x = np.linspace(0, 1, 512)
    y = np.power(x, gamma)
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("Input (L* normalized)")
    plt.ylabel("Output after gamma")
    plt.title(f"Gamma curve (γ={gamma})")
    plt.tight_layout()
    ensure_dir(path)
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def save_side_by_side(orig_rgb01: np.ndarray, corr_rgb01: np.ndarray, gamma: float, path: Path):
    plt.figure(figsize=(10, 4.5))
    plt.subplot(1, 2, 1)
    plt.imshow(orig_rgb01)
    plt.axis("off")
    plt.title("Original")
    plt.subplot(1, 2, 2)
    plt.imshow(corr_rgb01)
    plt.axis("off")
    plt.title(f"Gamma L* (γ={gamma})")
    plt.tight_layout()
    ensure_dir(path)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def gamma_correct_L(img_path: str, gamma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (original_rgb, corrected_rgb, original_L_normalized)."""
    if gamma <= 0:
        raise ValueError("gamma must be > 0")

    img = io.imread(img_path)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]

    rgb = img_as_float(img)
    lab = color.rgb2lab(rgb)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    L_norm = np.clip(L / 100.0, 0, 1)
    L_corr = np.power(L_norm, gamma) * 100.0
    lab_corr = np.stack([L_corr, a, b], axis=-1)
    rgb_corr = np.clip(color.lab2rgb(lab_corr), 0, 1)

    return rgb, rgb_corr, L_norm


def main():
    ap = argparse.ArgumentParser(
        description="Gamma-correct the L* channel in CIE Lab and save plots.")
    ap.add_argument("--input", default="assets/color-space.jpg",
                    help="Path to input RGB image")
    ap.add_argument("--outdir", default="outputs",
                    help="Directory for results")
    ap.add_argument("--gamma", type=float, default=0.6,
                    help="γ<1 brightens, γ>1 darkens (default 0.6)")
    ap.add_argument("--no-show", action="store_true",
                    help="Do not open interactive windows")
    args = ap.parse_args()

    outdir = Path(args.outdir)

    orig_rgb01, corr_rgb01, L_norm = gamma_correct_L(args.input, args.gamma)
    L_corr_norm = np.power(L_norm, args.gamma)

    save_image_rgb01(corr_rgb01, outdir / "q3_gamma_corrected_lab.png")
    save_histogram_L(L_norm, "Original L* histogram (normalized)",
                     outdir / "q3_L_hist_original.png")
    save_histogram_L(
        L_corr_norm, f"Corrected L* histogram (γ={args.gamma})", outdir / "q3_L_hist_corrected.png")
    save_gamma_curve(args.gamma, outdir / "q3_gamma_curve.png")
    save_side_by_side(orig_rgb01, corr_rgb01, args.gamma,
                      outdir / "q3_comparison.png")

    if not args.no_show:
        plt.figure()
        plt.imshow(orig_rgb01)
        plt.axis("off")
        plt.title("Original")
        plt.figure()
        plt.imshow(corr_rgb01)
        plt.axis("off")
        plt.title(f"Gamma L* (γ={args.gamma})")
        plt.show()

    print(f"Gamma used: {args.gamma}")
    print(f"Original L* mean={L_norm.mean():.4f}, std={L_norm.std():.4f}")
    print(
        f"Corrected L* mean={L_corr_norm.mean():.4f}, std={L_corr_norm.std():.4f}")


if __name__ == "__main__":
    main()
