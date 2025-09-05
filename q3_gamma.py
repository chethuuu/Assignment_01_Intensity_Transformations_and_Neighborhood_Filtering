import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, img_as_float, img_as_ubyte

def ensure_dir(p):
    p.parent.mkdir(parents=True, exist_ok=True)

def save_img01(rgb01, path):
    ensure_dir(path)
    io.imsave(str(path), img_as_ubyte(np.clip(rgb01, 0, 1)))
    print("Saved:", path)

def save_hist(L01, title, path):
    plt.figure()
    plt.hist(L01.ravel(), bins=256, range=(0, 1))
    plt.xlabel("L* (0–1)")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    ensure_dir(path)
    plt.savefig(path)
    plt.close()
    print("Saved:", path)

def save_gamma_curve(gamma, path):
    x = np.linspace(0, 1, 512)
    y = x ** gamma
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("Input (0–1)")
    plt.ylabel("Output")
    plt.title(f"Gamma (γ={gamma})")
    plt.tight_layout()
    ensure_dir(path)
    plt.savefig(path)
    plt.close()
    print("Saved:", path)

def save_side_by_side(orig01, corr01, gamma, path):
    plt.figure(figsize=(10, 4.5))
    plt.subplot(1, 2, 1); plt.imshow(orig01); plt.axis("off"); plt.title("Original")
    plt.subplot(1, 2, 2); plt.imshow(corr01); plt.axis("off"); plt.title(f"Gamma L* (γ={gamma})")
    plt.tight_layout()
    ensure_dir(path)
    plt.savefig(path, dpi=150)
    plt.close()
    print("Saved:", path)

def gamma_L(img_path, gamma):
    if gamma <= 0:
        raise ValueError("gamma must be > 0")

    img = io.imread(img_path)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]

    rgb01 = img_as_float(img)
    lab = color.rgb2lab(rgb01)
    L = lab[..., 0]             # 0..100
    a = lab[..., 1]
    b = lab[..., 2]

    L01 = np.clip(L / 100.0, 0, 1)
    L01_corr = np.clip(L01 ** gamma, 0, 1)

    lab_corr = np.stack([L01_corr * 100.0, a, b], axis=-1)
    rgb01_corr = np.clip(color.lab2rgb(lab_corr), 0, 1)

    return rgb01, rgb01_corr, L01, L01_corr

def main():
    p = argparse.ArgumentParser(description="Gamma-correct L* channel (Lab).")
    p.add_argument("--input", default="assets/color-space.jpg")
    p.add_argument("--outdir", default="outputs")
    p.add_argument("--gamma", type=float, default=0.6)
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    outdir = Path(args.outdir)

    orig01, corr01, L01, L01_corr = gamma_L(args.input, args.gamma)

    save_img01(corr01, outdir / "/q3/gamma_corrected_lab.png")
    save_hist(L01, "Original L* hist", outdir / "/q3/_hist_original.png")
    save_hist(L01_corr, f"Corrected L* hist (γ={args.gamma})", outdir / "/q3/_hist_corrected.png")
    save_gamma_curve(args.gamma, outdir / "/q3/amma_curve.png")
    save_side_by_side(orig01, corr01, args.gamma, outdir / "/q3/omparison.png")

    if not args.no_show:
        plt.figure(); plt.imshow(orig01); plt.axis("off"); plt.title("Original")
        plt.figure(); plt.imshow(corr01); plt.axis("off"); plt.title(f"Gamma L* (γ={args.gamma})")
        plt.show()

    print(f"Gamma: {args.gamma}")
    print(f"L* mean/std (orig): {L01.mean():.4f}/{L01.std():.4f}")
    print(f"L* mean/std (corr): {L01_corr.mean():.4f}/{L01_corr.std():.4f}")

if __name__ == "__main__":
    main()
