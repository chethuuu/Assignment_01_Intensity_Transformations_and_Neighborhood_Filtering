import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, img_as_float, img_as_ubyte

def ensure_dir(p):
    p.parent.mkdir(parents=True, exist_ok=True)

def boost_sat(x255, a, sigma):
    # f(x) = min(x + a*128*exp(-(x-128)^2/(2*sigma^2)), 255)
    bump = a * 128.0 * np.exp(-((x255 - 128.0) ** 2) / (2.0 * sigma ** 2))
    return np.minimum(x255 + bump, 255.0)

def save_curve(a, sigma, path):
    x = np.linspace(0, 255, 512)
    y = boost_sat(x, a, sigma)
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("S in (0–255)")
    plt.ylabel("S out (0–255)")
    plt.title(f"Vibrance transform (a={a}, σ={sigma})")
    plt.tight_layout()
    ensure_dir(path)
    plt.savefig(path)
    plt.close()
    print("Saved:", path)

def save_side_by_side(orig01, out01, a, sigma, path):
    plt.figure(figsize=(10, 4.5))
    plt.subplot(1, 2, 1); plt.imshow(orig01); plt.axis("off"); plt.title("Original")
    plt.subplot(1, 2, 2); plt.imshow(out01);  plt.axis("off"); plt.title(f"Vibrance (a={a}, σ={sigma})")
    plt.tight_layout()
    ensure_dir(path)
    plt.savefig(path, dpi=150)
    plt.close()
    print("Saved:", path)

def main():
    p = argparse.ArgumentParser(description="Q4: Vibrance via S boost (HSV)")
    p.add_argument("--input",  default="assets/spiderman.png")
    p.add_argument("--outdir", default="outputs")
    p.add_argument("--a",      type=float, default=0.6)
    p.add_argument("--sigma",  type=float, default=70.0)
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir)
    a, sigma = args.a, args.sigma

    # Read & to HSV
    img = io.imread(str(in_path))
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]
    rgb01 = img_as_float(img)
    hsv = color.rgb2hsv(rgb01)
    S = hsv[..., 1]

    # Boost S in 0..255 space
    S255 = np.clip(S * 255.0, 0, 255)
    S255_boost = boost_sat(S255, a, sigma)
    S_boost = np.clip(S255_boost / 255.0, 0, 1)

    # Recombine & save
    hsv_out = hsv.copy()
    hsv_out[..., 1] = S_boost
    rgb_out = np.clip(color.hsv2rgb(hsv_out), 0, 1)

    enhanced = outdir / "q4_vibrance_enhanced.png"
    ensure_dir(enhanced)
    io.imsave(str(enhanced), img_as_ubyte(rgb_out))
    print("Saved:", enhanced)

    save_curve(a, sigma, outdir / "q4_vibrance_transform_curve.png")
    save_side_by_side(rgb01, rgb_out, a, sigma, outdir / "q4_vibrance_comparison.png")

    if not args.no_show:
        plt.figure(); plt.imshow(rgb01); plt.axis("off"); plt.title("Original")
        plt.figure(); plt.imshow(rgb_out); plt.axis("off"); plt.title(f"Vibrance (a={a}, σ={sigma})")
        plt.show()

    print(f"Chosen a={a}, sigma={sigma}")

if __name__ == "__main__":
    main()
