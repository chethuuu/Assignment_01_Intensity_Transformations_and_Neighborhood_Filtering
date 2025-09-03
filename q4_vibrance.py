import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, img_as_float, img_as_ubyte


def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def boost_sat(x255: np.ndarray, a: float, sigma: float) -> np.ndarray:
    """
    f(x) = min( x + a*128*exp(-(x-128)^2 / (2*sigma^2)), 255 )
    x is saturation in [0,255].
    """
    bump = a * 128.0 * np.exp(-((x255 - 128.0) ** 2) / (2.0 * sigma ** 2))
    return np.minimum(x255 + bump, 255.0)


def save_transform_curve(a: float, sigma: float, path: Path):
    x = np.linspace(0, 255, 512)
    y = boost_sat(x, a=a, sigma=sigma)
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("S input (0–255)")
    plt.ylabel("S output (0–255)")
    plt.title(f"Vibrance transform (a={a}, σ={sigma})")
    plt.tight_layout()
    ensure_dir(path)
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def save_side_by_side(orig_rgb01: np.ndarray, out_rgb01: np.ndarray, a: float, sigma: float, path: Path):
    plt.figure(figsize=(10, 4.5))
    plt.subplot(1, 2, 1)
    plt.imshow(orig_rgb01)
    plt.axis("off")
    plt.title("Original")
    plt.subplot(1, 2, 2)
    plt.imshow(out_rgb01)
    plt.axis("off")
    plt.title(f"Vibrance (a={a}, σ={sigma})")
    plt.tight_layout()
    ensure_dir(path)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def main():
    ap = argparse.ArgumentParser(
        description="Q4: Vibrance via S-plane boost (HSV)")
    ap.add_argument("--input",  default="assets/spiderman.png",
                    help="Path to input image")
    ap.add_argument("--outdir", default="outputs",
                    help="Output directory")
    ap.add_argument("--a",      type=float, default=0.6,
                    help="Strength in [0,1]")
    ap.add_argument("--sigma",  type=float, default=70.0,
                    help="Gaussian sigma")
    ap.add_argument("--no-show", action="store_true",
                    help="Disable interactive windows")
    args = ap.parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir)
    a, sigma = float(args.a), float(args.sigma)

    # (a) Split into HSV
    img = io.imread(str(in_path))
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]
    rgb = img_as_float(img)                 # [0,1]
    hsv = color.rgb2hsv(rgb)                # H,S,V in [0,1]
    S = hsv[..., 1]

    # (b) Apply transform on S (work in 0..255 as per formula)
    S_255 = np.clip(S * 255.0, 0, 255)
    S_boost_255 = boost_sat(S_255, a=a, sigma=sigma)
    S_boost = np.clip(S_boost_255 / 255.0, 0, 1)

    # (c) 'a' is adjustable; printing for report
    print(f"Chosen a: {a}, sigma: {sigma}")

    # (d) Recombine
    hsv_out = hsv.copy()
    hsv_out[..., 1] = S_boost
    rgb_out = np.clip(color.hsv2rgb(hsv_out), 0, 1)

    # (e) Save results and transform curve
    enhanced = outdir / "q4_vibrance_enhanced.png"
    ensure_dir(enhanced)
    io.imsave(str(enhanced), img_as_ubyte(rgb_out))
    print(f"Saved: {enhanced}")

    curve = outdir / "q4_vibrance_transform_curve.png"
    save_transform_curve(a, sigma, curve)

    comp = outdir / "q4_vibrance_comparison.png"
    save_side_by_side(rgb, rgb_out, a, sigma, comp)

    if not args.no_show:
        plt.figure()
        plt.imshow(rgb)
        plt.axis("off")
        plt.title("Original")
        plt.figure()
        plt.imshow(rgb_out)
        plt.axis("off")
        plt.title(f"Vibrance (a={a}, σ={sigma})")
        plt.show()


if __name__ == "__main__":
    main()
