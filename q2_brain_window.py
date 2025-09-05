import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def build_lut(xs, ys):
    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    xall = np.arange(256, dtype=np.float32)
    yall = np.interp(xall, xs, ys)
    return np.clip(yall, 0, 255).astype(np.uint8)


def apply_lut(img_u8, lut):
    return lut[img_u8]


def save_curve(lut, out_path, title):
    x = np.arange(256, dtype=np.uint8)
    plt.figure()
    plt.plot(x, lut, linewidth=2)
    plt.title(title)
    plt.xlabel("Input intensity")
    plt.ylabel("Output intensity")
    plt.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    in_path = Path(args.input) if args.input else (
        script_dir / "assets" / "brain.png")
    out_dir = Path(args.outdir) if args.outdir else (
        script_dir / "outputs" / "q2")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load as grayscale
    img = Image.open(in_path).convert("L")
    arr = np.array(img, dtype=np.uint8)

    # Control points chosen to boost ~[60..120] range
    xs_wm = [0,  60, 120, 255]
    ys_wm = [0,   0, 255, 255]
    lut_wm = build_lut(xs_wm, ys_wm)
    out_wm = apply_lut(arr, lut_wm)
    Image.fromarray(out_wm).save(out_dir / "brain_white_matter.png")
    save_curve(lut_wm, out_dir / "curve_white_matter.png",
               "Q2 (a): Transform for White Matter")

    # Control points chosen to boost ~[90..170] range
    xs_gm = [0,  90, 170, 255]
    ys_gm = [0,   0, 255, 255]
    lut_gm = build_lut(xs_gm, ys_gm)
    out_gm = apply_lut(arr, lut_gm)
    Image.fromarray(out_gm).save(out_dir / "brain_gray_matter.png")
    save_curve(lut_gm, out_dir / "curve_gray_matter.png",
               "Q2 (b): Transform for Gray Matter")

    print("Saved:")
    print("  ", out_dir / "brain_white_matter.png")
    print("  ", out_dir / "curve_white_matter.png")
    print("  ", out_dir / "brain_gray_matter.png")
    print("  ", out_dir / "curve_gray_matter.png")


if __name__ == "__main__":
    main()
