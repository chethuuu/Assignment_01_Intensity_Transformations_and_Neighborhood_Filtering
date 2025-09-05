import argparse
from pathlib import Path
import numpy as np
from PIL import Image

def build_lut():
    xs = np.array([0, 50, 150, 255], dtype=np.float32)
    ys = np.array([0, 100, 255, 255], dtype=np.float32)
    x_all = np.arange(256, dtype=np.float32)
    y_all = np.interp(x_all, xs, ys)
    return y_all.astype(np.uint8)


def apply_transform(img_path: Path, out_path: Path):
    # Load as grayscale to apply the intensity mapping on a single channel
    img = Image.open(img_path).convert("L")
    arr = np.array(img, dtype=np.uint8)

    # Build LUT and map each pixel intensity
    lut = build_lut()
    out = lut[arr]

    # Save result
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(out_path)
    print("Saved:", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/assets/emma.jpg")
    parser.add_argument("--output", default="outputs/q1/emma-output.png")
    args = parser.parse_args()

    apply_transform(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
