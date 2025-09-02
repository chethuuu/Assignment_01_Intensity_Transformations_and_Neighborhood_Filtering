import argparse
from pathlib import Path
import numpy as np
from PIL import Image


def build_lut() -> np.ndarray:
    """Return a 256-value LUT for a piecewise linear transform."""
    lut = np.zeros(256, dtype=np.uint8)

    # 0..50 → identity
    lut[:51] = np.arange(0, 51, dtype=np.uint8)

    # 51..150 → line from (50,100) to (150,255)
    x = np.arange(51, 151)
    y = np.rint(100 + (x - 50) * (155.0 / 100.0))  # slope = 1.55
    lut[51:151] = np.clip(y, 0, 255).astype(np.uint8)

    # 151..255 → identity
    lut[151:] = np.arange(151, 256, dtype=np.uint8)

    return lut


def apply_transform(img_path: Path, out_path: Path):
    img = Image.open(img_path).convert("L")
    I = np.array(img, dtype=np.uint8)

    lut = build_lut()
    J = lut[I]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(J).save(out_path)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Piecewise intensity transform")
    parser.add_argument("--input", default="assets/emma.jpg",
                        help="Path to input image")
    parser.add_argument(
        "--output", default="outputs/emma-output.png", help="Path to save output")
    args = parser.parse_args()

    apply_transform(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
