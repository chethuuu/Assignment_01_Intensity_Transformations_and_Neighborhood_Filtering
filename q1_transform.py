import argparse
from pathlib import Path
import numpy as np
from PIL import Image


def build_lut() -> np.ndarray:
    """
    Build the 256-entry lookup table (LUT) from the figure:
      A) 0..50       -> y = x
      B) 51..150     -> line from (50,100) to (150,255)  => y = 100 + 1.55*(x-50)
      C) 151..255    -> y = x
    Note: we include x=150 in segment B so y(150)=255 (matches the plot).
    """
    lut = np.zeros(256, dtype=np.uint8)

    # Segment A: 0..50
    lut[:51] = np.arange(0, 51, dtype=np.uint8)

    # Segment B: 51..150
    x = np.arange(51, 151)
    y = np.rint(100 + (x - 50) * (155.0 / 100.0))  # slope = 1.55
    y = np.clip(y, 0, 255).astype(np.uint8)
    lut[51:151] = y

    # Segment C: 151..255
    lut[151:] = np.arange(151, 256, dtype=np.uint8)
    return lut


def apply_transform(img_path: Path, out_path: Path):
    # 1) load and force grayscale
    img = Image.open(img_path).convert("L")
    I = np.array(img, dtype=np.uint8)

    # 2) build LUT and apply
    lut = build_lut()
    J = lut[I]

    # 3) save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(J).save(out_path)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Piecewise intensity transform")
    parser.add_argument("--input",  default="assets/emma.jpg",
                        help="Path to input image")
    parser.add_argument(
        "--output", default="outputs/emma-output.png", help="Path to save output")
    args = parser.parse_args()

    apply_transform(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
