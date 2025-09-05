import argparse
from pathlib import Path
import numpy as np
from PIL import Image


def build_lut() -> np.ndarray:

    x = np.arange(256, dtype=np.float32)
    y = np.empty_like(x)

    # 0..50
    m1 = x <= 50
    y[m1] = 2.0 * x[m1]

    # 50..150  (line from 100 to 255 over 100 steps → slope 155/100 = 1.55)
    m2 = (x > 50) & (x <= 150)
    y[m2] = 100.0 + (155.0 / 100.0) * (x[m2] - 50.0)

    # >150  (identity)
    m3 = x > 150
    y[m3] = x[m3]
    y[150] = 255.0

    return np.clip(y, 0, 255).astype(np.uint8)


def apply_transform(img_path: Path, out_path: Path):
    # Load as grayscale & transform is defined on intensity
    img = Image.open(img_path).convert("L")
    arr = np.array(img, dtype=np.uint8)

    lut = build_lut()
    out = lut[arr]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(out_path)
    print("Saved:", out_path)


def main():
    parser = argparse.ArgumentParser(description="Fig. 1a intensity transform")
    parser.add_argument("--input", default="/assets/emma.jpg")
    parser.add_argument("--output", default="outputs/q1/emma-output.png")
    args = parser.parse_args()

    apply_transform(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
