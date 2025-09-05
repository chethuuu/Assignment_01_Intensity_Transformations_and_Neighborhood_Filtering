import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

def save(arr, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)).save(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="assets/flower.png")
    parser.add_argument("--outdir", default="outputs/q8")
    args = parser.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(args.input)
    h, w = img.shape[:2]

    # (a) GrabCut segmentation
    mask = np.zeros((h, w), np.uint8)
    rect = (10, 10, w-20, h-20)  # rectangle around flower
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    fg = img * mask2[:, :, np.newaxis]
    bg = img * (1 - mask2[:, :, np.newaxis])
    save(fg, outdir / "foreground.png")
    save(bg, outdir / "background.png")
    save(img, outdir / "original.png")

    # (b) Blur background
    blurred = cv2.GaussianBlur(img, (21, 21), 0)
    enhanced = fg + blurred * (1 - mask2[:, :, np.newaxis])
    save(enhanced, outdir / "enhanced.png")

if __name__ == "__main__":
    main()
