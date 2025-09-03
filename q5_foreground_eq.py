import argparse
from pathlib import Path
import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt

# ---------------------- utils ----------------------
def ensure_rgb(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)

def save_gray(a: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a).save(path)
    print(f"Saved: {path}")

def save_rgb(a: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a).save(path)
    print(f"Saved: {path}")

def plot_hist_and_cdf(vals: np.ndarray, out_hist: Path, out_cdf: Path):
    counts, bins = np.histogram(vals, bins=256, range=(0, 256))
    cdf = np.cumsum(counts)

    # histogram
    plt.figure()
    plt.bar(bins[:-1], counts, width=1.0)
    plt.title("Foreground Value-channel histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Count")
    plt.tight_layout()
    out_hist.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_hist)
    plt.close()
    print(f"Saved: {out_hist}")

    # cdf
    plt.figure()
    plt.plot(np.arange(256), cdf)
    plt.title("Cumulative sum (CDF) of foreground histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Cumulative count")
    plt.tight_layout()
    plt.savefig(out_cdf)
    plt.close()
    print(f"Saved: {out_cdf}")

    return counts, cdf

def first_existing(*cands: Path) -> Path:
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError("None of these exist: " + ", ".join(map(str, cands)))

# ---------------------- core steps ----------------------
def equalize_foreground_v_channel(hsv: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Histogram-equalize only the Value (V) channel for pixels where mask==255.
    Returns a new HSV array.
    """
    h, s, v = cv.split(hsv)
    v = v.copy()

    fg_vals = v[mask > 0]
    # (d) cumulative sum
    hist, _ = np.histogram(fg_vals, bins=256, range=(0, 256))
    cdf = np.cumsum(hist)

    # (e) classical HE mapping on foreground pixels only
    cdf_nonzero = cdf[cdf > 0]
    if len(cdf_nonzero) == 0:
        lut = np.arange(256, dtype=np.uint8)
    else:
        cdf_min = cdf_nonzero[0]
        lut = np.round((cdf - cdf_min) / (cdf[-1] - cdf_min + 1e-12) * 255.0)
        lut = np.clip(lut, 0, 255).astype(np.uint8)

    v_eq = v.copy()
    v_eq[mask > 0] = lut[v[mask > 0]]

    return cv.merge([h, s, v_eq]), hist, cdf

def main():
    ap = argparse.ArgumentParser(
        description="Q5 – Histogram equalization on foreground only (Fig. 5)"
    )
    ap.add_argument("--input", default="assets/jennifer.jpg",
                    help="Input color image. Tries jeniffer.jpg if not found.")
    ap.add_argument("--outdir", default="outputs/q5", help="Directory for results")
    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    in_path = Path(args.input)
    if not in_path.is_absolute():
        # tolerate the common misspelling too
        in_path = first_existing(
            base / args.input,
            base / "assets/jennifer.jpg",
            base / "assets/jeniffer.jpg",
        )
    img_rgb = ensure_rgb(in_path)
    outdir = Path(args.outdir)

    # (a) split to HSV and show planes in grayscale
    img_bgr = cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR)
    hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)  # OpenCV H∈[0,179], S,V∈[0,255]
    H, S, V = cv.split(hsv)

    # for visualization, scale H to 0..255
    H_vis = (H.astype(np.float32) * (255.0 / 179.0)).astype(np.uint8)
    save_gray(H_vis, outdir / "q5a_h_plane.png")
    save_gray(S,     outdir / "q5a_s_plane.png")
    save_gray(V,     outdir / "q5a_v_plane.png")

    # (b) select plane to threshold for foreground mask.
    # The background is gray and low-saturation; the subject is more saturated.
    # Use Otsu on S (after a small blur), then clean with morphology.
    S_blur = cv.GaussianBlur(S, (5, 5), 0)
    _, mask = cv.threshold(S_blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # morphological close->open to fill holes and remove specks
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN,  kernel, iterations=1)
    save_gray(mask, outdir / "q5b_mask.png")

    # (c) foreground only and histogram on V within mask
    fg_rgb = cv.bitwise_and(img_rgb, img_rgb, mask=mask)
    save_rgb(fg_rgb, outdir / "q5c_foreground_rgb.png")

    # histogram + CDF figures for the foreground V
    plot_hist_and_cdf(V[mask > 0], outdir / "q5d_fg_hist.png", outdir / "q5d_fg_cdf.png")

    # (d)+(e) equalize only foreground V using derived CDF (formulas from slides)
    hsv_eq, hist_fg, cdf_fg = equalize_foreground_v_channel(hsv, mask)

    # (f) combine background with equalized foreground
    bgr_eq = cv.cvtColor(hsv_eq, cv.COLOR_HSV2BGR)
    rgb_eq = cv.cvtColor(bgr_eq, cv.COLOR_BGR2RGB)
    save_rgb(img_rgb, outdir / "q5f_original_rgb.png")
    save_rgb(rgb_eq,  outdir / "q5f_result_rgb.png")

    # (f, explicit) also show bg + fg pieces
    inv = cv.bitwise_not(mask)
    bg_rgb = cv.bitwise_and(img_rgb, img_rgb, mask=inv)
    fg_eq_rgb = cv.bitwise_and(rgb_eq, rgb_eq, mask=mask)
    save_rgb(bg_rgb,     outdir / "q5f_background_rgb.png")
    save_rgb(fg_eq_rgb,  outdir / "q5f_foreground_eq_rgb.png")

    print(f"✅ Done. Results saved in {outdir.resolve()}")

if __name__ == "__main__":
    main()
