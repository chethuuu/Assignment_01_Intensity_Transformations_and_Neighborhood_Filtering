import argparse
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

def save_u8(img, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)
    print("Saved:", path)

def save_rgb(bgr, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(path)
    print("Saved:", path)

def fill_holes(bw: np.ndarray) -> np.ndarray:
    h, w = bw.shape
    ff = np.zeros((h+2, w+2), np.uint8)
    inv = 255 - bw
    flood = inv.copy()
    cv2.floodFill(flood, ff, (0, 0), 255)
    holes = 255 - flood
    return cv2.bitwise_or(bw, holes)

def main():
    p = argparse.ArgumentParser(description="Q10: Sapphires segmentation, morphology, area (pixels/mm^2).")
    p.add_argument("--input", default="assets/q10/sapphires.png")
    p.add_argument("--outdir", default="outputs/q10")
    p.add_argument("--f_mm", type=float, default=8.0, help="lens focal length f (mm)")
    p.add_argument("--Z_mm", type=float, default=480.0, help="camera height above table Z (mm)")
    p.add_argument("--pixel_pitch_um", type=float, default=4.0, help="sensor pixel pitch (micrometers)")
    args = p.parse_args()

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read {args.input}")
    save_rgb(bgr, out / "0_original.png")

    # (a) SEGMENTATION (HSV threshold for blue + cleanup)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # OpenCV H in [0,179]. Blue ~ [100, 140].
    lo1 = np.array([100, 60, 40], dtype=np.uint8)
    hi1 = np.array([140, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lo1, hi1)
    save_u8(mask, out / "1a_mask_raw.png")

    # (b) MORPHOLOGY to fill/clean (open->close, then hole fill)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask2 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, k, iterations=2)
    mask_filled = fill_holes(mask2)
    save_u8(mask_filled, out / "1b_mask_filled.png")

    # (c) CONNECTED COMPONENTS AREAS (pixels)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_filled, connectivity=8)
    # keep only sizeable components (remove crumbs)
    areas_px = []
    label_keep = np.zeros_like(labels, dtype=np.uint8)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area > 200:  # small debris removal
            areas_px.append(area)
            label_keep[labels == i] = 255
    save_u8(label_keep, out / "2_components.png")

    # overlay visualization
    vis = bgr.copy()
    rng = np.random.default_rng(0)
    for i, area in enumerate(areas_px, start=1):
        y, x = np.argwhere((labels>0) & (label_keep>0))[0]
    overlay = bgr.copy()
    colored = cv2.cvtColor(label_keep, cv2.COLOR_GRAY2BGR)
    colored[colored[:,:,0]>0] = [0,255,255]
    vis = cv2.addWeighted(bgr, 0.8, colored, 0.2, 0)
    save_rgb(vis, out / "2_components_overlay.png")

    # (d) ACTUAL AREAS IN mm^2  (requires pixel pitch)
    # Magnification m = f/Z  ⇒ world_length = (Z/f) * sensor_length
    # pixel pitch p (mm) = pixel_pitch_um / 1000.
    p_mm = args.pixel_pitch_um / 1000.0
    scale_world_per_px = (args.Z_mm / args.f_mm) * p_mm      # mm per pixel in the table plane
    px2mm2 = (scale_world_per_px ** 2)
    areas_mm2 = [a * px2mm2 for a in areas_px]

    # Save a small report
    with open(out / "areas.txt", "w") as f:
        f.write(f"f = {args.f_mm} mm,  Z = {args.Z_mm} mm,  pixel_pitch = {args.pixel_pitch_um} um\n")
        f.write(f"mm per pixel on table plane ≈ {scale_world_per_px:.6f} mm/px\n")
        for idx, (apx, amm2) in enumerate(zip(areas_px, areas_mm2), start=1):
            f.write(f"Sapphire {idx}: {apx} px  →  {amm2:.2f} mm^2\n")
    print(f"Estimated areas (mm^2): {['%.2f'%x for x in areas_mm2]}")
    save_rgb((bgr * (mask_filled[...,None]//255)).astype(np.uint8), out / "1c_foreground.png")
    save_rgb((bgr * (1 - mask_filled[...,None]//255)).astype(np.uint8), out / "1d_background.png")

if __name__ == "__main__":
    main()
