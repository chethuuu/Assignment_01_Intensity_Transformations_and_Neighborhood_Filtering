from pathlib import Path
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def imread_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def save_img(path: Path, arr: np.ndarray):
    ensure_dir(path.parent)
    cv2.imwrite(str(path), arr)


def otsu_segment(denoised: np.ndarray) -> tuple[np.ndarray, float]:
    t, mask = cv2.threshold(
        denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask, t


def morph_clean(mask: np.ndarray) -> np.ndarray:
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close, iterations=1)
    return closed


def count_components(bin_img: np.ndarray, min_area: int = 50):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        bin_img, connectivity=8)
    keep = []
    for lab in range(1, n):
        if stats[lab, cv2.CC_STAT_AREA] >= min_area:
            keep.append(lab)
    return labels, stats, keep


def colorize_labels(labels: np.ndarray, keep: list[int]) -> np.ndarray:
    """Assign a distinct color to each kept component."""
    h, w = labels.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    if not keep:
        return out
    # simple HSV wheel
    for i, lab in enumerate(keep):
        hue = int(179 * (i / max(1, len(keep))))
        col = cv2.cvtColor(
            np.uint8([[[hue, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
        out[labels == lab] = col
    return out


def draw_boxes(gray: np.ndarray, stats: np.ndarray, valid_labels: list[int]) -> np.ndarray:
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for idx, lab in enumerate(valid_labels, start=1):
        x = stats[lab, cv2.CC_STAT_LEFT]
        y = stats[lab, cv2.CC_STAT_TOP]
        w = stats[lab, cv2.CC_STAT_WIDTH]
        h = stats[lab, cv2.CC_STAT_HEIGHT]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # small id
        cv2.putText(vis, str(idx), (x, max(10, y - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
    return vis


def pipeline(in_path: Path, out_dir: Path, mode: str):
    """
    mode: 'g' (Gaussian) uses GaussianBlur, 's' (Salt-Pepper) uses medianBlur.
    Returns dict with everything needed for the montage.
    """
    ensure_dir(out_dir)
    gray = imread_gray(in_path)
    save_img(out_dir / "00_input.png", gray)

# Gaussian noise → Gaussian blur; Salt-pepper → median filter
    if mode == "g":
        den = cv2.GaussianBlur(gray, (5, 5), 0)
    elif mode == "s":
        den = cv2.medianBlur(gray, 3)
    else:
        raise ValueError("mode must be 'g' or 's'")
    save_img(out_dir / "01_denoised.png", den)

    mask_otsu, T = otsu_segment(den)
    save_img(out_dir / "02_otsu.png", mask_otsu)

    mask_clean = morph_clean(mask_otsu)
    save_img(out_dir / "03_morph.png", mask_clean)

    labels, stats, keep = count_components(mask_clean, min_area=50)
    count = len(keep)

    color_map = colorize_labels(labels, keep)
    save_img(out_dir / "04_counted_color.png", color_map)

    # optional: also keep a box overlay
    overlay_boxes = draw_boxes(gray, stats, keep)
    save_img(out_dir / "04_overlay_boxes.png", overlay_boxes)

    areas = [int(stats[lab, cv2.CC_STAT_AREA]) for lab in keep]
    report = [
        f"Input: {in_path.name}",
        f"Mode: {'Gaussian' if mode == 'g' else 'Salt-Pepper'}",
        f"Otsu threshold: {T:.2f}",
        f"Min area kept: 50 px",
        f"Counted rice grains: {count}",
        f"Areas (px): {areas}",
    ]
    (out_dir / "count.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))

    return {
        "gray": gray,
        "den": den,
        "seg": mask_otsu,
        "count_color": color_map[:, :, ::-1],  # BGR→RGB for matplotlib
        "count": count,
    }


def make_montage(res_g, res_s, out_path: Path):
    ensure_dir(out_path.parent)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    titles = ["Original", "Denoised", "Segmented", "Counted"]

    # Row 1: Gaussian
    imgs_g = [res_g["gray"], res_g["den"], res_g["seg"], res_g["count_color"]]
    cmaps_g = ["gray", "gray", "gray", None]
    for j in range(4):
        ax = axes[0, j]
        if cmaps_g[j]:
            ax.imshow(imgs_g[j], cmap=cmaps_g[j])
        else:
            ax.imshow(imgs_g[j])
        ax.set_title(titles[j] if j < 3 else f"Counted: {res_g['count']}")
        ax.axis("off")

    # Row 2: Salt-Pepper
    imgs_s = [res_s["gray"], res_s["den"], res_s["seg"], res_s["count_color"]]
    cmaps_s = ["gray", "gray", "gray", None]
    for j in range(4):
        ax = axes[1, j]
        if cmaps_s[j]:
            ax.imshow(imgs_s[j], cmap=cmaps_s[j])
        else:
            ax.imshow(imgs_s[j])
        ax.set_title(titles[j] if j < 3 else f"Counted: {res_s['count']}")
        ax.axis("off")

    plt.tight_layout()
    fig.suptitle(
        "Q9 – Rice Grain Counting (Top: Gaussian, Bottom: Salt-Pepper)", y=1.02, fontsize=12)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved montage: {out_path}")


def main():
    ROOT = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Q9 rice grain counting")
    parser.add_argument("--gauss", default=str(ROOT /
                        "assets/q9/rice_gaussian.png"))
    parser.add_argument("--salt",  default=str(ROOT /
                        "assets/q9/rice_saltpepper.png"))
    parser.add_argument("--out_g", default=str(ROOT / "outputs/q9/gaussian"))
    parser.add_argument("--out_s", default=str(ROOT /
                        "outputs/q9/salt-and-pepper/"))
    parser.add_argument("--montage", default=str(ROOT /
                        "outputs/q9/all.png"))
    args = parser.parse_args()

    res_g = pipeline(Path(args.gauss), Path(args.out_g), mode="g")
    res_s = pipeline(Path(args.salt),  Path(args.out_s), mode="s")
    make_montage(res_g, res_s, Path(args.montage))


if __name__ == "__main__":
    main()
