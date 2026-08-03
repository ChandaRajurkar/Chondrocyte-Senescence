"""
Generate Figure 11: Cellpose-based cell segmentation and size-filtering QC
High-end journal quality — 6-panel layout (2 rows × 3 columns):
  Row 1 (Control):   (a) Original  |  (b) Cellpose mask overlay  |  (c) Size-filtered contours
  Row 2 (Senescent): (d) Original  |  (e) Cellpose mask overlay  |  (f) Size-filtered contours
"""
import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch, FancyBboxPatch
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from cellpose import models
    HAS_CELLPOSE = True
except ImportError:
    HAS_CELLPOSE = False

from senescence_classifier.common import config as cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
MIN_AREA = 100
MAX_AREA = 50000

def _load_rgb(path):
    """Load an image as RGB numpy array."""
    bgr = cv2.imread(str(path))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _run_cellpose(img_rgb):
    """Run Cellpose cyto2 on an RGB image; return masks array."""
    if HAS_CELLPOSE:
        model = models.Cellpose(gpu=False, model_type="cyto2")
        masks, _, _, _ = model.eval(
            img_rgb, diameter=None, channels=[0, 0],
            flow_threshold=0.6, cellprob_threshold=-1.0,
        )
    else:
        # Fallback — adaptive threshold
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 51, 5,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        masks = np.zeros(gray.shape, dtype=np.int32)
        cid = 0
        for cnt in contours:
            cid += 1
            cv2.drawContours(masks, [cnt], -1, cid, -1)
    return masks


def _classify_contours(masks):
    """Split Cellpose masks into accepted / rejected contour lists."""
    valid, rejected = [], []
    for cid in np.unique(masks):
        if cid == 0:
            continue
        single = np.zeros(masks.shape, dtype=np.uint8)
        single[masks == cid] = 255
        cnts, _ = cv2.findContours(single, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if MIN_AREA <= area <= MAX_AREA:
                valid.append(c)
            else:
                rejected.append(c)
    return valid, rejected


def _colored_mask_overlay(img_rgb, masks, alpha=0.45):
    """Return the original image with a semi-transparent coloured instance mask."""
    overlay = img_rgb.copy().astype(np.float64)
    if masks.max() == 0:
        return overlay.astype(np.uint8)
    np.random.seed(42)
    colors = np.random.rand(masks.max() + 1, 3) * 0.8 + 0.2  # bright palette
    colors[0] = [0, 0, 0]
    mask_rgb = (colors[masks] * 255).astype(np.float64)
    fg = masks > 0
    overlay[fg] = (1 - alpha) * overlay[fg] + alpha * mask_rgb[fg]
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _contour_overlay(img_rgb, valid, rejected, line_w=2):
    """Draw accepted (green) and rejected (red) contours on the image."""
    out = img_rgb.copy()
    cv2.drawContours(out, valid, -1, (0, 220, 0), line_w)
    cv2.drawContours(out, rejected, -1, (220, 40, 40), line_w)
    return out


def _add_panel_label(ax, label, x=0.01, y=1.02, fontsize=16):
    """Add a bold panel letter (a, b, c …) positioned neatly above the top-left corner of each subplot."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold", va="bottom", ha="left",
            fontfamily="sans-serif")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def generate_cellpose_qc_figure():
    out_dir = cfg.OUTPUT_DIR / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- User-specified representative images ---
    ctrl_path = cfg.CONTROL_DIR / "image1_name.tif"
    sen_path  = cfg.SENESCENT_DIR / "image2_name.tif"

    if not ctrl_path.exists() or not sen_path.exists():
        print(f"ERROR: Required images not found:\n  {ctrl_path}\n  {sen_path}")
        return

    print(f"Control image : {ctrl_path.name}")
    print(f"Senescent image: {sen_path.name}")

    # Load images
    ctrl_rgb = _load_rgb(ctrl_path)
    sen_rgb  = _load_rgb(sen_path)

    # Run Cellpose on both
    print("Running Cellpose on control image...")
    ctrl_masks = _run_cellpose(ctrl_rgb)
    print(f"  → {ctrl_masks.max()} instances detected")

    print("Running Cellpose on senescent image...")
    sen_masks = _run_cellpose(sen_rgb)
    print(f"  → {sen_masks.max()} instances detected")

    # Classify contours
    ctrl_valid, ctrl_rej = _classify_contours(ctrl_masks)
    sen_valid, sen_rej   = _classify_contours(sen_masks)

    # --- Build the 6-panel figure ---
    fig = plt.figure(figsize=(18, 11), facecolor="white")
    gs = gridspec.GridSpec(2, 3, wspace=0.05, hspace=0.08,
                           left=0.04, right=0.98, top=0.90, bottom=0.02)

    # Row labels
    fig.text(0.015, 0.70, "Control", fontsize=14, fontweight="bold",
             rotation=90, va="center", ha="center", color="#2e7d32",
             fontfamily="sans-serif")
    fig.text(0.015, 0.25, "Senescent", fontsize=14, fontweight="bold",
             rotation=90, va="center", ha="center", color="#c62828",
             fontfamily="sans-serif")

    # Column headers positioned right above the top row
    col_titles = ["Original Bright-Field", "Cellpose Instance Mask", "Size-Filtered Contour QC"]
    for j, title in enumerate(col_titles):
        fig.text(0.04 + (j * 0.315) + 0.155, 0.935, title,
                 fontsize=13, fontweight="bold", ha="center", va="bottom",
                 fontfamily="sans-serif")

    panels = [
        # Row 0 — Control
        ("a", ctrl_rgb, None),
        ("b", _colored_mask_overlay(ctrl_rgb, ctrl_masks), None),
        ("c", _contour_overlay(ctrl_rgb, ctrl_valid, ctrl_rej),
         (ctrl_valid, ctrl_rej)),
        # Row 1 — Senescent
        ("d", sen_rgb, None),
        ("e", _colored_mask_overlay(sen_rgb, sen_masks), None),
        ("f", _contour_overlay(sen_rgb, sen_valid, sen_rej),
         (sen_valid, sen_rej)),
    ]

    for idx, (label, img, contour_info) in enumerate(panels):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(img)
        ax.axis("off")
        _add_panel_label(ax, label)

        # Add counts annotation for contour panels (c, f)
        if contour_info is not None:
            v, r = contour_info
            legend_els = [
                Line2D([0], [0], color="#00dc00", linewidth=2.5,
                       label=f"Accepted  (n = {len(v)})"),
                Line2D([0], [0], color="#dc2828", linewidth=2.5,
                       label=f"Rejected  (n = {len(r)})"),
            ]
            ax.legend(handles=legend_els, loc="lower left", fontsize=9,
                      frameon=True, fancybox=True,
                      facecolor="white", edgecolor="#999999",
                      framealpha=0.92, handlelength=1.6)

        # Instance count annotation for mask panels (b, e)
        if label in ("b", "e"):
            n_inst = ctrl_masks.max() if label == "b" else sen_masks.max()
            ax.text(0.98, 0.03, f"{n_inst} instances",
                    transform=ax.transAxes, fontsize=10, fontweight="bold",
                    color="white", ha="right", va="bottom",
                    bbox=dict(facecolor="black", alpha=0.6, pad=3,
                              edgecolor="none", boxstyle="round,pad=0.3"))

    # --- Suptitle ---
    fig.suptitle("Cellpose-Based Cell Segmentation and Size-Filtering Quality Control",
                 fontsize=17, fontweight="bold", y=0.98, fontfamily="sans-serif")

    # --- Save ---
    out_path = out_dir / "Figure11.png"
    fig.savefig(out_path, dpi=cfg.PLOT_DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)

    # Copy to figures/ folder
    fig_dir = cfg.OUTPUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(out_path, fig_dir / "cellpose_segmentation_qc.png")

    print(f"Figure 11 saved to: {out_path}")
    print(f"Individual copy saved to: {fig_dir / 'cellpose_segmentation_qc.png'}")


if __name__ == "__main__":
    generate_cellpose_qc_figure()
