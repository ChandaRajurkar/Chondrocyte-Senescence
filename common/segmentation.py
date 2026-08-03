"""
Cell segmentation and morphology feature extraction.

Ported from the original chondrocyte_cell_counting_and_morphology_analysis.py
with cleanup.  Produces a per-cell CSV (combined_morphology.csv) that feeds
the classical models and the MIL model.
"""
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

from . import config as cfg


# ---------------------------------------------------------------------------
# Core segmentation
# ---------------------------------------------------------------------------
def segment_cells(image_bgr: np.ndarray,
                  min_area: int = 100,
                  max_area: int = 50000) -> list[np.ndarray]:
    """Segment cells from a bright-field SA-β-gal image.

    Returns a list of contours (each is an ndarray of points).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 51, 10
    )

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Filter by area
    valid = [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]
    return valid


# ---------------------------------------------------------------------------
# Feature extraction for a single cell
# ---------------------------------------------------------------------------
def extract_cell_features(contour: np.ndarray,
                          gray_image: np.ndarray) -> dict | None:
    """Extract the 14 morphology + texture features for one cell contour."""
    area = cv2.contourArea(contour)
    if area < 1:
        return None

    perimeter = cv2.arcLength(contour, True)
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 0
    rect_area = w * h
    extent = float(area) / rect_area if rect_area > 0 else 0

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0

    equiv_diameter = np.sqrt(4 * area / np.pi)

    if len(contour) >= 5:
        _, (ma, MA), _ = cv2.fitEllipse(contour)
        eccentricity = np.sqrt(1 - (min(ma, MA) / max(ma, MA)) ** 2) if max(ma, MA) > 0 else 0
    else:
        eccentricity = 0

    # GLCM texture features from the cell ROI
    mask = np.zeros(gray_image.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    roi = cv2.bitwise_and(gray_image, gray_image, mask=mask)
    roi_crop = roi[y:y + h, x:x + w]

    if roi_crop.size == 0 or roi_crop.shape[0] < 2 or roi_crop.shape[1] < 2:
        texture = {
            "Texture_Contrast": 0, "Texture_Dissimilarity": 0,
            "Texture_Homogeneity": 0, "Texture_Energy": 0,
            "Texture_Correlation": 0, "Texture_ASM": 0,
        }
    else:
        glcm = graycomatrix(roi_crop, distances=[1], angles=[0],
                            levels=256, symmetric=True, normed=True)
        texture = {
            "Texture_Contrast":       float(graycoprops(glcm, "contrast")[0, 0]),
            "Texture_Dissimilarity":  float(graycoprops(glcm, "dissimilarity")[0, 0]),
            "Texture_Homogeneity":    float(graycoprops(glcm, "homogeneity")[0, 0]),
            "Texture_Energy":         float(graycoprops(glcm, "energy")[0, 0]),
            "Texture_Correlation":    float(graycoprops(glcm, "correlation")[0, 0]),
            "Texture_ASM":            float(graycoprops(glcm, "ASM")[0, 0]),
        }

    return {
        "Area": area,
        "Perimeter": perimeter,
        "Circularity": circularity,
        "Aspect_Ratio": aspect_ratio,
        "Extent": extent,
        "Solidity": solidity,
        "Equivalent_Diameter": equiv_diameter,
        "Eccentricity": eccentricity,
        **texture,
    }


# ---------------------------------------------------------------------------
# Process one image
# ---------------------------------------------------------------------------
def process_image(image_path: str, group_label: str) -> list[dict]:
    """Segment and extract features from all cells in one image.

    Returns a list of feature dicts, each with Cell_ID, Image, Group columns.
    """
    img = cv2.imread(image_path)
    if img is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contours = segment_cells(img)

    rows = []
    for i, cnt in enumerate(contours, start=1):
        feats = extract_cell_features(cnt, gray)
        if feats is not None:
            feats["Cell_ID"] = i
            feats["Image"] = Path(image_path).stem
            feats["Group"] = group_label
            rows.append(feats)
    return rows


# ---------------------------------------------------------------------------
# Run segmentation on the full dataset
# ---------------------------------------------------------------------------
def run_full_segmentation(
    control_dir: Path = cfg.CONTROL_DIR,
    senescent_dir: Path = cfg.SENESCENT_DIR,
    output_csv: Path | None = None,
) -> pd.DataFrame:
    """Segment all images and produce combined_morphology.csv.

    Returns the DataFrame and saves it to output_csv.
    """
    if output_csv is None:
        output_csv = cfg.MORPHOLOGY_OUTPUT / "combined_morphology.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for label, folder in [("Control", control_dir), ("Senescent", senescent_dir)]:
        for root, _, files in os.walk(folder):
            if os.path.basename(root).startswith("."):
                continue
            for fname in sorted(files):
                if fname.lower().endswith(cfg.IMG_EXTENSIONS):
                    fpath = os.path.join(root, fname)
                    rows = process_image(fpath, label)
                    all_rows.extend(rows)
                    print(f"  [{label}] {fname}: {len(rows)} cells")

    df = pd.DataFrame(all_rows)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(df)} cells from {df['Image'].nunique()} images → {output_csv}")
    return df
