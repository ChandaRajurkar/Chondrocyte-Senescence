"""
Unified dataset utilities for the senescence classification pipeline.

Provides:
  - SenescenceImageDataset : PyTorch Dataset for image-level models
  - load_morphology_csv    : loads per-cell CSV and aggregates to per-image
  - build_image_manifest   : creates (path, label, group) list from folders
"""
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from . import config as cfg
from .evaluation import image_group_id


# ---------------------------------------------------------------------------
# Image manifest builder
# ---------------------------------------------------------------------------
def build_image_manifest(
    control_dir: Path = cfg.CONTROL_DIR,
    senescent_dir: Path = cfg.SENESCENT_DIR,
) -> list[dict]:
    """Scan control / senescent folders and return a list of dicts with
    keys: path, label (0=control, 1=senescent), group, stem.
    """
    manifest = []
    for label, folder in enumerate([control_dir, senescent_dir]):
        for root, _, files in os.walk(folder):
            if os.path.basename(root).startswith("."):
                continue
            for fname in sorted(files):
                if fname.lower().endswith(cfg.IMG_EXTENSIONS):
                    p = Path(root) / fname
                    stem = p.stem
                    manifest.append({
                        "path":  str(p),
                        "label": label,
                        "group": image_group_id(stem),
                        "stem":  stem,
                    })
    return manifest


# ---------------------------------------------------------------------------
# PyTorch image dataset
# ---------------------------------------------------------------------------
class SenescenceImageDataset(Dataset):
    """Image dataset that returns (image_tensor, label) for deep models.

    Parameters
    ----------
    manifest : list[dict]
        Output of ``build_image_manifest()``.
    transform : torchvision transform, optional
        Applied to each PIL image before returning.
    """

    def __init__(self, manifest: list[dict], transform=None):
        self.manifest = manifest
        self.transform = transform
        self.paths  = [m["path"]  for m in manifest]
        self.labels = [m["label"] for m in manifest]
        self.groups = [m["group"] for m in manifest]

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ---------------------------------------------------------------------------
# Standard transforms
# ---------------------------------------------------------------------------
def get_train_transform(mean=None, std=None):
    """Training transform with augmentation."""
    mean = mean or cfg.IMAGENET_MEAN
    std  = std  or cfg.IMAGENET_STD
    return transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def get_val_transform(mean=None, std=None):
    """Validation / test transform (no augmentation)."""
    mean = mean or cfg.IMAGENET_MEAN
    std  = std  or cfg.IMAGENET_STD
    return transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# ---------------------------------------------------------------------------
# Morphology CSV → per-image feature matrix
# ---------------------------------------------------------------------------
def load_morphology_csv(
    csv_path: Path,
    agg_stats: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load the per-cell morphology CSV and aggregate to per-image vectors.

    Returns
    -------
    X : ndarray, shape (n_images, n_features)
        Per-image aggregated morphology vectors.
    y : ndarray, shape (n_images,)
        Labels (0 = control, 1 = senescent).
    groups : ndarray, shape (n_images,)
        Group IDs for CV splitting (= image stem).
    feature_names : list[str]
        Column names of the aggregated feature matrix.
    """
    agg_stats = agg_stats or cfg.AGG_STATS

    df = pd.read_csv(csv_path)

    # Drop the feature we excluded
    if cfg.DROPPED_FEATURE in df.columns:
        df = df.drop(columns=[cfg.DROPPED_FEATURE])

    # Clean group labels
    df["Group"] = df["Group"].astype(str).str.strip().str.capitalize()
    df = df[df["Group"].isin(["Control", "Senescent"])].copy()

    # Clean numeric features
    feats = df[cfg.FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    feats = feats.apply(pd.to_numeric, errors="coerce")
    feats = feats.fillna(feats.median(numeric_only=True))
    df[cfg.FEATURE_COLUMNS] = feats

    # Need Image column for grouping
    if "Image" not in df.columns:
        raise ValueError("CSV must have an 'Image' column for per-image aggregation")

    # Aggregate per-cell → per-image
    agg_dict = {}
    for col in cfg.FEATURE_COLUMNS:
        for stat in agg_stats:
            if stat == "p25":
                agg_dict[(col, stat)] = (col, lambda x: np.percentile(x, 25))
            elif stat == "p75":
                agg_dict[(col, stat)] = (col, lambda x: np.percentile(x, 75))
            else:
                agg_dict[(col, stat)] = (col, stat)

    rows = []
    feature_names = []
    first = True
    for image_name, grp in df.groupby("Image"):
        label = 0 if grp["Group"].iloc[0] == "Control" else 1
        vec = []
        names = []
        for col in cfg.FEATURE_COLUMNS:
            vals = grp[col].values
            for stat in agg_stats:
                if stat == "p25":
                    vec.append(np.percentile(vals, 25))
                elif stat == "p75":
                    vec.append(np.percentile(vals, 75))
                else:
                    vec.append(getattr(np, f"nan{stat}")(vals) if stat != "std"
                               else np.nanstd(vals))
                names.append(f"{col}_{stat}")
        # Add log cell count
        vec.append(np.log1p(len(grp)))
        names.append("log_cell_count")

        rows.append({
            "image": image_name,
            "label": label,
            "group": image_group_id(image_name),
            "features": vec,
        })
        if first:
            feature_names = names
            first = False

    X = np.array([r["features"] for r in rows], dtype=np.float32)
    y = np.array([r["label"]    for r in rows], dtype=int)
    groups = np.array([r["group"] for r in rows])

    # Replace any remaining NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y, groups, feature_names


# ---------------------------------------------------------------------------
# MIL bag builder (per-cell features grouped by image)
# ---------------------------------------------------------------------------
def build_mil_bags(csv_path: Path) -> list[dict]:
    """Build MIL bags from the per-cell morphology CSV.

    Each bag = one imaging field; each instance = one cell's 14 features.

    Returns a list of dicts with keys:
        image, label, group, features (ndarray shape [n_cells, 14])
    """
    df = pd.read_csv(csv_path)
    if cfg.DROPPED_FEATURE in df.columns:
        df = df.drop(columns=[cfg.DROPPED_FEATURE])

    df["Group"] = df["Group"].astype(str).str.strip().str.capitalize()
    df = df[df["Group"].isin(["Control", "Senescent"])].copy()

    feats = df[cfg.FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    feats = feats.apply(pd.to_numeric, errors="coerce")
    feats = feats.fillna(feats.median(numeric_only=True))
    df[cfg.FEATURE_COLUMNS] = feats

    bags = []
    for image_name, grp in df.groupby("Image"):
        label = 0 if grp["Group"].iloc[0] == "Control" else 1
        cell_features = grp[cfg.FEATURE_COLUMNS].values.astype(np.float32)
        bags.append({
            "image":    image_name,
            "label":    label,
            "group":    image_group_id(image_name),
            "features": cell_features,  # shape (n_cells, 14)
        })
    return bags
