"""
Shared evaluation protocol — every model in the paper uses IDENTICAL folds,
metrics, and curves.  This is what makes Table 1 a fair comparison.

Key design decisions (all address reviewer concerns):
  - StratifiedGroupKFold with image-level grouping → no within-image leakage
  - Primary metrics: MCC + PR-AUC  (Chicco & Jurman 2023)
  - Secondary: balanced accuracy, sensitivity, specificity, F1, ROC-AUC
  - Pooled out-of-fold predictions for PR / ROC curves
"""
import json
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix, roc_curve, precision_recall_curve,
)
import torch

from . import config as cfg


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = cfg.SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Group extraction
# ---------------------------------------------------------------------------
def image_group_id(stem: str) -> str:
    """Return the grouping key for an image.

    Since the filenames carry no well/plate metadata (confirmed by user),
    each image is its own group.  StratifiedGroupKFold then behaves like
    StratifiedKFold — but the code is ready for real grouping if metadata
    becomes available in the future.
    """
    return str(stem).strip()


# ---------------------------------------------------------------------------
# Cross-validation splitter
# ---------------------------------------------------------------------------
def make_cv(n_splits: int = cfg.NUM_FOLDS):
    """Return the canonical CV splitter used by every model."""
    return StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=cfg.SEED
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_prob):
    """Compute the full metric suite from true labels, predictions, and probs.

    Returns a dict with: accuracy, balanced_accuracy, precision,
    recall_sensitivity, specificity, f1, roc_auc, pr_auc, mcc, confusion_matrix.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    two_classes = len(np.unique(y_true)) > 1

    return {
        "accuracy":           accuracy_score(y_true, y_pred),
        "balanced_accuracy":  balanced_accuracy_score(y_true, y_pred),
        "precision":          precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_sensitivity": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "specificity":        specificity,
        "f1":                 f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "roc_auc":            roc_auc_score(y_true, y_prob) if two_classes else float("nan"),
        "pr_auc":             average_precision_score(y_true, y_prob) if two_classes else float("nan"),
        "mcc":                matthews_corrcoef(y_true, y_pred),
        "confusion_matrix":   cm.tolist(),
    }


def summarize_folds(per_fold: list[dict]) -> dict:
    """Aggregate per-fold metrics into mean ± std."""
    keys = [
        "accuracy", "balanced_accuracy", "precision", "recall_sensitivity",
        "specificity", "f1", "roc_auc", "pr_auc", "mcc",
    ]
    return {
        k: {
            "mean": float(np.nanmean([f[k] for f in per_fold])),
            "std":  float(np.nanstd([f[k] for f in per_fold])),
        }
        for k in keys
    }


# ---------------------------------------------------------------------------
# Curve plotting (pooled out-of-fold)
# ---------------------------------------------------------------------------
def save_pr_roc(name: str, y_true, y_prob, out: Path):
    """Save publication-quality PR and ROC curves from pooled OOF predictions."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        return

    out.mkdir(parents=True, exist_ok=True)

    # Precision-Recall
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, lw=2, label=f"PR-AUC = {ap:.3f}")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title(f"{name} — Precision-Recall (OOF)")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.savefig(out / f"{name}_pr_oof.png", dpi=cfg.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC-AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{name} — ROC (OOF)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.savefig(out / f"{name}_roc_oof.png", dpi=cfg.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(name: str, cm, out: Path):
    """Save a labelled confusion matrix heatmap."""
    import seaborn as sns

    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        np.array(cm), annot=True, fmt="d", cmap="Blues",
        xticklabels=["Control", "Senescent"],
        yticklabels=["Control", "Senescent"],
        ax=ax,
    )
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    ax.set_title(f"{name} — Confusion Matrix")
    fig.savefig(out / f"{name}_cm.png", dpi=cfg.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# JSON-safe serialisation
# ---------------------------------------------------------------------------
def save_metrics(metrics: dict, path: Path):
    """Write metrics dict to JSON, handling numpy types."""
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        return obj

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_convert(metrics), f, indent=2)
