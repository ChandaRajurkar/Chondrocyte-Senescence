"""
Shared deep-model training loop for image-level classifiers.

Used by ResNet-18, EfficientNet-B0, and DINOv2 training scripts.
Implements the unified group-aware CV protocol with balanced loss.
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from senescence_classifier.common import config as cfg
from senescence_classifier.common.dataset import (
    SenescenceImageDataset, build_image_manifest,
    get_train_transform, get_val_transform,
)
from senescence_classifier.common.evaluation import (
    set_seed, make_cv, compute_metrics, summarize_folds,
    save_pr_roc, save_confusion_matrix, save_metrics,
)

logger = logging.getLogger("deep")


def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    num_epochs: int = cfg.NUM_EPOCHS,
    lr: float = cfg.LEARNING_RATE,
    patience: int = cfg.PATIENCE,
) -> tuple[list, list]:
    """Train a model for one fold. Returns (val_preds, val_probs)."""

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(num_epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = torch.tensor(labels, dtype=torch.long).to(device) if not isinstance(labels, torch.Tensor) else labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        scheduler.step()
        train_loss /= len(train_loader.dataset)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = torch.tensor(labels, dtype=torch.long).to(device) if not isinstance(labels, torch.Tensor) else labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(val_loader.dataset)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    # Restore best
    if best_state:
        model.load_state_dict(best_state)
    model.to(device)

    # Final validation predictions
    model.eval()
    all_preds, all_probs = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())

    return all_preds, all_probs


def run_cv_deep(
    model_factory,
    model_name: str,
    output_dir: Path,
    num_epochs: int = cfg.NUM_EPOCHS,
    lr: float = cfg.LEARNING_RATE,
):
    """Run full group-aware K-fold CV for a deep image model.

    Parameters
    ----------
    model_factory : callable
        A function that returns a fresh nn.Module instance.
    model_name : str
        Used for logging and file names.
    output_dir : Path
    """
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Device: {device}")

    manifest = build_image_manifest()
    labels = np.array([m["label"] for m in manifest])
    groups = np.array([m["group"] for m in manifest])

    logger.info(f"Dataset: {len(manifest)} images  "
                f"(control={sum(labels==0)}, senescent={sum(labels==1)})")

    # Class weights
    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=labels)
    class_weights = torch.tensor(cw, dtype=torch.float32)

    cv = make_cv()
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    per_fold = []
    oof_probs = np.zeros(len(labels), dtype=float)
    oof_preds = np.full(len(labels), -1, dtype=int)

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(
            np.arange(len(labels)), labels, groups)):
        logger.info(f"\n--- {model_name} Fold {fold_idx+1}/{cfg.NUM_FOLDS} "
                     f"(train={len(train_idx)}, val={len(val_idx)}) ---")

        train_manifest = [manifest[i] for i in train_idx]
        val_manifest   = [manifest[i] for i in val_idx]

        train_ds = SenescenceImageDataset(train_manifest, transform=get_train_transform())
        val_ds   = SenescenceImageDataset(val_manifest,   transform=get_val_transform())

        train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                                  shuffle=True, num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE,
                                  shuffle=False, num_workers=0)

        model = model_factory().to(device)
        preds, probs = train_one_fold(
            model, train_loader, val_loader, class_weights,
            device, num_epochs=num_epochs, lr=lr,
        )

        oof_probs[val_idx] = probs
        oof_preds[val_idx] = preds

        y_val = labels[val_idx]
        fold_metrics = compute_metrics(y_val, preds, probs)
        per_fold.append(fold_metrics)
        logger.info(f"  MCC={fold_metrics['mcc']:.3f}  "
                     f"PR-AUC={fold_metrics['pr_auc']:.3f}")

    # Summarise
    summary = summarize_folds(per_fold)
    logger.info(f"\n{model_name} SUMMARY: "
                f"MCC={summary['mcc']['mean']:.3f}±{summary['mcc']['std']:.3f}  "
                f"PR-AUC={summary['pr_auc']['mean']:.3f}±{summary['pr_auc']['std']:.3f}")

    valid = oof_preds >= 0
    pooled = compute_metrics(labels[valid], oof_preds[valid], oof_probs[valid])

    # Save metrics JSON
    save_metrics({"per_fold": per_fold, "summary": summary, "pooled_oof": pooled},
                 model_dir / "metrics.json")
    
    # Save OOF predictions CSV
    df_oof = pd.DataFrame({
        "image": [manifest[i]["stem"] for i in range(len(manifest))],
        "true_label": labels,
        "prob_senescent": oof_probs,
        "pred_label": oof_preds
    })
    df_oof.to_csv(model_dir / "oof_predictions.csv", index=False)
    logger.info(f"Saved OOF predictions → {model_dir / 'oof_predictions.csv'}")

    save_pr_roc(model_name, labels[valid], oof_probs[valid], model_dir)
    save_confusion_matrix(model_name, pooled["confusion_matrix"], model_dir)

    return summary
