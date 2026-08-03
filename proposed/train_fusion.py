"""
Tier 3 — Proposed Model A: MorphoFusionNet.

Fuses features from a frozen DINOv2 backbone (image stream)
and a multi-layer perceptron (aggregated morphology stream)
using a learned gated attention mechanism.

Directly addresses reviewer concerns:
  - #1  Leakage-free evaluation using unified group-aware CV
  - #2  Balanced cross-entropy loss based on class frequencies
  - #3  MCC + PR-AUC primary metrics
  - #5  Provides interpretability via the gate weight distribution
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from senescence_classifier.common import config as cfg
from senescence_classifier.common.dataset import (
    build_image_manifest, load_morphology_csv,
    get_train_transform, get_val_transform,
)
from senescence_classifier.common.evaluation import (
    set_seed, make_cv, compute_metrics, summarize_folds,
    save_pr_roc, save_confusion_matrix, save_metrics,
)

logger = logging.getLogger("fusion")


# ---------------------------------------------------------------------------
# Fusion Dataset Class
# ---------------------------------------------------------------------------
class FusionDataset(Dataset):
    """Dataset returning (image, morphology_vector, label) for each sample."""

    def __init__(self, manifest: list[dict], morph_map: dict[str, np.ndarray], transform=None):
        self.manifest = manifest
        self.morph_map = morph_map
        self.transform = transform

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        item = self.manifest[idx]
        stem = item["stem"]
        label = item["label"]
        
        # Load image
        from PIL import Image
        img = Image.open(item["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
            
        # Get aggregated morphology vector
        morph_vec = torch.tensor(self.morph_map[stem], dtype=torch.float32)
        
        return img, morph_vec, label


# ---------------------------------------------------------------------------
# MorphoFusionNet Architecture
# ---------------------------------------------------------------------------
class MorphoFusionNet(nn.Module):
    """Dynamic Gated Fusion Network combining image & morphology streams."""

    def __init__(
        self,
        n_morph: int,
        num_classes: int = 2,
        model_name: str = "facebook/dinov2-small",
        fuse_dim: int = 128
    ):
        super().__init__()
        logger.info(f"Fusion backbone: Loading DINOv2 pre-trained weights ({model_name})")
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # Freeze DINOv2 backbone completely
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.img_proj = nn.Sequential(
            nn.Linear(self.backbone.config.hidden_size, fuse_dim),
            nn.BatchNorm1d(fuse_dim),
            nn.ReLU(inplace=True)
        )
        
        self.morph_mlp = nn.Sequential(
            nn.Linear(n_morph, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, fuse_dim),
            nn.BatchNorm1d(fuse_dim),
            nn.ReLU(inplace=True)
        )
        
        # Gate learns relative contribution of image vs morphology stream
        self.gate = nn.Sequential(
            nn.Linear(2 * fuse_dim, 2),
            nn.Softmax(dim=1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(fuse_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, img, morph, return_gate=False):
        # Extract image features from CLS token
        with torch.no_grad():
            outputs = self.backbone(img)
            cls_token = outputs.pooler_output
            
        f_img = self.img_proj(cls_token)
        f_morph = self.morph_mlp(morph)
        
        # Concatenate features and compute gating weights
        g = self.gate(torch.cat([f_img, f_morph], dim=1))  # (B, 2)
        
        # Weighted sum based on the gating mechanism
        fused = g[:, 0:1] * f_img + g[:, 1:2] * f_morph
        logits = self.classifier(fused)
        
        if return_gate:
            return logits, g
        return logits


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_fusion_fold(
    model: MorphoFusionNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
) -> tuple[list, list, list]:
    """Train fusion model for one fold. Returns (preds, probs, gate_weights)."""
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.NUM_EPOCHS)

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(cfg.NUM_EPOCHS):
        # Train
        model.train()
        for images, morphs, labels in train_loader:
            images = images.to(device)
            morphs = morphs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, morphs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, morphs, labels in val_loader:
                images = images.to(device)
                morphs = morphs.to(device)
                labels = labels.to(device)
                outputs = model(images, morphs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= cfg.PATIENCE:
                break

    # Restore best model state
    if best_state:
        model.load_state_dict(best_state)
    model.to(device)

    # Final evaluation on validation set
    model.eval()
    all_preds, all_probs, all_gates = [], [], []
    with torch.no_grad():
        for images, morphs, _ in val_loader:
            images = images.to(device)
            morphs = morphs.to(device)
            outputs, gates = model(images, morphs, return_gate=True)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())
            all_gates.extend(gates.cpu().numpy().tolist())

    return all_preds, all_probs, all_gates


def main():
    parser = argparse.ArgumentParser(description="Train MorphoFusionNet")
    parser.add_argument("--morphology-csv", type=Path,
                        default=cfg.MORPHOLOGY_OUTPUT / "combined_morphology.csv")
    parser.add_argument("--output-dir", type=Path, default=cfg.PROPOSED_OUTPUT)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load morphology aggregation
    logger.info(f"Loading morphology aggregation from {args.morphology_csv}")
    X_morph, y_morph, groups_morph, _ = load_morphology_csv(args.morphology_csv)
    morph_map = {stem: X_morph[i] for i, stem in enumerate(groups_morph)}

    # Build image manifest and sync
    manifest = build_image_manifest()
    synced_manifest = [m for m in manifest if m["stem"] in morph_map]
    
    logger.info(f"Sync complete. Unified dataset has {len(synced_manifest)} images "
                f"(Dropped {len(manifest) - len(synced_manifest)} due to missing morphology rows)")

    labels = np.array([m["label"] for m in synced_manifest])
    groups = np.array([m["group"] for m in synced_manifest])

    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=labels)
    class_weights = torch.tensor(cw, dtype=torch.float32)

    cv = make_cv()
    model_dir = args.output_dir / "MorphoFusionNet"
    model_dir.mkdir(parents=True, exist_ok=True)

    per_fold = []
    oof_probs = np.zeros(len(labels), dtype=float)
    oof_preds = np.full(len(labels), -1, dtype=int)
    oof_gates = np.zeros((len(labels), 2), dtype=float)

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(
            np.arange(len(labels)), labels, groups)):
        logger.info(f"\n--- MorphoFusionNet Fold {fold_idx+1}/{cfg.NUM_FOLDS} ---")

        train_manifest = [synced_manifest[i] for i in train_idx]
        val_manifest   = [synced_manifest[i] for i in val_idx]

        train_ds = FusionDataset(train_manifest, morph_map, transform=get_train_transform())
        val_ds   = FusionDataset(val_manifest,   morph_map, transform=get_val_transform())

        train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                                  shuffle=True, num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE,
                                  shuffle=False, num_workers=0)

        # Build model
        model = MorphoFusionNet(n_morph=X_morph.shape[1], num_classes=2).to(device)
        
        preds, probs, gates = train_fusion_fold(
            model, train_loader, val_loader, class_weights, device
        )

        oof_probs[val_idx] = probs
        oof_preds[val_idx] = preds
        oof_gates[val_idx] = gates

        y_val = labels[val_idx]
        fold_metrics = compute_metrics(y_val, preds, probs)
        per_fold.append(fold_metrics)
        logger.info(f"  Fold MCC={fold_metrics['mcc']:.3f}  "
                     f"PR-AUC={fold_metrics['pr_auc']:.3f}")

    # Summary
    summary = summarize_folds(per_fold)
    mean_gate = oof_gates.mean(axis=0)
    logger.info(f"\nFusion summary:")
    logger.info(f"  MCC    = {summary['mcc']['mean']:.3f} ± {summary['mcc']['std']:.3f}")
    logger.info(f"  PR-AUC = {summary['pr_auc']['mean']:.3f} ± {summary['pr_auc']['std']:.3f}")
    logger.info(f"  Gate weights (Image / Morphology): {mean_gate[0]:.3f} / {mean_gate[1]:.3f}")

    valid = oof_preds >= 0
    pooled = compute_metrics(labels[valid], oof_preds[valid], oof_probs[valid])
    
    # Save results
    save_metrics({
        "per_fold": per_fold,
        "summary": summary,
        "pooled_oof": pooled,
        "mean_gate_weights": mean_gate.tolist()
    }, model_dir / "metrics.json")
    
    # Save OOF predictions CSV
    import pandas as pd
    df_oof = pd.DataFrame({
        "image": [m["stem"] for m in synced_manifest],
        "true_label": labels,
        "prob_senescent": oof_probs,
        "pred_label": oof_preds,
        "gate_image": oof_gates[:, 0],
        "gate_morph": oof_gates[:, 1]
    })
    df_oof.to_csv(model_dir / "oof_predictions.csv", index=False)
    logger.info(f"    Saved OOF predictions → {model_dir / 'oof_predictions.csv'}")
    
    save_pr_roc("MorphoFusionNet", labels[valid], oof_probs[valid], model_dir)
    save_confusion_matrix("MorphoFusionNet", pooled["confusion_matrix"], model_dir)
    
    logger.info(f"MorphoFusionNet results successfully saved to {model_dir}")


if __name__ == "__main__":
    main()
