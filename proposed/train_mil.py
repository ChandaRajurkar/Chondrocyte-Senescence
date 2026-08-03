"""
Tier 3 — Proposed Model B: Attention-based Multiple Instance Learning (Attention-MIL).

Treats each imaging field as a bag and each segmented cell as an instance.
Uses gated attention pooling (Ilse et al., 2018) to aggregate instance embeddings.
Computes a continuous "senescence burden" as the attention-weighted average
of instance-level senescent probabilities.

Directly addresses reviewer concerns:
  - #1  Leakage-free evaluation using unified group-aware CV
  - #2  Balanced cross-entropy loss based on class frequencies
  - #3  MCC + PR-AUC primary metrics
  - #5  Interpretability: provides cell-level attention weights and senescence burden
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from senescence_classifier.common import config as cfg
from senescence_classifier.common.dataset import build_mil_bags
from senescence_classifier.common.evaluation import (
    set_seed, make_cv, compute_metrics, summarize_folds,
    save_pr_roc, save_confusion_matrix, save_metrics,
)

logger = logging.getLogger("mil")


# ---------------------------------------------------------------------------
# Attention-MIL Architecture
# ---------------------------------------------------------------------------
class AttentionMIL(nn.Module):
    """Gated Attention-based Multiple Instance Learning (Ilse et al., 2018)."""

    def __init__(self, n_features: int, hidden: int = 128, attn: int = 64, dropout: float = 0.3):
        super().__init__()
        
        # Instance encoder (maps cell features to embedding space)
        # Use LayerNorm instead of BatchNorm1d to handle dynamic bag sizes (even size 1) safely
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True)
        )
        
        # Gated attention pooling mechanism
        self.attn_V = nn.Linear(hidden, attn)
        self.attn_U = nn.Linear(hidden, attn)
        self.attn_w = nn.Linear(attn, 1)
        
        # Classification heads
        self.bag_head = nn.Linear(hidden, 2)

    def forward(self, x):
        """Forward pass for a single bag.
        
        x: tensor of shape (n_instances, n_features)
        
        Returns:
            bag_logits: tensor of shape (2,)
            alpha: attention weights of shape (n_instances,)
            inst_prob: instance-level probabilities of shape (n_instances,)
        """
        h = self.encoder(x)  # (n_instances, hidden)
        
        # Gated attention computation
        a = self.attn_w(torch.tanh(self.attn_V(h)) * torch.sigmoid(self.attn_U(h)))  # (n_instances, 1)
        alpha = torch.softmax(a, dim=0)  # (n_instances, 1)
        
        # Attention pooling
        z = torch.sum(alpha * h, dim=0, keepdim=True)  # (1, hidden)
        
        bag_logits = self.bag_head(z).squeeze(0)  # (2,)
        
        # Compute instance-level probabilities using the trained bag_head
        inst_logits = self.bag_head(h)  # (n_instances, 2)
        inst_prob = torch.softmax(inst_logits, dim=1)[:, 1]  # (n_instances,)
        
        return bag_logits, alpha.squeeze(1), inst_prob


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_mil_fold(
    train_bags: list[dict],
    val_bags: list[dict],
    scaler: StandardScaler,
    class_weights: torch.Tensor,
    device: torch.device,
    accum_steps: int = 8,
) -> tuple[list, list, list]:
    """Train MIL model for one fold. Returns (preds, probs, burden_records)."""
    model = AttentionMIL(n_features=len(cfg.FEATURE_COLUMNS)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    
    # Pre-scale bags once to avoid high sklearn/conversion overhead inside the loop
    scaled_train_bags = []
    for b in train_bags:
        if len(b["features"]) > 0:
            scaled_feats = scaler.transform(b["features"])
            scaled_train_bags.append({
                "image": b["image"],
                "label": b["label"],
                "tensor": torch.tensor(scaled_feats, dtype=torch.float32, device=device)
            })
            
    scaled_val_bags = []
    for b in val_bags:
        if len(b["features"]) > 0:
            scaled_feats = scaler.transform(b["features"])
            scaled_val_bags.append({
                "image": b["image"],
                "label": b["label"],
                "tensor": torch.tensor(scaled_feats, dtype=torch.float32, device=device),
                "raw_len": len(b["features"])
            })
        else:
            scaled_val_bags.append({
                "image": b["image"],
                "label": b["label"],
                "tensor": None,
                "raw_len": 0
            })

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(60):  # Higher epochs for MIL
        model.train()
        order = np.random.permutation(len(scaled_train_bags))
        optimizer.zero_grad()
        
        for step, idx in enumerate(order, 1):
            bag = scaled_train_bags[idx]
            logits, _, _ = model(bag["tensor"])
            target = torch.tensor([bag["label"]], dtype=torch.long, device=device)
            loss = criterion(logits.unsqueeze(0), target) / accum_steps
            loss.backward()
            
            if step % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
        optimizer.step()
        optimizer.zero_grad()

        # Validate (bag-level loss)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bag in scaled_val_bags:
                if bag["tensor"] is None:
                    continue
                logits, _, _ = model(bag["tensor"])
                target = torch.tensor([bag["label"]], dtype=torch.long, device=device)
                val_loss += criterion(logits.unsqueeze(0), target).item()
        val_loss /= max(len(scaled_val_bags), 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= cfg.PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.to(device)

    # Evaluate out-of-fold predictions and calculate continuous senescence burden
    model.eval()
    all_preds, all_probs, all_records = [], [], []
    with torch.no_grad():
        for bag in scaled_val_bags:
            if bag["tensor"] is None:
                # Fallback for empty bags
                all_preds.append(0)
                all_probs.append(0.0)
                all_records.append({
                    "image": bag["image"],
                    "true_label": bag["label"],
                    "prob_senescent": 0.0,
                    "senescence_burden": 0.0,
                    "n_cells": 0,
                })
                continue

            logits, alpha, inst_prob = model(bag["tensor"])
            prob = torch.softmax(logits, dim=0)[1].item()
            
            # Continuous burden = attention-weighted senescent probability of cells
            burden = torch.sum(alpha * inst_prob).item()
            
            all_preds.append(int(prob >= 0.5))
            all_probs.append(prob)
            all_records.append({
                "image": bag["image"],
                "true_label": bag["label"],
                "prob_senescent": prob,
                "senescence_burden": burden,
                "n_cells": bag["raw_len"],
            })

    return all_preds, all_probs, all_records


def main():
    parser = argparse.ArgumentParser(description="Train Attention-MIL model")
    parser.add_argument("--morphology-csv", type=Path,
                        default=cfg.MORPHOLOGY_OUTPUT / "combined_morphology.csv")
    parser.add_argument("--output-dir", type=Path, default=cfg.PROPOSED_OUTPUT)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    
    set_seed()
    # Force CPU for MIL: because bag-by-bag training involves millions of tiny kernel launches,
    # GPU launch latency overhead dominates the compute time. CPU has 0 launch latency and is much faster here.
    device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Build bags from cell-level features
    logger.info(f"Building bags from {args.morphology_csv}")
    bags = build_mil_bags(args.morphology_csv)
    
    y = np.array([b["label"] for b in bags])
    groups = np.array([b["group"] for b in bags])
    
    logger.info(f"Loaded {len(bags)} bags "
                f"(control={sum(y==0)}, senescent={sum(y==1)})")

    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
    class_weights = torch.tensor(cw, dtype=torch.float32)

    cv = make_cv()
    model_dir = args.output_dir / "AttentionMIL"
    model_dir.mkdir(parents=True, exist_ok=True)

    per_fold = []
    oof_probs = np.zeros(len(y), dtype=float)
    oof_preds = np.full(len(y), -1, dtype=int)
    all_records = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(np.arange(len(y)), y, groups)):
        logger.info(f"\n--- Attention-MIL Fold {fold_idx+1}/{cfg.NUM_FOLDS} ---")

        train_bags = [bags[i] for i in train_idx]
        val_bags   = [bags[i] for i in val_idx]

        # Fit morphology scaler on train fold instances only (no data leakage)
        train_instances = np.vstack([b["features"] for b in train_bags if len(b["features"]) > 0])
        scaler = StandardScaler()
        scaler.fit(train_instances)

        preds, probs, records = train_mil_fold(
            train_bags, val_bags, scaler, class_weights, device
        )

        oof_probs[val_idx] = probs
        oof_preds[val_idx] = preds
        all_records.extend(records)

        fold_metrics = compute_metrics(y[val_idx], preds, probs)
        per_fold.append(fold_metrics)
        logger.info(f"  Fold MCC={fold_metrics['mcc']:.3f}  "
                     f"PR-AUC={fold_metrics['pr_auc']:.3f}")

    # Summary
    summary = summarize_folds(per_fold)
    logger.info(f"\nMIL Summary:")
    logger.info(f"  MCC    = {summary['mcc']['mean']:.3f} ± {summary['mcc']['std']:.3f}")
    logger.info(f"  PR-AUC = {summary['pr_auc']['mean']:.3f} ± {summary['pr_auc']['std']:.3f}")

    valid = oof_preds >= 0
    pooled = compute_metrics(y[valid], oof_preds[valid], oof_probs[valid])
    
    # Save results
    save_metrics({
        "per_fold": per_fold,
        "summary": summary,
        "pooled_oof": pooled
    }, model_dir / "metrics.json")
    
    # Save senescence burden per field
    df_burden = pd.DataFrame(all_records)
    df_burden.to_csv(model_dir / "senescence_burden_per_field.csv", index=False)
    logger.info(f"Saved burden estimates per field → {model_dir / 'senescence_burden_per_field.csv'}")

    # Save OOF predictions CSV
    df_oof = pd.DataFrame({
        "image": [r["image"] for r in all_records],
        "true_label": [r["true_label"] for r in all_records],
        "prob_senescent": [r["prob_senescent"] for r in all_records],
        "pred_label": [int(r["prob_senescent"] >= 0.5) for r in all_records]
    })
    df_oof.to_csv(model_dir / "oof_predictions.csv", index=False)
    logger.info(f"    Saved OOF predictions → {model_dir / 'oof_predictions.csv'}")

    save_pr_roc("AttentionMIL", y[valid], oof_probs[valid], model_dir)
    save_confusion_matrix("AttentionMIL", pooled["confusion_matrix"], model_dir)
    
    logger.info(f"Attention-MIL results successfully saved to {model_dir}")


if __name__ == "__main__":
    main()
