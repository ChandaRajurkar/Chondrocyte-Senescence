"""
Tier 1 — Classical morphology-based models.

Models: SVM (linear), SVM (RBF), Random Forest, XGBoost.
All use per-image aggregated morphology vectors and the shared
group-aware CV protocol.

Addresses reviewer concerns:
  - #2  class_weight='balanced' for all models
  - #3  MCC + PR-AUC as primary metrics
  - #4  identical folds, identical features → fair comparison
  - #5  SVM linear + RBF tested (reviewer asked about kernel choice)

Run:
  python -m senescence_classifier.classical.train_classical \\
      --morphology-csv senescence_classifier/Outputs/morphology/combined_morphology.csv
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Try to import xgboost; fall back gracefully
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from senescence_classifier.common import config as cfg
from senescence_classifier.common.dataset import load_morphology_csv
from senescence_classifier.common.evaluation import (
    set_seed, make_cv, compute_metrics, summarize_folds,
    save_pr_roc, save_confusion_matrix, save_metrics,
)

logger = logging.getLogger("classical")


def get_models() -> list[tuple[str, object]]:
    """Return (name, model) pairs for all classical models."""
    models = [
        ("SVM_Linear", SVC(
            kernel="linear", probability=True,
            class_weight="balanced", random_state=cfg.SEED)),
        ("SVM_RBF", SVC(
            kernel="rbf", probability=True,
            class_weight="balanced", random_state=cfg.SEED)),
        ("RandomForest", RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=cfg.SEED, n_jobs=-1)),
    ]
    if HAS_XGB:
        models.append(("XGBoost", XGBClassifier(
            n_estimators=300, use_label_encoder=False,
            eval_metric="logloss", random_state=cfg.SEED,
        )))
    return models


def train_and_evaluate(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray,
    output_dir: Path,
):
    """Train all classical models under identical group-aware 5-fold CV."""
    set_seed()
    output_dir.mkdir(parents=True, exist_ok=True)
    cv = make_cv()

    # For XGBoost: compute scale_pos_weight
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    all_results = {}

    for model_name, model in get_models():
        logger.info(f"\n{'='*60}\n  {model_name}\n{'='*60}")
        model_dir = output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Set XGBoost scale_pos_weight
        if model_name == "XGBoost":
            model.set_params(scale_pos_weight=scale_pos_weight)

        per_fold = []
        oof_probs = np.zeros(len(y), dtype=float)
        oof_preds = np.full(len(y), -1, dtype=int)

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
            logger.info(f"  Fold {fold_idx + 1}/{cfg.NUM_FOLDS} "
                        f"(train={len(train_idx)}, val={len(val_idx)})")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Scale using training-fold stats only
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s   = scaler.transform(X_val)

            model_clone = type(model)(**model.get_params())
            model_clone.fit(X_train_s, y_train)

            y_pred = model_clone.predict(X_val_s)
            y_prob = model_clone.predict_proba(X_val_s)[:, 1]

            oof_probs[val_idx] = y_prob
            oof_preds[val_idx] = y_pred

            fold_metrics = compute_metrics(y_val, y_pred, y_prob)
            per_fold.append(fold_metrics)
            logger.info(f"    MCC={fold_metrics['mcc']:.3f}  "
                        f"PR-AUC={fold_metrics['pr_auc']:.3f}  "
                        f"Bal.Acc={fold_metrics['balanced_accuracy']:.3f}")

        # Summarise
        summary = summarize_folds(per_fold)
        logger.info(f"\n  {model_name} SUMMARY:")
        logger.info(f"    MCC    = {summary['mcc']['mean']:.3f} ± {summary['mcc']['std']:.3f}")
        logger.info(f"    PR-AUC = {summary['pr_auc']['mean']:.3f} ± {summary['pr_auc']['std']:.3f}")

        # Pooled OOF metrics
        valid = oof_preds >= 0
        pooled = compute_metrics(y[valid], oof_preds[valid], oof_probs[valid])

        # Save metrics JSON
        save_metrics({"per_fold": per_fold, "summary": summary, "pooled_oof": pooled},
                     model_dir / "metrics.json")
        
        # Save OOF predictions CSV
        df_oof = pd.DataFrame({
            "image": groups[valid],
            "true_label": y[valid],
            "prob_senescent": oof_probs[valid],
            "pred_label": oof_preds[valid]
        })
        df_oof.to_csv(model_dir / "oof_predictions.csv", index=False)
        logger.info(f"    Saved OOF predictions → {model_dir / 'oof_predictions.csv'}")

        save_pr_roc(model_name, y[valid], oof_probs[valid], model_dir)
        save_confusion_matrix(model_name, pooled["confusion_matrix"], model_dir)

        all_results[model_name] = summary

    # Save combined summary
    save_metrics(all_results, output_dir / "classical_summary.json")
    logger.info(f"\nAll classical results saved to {output_dir}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Train classical morphology models")
    parser.add_argument("--morphology-csv", type=Path,
                        default=cfg.MORPHOLOGY_OUTPUT / "combined_morphology.csv")
    parser.add_argument("--output-dir", type=Path, default=cfg.CLASSICAL_OUTPUT)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])

    logger.info(f"Loading morphology from {args.morphology_csv}")
    X, y, groups, feature_names = load_morphology_csv(args.morphology_csv)
    logger.info(f"Loaded {X.shape[0]} images, {X.shape[1]} features, "
                f"control={sum(y==0)}, senescent={sum(y==1)}")

    train_and_evaluate(X, y, groups, args.output_dir)


if __name__ == "__main__":
    main()
