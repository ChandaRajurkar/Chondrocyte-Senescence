"""
Verification script for the senescence classification pipeline.

Checks:
  1. Configuration paths and outputs creation.
  2. Dataset loading and class balancing.
  3. Feature extraction and morphology aggregation.
  4. Train-validation split group check (leakage prevention).
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from senescence_classifier.common import config as cfg
from senescence_classifier.common.dataset import build_image_manifest, load_morphology_csv
from senescence_classifier.common.evaluation import make_cv


def verify_all():
    print("="*60)
    print(" Verifying Senescence Pipeline Infrastructure")
    print("="*60)

    # 1. Config Check
    print("\n1. Checking configuration...")
    print(f"Workspace Root: {cfg.WORKSPACE_ROOT}")
    print(f"Control directory exists: {cfg.CONTROL_DIR.exists()} ({cfg.CONTROL_DIR})")
    print(f"Senescent directory exists: {cfg.SENESCENT_DIR.exists()} ({cfg.SENESCENT_DIR})")
    print(f"Output directory: {cfg.OUTPUT_DIR}")

    # 2. Manifest Check
    print("\n2. Checking image manifest...")
    manifest = build_image_manifest()
    print(f"Found {len(manifest)} total images.")
    n_ctrl = sum(1 for m in manifest if m["label"] == 0)
    n_sens = sum(1 for m in manifest if m["label"] == 1)
    print(f"  Control images:   {n_ctrl}")
    print(f"  Senescent images: {n_sens}")

    if len(manifest) == 0:
        print("  WARNING: No images found. Check that the Dataset folder exists and is populated.")

    # 3. Leakage Prevention Check
    print("\n3. Checking cross-validation and leakage prevention...")
    if len(manifest) > 0:
        cv = make_cv()
        labels = [m["label"] for m in manifest]
        groups = [m["group"] for m in manifest]
        
        has_leakage = False
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(manifest, labels, groups)):
            train_groups = set(groups[i] for i in train_idx)
            val_groups = set(groups[i] for i in val_idx)
            overlap = train_groups.intersection(val_groups)
            
            print(f"  Fold {fold_idx+1}: train_size={len(train_idx)}, val_size={len(val_idx)}, group_overlap={len(overlap)}")
            if len(overlap) > 0:
                print(f"    CRITICAL: Group overlap detected in Fold {fold_idx+1}: {overlap}")
                has_leakage = True
                
        if not has_leakage:
            print("  SUCCESS: Leakage check passed. No groups overlap between train and validation folds.")
        else:
            print("  FAIL: Leakage check failed.")

    # 4. Check morphology CSV if it exists
    print("\n4. Checking morphology CSV...")
    morph_csv = cfg.MORPHOLOGY_OUTPUT / "combined_morphology.csv"
    if morph_csv.exists():
        print(f"Morphology CSV found: {morph_csv}")
        X, y, groups, feat_names = load_morphology_csv(morph_csv)
        print(f"  Aggregated morphology shape: {X.shape}")
        print(f"  Number of unique images: {len(np.unique(groups)) if 'np' in globals() else len(set(groups))}")
        print(f"  Number of features: {len(feat_names)}")
    else:
        print("  Morphology CSV not generated yet (Stage 1 has not been run).")

    print("\n" + "="*60)
    print(" Verification complete!")
    print("="*60)


if __name__ == "__main__":
    import numpy as np  # locally import for validation if needed
    verify_all()
