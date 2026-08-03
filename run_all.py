"""
End-to-end master orchestrator for the senescence classification pipeline.

Runs:
  Stage 1: Segmentation and feature extraction (produces combined_morphology.csv)
  Stage 2: Classical models training (SVM linear/RBF, Random Forest, XGBoost)
  Stage 3: Deep learning models training (ResNet-18, EfficientNet-B0, DINOv2 probe)
  Stage 4: Proposed models training (MorphoFusionNet, Attention-MIL)
  Stage 5: Result compilation (produces model_comparison_table.csv) and figure generation

Usage:
  python senescence_classifier/run_all.py                 # run all stages
  python senescence_classifier/run_all.py --skip-segment  # reuse existing morphology CSV
"""
import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from senescence_classifier.common import config as cfg
from senescence_classifier.common.segmentation import run_full_segmentation

logger = logging.getLogger("orchestrator")


def run_command(command: list[str], description: str):
    """Run a terminal command via subprocess and stream outputs."""
    logger.info(f"\n>>> Running {description}...")
    logger.info(f"Command: {' '.join(command)}")
    
    result = subprocess.run(command, stdout=sys.stdout, stderr=sys.stderr)
    
    if result.returncode != 0:
        logger.error(f"Error occurred in {description} (exit code: {result.returncode})")
        sys.exit(result.returncode)
    logger.info(f">>> {description} completed successfully.")


def compile_results(output_csv: Path):
    """Scan the metrics output directories and compile Table 1."""
    logger.info("\n>>> Compiling all model metrics into final comparison table...")
    
    # Define models, their metrics.json paths, and their display names
    models_config = [
        # Classical (read from classical_summary.json)
        ("SVM_Linear",      cfg.CLASSICAL_OUTPUT / "SVM_Linear" / "metrics.json"),
        ("SVM_RBF",         cfg.CLASSICAL_OUTPUT / "SVM_RBF" / "metrics.json"),
        ("RandomForest",    cfg.CLASSICAL_OUTPUT / "RandomForest" / "metrics.json"),
        ("XGBoost",         cfg.CLASSICAL_OUTPUT / "XGBoost" / "metrics.json"),
        # Deep
        ("ResNet18",        cfg.DEEP_OUTPUT / "ResNet18" / "metrics.json"),
        ("EfficientNetB0",  cfg.DEEP_OUTPUT / "EfficientNetB0" / "metrics.json"),
        ("DINOv2_Probe",    cfg.DEEP_OUTPUT / "DINOv2_Probe" / "metrics.json"),
        # Proposed
        ("MorphoFusionNet", cfg.PROPOSED_OUTPUT / "MorphoFusionNet" / "metrics.json"),
        ("AttentionMIL",    cfg.PROPOSED_OUTPUT / "AttentionMIL" / "metrics.json"),
    ]
    
    compiled_rows = []
    
    for name, path in models_config:
        if not path.exists():
            logger.warning(f"Metrics file missing for {name}: {path}. Skipping.")
            continue
            
        with open(path, "r") as f:
            data = json.load(f)
            
        summary = data.get("summary", {})
        
        # Helper to format mean ± std
        def fmt(metric_name):
            m = summary.get(metric_name, {})
            val = m.get("mean", float("nan"))
            std = m.get("std", float("nan"))
            if pd.isna(val) or pd.isna(std):
                return "N/A"
            return f"{val:.3f} ± {std:.3f}"
            
        compiled_rows.append({
            "Model":             name,
            "Balanced Accuracy": fmt("balanced_accuracy"),
            "Sensitivity":       fmt("recall_sensitivity"),
            "Specificity":       fmt("specificity"),
            "F1-Score":          fmt("f1"),
            "ROC-AUC":           fmt("roc_auc"),
            "PR-AUC":            fmt("pr_auc"),
            "MCC":               fmt("mcc"),
        })
        
    df = pd.DataFrame(compiled_rows)
    df.to_csv(output_csv, index=False)
    
    # Save a markdown version as well
    md_path = output_csv.with_suffix(".md")
    df.to_markdown(md_path, index=False)
    
    logger.info(f"\nFinal Comparison Table:")
    logger.info("\n" + df.to_string(index=False))
    logger.info(f"\nSaved CSV comparison table → {output_csv}")
    logger.info(f"Saved Markdown comparison table → {md_path}")


def main():
    parser = argparse.ArgumentParser(description="End-to-end pipeline run")
    parser.add_argument("--skip-segmentation", action="store_true",
                        help="Skip cell segmentation and reuse existing morphology CSV")
    parser.add_argument("--morphology-csv", type=str, default=None,
                        help="Path to pre-computed morphology CSV to train models on (skips segmentation)")
    parser.add_argument("--skip-deep", action="store_true",
                        help="Skip training deep models (ResNet-18, EfficientNet, DINOv2 Probe) as they only use raw images and are independent of morphology features")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])

    py_exe = sys.executable
    
    if args.morphology_csv:
        morph_csv = Path(args.morphology_csv).resolve()
        skip_seg = True
        logger.info(f"Using custom morphology CSV: {morph_csv}")
    else:
        morph_csv = cfg.MORPHOLOGY_OUTPUT / "combined_morphology.csv"
        skip_seg = args.skip_segmentation

    # ---------------------------------------------------------------------------
    # STAGE 1: Segmentation
    # ---------------------------------------------------------------------------
    if skip_seg:
        logger.info(f">>> Stage 1: Skipping segmentation (reuse existing CSV: {morph_csv})")
        if not morph_csv.exists():
            logger.error(f"Provided morphology CSV does not exist: {morph_csv}")
            sys.exit(1)
    else:
        logger.info(">>> Stage 1: Segmenting cells and extracting morphology features...")
        run_full_segmentation(output_csv=morph_csv)

    # ---------------------------------------------------------------------------
    # STAGE 2: Classical Models
    # ---------------------------------------------------------------------------
    run_command(
        [py_exe, "-m", "senescence_classifier.classical.train_classical",
         "--morphology-csv", str(morph_csv)],
        "Stage 2 (Classical Models Benchmarking)"
    )

    # ---------------------------------------------------------------------------
    # STAGE 3: Deep Models
    # ---------------------------------------------------------------------------
    if args.skip_deep:
        logger.info(">>> Stage 3: Skipping deep models (reusing existing deep model results)")
    else:
        run_command(
            [py_exe, "-m", "senescence_classifier.deep.train_resnet18"],
            "Stage 3a (ResNet-18 Training)"
        )
        
        run_command(
            [py_exe, "-m", "senescence_classifier.deep.train_efficientnet"],
            "Stage 3b (EfficientNet-B0 Training)"
        )
        
        run_command(
            [py_exe, "-m", "senescence_classifier.deep.train_dinov2_probe"],
            "Stage 3c (DINOv2 Probe Training)"
        )

    # ---------------------------------------------------------------------------
    # STAGE 4: Proposed Models
    # ---------------------------------------------------------------------------
    run_command(
        [py_exe, "-m", "senescence_classifier.proposed.train_fusion",
         "--morphology-csv", str(morph_csv)],
        "Stage 4a (MorphoFusionNet Training)"
    )
    
    run_command(
        [py_exe, "-m", "senescence_classifier.proposed.train_mil",
         "--morphology-csv", str(morph_csv)],
        "Stage 4b (Attention-MIL Training)"
    )

    # ---------------------------------------------------------------------------
    # STAGE 5: Result compilation
    # ---------------------------------------------------------------------------
    output_table = cfg.OUTPUT_DIR / "model_comparison_table.csv"
    compile_results(output_table)

    # Trigger individual figure generation script
    indiv_figures_script = Path(__file__).resolve().parent / "generate_figures.py"
    if indiv_figures_script.exists():
        run_command(
            [py_exe, str(indiv_figures_script)],
            "Stage 5b (Standalone Figure Generation)"
        )

    # Trigger paper figure compilation script
    paper_figures_script = Path(__file__).resolve().parent / "generate_paper_figures.py"
    if paper_figures_script.exists():
        run_command(
            [py_exe, str(paper_figures_script)],
            "Stage 5c (Publication-ready Paper Figures)"
        )

    logger.info("\n" + "="*80)
    logger.info("  ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
    logger.info(f"  Final comparison table: {output_table}")
    logger.info(f"  Figures saved to:      {cfg.OUTPUT_DIR / 'figures'} and {cfg.OUTPUT_DIR / 'paper_figures'}")
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    main()
