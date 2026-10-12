# Leakage-Controlled Multimodal Fusion and Attention-Based Multiple-Instance Learning for Automated Quantification of Articular Chondrocyte Senescence

This repository provides the official source code, evaluation pipeline, and benchmark metrics for automated SA-β-gal chondrocyte senescence classification.

## Key Features
- **Leakage-Controlled Protocol**: Strict image-field-level grouped splitting (`StratifiedGroupKFold`) preventing data leakage across training and test folds.
- **SenesFuseNet**: Gated multimodal fusion combining raw image representations (DINOv2) with handcrafted morphometric descriptors via an inspectable gating mechanism.
- **Attention-MIL**: Multiple-instance learning framework providing cell-level attention maps and continuous field-level senescence burden prediction.
- **8-Model Benchmark**: Baseline comparison covering classical ML (Linear SVM, RBF SVM, Random Forest), deep baselines (ResNet-18, EfficientNet-B0, DINOv2 Probe), and proposed architectures.

---

## Directory Structure

```
Code_Release_Package/
├── common/                  # Configuration, Dataset Loaders, Cellpose Segmentation & Evaluation Metrics
├── classical/               # Classical ML Trainers (SVM, Random Forest)
├── deep/                    # Deep Baseline Trainers (ResNet-18, EfficientNet-B0, DINOv2 Probe)
├── proposed/                # Proposed Architectures (SenesFuseNet & Attention-MIL)
├── run_all.py               # One-click Master Reproduction Script
├── generate_paper_figures.py # Paper Figures Generation Script (Figures 1-9)
├── generate_cellpose_qc_figure.py # Cellpose QC Figure Generation Script (Figure 11)
├── requirements.txt         # Python Package Dependencies
└── README.md                # Project Overview & Usage Instructions
```

---

## Quick Start & Installation

### 1. Environment Setup
```bash
git clone https://github.com/YourUsername/Chondrocyte-Senescence-ML.git
cd Chondrocyte-Senescence-ML
pip install -r requirements.txt
```

### 2. Running Full Benchmark
To run the end-to-end training and cross-validation pipeline across all 8 models:
```bash
python run_all.py
```

### 3. Generating Publication Figures
To regenerate all 11 high-resolution figures:
```bash
python generate_paper_figures.py
python generate_cellpose_qc_figure.py
```

---

## Citation & DOI
If you use this code or dataset in your research, please cite our manuscript and the archived Zenodo repository (DOI: `10.5281/zenodo.21769812`).
