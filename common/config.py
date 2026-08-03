"""
Central configuration for the senescence classification pipeline.

All paths, hyperparameters, feature definitions, and seeds are defined here
so every script in the project shares identical settings.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths  (relative to the workspace root: Work1_Comp/)
# ---------------------------------------------------------------------------
# Adjust WORKSPACE_ROOT if running from a different working directory.
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent   # …/Work1_Comp

CONTROL_DIR   = WORKSPACE_ROOT / "Dataset" / "control"
SENESCENT_DIR = WORKSPACE_ROOT / "Dataset" / "senescence"
OUTPUT_DIR    = WORKSPACE_ROOT / "senescence_classifier" / "Outputs"

# Sub-directories created automatically
MORPHOLOGY_OUTPUT = OUTPUT_DIR / "morphology"
CLASSICAL_OUTPUT  = OUTPUT_DIR / "classical"
DEEP_OUTPUT       = OUTPUT_DIR / "deep"
PROPOSED_OUTPUT   = OUTPUT_DIR / "proposed"
FIGURES_OUTPUT    = OUTPUT_DIR / "figures"

for _d in [MORPHOLOGY_OUTPUT, CLASSICAL_OUTPUT, DEEP_OUTPUT,
           PROPOSED_OUTPUT, FIGURES_OUTPUT]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "Area", "Perimeter", "Circularity", "Aspect_Ratio", "Extent",
    "Solidity", "Equivalent_Diameter", "Eccentricity",
    "Texture_Contrast", "Texture_Dissimilarity", "Texture_Homogeneity",
    "Texture_Energy", "Texture_Correlation", "Texture_ASM",
]
DROPPED_FEATURE = "Nuclear_Envelope_Irregularity"

# Aggregation statistics for per-cell → per-image morphology vectors
AGG_STATS = ["mean", "std", "median", "min", "max"]
# Per-image vector length: len(FEATURE_COLUMNS) * len(AGG_STATS) + 1 (log cell count)
MORPH_VECTOR_DIM = len(FEATURE_COLUMNS) * len(AGG_STATS) + 1  # 71

# ---------------------------------------------------------------------------
# Image / model constants
# ---------------------------------------------------------------------------
IMAGE_SIZE    = 224
BATCH_SIZE    = 16
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# ---------------------------------------------------------------------------
# Cross-validation & reproducibility
# ---------------------------------------------------------------------------
SEED      = 42
NUM_FOLDS = 5

# ---------------------------------------------------------------------------
# Training hyperparameters (deep models)
# ---------------------------------------------------------------------------
NUM_EPOCHS   = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 0.05
PATIENCE      = 10        # early stopping patience
PLOT_DPI      = 600

# ---------------------------------------------------------------------------
# ImageNet normalisation (used for ResNet / EfficientNet / DINOv2)
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
