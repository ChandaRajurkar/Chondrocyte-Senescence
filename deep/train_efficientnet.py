"""
Tier 2 — EfficientNet-B0 transfer learning.

ImageNet-pretrained EfficientNet-B0 with frozen early layers and a custom binary head.
"""
import logging
import sys
from pathlib import Path

import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from senescence_classifier.common import config as cfg
from senescence_classifier.deep.train_loop import run_cv_deep

logger = logging.getLogger("efficientnet")


def make_efficientnet():
    """Create an EfficientNet-B0 with ImageNet weights and a binary classification head."""
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Freeze all layers except features.7 and classifier
    for name, param in model.named_parameters():
        if "features.7" not in name and "classifier" not in name:
            param.requires_grad = False

    # Replace final classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 2),
    )
    return model


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    logger.info("Training EfficientNet-B0 (transfer learning)")
    run_cv_deep(make_efficientnet, "EfficientNetB0", cfg.DEEP_OUTPUT)


if __name__ == "__main__":
    main()
