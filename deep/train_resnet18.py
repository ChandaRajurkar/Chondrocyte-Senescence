"""
Tier 2 — ResNet-18 transfer learning.

ImageNet-pretrained ResNet-18 with frozen early layers and a custom binary head.
"""
import logging
import sys
from pathlib import Path

import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from senescence_classifier.common import config as cfg
from senescence_classifier.deep.train_loop import run_cv_deep

logger = logging.getLogger("resnet18")


def make_resnet18():
    """Create a ResNet-18 with ImageNet weights and a binary classification head."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # Freeze all layers except layer4 and fc
    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False

    # Replace final FC
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 2),
    )
    return model


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    logger.info("Training ResNet-18 (transfer learning)")
    run_cv_deep(make_resnet18, "ResNet18", cfg.DEEP_OUTPUT)


if __name__ == "__main__":
    main()
