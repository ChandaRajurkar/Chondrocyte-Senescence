"""
Tier 2 — DINOv2 frozen backbone + linear probe.

Extracts features using a frozen pre-trained DINOv2 (ViT-Small) backbone
and trains a simple linear classifier head on top.
This model is highly robust, trains extremely fast, and has low risk of overfitting.
"""
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from senescence_classifier.common import config as cfg
from senescence_classifier.deep.train_loop import run_cv_deep

logger = logging.getLogger("dinov2")


class DINOv2LinearProbe(nn.Module):
    """Frozen DINOv2 backbone with a trainable linear classification head."""

    def __init__(self, model_name: str = "facebook/dinov2-small", num_classes: int = 2):
        super().__init__()
        logger.info(f"Loading pre-trained DINOv2 backbone: {model_name}")
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # Freeze the backbone completely
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Linear classifier on top of the CLS token embedding
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.backbone.config.hidden_size, num_classes)
        )

    def forward(self, x):
        # Pre-trained DINOv2 expects images normalised to ImageNet stats, size 224x224
        # We perform backbone forward in no_grad context for safety
        with torch.no_grad():
            outputs = self.backbone(x)
            # DINOv2 outputs class token in pooler_output (or last_hidden_state[:, 0])
            cls_token = outputs.pooler_output
            
        return self.classifier(cls_token)


def make_dinov2_probe():
    return DINOv2LinearProbe(model_name="facebook/dinov2-small", num_classes=2)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    logger.info("Training DINOv2 Linear Probe (frozen backbone)")
    # Since DINOv2 is frozen, we can use a higher learning rate for the linear head
    run_cv_deep(make_dinov2_probe, "DINOv2_Probe", cfg.DEEP_OUTPUT, lr=1e-3)


if __name__ == "__main__":
    main()
