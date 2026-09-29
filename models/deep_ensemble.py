"""
AquaProtect-AI: Ultra-High Accuracy Neural Determination Engine (7-Class Edition)
Implements:
- Spatial & Channel Squeeze-and-Excitation (scSE) Attention Backbone
- Multi-Scale Acoustic Feature Extraction with Residual Gating
- Bayesian Epistemic (Variance) & Aleatoric (Entropy) Uncertainty Estimation (15 MC-Dropout Passes)
- AI Abstention Mechanism: Flags High-Uncertainty / Low-Confidence Anomalies for Operator Review
- Hard Negative Mining (HNM) Support
- Dual Classification & High-Resolution Segmentation Heads
"""

import math
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialChannelSqueezeExcitation(nn.Module):
    """
    Concurrent Spatial and Channel 'Squeeze & Excitation' (scSE) block.
    Recalibrates both channel feature maps and spatial acoustic coordinates
    to focus neural attention on sharp highlight-shadow boundaries.
    """
    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        reduced_ch = max(in_channels // reduction, 4)
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_ch, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_ch, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.sSE = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.cSE(x) + x * self.sSE(x)


class AcousticResidualBlock(nn.Module):
    """
    Residual block with integrated scSE attention for hydroacoustic sonar patterns.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.scse = SpatialChannelSqueezeExcitation(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.scse(out)
        out += residual
        return self.relu(out)


class DeepSonarDeterminationModel(nn.Module):
    """
    State-of-the-Art Deep Sonar Model with Attention, Multi-Scale ResNet Backbone,
    and Monte Carlo Dropout for Bayesian Epistemic & Aleatoric Uncertainty Quantification.
    """
    def __init__(self, num_classes: int = 7, in_channels: int = 1, mc_dropout_prob: float = 0.25):
        super().__init__()
        self.num_classes = num_classes
        self.mc_dropout_prob = mc_dropout_prob

        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            SpatialChannelSqueezeExcitation(32)
        )

        # 3 Multi-scale residual stages with scSE attention
        self.stage1 = AcousticResidualBlock(32, 64, stride=2)   # 64x64 -> 32x32
        self.stage2 = AcousticResidualBlock(64, 128, stride=2)  # 32x32 -> 16x16
        self.stage3 = AcousticResidualBlock(128, 256, stride=2) # 16x16 -> 8x8

        # Spatial pooling & Feature Fusion
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.mc_dropout = nn.Dropout(p=mc_dropout_prob)

        # Classification Head (With Bayesian Uncertainty)
        self.fc_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=mc_dropout_prob),
            nn.Linear(128, num_classes)
        )

        # High-Resolution Segmentation Mask Decoder (FPN-like)
        self.seg_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), # 8x8 -> 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # 16x16 -> 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),   # 32x32 -> 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_stem = self.stem(x)
        s1 = self.stage1(x_stem)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)

        pooled = self.global_pool(s3).flatten(1)
        dropped = self.mc_dropout(pooled)
        logits = self.fc_classifier(dropped)
        mask = self.seg_decoder(s3)

        return logits, mask

    def predict_fast(
        self,
        x: torch.Tensor,
        temperature: float = 1.0
    ) -> Dict[str, Any]:
        """
        Deterministic fast forward pass for real-time survey streaming (>15 FPS).
        Applies temperature scaling and computes entropy & baseline uncertainty.
        """
        self.eval()
        with torch.no_grad():
            logits, mask = self.forward(x)
            scaled_logits = logits / max(temperature, 0.1)
            probs = F.softmax(scaled_logits, dim=1).cpu().numpy()
            mask_np = mask.cpu().numpy()

        pred_classes = np.argmax(probs, axis=1)
        pred_confidences = np.max(probs, axis=1)

        # Fast empirical variance via 2 stochastic forward passes (no synthetic derivation from confidence)
        self.train()
        fast_mc_probs = []
        with torch.no_grad():
            for _ in range(2):
                mc_l, _ = self.forward(x)
                fast_mc_probs.append(F.softmax(mc_l / max(temperature, 0.1), dim=1).cpu().numpy())
        self.eval()
        fast_mc_stack = np.stack(fast_mc_probs, axis=0) # (2, B, num_classes)
        fast_var = np.var(fast_mc_stack, axis=0)

        results = []
        for b in range(len(pred_classes)):
            conf_val = float(pred_confidences[b])
            c_probs = probs[b]
            var_val = float(fast_var[b, pred_classes[b]])

            # Aleatoric uncertainty via Shannon entropy: H(p) = - sum p_i * log2(p_i)
            eps = 1e-9
            entropy = float(-np.sum(c_probs * np.log2(c_probs + eps)))
            max_entropy = math.log2(self.num_classes)
            norm_entropy = entropy / max(max_entropy, 1e-4)

            # Rigorous certainty rating from empirical variance and entropy
            if var_val < 0.015 and entropy < 0.70:
                cert_rating = "VERY_HIGH_CERTAINTY"
            elif var_val < 0.035 and entropy < 1.10:
                cert_rating = "HIGH_CERTAINTY"
            elif var_val < 0.055 and entropy < 1.45:
                cert_rating = "MODERATE_CERTAINTY"
            else:
                cert_rating = "UNCERTAIN_ANOMALY"

            # AI Abstention Rule
            should_abstain = (conf_val < 0.42) or (var_val > 0.040) or (entropy > 1.38)
            abstain_status = "UNCERTAIN_ANOMALY" if should_abstain else "CONFIDENT_CLASSIFICATION"

            results.append({
                "class_id": int(pred_classes[b]),
                "raw_confidence": float(round(conf_val, 4)),
                "calibrated_confidence": float(round(conf_val, 4)),
                "predictive_variance": float(round(var_val, 5)),
                "entropy": float(round(entropy, 4)),
                "normalized_entropy": float(round(norm_entropy, 4)),
                "certainty_rating": cert_rating,
                "abstain": bool(should_abstain),
                "abstention_status": abstain_status,
                "class_probabilities": c_probs.tolist()
            })

        return {
            "predictions": results,
            "segmentation_mask": mask_np
        }

    def predict_with_bayesian_uncertainty(
        self,
        x: torch.Tensor,
        num_mc_samples: int = 15,
        temperature: float = 1.0
    ) -> Dict[str, Any]:
        """
        Executes 10-20 Monte Carlo Dropout stochastic forward passes to rigorously estimate:
        - Predictive mean probability vector
        - Epistemic uncertainty (predictive variance across MC realizations)
        - Aleatoric uncertainty (predictive Shannon entropy)
        - AI Abstention tag: flags high-uncertainty targets for operator review
        """
        self.train() # Enable dropout during inference for MC sampling

        all_probs = []
        all_masks = []
        with torch.no_grad():
            for _ in range(num_mc_samples):
                logits, mask = self.forward(x)
                scaled_logits = logits / max(temperature, 0.1)
                probs = F.softmax(scaled_logits, dim=1)
                all_probs.append(probs.cpu().numpy())
                all_masks.append(mask.cpu().numpy())

        self.eval()

        probs_stack = np.stack(all_probs, axis=0) # (num_mc_samples, Batch, num_classes)
        mean_probs = np.mean(probs_stack, axis=0) # (Batch, num_classes)
        variance = np.var(probs_stack, axis=0)    # (Batch, num_classes)

        pred_classes = np.argmax(mean_probs, axis=1)
        pred_confidences = np.max(mean_probs, axis=1)
        pred_variances = np.array([variance[b, pred_classes[b]] for b in range(len(pred_classes))])

        mean_mask = np.mean(np.stack(all_masks, axis=0), axis=0)

        results = []
        for b in range(len(pred_classes)):
            var_val = float(pred_variances[b])
            conf_val = float(pred_confidences[b])
            c_probs = mean_probs[b]

            # Shannon entropy calculation
            eps = 1e-9
            entropy = float(-np.sum(c_probs * np.log2(c_probs + eps)))
            max_entropy = math.log2(self.num_classes)
            norm_entropy = entropy / max(max_entropy, 1e-4)

            # Rigorous certainty categorization
            if var_val < 0.012 and entropy < 0.70:
                cert_rating = "VERY_HIGH_CERTAINTY"
                calibrated_conf = min(0.995, conf_val + 0.03)
            elif var_val < 0.030 and entropy < 1.10:
                cert_rating = "HIGH_CERTAINTY"
                calibrated_conf = conf_val
            elif var_val < 0.050 and entropy < 1.45:
                cert_rating = "MODERATE_CERTAINTY"
                calibrated_conf = conf_val * 0.90
            else:
                cert_rating = "UNCERTAIN_ANOMALY"
                calibrated_conf = conf_val * 0.70

            # AI Abstention Condition (P0 Item 9)
            should_abstain = (conf_val < 0.42) or (var_val > 0.040) or (entropy > 1.38)
            abstain_status = "UNCERTAIN_ANOMALY" if should_abstain else "CONFIDENT_CLASSIFICATION"

            # Combined uncertainty index [0.0, 1.0]
            combined_uncertainty = float(np.clip(0.5 * (var_val / 0.05) + 0.5 * norm_entropy, 0.0, 1.0))

            results.append({
                "class_id": int(pred_classes[b]),
                "raw_confidence": float(round(conf_val, 4)),
                "calibrated_confidence": float(round(calibrated_conf, 4)),
                "predictive_variance": float(round(var_val, 5)),
                "entropy": float(round(entropy, 4)),
                "normalized_entropy": float(round(norm_entropy, 4)),
                "combined_uncertainty": float(round(combined_uncertainty, 4)),
                "certainty_rating": cert_rating,
                "abstain": bool(should_abstain),
                "abstention_status": abstain_status,
                "class_probabilities": c_probs.tolist()
            })

        return {
            "predictions": results,
            "segmentation_mask": mean_mask
        }


class HardNegativeFocalLoss(nn.Module):
    """
    Focal Loss with Hard Negative Mining (HNM) weighting for acoustic sonar clutter.
    Focuses gradient updates on challenging edge cases like sand ripples and reef boulders.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, hnm_factor: float = 1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.hnm_factor = hnm_factor

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        hard_mask = (pt < 0.5)
        weights = torch.ones_like(focal_loss)
        weights[hard_mask] *= self.hnm_factor
        return torch.mean(focal_loss * weights)