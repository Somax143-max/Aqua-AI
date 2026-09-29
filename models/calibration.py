"""
AquaProtect-AI: Scientific Confidence Calibration & Temperature Scaling Engine
Implements:
- Post-hoc Temperature Scaling (Guo et al., 2017)
- Expected Calibration Error (ECE) & Maximum Calibration Error (MCE)
- Multi-class Brier Score calculation
- Empirical Reliability Diagram generation
- Calibrated certainty classification (Confidence vs Uncertainty verification)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Tuple, List, Optional

class TemperatureScaler(nn.Module):
    """
    Optimizes a single temperature parameter T > 0 on validation set logits
    to calibrate neural network probability estimates without altering classification accuracy.
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.2)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = torch.clamp(self.temperature, min=0.1, max=10.0)
        return logits / temperature

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor, lr: float = 0.01, max_iter: int = 100) -> float:
        """
        Fits temperature T by minimizing cross-entropy (Negative Log Likelihood) on validation data.
        """
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.forward(val_logits)
            loss = nll_criterion(scaled_logits, val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        return float(self.temperature.item())


class ConfidenceCalibrator:
    """
    Computes rigorous hydroacoustic calibration metrics:
    - ECE (Expected Calibration Error)
    - Brier Score
    - Reliability diagrams
    """
    @staticmethod
    def calculate_ece(
        probs: np.ndarray,
        labels: np.ndarray,
        num_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Calculates Expected Calibration Error (ECE) and bin data for reliability diagrams.
        """
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels).astype(float)

        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        mce = 0.0
        reliability_diagram = []

        total_samples = len(labels)
        if total_samples == 0:
            return {"ece": 0.0, "mce": 0.0, "diagram": []}

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = float(np.mean(in_bin))

            if np.sum(in_bin) > 0:
                accuracy_in_bin = float(np.mean(accuracies[in_bin]))
                avg_confidence_in_bin = float(np.mean(confidences[in_bin]))
                abs_diff = abs(avg_confidence_in_bin - accuracy_in_bin)
                ece += abs_diff * prop_in_bin
                mce = max(mce, abs_diff)

                reliability_diagram.append({
                    "bin_range": [round(float(bin_lower), 2), round(float(bin_upper), 2)],
                    "confidence": round(avg_confidence_in_bin, 4),
                    "accuracy": round(accuracy_in_bin, 4),
                    "count": int(np.sum(in_bin))
                })
            else:
                reliability_diagram.append({
                    "bin_range": [round(float(bin_lower), 2), round(float(bin_upper), 2)],
                    "confidence": round((bin_lower + bin_upper) / 2.0, 4),
                    "accuracy": 0.0,
                    "count": 0
                })

        return {
            "ece": round(float(ece) * 100.0, 2),       # In percentage
            "mce": round(float(mce) * 100.0, 2),
            "diagram": reliability_diagram,
            "rating": "GOOD" if ece < 0.05 else ("MODERATE" if ece < 0.10 else "POOR")
        }

    @staticmethod
    def calculate_brier_score(probs: np.ndarray, labels: np.ndarray, num_classes: int = 7) -> float:
        """
        Calculates multi-class Brier score (lower is better; 0 = perfect probability assignment).
        """
        N = len(labels)
        if N == 0:
            return 0.0
        one_hot = np.zeros((N, num_classes), dtype=float)
        for i, lbl in enumerate(labels):
            if 0 <= lbl < num_classes:
                one_hot[i, lbl] = 1.0
        brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
        return round(float(brier), 4)