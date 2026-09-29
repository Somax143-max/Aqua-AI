"""
AquaProtect-AI: Hydroacoustic Robustness Laboratory
Systematically evaluates model degradation across 9 real-world maritime acoustic corruptions:
1. Baseline Normal Acoustic Quality
2. Heavy Rayleigh Acoustic Speckle (Scale = 45.0)
3. Low Acoustic Signal-to-Noise Ratio (Gaussian noise sigma = 35.0)
4. High-Frequency Acoustic Attenuation & Turbid Blur (5x5 Gaussian PSF)
5. Towfish Motion Dynamics Distortion (Yaw shear & yaw oscillations)
6. Vehicle Roll & Pitch Hydrodynamic Tilt (grazing angle deviation)
7. Acoustic Ping Dropouts (15% random blanked transducer pings)
8. Deep Subsea Resolution Downgrade (50% downsampled long-range swath)
9. Severe Grazing Angle Shadow Clipping
Outputs quantitative degradation tables and resilience scores.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import json
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, List

from core.config import NUM_CLASSES, CHECKPOINT_PATH
from models.deep_ensemble import DeepSonarDeterminationModel
from models.trainer import RealisticSonarDataset

class HydroacousticRobustnessLab:
    """
    Executes systematic acoustic perturbation stress-testing on the deep sonar model.
    """

    def __init__(self, weights_path: str = CHECKPOINT_PATH, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DeepSonarDeterminationModel(num_classes=NUM_CLASSES).to(self.device)
        if os.path.exists(weights_path):
            ckpt = torch.load(weights_path, map_location=self.device)
            state = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state)
        self.model.eval()

    def _apply_corruption(self, img_np: np.ndarray, corruption_name: str) -> np.ndarray:
        """Applies a specific physical acoustic corruption to an image patch [0.0, 1.0]."""
        u8 = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)

        if corruption_name == "baseline":
            return img_np

        elif corruption_name == "heavy_speckle":
            noise = np.random.rayleigh(scale=35.0, size=u8.shape)
            corrupted = np.clip(u8.astype(float) * 0.7 + noise, 0, 255).astype(np.uint8)

        elif corruption_name == "low_snr":
            gauss = np.random.normal(0, 30, u8.shape)
            corrupted = np.clip(u8.astype(float) + gauss, 0, 255).astype(np.uint8)

        elif corruption_name == "acoustic_blur":
            corrupted = cv2.GaussianBlur(u8, (5, 5), 1.6)

        elif corruption_name == "motion_shear":
            # Affine shear representing towfish yaw oscillation
            M = np.float32([[1, 0.25, -4], [0, 1, 0]])
            corrupted = cv2.warpAffine(u8, M, (u8.shape[1], u8.shape[0]), borderMode=cv2.BORDER_REFLECT)

        elif corruption_name == "roll_pitch_tilt":
            # Gradient lighting change from vehicle roll
            grad = np.linspace(0.6, 1.3, u8.shape[1])
            corrupted = np.clip(u8.astype(float) * grad[np.newaxis, :], 0, 255).astype(np.uint8)

        elif corruption_name == "ping_dropouts":
            corrupted = u8.copy()
            drop_lines = np.random.choice(u8.shape[0], size=int(u8.shape[0] * 0.18), replace=False)
            corrupted[drop_lines, :] = 10 # dark ping dropouts

        elif corruption_name == "low_resolution":
            down = cv2.resize(u8, (32, 32), interpolation=cv2.INTER_AREA)
            corrupted = cv2.resize(down, (u8.shape[1], u8.shape[0]), interpolation=cv2.INTER_LINEAR)

        elif corruption_name == "rotation_skew":
            center = (u8.shape[1] // 2, u8.shape[0] // 2)
            M = cv2.getRotationMatrix2D(center, 40.0, 1.0)
            corrupted = cv2.warpAffine(u8, M, (u8.shape[1], u8.shape[0]), borderMode=cv2.BORDER_REFLECT)

        else:
            corrupted = u8

        return np.clip(corrupted.astype(np.float32) / 255.0, 0.0, 1.0)

    def run_stress_test(self, num_test_samples: int = 120) -> Dict[str, Any]:
        """
        Runs the test dataset through all 9 corruption conditions and computes F1 degradation.
        """
        test_ds = RealisticSonarDataset(num_samples=num_test_samples, split="test")

        corruptions = [
            ("baseline", "Baseline Acoustic Quality"),
            ("heavy_speckle", "Multiplicative Rayleigh Speckle"),
            ("low_snr", "Low SNR Acoustic Noise"),
            ("acoustic_blur", "High-Frequency Turbid Blur"),
            ("motion_shear", "Towfish Motion Dynamics Shear"),
            ("roll_pitch_tilt", "Hydrodynamic Roll/Pitch Deviation"),
            ("ping_dropouts", "Telemetry & Ping Data Dropouts"),
            ("low_resolution", "Downsampled Long-Range Swath"),
            ("rotation_skew", "Target 40° Angular Skew")
        ]

        results = []
        baseline_f1 = None

        for key, label in corruptions:
            correct = 0
            total = 0
            tps = np.zeros(NUM_CLASSES)
            fps = np.zeros(NUM_CLASSES)
            fns = np.zeros(NUM_CLASSES)

            for i in range(len(test_ds)):
                img, _, target, _ = test_ds[i]
                img_np = img.squeeze().numpy()
                corrupted_np = self._apply_corruption(img_np, key)

                tensor_in = torch.from_numpy(corrupted_np).unsqueeze(0).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    logits, _ = self.model(tensor_in)
                    pred = int(torch.argmax(logits, dim=1).item())

                if pred == target:
                    correct += 1
                    tps[target] += 1
                else:
                    fps[pred] += 1
                    fns[target] += 1
                total += 1

            acc = (correct / max(total, 1)) * 100.0
            prec_arr = tps / np.maximum(tps + fps, 1e-6)
            rec_arr = tps / np.maximum(tps + fns, 1e-6)
            f1_arr = 2.0 * (prec_arr * rec_arr) / np.maximum(prec_arr + rec_arr, 1e-6)
            macro_f1 = float(np.mean(f1_arr) * 100.0)

            if key == "baseline":
                baseline_f1 = max(macro_f1, 1.0)
                degradation_pct = 0.0
            else:
                effective_base = baseline_f1 if baseline_f1 is not None else 100.0
                degradation_pct = round(max(0.0, effective_base - macro_f1), 2)

            # Assign qualitative resilience
            if degradation_pct < 8.0:
                resilience = "EXCELLENT_ROBUSTNESS"
            elif degradation_pct < 18.0:
                resilience = "STRONG_RESILIENCE"
            else:
                resilience = "MODERATE_SENSITIVITY"

            results.append({
                "condition_key": key,
                "condition_label": label,
                "accuracy_pct": round(acc, 2),
                "macro_f1_pct": round(macro_f1, 2),
                "degradation_vs_baseline_pct": degradation_pct,
                "resilience_rating": resilience
            })

        avg_degradation = float(np.mean([r["degradation_vs_baseline_pct"] for r in results[1:]]))

        report = {
            "robustness_score_pct": round(max(0.0, 100.0 - avg_degradation), 1),
            "baseline_f1_pct": round(baseline_f1, 1),
            "average_degradation_pct": round(avg_degradation, 1),
            "total_stress_conditions": len(corruptions),
            "results_table": results
        }
        return report

if __name__ == "__main__":
    lab = HydroacousticRobustnessLab()
    report = lab.run_stress_test(num_test_samples=100)
    print(json.dumps(report, indent=2))