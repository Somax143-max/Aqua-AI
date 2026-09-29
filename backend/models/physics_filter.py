"""
AquaProtect-AI: Physics-Informed Acoustic Signature Validation (AHSA)
Validates neural proposals against fundamental laws of hydroacoustic propagation:
- Specular Highlight Intensity vs Ambient Seabed Reverberation
- Acoustic Shadow Geometry & Directional Sweep Alignment
- Highlight-to-Shadow Contrast Ratio (HSCR)
- Scientifically Grounded Multi-Factor Confidence Calibration:
  Calibrated Confidence = w_ai * P_neural + w_phys * S_physics + w_qual * Q_image - lambda * Var(epistemic)
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional


class AcousticPhysicsFilter:
    """
    Validates hydroacoustic signatures against side-scan sonar physical geometry.
    Suppresses natural geological false alarms (sand ripples, rocky outcrops, coral heads).
    """

    def __init__(
        self,
        min_contrast_ratio: float = 0.35,
        min_shadow_drop_ratio: float = 0.20,
        max_aspect_ratio_rock: float = 1.3
    ):
        self.min_contrast_ratio = min_contrast_ratio
        self.min_shadow_drop_ratio = min_shadow_drop_ratio
        self.max_aspect_ratio_rock = max_aspect_ratio_rock

    def analyze_acoustic_signature(
        self,
        sonar_crop: np.ndarray,
        is_starboard: bool
    ) -> Dict[str, Any]:
        """
        Analyzes a localized sonar patch containing a candidate target.
        In SSS:
        - Starboard: Acoustic ping travels Left -> Right.
          Highlight appears on the LEFT, Shadow stretches to the RIGHT.
        - Port: Acoustic ping travels Right -> Left.
          Highlight appears on the RIGHT, Shadow stretches to the LEFT.
        """
        if len(sonar_crop.shape) == 3:
            gray = cv2.cvtColor(sonar_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = sonar_crop.copy()

        h, w = gray.shape
        if h < 5 or w < 5:
            return {
                "verified": False,
                "physics_consistency_score": 0.20,
                "image_quality_score": 0.30,
                "confidence_adjustment": -0.30,
                "reason": "Crop too small for acoustic verification",
                "status": "INSUFFICIENT_DATA"
            }

        # Divide patch along across-track direction into Insonified Face vs Shadow Zone
        mid_x = w // 2
        
        if is_starboard:
            # Highlight should be on the inner side (left), shadow on outer (right)
            highlight_zone = gray[:, :mid_x]
            shadow_zone = gray[:, mid_x:]
        else:
            # Port: Highlight on inner side (right), shadow on outer (left)
            highlight_zone = gray[:, mid_x:]
            shadow_zone = gray[:, :mid_x]

        i_high_mean = float(np.percentile(highlight_zone, 90)) # Specular return
        i_shad_mean = float(np.percentile(shadow_zone, 10))    # Shadow return
        i_bg_mean = float(np.median(gray))                     # Ambient seabed backscatter

        # Contrast Ratio: (I_high - I_shad) / (I_high + I_shad + eps)
        contrast_ratio = (i_high_mean - i_shad_mean) / (i_high_mean + i_shad_mean + 1e-5)
        
        # Shadow Drop: relative darkness of shadow compared to seabed background
        shadow_drop = (i_bg_mean - i_shad_mean) / (i_bg_mean + 1e-5)
        
        # Highlight Boost: specular reflectance compared to seabed
        highlight_boost = (i_high_mean - i_bg_mean) / (i_bg_mean + 1e-5)

        # Physics criteria validation
        has_highlight = highlight_boost > 0.15
        has_acoustic_shadow = (shadow_drop > 0.20) or (i_shad_mean < 45)
        good_contrast = contrast_ratio >= self.min_contrast_ratio

        # Physics consistency score S_physics in [0, 1]
        phys_score = float(np.clip(
            0.4 * min(contrast_ratio / 0.5, 1.0) +
            0.3 * min(max(shadow_drop, 0.0) / 0.4, 1.0) +
            0.3 * min(max(highlight_boost, 0.0) / 0.5, 1.0),
            0.05, 1.0
        ))

        # Local Image Quality Score Q_image in [0, 1] based on SNR and gradient sharpness
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        image_quality = float(np.clip(min(laplacian_var / 400.0, 1.0) * 0.7 + 0.3, 0.3, 1.0))

        # Determine acoustic validity
        is_physically_consistent = has_acoustic_shadow and good_contrast

        if is_physically_consistent:
            status = "VERIFIED_ARTIFICIAL_ANOMALY"
            confidence_adjustment = 0.08 * min(contrast_ratio, 1.0)
        elif has_highlight and not has_acoustic_shadow:
            status = "SEABED_TEXTURE_OR_RIPPLE"
            confidence_adjustment = -0.20
            phys_score *= 0.4
        elif not has_highlight and has_acoustic_shadow:
            status = "NATURAL_DEPRESSION_OR_POCKMARK"
            confidence_adjustment = -0.15
            phys_score *= 0.5
        else:
            status = "NATURAL_REVERBERATION_FALSE_ALARM"
            confidence_adjustment = -0.30
            phys_score *= 0.2

        return {
            "verified": is_physically_consistent,
            "status": status,
            "contrast_ratio": round(contrast_ratio, 3),
            "shadow_drop": round(shadow_drop, 3),
            "highlight_boost": round(highlight_boost, 3),
            "physics_consistency_score": round(phys_score, 3),
            "image_quality_score": round(image_quality, 3),
            "confidence_adjustment": round(confidence_adjustment, 3),
            "highlight_intensity": round(i_high_mean, 1),
            "shadow_intensity": round(i_shad_mean, 1),
            "background_intensity": round(i_bg_mean, 1)
        }

    def calibrate_confidence(
        self,
        base_confidence: float,
        acoustic_analysis: Dict[str, Any],
        epistemic_variance: float = 0.008
    ) -> float:
        """
        Scientifically calibrated confidence fusion:
        Fuses neural probability (60%), physical highlight-shadow consistency (25%),
        and image quality (15%), penalized by epistemic uncertainty variance.
        Guarantees realistic, non-inflated confidence values bounded in [0.10, 0.95].
        """
        phys_score = acoustic_analysis.get("physics_consistency_score", 0.5)
        qual_score = acoustic_analysis.get("image_quality_score", 0.7)
        
        # Multi-factor fusion
        w_ai = 0.60
        w_phys = 0.25
        w_qual = 0.15
        lambda_unc = 1.5

        fused = (
            w_ai * base_confidence +
            w_phys * phys_score +
            w_qual * qual_score -
            lambda_unc * max(0.0, epistemic_variance)
        )

        # If physically inconsistent, apply penalty floor
        if not acoustic_analysis.get("verified", True):
            fused = min(fused, 0.42)

        # Realistic confidence bounds (never fake 99.9% on real ocean pings)
        return float(np.clip(round(fused, 3), 0.10, 0.95))
