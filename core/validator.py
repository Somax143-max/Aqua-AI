"""
AquaProtect-AI: Input Validation & Structured Defensive Verifier
Provides robust verification for uploaded sonar waterfalls, navigational streams,
and model checkpoints to ensure uninterrupted, crash-proof autonomous operations.
"""

import os
import cv2
import numpy as np
import torch
from typing import Tuple, Dict, Any, Optional


class SonarInputValidator:
    """
    Validates hydroacoustic inputs, telemetry streams, and weights against physical standards.
    """

    @staticmethod
    def validate_sonar_image(image: Any) -> Tuple[bool, str, Optional[np.ndarray]]:
        """
        Validates an incoming sonar image/waterfall strip.
        Ensures proper 2D array structure, valid dimensions, and valid bit-depth.
        """
        if image is None:
            return False, "Image object is None.", None

        if not isinstance(image, np.ndarray):
            return False, f"Expected numpy.ndarray, got {type(image).__name__}", None

        if image.size == 0:
            return False, "Input image has 0 pixels.", None

        # Dimensions check
        if len(image.shape) == 2:
            h, w = image.shape
            channels = 1
        elif len(image.shape) == 3:
            h, w, channels = image.shape
            if channels not in [1, 3, 4]:
                return False, f"Unsupported channel count: {channels}. Must be 1, 3, or 4.", None
        else:
            return False, f"Invalid array dimensions: {image.shape}. Must be 2D or 3D.", None

        if h < 24 or w < 24:
            return False, f"Image dimensions ({h}x{w}) too small. Minimum resolution is 24x24 px.", None

        if h > 65536 or w > 16384:
            return False, f"Image dimensions ({h}x{w}) exceed maximum hydrographic buffer (65536x16384 px).", None

        # Convert to single-channel uint8
        try:
            if len(image.shape) == 3:
                if channels == 4:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
                else:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            if gray.dtype != np.uint8:
                if np.max(gray) <= 1.0:
                    gray = (gray * 255.0).astype(np.uint8)
                else:
                    gray = np.clip(gray, 0, 255).astype(np.uint8)

            # 1. Distinguish between Natural Outdoor Camera Photos vs Hydroacoustic Sonar Displays
            # Acoustic sonar data is either pure grayscale or a 1D pseudocolor map (Amber, Bronze, Copper, Sepia, Cyan).
            # Natural camera photographs contain multi-spectral color conflicts (vibrant foliage and daylight sky).
            if len(image.shape) == 3 and channels == 3:
                b, g, r = cv2.split(image)
                
                # Check for vibrant green foliage (trees, grass, leaves in optical photos)
                green_foliage = (g > 60) & (g.astype(np.int16) - r.astype(np.int16) > 25) & (g.astype(np.int16) - b.astype(np.int16) > 15)
                gf_pct = float(np.mean(green_foliage) * 100.0)
                
                # Check for daylight blue sky
                blue_sky = (b > 110) & (b.astype(np.int16) - r.astype(np.int16) > 25)
                bs_pct = float(np.mean(blue_sky) * 100.0)

                # Natural outdoor camera photograph rejection:
                # Triggers when daylight sky is present alongside green foliage, or extreme foliage dominates.
                if (gf_pct > 8.0 and bs_pct > 15.0) or (gf_pct > 20.0) or (bs_pct > 35.0):
                    return False, f"Domain Reject (OOD): Detected natural outdoor camera photograph (Foliage: {gf_pct:.1f}%, Sky: {bs_pct:.1f}%). System requires acoustic sonar data.", None

            # 2. Acoustic Spatial Frequency / Stationarity Check (guards against grayscale photos)
            if h >= 128 and w >= 128:
                col_var = np.var(np.mean(gray, axis=0))
                row_var = np.var(np.mean(gray, axis=1))
                var_ratio = col_var / (row_var + 1e-6)
                inv_ratio = row_var / (col_var + 1e-6)
                
                # Natural isotropic scenes have low variance in both axes, unlike directional sonar beams
                if var_ratio < 2.0 and inv_ratio < 2.0 and col_var < 50.0 and row_var < 50.0:
                    return False, f"Domain Reject (OOD): Isotropic spatial variance too low (ColVar: {col_var:.1f}, RowVar: {row_var:.1f}). Lacks hydroacoustic across-track profile.", None

            # 2. Reject excessively smooth synthetic graphics
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if lap_var < 15.0:
                return False, f"Domain Reject (OOD): Acoustic speckle variance too low ({lap_var:.1f}). Missing characteristic sonar reverberation.", None
                
            # 3. Sonar specific intensity check
            mean_intensity = np.mean(gray)
            if mean_intensity < 5.0 or mean_intensity > 245.0:
                return False, f"Domain Reject (OOD): Extreme exposure ({mean_intensity:.1f}). Not a valid sonar dynamic range.", None

            return True, "Valid sonar imagery", gray
        except Exception as exc:
            return False, f"Failed converting image to grayscale uint8: {str(exc)}", None

    @staticmethod
    def validate_nav_telemetry(nav: Optional[Dict[str, Any]]) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validates vessel and towfish navigation telemetry.
        Enforces geographic coordinate bounds and physical depth/altitude constraints.
        """
        defaults = {
            "latitude": 13.0827,
            "longitude": 80.3705,
            "heading": 90.0,
            "altitude": 12.0,
            "depth": 35.0,
            "speed_mps": 1.5,
            "ping_rate_hz": 10.0,
            "gps_accuracy_m": 1.2,
            "roll": 0.0,
            "pitch": 0.0
        }

        if not nav or not isinstance(nav, dict):
            return False, "Telemetry record is missing or not a dictionary. Using standard coastal defaults.", defaults

        cleaned = defaults.copy()
        errors = []

        # Validate Latitude
        if "latitude" in nav:
            try:
                lat = float(nav["latitude"])
                if -90.0 <= lat <= 90.0:
                    cleaned["latitude"] = lat
                else:
                    errors.append(f"Latitude {lat} out of range [-90, 90].")
            except (ValueError, TypeError):
                errors.append("Latitude value is non-numeric.")

        # Validate Longitude
        if "longitude" in nav:
            try:
                lon = float(nav["longitude"])
                if -180.0 <= lon <= 180.0:
                    cleaned["longitude"] = lon
                else:
                    errors.append(f"Longitude {lon} out of range [-180, 180].")
            except (ValueError, TypeError):
                errors.append("Longitude value is non-numeric.")

        # Validate Heading
        if "heading" in nav:
            try:
                hdg = float(nav["heading"]) % 360.0
                cleaned["heading"] = hdg
            except (ValueError, TypeError):
                pass

        # Validate Altitude and Depth
        if "altitude" in nav:
            try:
                alt = max(0.5, float(nav["altitude"]))
                cleaned["altitude"] = alt
            except (ValueError, TypeError):
                pass

        if "depth" in nav:
            try:
                dep = max(1.0, float(nav["depth"]))
                cleaned["depth"] = dep
            except (ValueError, TypeError):
                pass

        # Ensure towfish is not deeper than seafloor
        if cleaned["altitude"] >= cleaned["depth"]:
            cleaned["depth"] = cleaned["altitude"] + 15.0

        is_valid = len(errors) == 0
        msg = "Valid navigation telemetry" if is_valid else "; ".join(errors)
        return is_valid, msg, cleaned

    @staticmethod
    def validate_model_checkpoint(filepath: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validates neural network weights checkpoint file integrity.
        """
        if not os.path.exists(filepath):
            return False, f"File not found: {filepath}", {}

        if os.path.getsize(filepath) < 1024:
            return False, f"File size too small ({os.path.getsize(filepath)} bytes). Likely truncated.", {}

        try:
            ckpt = torch.load(filepath, map_location="cpu")
            if not isinstance(ckpt, dict):
                return False, f"Expected dict in checkpoint, got {type(ckpt).__name__}", {}

            if "model_state_dict" not in ckpt:
                return False, "Missing 'model_state_dict' key in checkpoint dictionary.", {}

            state_dict = ckpt["model_state_dict"]
            num_tensors = len(state_dict)
            return True, f"Valid checkpoint containing {num_tensors} weight tensors", ckpt
        except Exception as exc:
            return False, f"Corrupted or unreadable checkpoint: {str(exc)}", {}
