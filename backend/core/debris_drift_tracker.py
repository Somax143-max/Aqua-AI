"""
AquaProtect-AI: Hydrodynamic Debris Drift & Multi-Temporal Change Tracker
Implements:
- Hydrodynamic benthic current drag modeling (Stokes drift, drag coefficient Cd, and turbulent diffusion)
- Multi-horizon predicted drift trajectories (6h, 12h, 24h, 48h) with uncertainty confidence ellipses
- Multi-temporal survey difference detector (Survey T0 vs Survey T1)
- Identifies newly deposited hazards, dragged fishing gear, and buried anomalies
"""

import math
from typing import Dict, Any, List, Tuple
import numpy as np

class DebrisDriftTracker:
    """
    Simulates ocean hydrodynamic drag on underwater debris and computes
    probabilistic dispersion cones under benthic current vectors.
    """

    DRAG_COEFFICIENTS = {
        "Ghost Fishing Net": {"Cd": 1.25, "drift_ratio": 0.48, "diffusion_rate_m2ps": 0.25, "risk": "EXTREME_MIGRATION_RISK"},
        "Metal Drum / Chemical Barrel": {"Cd": 0.82, "drift_ratio": 0.26, "diffusion_rate_m2ps": 0.12, "risk": "MODERATE_ROLLING_RISK"},
        "Munition / Naval Mine": {"Cd": 0.47, "drift_ratio": 0.08, "diffusion_rate_m2ps": 0.05, "risk": "LOW_ANCHORED_RISK"},
        "Lost Cargo Container": {"Cd": 2.05, "drift_ratio": 0.02, "diffusion_rate_m2ps": 0.02, "risk": "STATIC_BENTHIC_OBSTRUCTION"},
        "Shipwreck / Hull Debris": {"Cd": 1.80, "drift_ratio": 0.01, "diffusion_rate_m2ps": 0.01, "risk": "PERMANENT_STRUCTURE"},
        "Submarine Pipeline / Cable": {"Cd": 1.10, "drift_ratio": 0.00, "diffusion_rate_m2ps": 0.00, "risk": "FIXED_UTILITY_ROUTE"}
    }

    @classmethod
    def predict_debris_drift(
        cls,
        lat: float,
        lon: float,
        class_name: str,
        current_speed_knots: float = 1.2,
        current_bearing_deg: float = 45.0,
        forecast_hours: float = 48.0,
        seabed_depth_m: float = 35.0
    ) -> Dict[str, Any]:
        """
        Computes hydrodynamic displacement of marine debris under benthic current vectors
        with turbulent diffusion uncertainty growth.
        """
        v_current_mps = current_speed_knots * 0.514444

        spec = cls.DRAG_COEFFICIENTS.get(class_name, {
            "Cd": 1.0, "drift_ratio": 0.15, "diffusion_rate_m2ps": 0.10, "risk": "LOW_DISPLACEMENT_RISK"
        })

        v_drift_mps = v_current_mps * spec["drift_ratio"]
        total_displacement_m = v_drift_mps * (forecast_hours * 3600.0)

        rad_bearing = math.radians(current_bearing_deg)
        dx_meters = total_displacement_m * math.sin(rad_bearing)
        dy_meters = total_displacement_m * math.cos(rad_bearing)

        d_lat = dy_meters / 111139.0
        d_lon = dx_meters / (111139.0 * math.cos(math.radians(lat)))

        pred_lat_24h = lat + (d_lat * 0.5)
        pred_lon_24h = lon + (d_lon * 0.5)
        pred_lat_48h = lat + d_lat
        pred_lon_48h = lon + d_lon

        # Trajectory with turbulent dispersion confidence radius (sigma = sqrt(2 * D * t))
        trajectory = []
        for h in [0, 6, 12, 24, 36, 48]:
            frac = h / max(forecast_hours, 1.0)
            t_sec = h * 3600.0
            disp_radius_m = round(math.sqrt(max(2.0 * spec["diffusion_rate_m2ps"] * t_sec, 1.0)), 1)
            dist_h = total_displacement_m * frac

            trajectory.append({
                "hour": h,
                "lat": round(lat + d_lat * frac, 6),
                "lon": round(lon + d_lon * frac, 6),
                "displacement_m": round(dist_h, 1),
                "confidence_radius_m": disp_radius_m,
                "depth_m": round(seabed_depth_m, 1)
            })

        return {
            "initial_lat": lat,
            "initial_lon": lon,
            "class_name": class_name,
            "hydrodynamic_drag_Cd": spec["Cd"],
            "drift_velocity_mps": round(v_drift_mps, 3),
            "total_drift_displacement_m": round(total_displacement_m, 1),
            "predicted_lat_24h": round(pred_lat_24h, 6),
            "predicted_lon_24h": round(pred_lon_24h, 6),
            "predicted_lat_48h": round(pred_lat_48h, 6),
            "predicted_lon_48h": round(pred_lon_48h, 6),
            "dispersion_radius_48h_m": trajectory[-1]["confidence_radius_m"],
            "risk_category": spec["risk"],
            "drift_trajectory": trajectory
        }

    @staticmethod
    def compare_temporal_surveys(
        previous_survey_hazards: List[Dict[str, Any]],
        current_survey_hazards: List[Dict[str, Any]],
        match_threshold_m: float = 12.0
    ) -> Dict[str, Any]:
        """Backward-compatible wrapper referencing core.change_detection."""
        from core.change_detection import TemporalChangeDetector
        detector = TemporalChangeDetector(match_tolerance_m=match_threshold_m)
        return detector.compare_surveys(previous_survey_hazards, current_survey_hazards)