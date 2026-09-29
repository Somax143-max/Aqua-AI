"""
AquaProtect-AI: Defense-Grade Marine Hazard Risk Engine
Computes multi-factor nautical and environmental risk index (0 - 100):
- Threat Factors:
  * Inherent Hazard Weight (Ghost Nets = Entanglement, Munitions = Detonation, Drums = Toxic Spills)
  * Vertical Navigational Encroachment (Relief height / Water depth ratio)
  * Hydrodynamic Drift Mobility (Dispersion velocity and dragging potential)
  * Proximity to Critical Maritime Infrastructure (Shipping fairways, submarine cables, coral MPAs)
  * AI Certainty Penalty (High-uncertainty detections get flagged for expedited verification)
- Risk Tiers: LOW, MEDIUM, HIGH, CRITICAL
- Operational Action Recommendations for Coast Guard & Port Authorities
"""

import math
from typing import Dict, Any, Optional

class MarineHazardRiskEngine:
    """
    Computes rigorous risk scores and operational action plans for marine anomalies.
    """

    INHERENT_HAZARD_WEIGHTS = {
        "GHOST_NET": 0.95,        # Entanglement, ghost fishing, marine mammal fatality
        "UXO_MINE": 1.00,         # High explosive detonation, diver and vessel hull rupture
        "DRUM_BARREL": 0.82,      # Chemical / petroleum toxicity spill
        "SHIPWRECK": 0.75,        # Navigational hull strike, fuel oil leak
        "PIPELINE_CABLE": 0.70,   # Subsea infrastructure anchor dragging risk
        "CARGO_CONTAINER": 0.60,  # Physical seafloor obstruction
        "NON_DEBRIS": 0.10        # Natural geology (sand ripple, rock)
    }

    @classmethod
    def evaluate_hazard_risk(
        cls,
        detection: Dict[str, Any],
        proximity_to_shipping_lane_m: float = 450.0,
        is_coral_habitat: bool = True
    ) -> Dict[str, Any]:
        """
        Calculates composite risk score [0 - 100] and assigns operational response level.
        """
        code = detection.get("class_code", "UNKNOWN")
        inherent_w = cls.INHERENT_HAZARD_WEIGHTS.get(code, 0.50)

        # 1. Depth clearance ratio (relief height vs total water depth)
        dim = detection.get("dimensions", {})
        gt = detection.get("geotag", {})
        relief_m = dim.get("relief_height_m", 1.5)
        depth_m = max(gt.get("depth_m", 35.0), 5.0)
        keel_clearance_hazard = min(1.0, (relief_m / depth_m) * 4.0)

        # 2. Footprint factor (target area)
        area_m2 = dim.get("length_m", 2.0) * dim.get("width_m", 2.0)
        area_factor = min(1.0, area_m2 / 100.0)

        # 3. Infrastructure & Navigation Proximity factor
        if proximity_to_shipping_lane_m < 250.0:
            prox_factor = 1.0
        elif proximity_to_shipping_lane_m < 750.0:
            prox_factor = 0.75
        elif proximity_to_shipping_lane_m < 2000.0:
            prox_factor = 0.45
        else:
            prox_factor = 0.20

        # 4. Ecological MPA habitat factor
        eco_factor = 1.25 if is_coral_habitat and code in ["GHOST_NET", "DRUM_BARREL"] else 1.0

        # 5. Epistemic uncertainty penalty
        bayes = detection.get("bayesian_certainty", {})
        variance = bayes.get("predictive_variance", 0.010)
        uncertainty_penalty = 1.15 if variance > 0.035 else 1.0

        # Composite score calculation (0 - 100)
        raw_score = (
            0.40 * (inherent_w * 100.0) +
            0.25 * (keel_clearance_hazard * 100.0) +
            0.20 * (prox_factor * 100.0) +
            0.15 * (area_factor * 100.0)
        ) * eco_factor * uncertainty_penalty

        risk_score = round(float(np.clip(raw_score, 0.0, 100.0)), 1) if "np" in globals() else round(min(100.0, max(0.0, raw_score)), 1)

        # Determine Tier
        if code == "NON_DEBRIS":
            tier = "LOW"
            priority = "P4_BENIGN_GEOLOGY"
            action = "No intervention needed. Geological seafloor benthos."
        elif risk_score >= 80.0:
            tier = "CRITICAL"
            priority = "P1_IMMEDIATE_ACTION"
            action = "Issue immediate Notice to Mariners (NOTMAR). Dispatch ROV / clearance vessel within 24h."
        elif risk_score >= 60.0:
            tier = "HIGH"
            priority = "P2_URGENT_INTERVENTION"
            action = "Establish 500m maritime safety exclusion zone. Schedule recovery in next survey leg."
        elif risk_score >= 35.0:
            tier = "MEDIUM"
            priority = "P3_SCHEDULED_REMOVAL"
            action = "Log on hydrographic hazard registry. Monitor acoustic drift."
        else:
            tier = "LOW"
            priority = "P4_ROUTINE_MONITOR"
            action = "Low risk benthic anomaly. Retain in survey historical database."

        return {
            "risk_score": risk_score,
            "risk_tier": tier,
            "operational_priority": priority,
            "inherent_threat_rating": round(inherent_w * 100.0, 1),
            "navigational_clearance_hazard": round(keel_clearance_hazard * 100.0, 1),
            "fairway_proximity_m": round(proximity_to_shipping_lane_m, 1),
            "action_directive": action
        }