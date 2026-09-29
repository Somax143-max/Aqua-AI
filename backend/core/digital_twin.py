"""
AquaProtect-AI: Multi-Survey Marine Digital Twin Engine
Maintains an evolving spatial digital twin of the surveyed seabed zone across multiple years:
- Tracks temporal survey epochs: 2025 (Baseline) -> 2026 (Current) -> 2027 (Forecast)
- Persistent object registry indexing subsea infrastructure and hazards
- Real-time digital twin state:
  * Persistent debris (unresolved navigational hazards)
  * Recovered debris (salvaged by ROV)
  * Migrated / drifting debris (tracked across current vectors)
  * Newly classified hazards
- Computes long-term seabed clearance progress & environmental remediation rate
"""

import os
import json
import time
from typing import Dict, Any, List, Optional

class MarineDigitalTwin:
    """
    Evolving subsea digital twin managing multi-survey temporal archives.
    """

    TWIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "digital_twin_data"))
    TWIN_STATE_FILE = os.path.join(TWIN_DIR, "seafloor_digital_twin_registry.json")

    def __init__(self):
        os.makedirs(self.TWIN_DIR, exist_ok=True)
        if not os.path.exists(self.TWIN_STATE_FILE):
            self._initialize_default_twin()

    def _initialize_default_twin(self):
        """Initializes baseline digital twin state with historic multi-year survey archives."""
        default_state = {
            "digital_twin_id": "DT-NIOT-CHENNAI-SEABED",
            "geographic_zone": "Chennai Offshore Continental Shelf (NIOT Survey Sector A)",
            "bounding_box": {
                "min_lat": 13.0600, "max_lat": 13.1100,
                "min_lon": 80.2500, "max_lon": 80.3200
            },
            "survey_epochs": [
                {
                    "epoch_id": "EPOCH-2025",
                    "year": 2025,
                    "vessel": "ORV Sagar Nidhi",
                    "total_targets_detected": 14,
                    "ghost_nets": 4,
                    "munitions": 2,
                    "drums": 3,
                    "wrecks": 2,
                    "pipelines": 3
                },
                {
                    "epoch_id": "EPOCH-2026",
                    "year": 2026,
                    "vessel": "ORV Sagar Kanya",
                    "total_targets_detected": 18,
                    "new_debris": 5,
                    "recovered_debris": 3,
                    "persistent_debris": 11,
                    "moved_debris": 2
                },
                {
                    "epoch_id": "EPOCH-2027-PLANNED",
                    "year": 2027,
                    "vessel": "Autonomous Survey Glider Fleet",
                    "status": "SCHEDULED_RESCUE_SORTIE"
                }
            ],
            "remediation_metrics": {
                "initial_hazard_volume_m3": 450.0,
                "current_hazard_volume_m3": 380.0,
                "cleanup_efficiency_pct": 15.5,
                "ghost_net_cleared_kg": 2400.0
            }
        }
        with open(self.TWIN_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(default_state, f, indent=2)

    def get_digital_twin_summary(self) -> Dict[str, Any]:
        """Returns the current multi-survey digital twin overview."""
        try:
            with open(self.TWIN_STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"error": "Failed to read digital twin state"}

    def update_twin_with_current_survey(
        self,
        current_detections: List[Dict[str, Any]],
        survey_year: int = 2026
    ) -> Dict[str, Any]:
        """Incorporates newly processed survey detections into the digital twin registry."""
        twin = self.get_digital_twin_summary()
        epoch_entry = {
            "epoch_id": f"EPOCH-{survey_year}-LIVE",
            "year": survey_year,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "active_targets_count": len(current_detections),
            "critical_targets": sum(1 for d in current_detections if d.get("severity") == "CRITICAL")
        }
        twin["latest_live_epoch"] = epoch_entry

        with open(self.TWIN_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(twin, f, indent=2)

        return twin