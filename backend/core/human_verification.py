"""
AquaProtect-AI: Human-in-the-Loop (HITL) Verification System
Provides the operator audit workflow for naval and oceanographic personnel:
- Operator Decisions:
  * CONFIRM (Operator agrees with AI neural & physics determination)
  * REJECT (Operator identifies false alarm or geological clutter)
  * RECLASSIFY (Operator corrects predicted class, e.g. Barrel -> Mine)
  * UNCERTAIN (Target earmarked for physical ROV camera inspection)
- Stores persistent feedback records used to drive the Active Learning retraining queue.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional

class HumanVerificationManager:
    """
    Manages operator verification decisions and audit persistence.
    """

    FEEDBACK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "feedback_data"))
    FEEDBACK_FILE = os.path.join(FEEDBACK_DIR, "operator_verifications.json")

    def __init__(self):
        os.makedirs(self.FEEDBACK_DIR, exist_ok=True)
        if not os.path.exists(self.FEEDBACK_FILE):
            with open(self.FEEDBACK_FILE, "w", encoding="utf-8") as f:
                json.dump([], f)

    def log_operator_decision(
        self,
        target_id: str,
        predicted_class: str,
        predicted_confidence: float,
        operator_action: str, # "CONFIRM", "REJECT", "RECLASSIFY", "UNCERTAIN"
        operator_id: str = "NAV-OP-01",
        corrected_class: Optional[str] = None,
        notes: str = ""
    ) -> Dict[str, Any]:
        """
        Logs a human verification entry and persists to feedback file.
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

        record = {
            "record_id": f"HITL-{int(time.time()*1000)%1000000:06d}",
            "timestamp": timestamp,
            "operator_id": operator_id,
            "target_id": target_id,
            "predicted_class": predicted_class,
            "predicted_confidence": predicted_confidence,
            "operator_action": operator_action.upper(),
            "final_verified_class": corrected_class if operator_action == "RECLASSIFY" else (predicted_class if operator_action == "CONFIRM" else "REJECTED_NON_DEBRIS"),
            "notes": notes or ("Verified by operator" if operator_action == "CONFIRM" else "Marked as clutter/reclassified")
        }

        # Load existing records
        try:
            with open(self.FEEDBACK_FILE, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception:
            records = []

        records.append(record)

        with open(self.FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

        return record

    def get_all_verifications(self) -> List[Dict[str, Any]]:
        """Retrieves all historical human operator feedback records."""
        if not os.path.exists(self.FEEDBACK_FILE):
            return []
        try:
            with open(self.FEEDBACK_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []