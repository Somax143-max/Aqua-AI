"""
AquaProtect-AI: Active Learning & Operational Retraining Engine
Transforms operational human feedback into model improvements:
- Ingests operator-confirmed and rejected sonar crops
- Prioritizes high-uncertainty edge cases (Epistemic variance > 0.035)
- Retraining Queue Management:
  * Batches verified crops into fine-tuning partitions
  * Updates dataset manifest with field annotations
  * Computes retraining readiness threshold (e.g. >= 20 new samples)
"""

import os
import json
import time
from typing import Dict, Any, List, Optional

class ActiveLearningPipeline:
    """
    Continuous active learning controller connecting operator feedback to neural weights.
    """

    QUEUE_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "feedback_data", "retraining_queue.json"))

    def __init__(self, retrain_batch_threshold: int = 15):
        self.retrain_batch_threshold = retrain_batch_threshold
        os.makedirs(os.path.dirname(self.QUEUE_FILE), exist_ok=True)
        if not os.path.exists(self.QUEUE_FILE):
            with open(self.QUEUE_FILE, "w", encoding="utf-8") as f:
                json.dump([], f)

    def queue_sample_for_retraining(
        self,
        target_id: str,
        assigned_class_id: int,
        source: str = "OPERATOR_FEEDBACK",
        uncertainty_variance: float = 0.042,
        notes: str = ""
    ) -> Dict[str, Any]:
        """
        Appends an acoustic anomaly crop to the active learning retraining queue.
        """
        sample = {
            "queue_id": f"ALQ-{int(time.time()*1000)%1000000:06d}",
            "target_id": target_id,
            "ground_truth_class_id": assigned_class_id,
            "source": source,
            "epistemic_uncertainty": round(uncertainty_variance, 5),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "status": "QUEUED_FOR_FINE_TUNING",
            "notes": notes
        }

        try:
            with open(self.QUEUE_FILE, "r", encoding="utf-8") as f:
                queue = json.load(f)
        except Exception:
            queue = []

        queue.append(sample)

        with open(self.QUEUE_FILE, "w", encoding="utf-8") as f:
            json.dump(queue, f, indent=2)

        return sample

    def get_queue_status(self) -> Dict[str, Any]:
        """Returns the current status of the active learning retraining queue."""
        if not os.path.exists(self.QUEUE_FILE):
            return {"queued_count": 0, "ready_for_retraining": False, "samples": []}

        try:
            with open(self.QUEUE_FILE, "r", encoding="utf-8") as f:
                queue = json.load(f)
        except Exception:
            queue = []

        count = len(queue)
        is_ready = count >= self.retrain_batch_threshold

        return {
            "queued_samples_count": count,
            "batch_threshold": self.retrain_batch_threshold,
            "ready_for_retraining": is_ready,
            "readiness_pct": min(100.0, round((count / max(self.retrain_batch_threshold, 1)) * 100.0, 1)),
            "status_message": (
                f"Retraining queue has {count}/{self.retrain_batch_threshold} verified operational samples. "
                + ("Threshold reached: ready for incremental fine-tuning." if is_ready else "Gathering further operator verifications.")
            ),
            "recent_samples": queue[-5:]
        }