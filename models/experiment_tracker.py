"""
AquaProtect-AI: Neural Experiment & Training Run Tracker
Maintains an immutable registry of all training runs, hyperparameter sweeps, and evaluations:
- Experiment ID & Run Name
- Training Hyperparameters (Batch Size, LR, Epochs, Loss Type, HNM Factor)
- Training & Validation Convergence Metrics
- Independent Test Benchmark Metrics (mAP@50, mAP@50:95, ECE, Dice)
- Checkpoint SHA-256 Hash and Artifact Links
"""

import os
import json
import time
import hashlib
from typing import Dict, Any, List, Optional

EXPERIMENTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "benchmark_reports", "experiments"))

class ExperimentTracker:
    """Logs and indexes machine learning training runs."""

    def __init__(self):
        os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
        self.index_file = os.path.join(EXPERIMENTS_DIR, "experiment_registry.json")
        if not os.path.exists(self.index_file):
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump([], f)

    def log_experiment(
        self,
        experiment_name: str,
        hyperparameters: Dict[str, Any],
        val_metrics: Dict[str, Any],
        test_metrics: Dict[str, Any],
        checkpoint_path: str,
        duration_seconds: float = 120.0
    ) -> Dict[str, Any]:
        """Logs a completed training experiment."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        exp_id = f"EXP-{int(time.time())}"

        # Hash weights file if exists
        ckpt_hash = "None"
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, "rb") as f:
                ckpt_hash = f"SHA256:{hashlib.sha256(f.read()).hexdigest()[:16]}"

        record = {
            "experiment_id": exp_id,
            "experiment_name": experiment_name,
            "timestamp": timestamp,
            "duration_seconds": round(duration_seconds, 1),
            "hyperparameters": hyperparameters,
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "checkpoint": {
                "path": os.path.abspath(checkpoint_path),
                "hash": ckpt_hash
            }
        }

        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                registry = json.load(f)
        except Exception:
            registry = []

        registry.append(record)

        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)

        return record

    def list_experiments(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.index_file):
            return []
        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []