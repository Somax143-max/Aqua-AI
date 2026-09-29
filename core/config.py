"""
AquaProtect-AI: Centralized System Configuration
Contains all hydroacoustic, neural, and operational parameters for SIH26057.
"""

import os
from typing import Dict, Any, List

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models_output")
CHECKPOINT_PATH = os.path.join(MODELS_DIR, "best_sonar_model.pt")
METRICS_OUTPUT_DIR = os.path.join(BASE_DIR, "benchmark_reports")
REPORTS_DIR = os.path.join(BASE_DIR, "reports_output")

for d in [MODELS_DIR, METRICS_OUTPUT_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

CLASSES: List[Dict[str, Any]] = [
    {
        "id": 0,
        "name": "Ghost Net",
        "code": "GHOST_NET",
        "severity": "CRITICAL",
        "color": (0, 0, 255),       # Red BGR
        "hex": "#FF3333",
        "description": "Entangled synthetic nylon netting posing critical wildlife & propeller entanglement danger."
    },
    {
        "id": 1,
        "name": "Shipwreck",
        "code": "SHIPWRECK",
        "severity": "HIGH",
        "color": (255, 0, 200),     # Magenta BGR
        "hex": "#FF00C8",
        "description": "Submerged vessel wreckage or metallic hull creating shallow navigational obstruction."
    },
    {
        "id": 2,
        "name": "Pipeline/Cable",
        "code": "PIPELINE_CABLE",
        "severity": "HIGH",
        "color": (0, 165, 255),     # Orange BGR
        "hex": "#FFA500",
        "description": "Exposed benthic subsea pipeline or communication cable route requiring anchor exclusion."
    },
    {
        "id": 3,
        "name": "Metal Drum",
        "code": "DRUM_BARREL",
        "severity": "HIGH",
        "color": (255, 255, 0),     # Cyan BGR
        "hex": "#00E5FF",
        "description": "Subsea drum container with hazardous chemical or petrochemical contamination risk."
    },
    {
        "id": 4,
        "name": "Cargo Container",
        "code": "CARGO_CONTAINER",
        "severity": "MEDIUM",
        "color": (0, 255, 128),     # Green BGR
        "hex": "#00FF80",
        "description": "Jettisoned standard intermodal container creating seafloor physical obstruction."
    },
    {
        "id": 5,
        "name": "Naval Mine / UXO",
        "code": "UXO_MINE",
        "severity": "CRITICAL",
        "color": (50, 50, 255),     # Bright Orange/Red BGR
        "hex": "#FF4500",
        "description": "High-density cylindrical unexploded ordnance (UXO) posing explosive detonation hazard."
    },
    {
        "id": 6,
        "name": "Natural Seafloor",
        "code": "NON_DEBRIS",
        "severity": "LOW",
        "color": (160, 160, 160),   # Neutral Gray BGR
        "hex": "#A0A0A0",
        "description": "Non-hazardous geological benthos including sand ripples, granite boulders, coral outcrops, and mudflats."
    }
]

CLASS_NAMES = [c["name"] for c in CLASSES]
CLASS_CODES = [c["code"] for c in CLASSES]
NUM_CLASSES = len(CLASSES)

DEFAULT_CONFIDENCE_THRESHOLD = 0.45
DEFAULT_NMS_THRESHOLD = 0.35
INPUT_PATCH_SIZE = (64, 64)
MC_DROPOUT_EVAL_SAMPLES = 15
MC_DROPOUT_DEEP_SAMPLES = 20

# AI Abstention Thresholds
ABSTAIN_UNCERTAINTY_VARIANCE_THRESH = 0.040
ABSTAIN_ENTROPY_THRESH = 1.38
ABSTAIN_CONFIDENCE_MIN = 0.42

AHSA_MIN_HIGHLIGHT_INTENSITY = 140
AHSA_MAX_SHADOW_INTENSITY = 60
AHSA_MIN_SHADOW_LENGTH_PX = 6
AHSA_CONFIDENCE_FLOOR = 0.30

DEFAULT_SOUND_SPEED_MPS = 1540.0
MACKENZIE_SURFACE_TEMP_C = 28.5
MACKENZIE_SALINITY_PPT = 34.8

