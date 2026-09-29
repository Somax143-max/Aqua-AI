"""
AquaProtect-AI: Autonomous Marine Debris Command Center - Backend REST API
National Institute of Ocean Technology (NIOT) &bull; Ministry of Earth Sciences (MoES)
Smart India Hackathon (SIH26057) - Defense Edition v2.2

High-performance FastAPI service providing hydroacoustic processing, deep neural detection,
physics validation, drift trajectory modeling, digital twin synchronization, continuous self-correction,
and IHO S-57 / PDF hydrographic report generation.
"""

import os
import sys
import io
import time
import base64
import json
from typing import Dict, Any, List, Optional
import numpy as np
import cv2
from PIL import Image

# Ensure backend directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field

# Core Hydroacoustic & AI Modules
from core.sonar_physics import SonarPhysics
from core.validator import SonarInputValidator
from core.preprocessor import SonarPreprocessor
from core.geotagging import SonarGeotagger
from core.motion_compensation import MotionCompensator
from core.sonar_parser import SonarDataParser
from core.mission_planner import OceanCleanupMissionPlanner
from core.material_classifier import AcousticMaterialClassifier
from core.debris_drift_tracker import DebrisDriftTracker
from core.multi_ping_tracker import MultiPingObjectTracker
from core.hazard_engine import MarineHazardRiskEngine
from core.provenance import DataProvenanceLedger
from core.human_verification import HumanVerificationManager
from core.reproducibility import ReproducibilityManifest
from core.versioning import SystemVersioning
from core.digital_twin import MarineDigitalTwin
from models.detector import SonarDebrisDetector, DEBRIS_CLASSES
from models.physics_filter import AcousticPhysicsFilter
from models.ghost_net_analyzer import GhostNetAnalyzer
from models.edge_profiler import EdgeHardwareProfiler
from models.continuous_self_correction import ContinuousSelfCorrectionEngine
from models.active_learning import ActiveLearningPipeline
from data.sample_missions import SAMPLE_MISSIONS, load_mission_data
from reports.report_generator import HazardReportGenerator
from reports.hazard_map_generator import HydrographicHazardMapGenerator
from reports.s57_enc_exporter import S57ENCExporter

# FastAPI Application Definition
app = FastAPI(
    title="AquaProtect-AI Backend API",
    description="Autonomous Marine Debris Command Center REST API for NIOT & MoES Ocean Missions",
    version="2.2.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for cross-origin access (Streamlit, Web Dashboards, AUV Comms)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Application Singletons
physics = SonarPhysics()
validator = SonarInputValidator()
preprocessor = SonarPreprocessor()
motion_comp = MotionCompensator()
sonar_parser = SonarDataParser()
mission_planner = OceanCleanupMissionPlanner()
material_classifier = AcousticMaterialClassifier()
drift_tracker = DebrisDriftTracker()
object_tracker = MultiPingObjectTracker()
hazard_engine = MarineHazardRiskEngine()
provenance_ledger = DataProvenanceLedger()
human_verifier = HumanVerificationManager()
system_versioning = SystemVersioning()
digital_twin = MarineDigitalTwin()
edge_profiler = EdgeHardwareProfiler()
active_learning = ActiveLearningPipeline()
report_gen = HazardReportGenerator(output_dir=os.path.join(BASE_DIR, "reports_output"))
enc_exporter = S57ENCExporter(output_dir=os.path.join(BASE_DIR, "reports_output"))
self_correction_engine = ContinuousSelfCorrectionEngine()

# Lazy detector cache
_detector_instances: Dict[float, SonarDebrisDetector] = {}

def get_detector(conf_threshold: float = 0.45) -> SonarDebrisDetector:
    thresh_key = round(conf_threshold, 2)
    if thresh_key not in _detector_instances:
        _detector_instances[thresh_key] = SonarDebrisDetector(confidence_threshold=thresh_key)
    return _detector_instances[thresh_key]


# Helper: Image encoding/decoding
def image_to_base64(img_np: np.ndarray, format_ext: str = ".jpg") -> str:
    success, buffer = cv2.imencode(format_ext, img_np)
    if not success:
        return ""
    return base64.b64encode(buffer).decode("utf-8")


def base64_to_image(b64_str: str) -> Optional[np.ndarray]:
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_str)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Pydantic Request / Response Models
# ---------------------------------------------------------------------------
class PingParseRequest(BaseModel):
    raw_content: str = Field(..., description="Raw text or CSV content of ping navigation stream")

class FeedbackRequest(BaseModel):
    detection_id: str
    class_name: str
    verified_class: str
    is_correct: bool
    operator_id: str = "OPERATOR_CHIEF_01"
    notes: Optional[str] = None

class DriftRequest(BaseModel):
    target: Dict[str, Any]
    current_speed_knots: float = 1.8
    current_heading_deg: float = 65.0
    hours: int = 24

class SelfCorrectionRequest(BaseModel):
    iterations: int = Field(default=5, ge=1, le=100, description="Number of iterative correction batches")
    samples_per_iteration: int = Field(default=10, ge=1, le=50, description="Samples per training loop")

class ReportGenerateRequest(BaseModel):
    mission_telemetry: Dict[str, Any]
    detections: List[Dict[str, Any]]
    surveyor_name: str = "MoES/NIOT Chief Hydrographer"
    remarks: str = "Certified hydroacoustic survey conducted via AUV Matsya-6000."


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root_info():
    """Root info endpoint."""
    return {
        "system": "AquaProtect-AI Defense Edition v2.2",
        "organization": "National Institute of Ocean Technology (NIOT) / MoES",
        "status": "ONLINE",
        "endpoints": {
            "health": "/health",
            "documentation": "/docs",
            "missions": "/api/missions",
            "detect": "/api/detect",
            "parse_pings": "/api/parse_pings",
            "drift": "/api/drift",
            "digital_twin": "/api/digital_twin/state",
            "self_correction": "/api/self_correction/run_batch",
            "reports": "/api/reports/generate"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint indicating model state, device, and versions."""
    import torch
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "status": "HEALTHY",
        "device": device_name,
        "cuda_available": torch.cuda.is_available(),
        "version": "v2.2-Defense-Edition",
        "timestamp": time.time(),
        "models_loaded": True
    }


@app.get("/api/missions")
def list_missions():
    """List all available pre-configured Indian Maritime survey missions."""
    missions_summary = []
    for key, data in SAMPLE_MISSIONS.items():
        missions_summary.append({
            "key": key,
            "title": data.get("title", key),
            "location_name": data.get("location_name", ""),
            "agency": data.get("agency", ""),
            "vessel_name": data.get("vessel_name", ""),
            "seabed": data.get("seabed", ""),
            "start_lat": data.get("start_lat", 0.0),
            "start_lon": data.get("start_lon", 0.0),
            "depth_m": data.get("depth_m", 0.0),
            "altitude_m": data.get("altitude_m", 0.0),
            "description": data.get("description", ""),
            "target_count": len(data.get("targets", []))
        })
    return {"missions": missions_summary}


@app.get("/api/missions/{mission_key}")
def get_mission_detail(mission_key: str):
    """Retrieve full mission details including simulated sonar waterfall and nav telemetry."""
    if mission_key not in SAMPLE_MISSIONS:
        raise HTTPException(status_code=404, detail=f"Mission '{mission_key}' not found.")
    
    mission_data = load_mission_data(mission_key)
    sonar_img = mission_data["sonar_image"]
    img_b64 = image_to_base64(sonar_img, format_ext=".png")
    
    return {
        "mission_key": mission_key,
        "config": mission_data["config"],
        "nav_telemetry": mission_data["nav_telemetry"],
        "sonar_image_b64": img_b64,
        "image_shape": list(sonar_img.shape)
    }


@app.post("/api/validate_image")
async def validate_image(file: UploadFile = File(...)):
    """Stage-1 Out-Of-Distribution validation to prevent false daylight RGB uploads."""
    contents = await file.read()
    arr = np.frombuffer(contents, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode uploaded file as an image.")
    
    is_valid, validation_msg, _ = validator.validate_sonar_image(img)
    return {
        "valid": is_valid,
        "message": validation_msg
    }


@app.post("/api/parse_pings")
def parse_pings(payload: PingParseRequest):
    """Parse hydrographic navigation ping stream CSV/TXT into structured telemetry."""
    raw_text = payload.raw_content
    # Check for binary null bytes
    if "\x00" in raw_text[:1024]:
        raise HTTPException(status_code=400, detail="Payload contains binary null bytes and is not valid text telemetry.")
    
    try:
        pings = sonar_parser.parse_ping_csv(raw_text)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Error parsing ping telemetry: {str(exc)}")
    
    if not pings:
        raise HTTPException(status_code=400, detail="No valid ping records found in input.")
    
    first_ping = pings[0]
    telemetry = {
        "mission_title": "Hydrographic Ping Stream",
        "location_name": "Survey Swath Coordinates",
        "agency": "MoES / NIOT",
        "vessel_name": "AUV Navigational Stream",
        "latitude": first_ping.get("latitude", 13.0827),
        "longitude": first_ping.get("longitude", 80.3705),
        "heading": first_ping.get("heading", 90.0),
        "altitude": first_ping.get("altitude", 12.0),
        "depth": first_ping.get("depth", 38.0),
        "speed_mps": first_ping.get("speed_mps", 1.5),
        "speed_knots": first_ping.get("speed_knots", 2.9),
        "ping_rate_hz": 10.0,
        "max_range_m": 75.0,
        "roll": first_ping.get("roll", 0.0),
        "pitch": first_ping.get("pitch", 0.0)
    }
    
    return {
        "ping_count": len(pings),
        "telemetry": telemetry,
        "pings": pings[:200]  # return top 200 for preview
    }


@app.post("/api/detect")
async def detect_debris(
    file: Optional[UploadFile] = File(None),
    image_base64_str: Optional[str] = Form(None),
    filter_type: str = Form("lee"),
    cmap: str = Form("bronze"),
    confidence_thresh: float = Form(0.45),
    enable_src: bool = Form(True),
    enable_motion_comp: bool = Form(True),
    enable_bayes_eval: bool = Form(False),
    include_benthos: bool = Form(False),
    telemetry_json: Optional[str] = Form(None)
):
    """
    Core Hydroacoustic AI Pipeline:
    1. Stage-1 OOD Acoustic Validation
    2. Slant-to-Ground Range Correction (SRC)
    3. Acoustic Speckle Filtering & CLAHE Preprocessing
    4. Towfish Motion Dynamics Compensation
    5. Dual-Stream scSE Attention Neural Inference
    6. Acoustic Highlight-Shadow Association (AHSA) Physics Filter
    7. High-Precision Geo-Referencing (WGS-84)
    8. Acoustic Backscatter Material Classification
    9. 24-Hour Hydrodynamic Drift Simulation
    """
    t_start = time.time()
    
    # 1. Ingest Sonar Image
    raw_img = None
    if file is not None:
        contents = await file.read()
        arr = np.frombuffer(contents, dtype=np.uint8)
        raw_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    elif image_base64_str:
        raw_img = base64_to_image(image_base64_str)
    
    if raw_img is None:
        raise HTTPException(status_code=400, detail="No valid image provided via file upload or base64 parameter.")
    
    # Parse telemetry
    telemetry = {
        "mission_title": "Acoustic Survey Swath",
        "location_name": "Indian Continental Shelf",
        "latitude": 13.0827,
        "longitude": 80.3705,
        "heading": 90.0,
        "altitude": 12.0,
        "depth": 38.0,
        "speed_mps": 1.5,
        "speed_knots": 2.9,
        "max_range_m": 75.0,
        "roll": 1.2,
        "pitch": 0.5
    }
    if telemetry_json:
        try:
            parsed_tel = json.loads(telemetry_json)
            telemetry.update(parsed_tel)
        except Exception:
            pass
    
    # 2. Stage-1 Natural Gate / OOD Validation
    is_valid, validation_msg, gray_img = validator.validate_sonar_image(raw_img)
    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail=f"Image Validation Failed (OOD): {validation_msg}"
        )
    
    # Grayscale conversion fallback
    if gray_img is None:
        if len(raw_img.shape) == 3:
            gray_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = raw_img.copy()
    
    # 3. Slant-to-Ground Range Correction
    processing_img = gray_img
    if enable_src:
        try:
            processing_img = physics.slant_to_ground_range(processing_img, altitude_m=telemetry.get("altitude", 12.0))
        except Exception as e:
            # Fallback to uncorrected if altitude invalid
            pass
    
    # 4. Preprocessing (Speckle Filter + CLAHE)
    denoised_img, filtered_img = preprocessor.denoise_and_enhance(processing_img, filter_type=filter_type)
    
    # 5. Motion Compensation
    if enable_motion_comp:
        try:
            motion_res = motion_comp.compensate(
                filtered_img,
                roll_deg=telemetry.get("roll", 1.2),
                pitch_deg=telemetry.get("pitch", 0.5)
            )
            filtered_img = motion_res["compensated_image"]
        except Exception:
            pass
    
    # 6. Deep Neural Detection
    # 6. Deep Neural Detection Pipeline
    detector = get_detector(conf_threshold=confidence_thresh)
    detector_result = detector.detect(
        sonar_img=processing_img,
        nav_record=telemetry,
        filter_type=filter_type,
        apply_motion_comp=enable_motion_comp,
        compute_uncertainty=enable_bayes_eval,
        include_natural_benthos=include_benthos
    )
    
    detections = detector_result.get("detections", [])
    annotated = detector_result.get("annotated_image", raw_img)
    
    # Material Classification & Drift Prediction Enrichment
    for det in detections:
        bbox = det["bbox"]
        crop = filtered_img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        if crop.size > 0:
            mat_res = material_classifier.classify_crop(crop)
            det["material"] = mat_res["material"]
            det["material_confidence"] = mat_res["confidence"]
            det["acoustic_signature"] = mat_res.get("signature", {})
        else:
            det["material"] = "Acoustic Composite"
            det["material_confidence"] = 0.85
        
        # Drift prediction
        drift_res = drift_tracker.predict_drift(det, hours=24)
        det["drift_trajectory"] = drift_res.get("trajectory", [])
        det["drift_risk"] = drift_res.get("risk_level", "LOW")
    
    color_map = {
        "Ghost Net": (0, 0, 255),          # Red
        "Shipwreck": (200, 0, 255),        # Magenta
        "Pipeline/Cable": (0, 165, 255),    # Orange
        "Metal Drum": (255, 255, 0),       # Cyan
        "Cargo Container": (0, 255, 128),  # Green
        "Naval Mine / UXO": (50, 50, 255), # Bright Red-Orange
        "Natural Seafloor": (160, 160, 160)# Gray
    }
    
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls_name = det.get("class_name", "Debris")
        conf = det.get("confidence", 0.0)
        col = color_map.get(cls_name, (0, 255, 255))
        
        cv2.rectangle(annotated, (x1, y1), (x2, y2), col, 2)
        label = f"{cls_name} {conf:.2f}"
        cv2.putText(annotated, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Shadow connection line if shadow detected
        if "shadow_bbox" in det and det["shadow_bbox"]:
            sx1, sy1, sx2, sy2 = det["shadow_bbox"]
            cv2.rectangle(annotated, (sx1, sy1), (sx2, sy2), (80, 80, 80), 1)
            cv2.line(annotated, (x2, (y1 + y2) // 2), (sx1, (sy1 + sy2) // 2), (180, 180, 180), 1, cv2.LINE_AA)
    
    annotated_b64 = image_to_base64(annotated, format_ext=".jpg")
    
    # Calculate summary metrics
    crit_count = sum(1 for d in detections if d.get("severity") == "CRITICAL")
    high_count = sum(1 for d in detections if d.get("severity") == "HIGH")
    med_count = sum(1 for d in detections if d.get("severity") == "MEDIUM")
    mean_conf = float(np.mean([d.get("confidence", 0.0) for d in detections])) if detections else 0.0
    latency_ms = round((time.time() - t_start) * 1000.0, 1)
    
    return {
        "status": "success",
        "latency_ms": latency_ms,
        "detection_count": len(detections),
        "metrics": {
            "total_targets": len(detections),
            "critical_hazards": crit_count,
            "high_hazards": high_count,
            "medium_hazards": med_count,
            "mean_confidence": round(mean_conf, 3),
            "fps": round(1000.0 / max(latency_ms, 1.0), 1)
        },
        "telemetry": telemetry,
        "detections": detections,
        "annotated_image_b64": annotated_b64
    }


@app.post("/api/drift")
def calculate_drift(payload: DriftRequest):
    """Calculate 24-hour hydrodynamic drift trajectory for a detected object."""
    target = payload.target
    speed = payload.current_speed_knots
    heading = payload.current_heading_deg
    hours = payload.hours
    
    res = drift_tracker.predict_drift(
        target,
        hours=hours,
        current_speed_knots=speed,
        current_heading_deg=heading
    )
    return res


@app.get("/api/digital_twin/state")
def get_digital_twin_state():
    """Retrieve full digital twin seafloor registry and synchronized objects."""
    state = digital_twin.get_digital_twin_summary()
    return state


@app.post("/api/digital_twin/sync")
def sync_digital_twin(detections: List[Dict[str, Any]], telemetry: Dict[str, Any]):
    """Synchronize marine digital twin with real-time AUV detections."""
    result = digital_twin.update_twin_with_current_survey(detections, survey_year=2026)
    return {"status": "synced", "result": result}


@app.post("/api/self_correction/run_batch")
def run_self_correction_batch(payload: SelfCorrectionRequest):
    """
    Execute autonomous continuous self-correction training iteration.
    Simulates / scans waterfall strips, identifies residual errors, and trains backprop weights.
    """
    res = self_correction_engine.run_self_correction_cycle(
        num_iterations=payload.iterations,
        batch_size=payload.samples_per_iteration
    )
    return res


@app.get("/api/self_correction/status")
def get_self_correction_status():
    """Get continuous self-correction telemetry, iteration count, and loss curve."""
    return {
        "status": "READY",
        "training_log_count": len(self_correction_engine.training_log)
    }


@app.post("/api/active_learning/feedback")
def submit_operator_feedback(payload: FeedbackRequest):
    """Record operator verification or correction for Human-in-the-Loop retraining."""
    record = human_verifier.record_verification(
        detection_id=payload.detection_id,
        predicted_class=payload.class_name,
        verified_class=payload.verified_class,
        is_correct=payload.is_correct,
        operator_id=payload.operator_id,
        notes=payload.notes
    )
    active_learning.queue_sample_for_retraining({
        "detection_id": payload.detection_id,
        "verified_class": payload.verified_class,
        "timestamp": time.time()
    })
    return {"status": "recorded", "record": record}


@app.get("/api/edge_profile")
def get_edge_profile():
    """Run edge hardware profiler benchmark and ONNX runtime verifier."""
    metrics = edge_profiler.profile_model()
    return metrics


@app.post("/api/reports/generate")
def generate_reports(payload: ReportGenerateRequest):
    """Generate certified hydrographic survey PDF, S-57 ENC layer, JSON, and CSV."""
    telemetry = payload.mission_telemetry
    detections = payload.detections
    
    # 1. PDF Report
    pdf_path = report_gen.generate_pdf_report(
        detections=detections,
        mission_metadata=telemetry,
        surveyor_name=payload.surveyor_name,
        remarks=payload.remarks
    )
    
    # 2. S-57 ENC Layer
    s57_path = enc_exporter.export_s57(
        detections=detections,
        mission_metadata=telemetry
    )
    
    # 3. JSON Export
    json_path = os.path.join(BASE_DIR, "reports_output", "hydrographic_survey_export.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "mission": telemetry,
            "detections": detections,
            "generated_at": time.time()
        }, f, indent=2)
        
    return {
        "status": "success",
        "pdf_report": pdf_path,
        "s57_enc": s57_path,
        "json_export": json_path
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("BACKEND_PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)
