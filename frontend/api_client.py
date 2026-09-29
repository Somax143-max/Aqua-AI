"""
AquaProtect-AI: Frontend API Client
National Institute of Ocean Technology (NIOT) / Ministry of Earth Sciences (MoES)
Smart India Hackathon (SIH26057)

Client for communicating with the AquaProtect-AI FastAPI Backend service.
Supports both Microservice REST mode (via HTTP) and Local Fallback mode (direct Python invocation).
"""

import os
import sys
import json
import base64
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import requests
import cv2

# Configuration
DEFAULT_BACKEND_URLS = [
    os.environ.get("BACKEND_URL", ""),
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "http://127.0.0.1:8001",
    "http://localhost:8001",
    "http://backend:8000"
]

class AquaProtectAPIClient:
    """Client for AquaProtect-AI backend REST API with seamless local fallback."""

    def __init__(self, backend_url: Optional[str] = None):
        self.backend_url = backend_url or os.environ.get("BACKEND_URL")
        self._is_online = False
        self._last_health_check = 0.0
        self._auto_detect_backend()

    def _auto_detect_backend(self) -> bool:
        """Probe potential backend URLs to discover active REST service."""
        candidates = [self.backend_url] if self.backend_url else []
        candidates.extend([u for u in DEFAULT_BACKEND_URLS if u and u not in candidates])
        
        for url in candidates:
            try:
                res = requests.get(f"{url.rstrip('/')}/health", timeout=1.0)
                if res.status_code == 200:
                    self.backend_url = url.rstrip("/")
                    self._is_online = True
                    self._last_health_check = time.time()
                    return True
            except Exception:
                continue
        
        self._is_online = False
        return False

    def is_online(self) -> bool:
        """Check if backend REST API is responsive."""
        if time.time() - self._last_health_check > 5.0:
            return self._auto_detect_backend()
        return self._is_online

    def get_health(self) -> Dict[str, Any]:
        """Fetch backend health status."""
        if self.is_online():
            try:
                res = requests.get(f"{self.backend_url}/health", timeout=3.0)
                if res.status_code == 200:
                    return res.json()
            except Exception:
                pass
        return {"status": "OFFLINE_LOCAL_MODE", "device": "local", "models_loaded": True}

    def get_missions(self) -> List[Dict[str, Any]]:
        """List pre-configured Indian Maritime survey zones."""
        if self.is_online():
            try:
                res = requests.get(f"{self.backend_url}/api/missions", timeout=5.0)
                if res.status_code == 200:
                    return res.json().get("missions", [])
            except Exception:
                pass
        
        # Local fallback
        try:
            from data.sample_missions import SAMPLE_MISSIONS
            missions = []
            for k, v in SAMPLE_MISSIONS.items():
                missions.append({
                    "key": k,
                    "title": v.get("title", k),
                    "location_name": v.get("location_name", ""),
                    "description": v.get("description", "")
                })
            return missions
        except Exception:
            return []

    def get_mission_detail(self, mission_key: str) -> Optional[Dict[str, Any]]:
        """Load specific mission sonar waterfall and telemetry."""
        if self.is_online():
            try:
                res = requests.get(f"{self.backend_url}/api/missions/{mission_key}", timeout=10.0)
                if res.status_code == 200:
                    data = res.json()
                    # Decode image
                    img_b64 = data.get("sonar_image_b64", "")
                    img_bytes = base64.b64decode(img_b64)
                    arr = np.frombuffer(img_bytes, dtype=np.uint8)
                    sonar_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    return {
                        "config": data["config"],
                        "nav_telemetry": data["nav_telemetry"],
                        "sonar_image": sonar_img
                    }
            except Exception:
                pass
        
        # Local fallback
        try:
            from data.sample_missions import load_mission_data
            return load_mission_data(mission_key)
        except Exception:
            return None

    def validate_image(self, img_np: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        """Stage-1 Acoustic Validation (OOD check)."""
        if self.is_online():
            try:
                _, buf = cv2.imencode(".png", img_np)
                files = {"file": ("sonar.png", buf.tobytes(), "image/png")}
                res = requests.post(f"{self.backend_url}/api/validate_image", files=files, timeout=5.0)
                if res.status_code == 200:
                    data = res.json()
                    return data.get("valid", False), data.get("validation_info", {})
            except Exception:
                pass
                
        # Local fallback
        try:
            from core.validator import SonarInputValidator
            val = SonarInputValidator()
            return val.validate_sonar_image(img_np)
        except Exception as e:
            return True, {"valid": True, "error": str(e)}

    def parse_pings(self, raw_text: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Parse hydrographic navigation stream."""
        if self.is_online():
            try:
                payload = {"raw_content": raw_text}
                res = requests.post(f"{self.backend_url}/api/parse_pings", json=payload, timeout=5.0)
                if res.status_code == 200:
                    data = res.json()
                    return data.get("pings", []), data.get("telemetry", {})
            except Exception:
                pass
                
        # Local fallback
        try:
            from core.sonar_parser import SonarDataParser
            p = SonarDataParser()
            pings = p.parse_ping_csv(raw_text)
            if pings:
                first = pings[0]
                tel = {
                    "mission_title": "Hydrographic Ping Stream",
                    "location_name": "Survey Swath Coordinates",
                    "agency": "MoES / NIOT",
                    "vessel_name": "AUV Navigational Stream",
                    "latitude": first.get("latitude", 13.0827),
                    "longitude": first.get("longitude", 80.3705),
                    "heading": first.get("heading", 90.0),
                    "altitude": first.get("altitude", 12.0),
                    "depth": first.get("depth", 38.0),
                    "speed_mps": first.get("speed_mps", 1.5),
                    "speed_knots": first.get("speed_knots", 2.9),
                    "ping_rate_hz": 10.0,
                    "max_range_m": 75.0,
                    "roll": first.get("roll", 0.0),
                    "pitch": first.get("pitch", 0.0)
                }
                return pings, tel
        except Exception:
            pass
        return [], {}

    def detect(
        self,
        image_np: np.ndarray,
        filter_type: str = "lee",
        cmap: str = "bronze",
        confidence_thresh: float = 0.45,
        enable_src: bool = True,
        enable_motion_comp: bool = True,
        enable_bayes_eval: bool = False,
        include_benthos: bool = False,
        telemetry: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute full acoustic AI detection pipeline via REST API."""
        if self.is_online():
            try:
                _, buf = cv2.imencode(".png", image_np)
                files = {"file": ("sonar.png", buf.tobytes(), "image/png")}
                data = {
                    "filter_type": filter_type,
                    "cmap": cmap,
                    "confidence_thresh": str(confidence_thresh),
                    "enable_src": str(enable_src),
                    "enable_motion_comp": str(enable_motion_comp),
                    "enable_bayes_eval": str(enable_bayes_eval),
                    "include_benthos": str(include_benthos),
                    "telemetry_json": json.dumps(telemetry or {})
                }
                res = requests.post(f"{self.backend_url}/api/detect", files=files, data=data, timeout=30.0)
                if res.status_code == 200:
                    resp_json = res.json()
                    # Decode annotated image if present
                    if "annotated_image_b64" in resp_json:
                        b64 = resp_json["annotated_image_b64"]
                        img_bytes = base64.b64decode(b64)
                        arr = np.frombuffer(img_bytes, dtype=np.uint8)
                        resp_json["annotated_image"] = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    return resp_json
            except Exception as e:
                # If network fails, fall through to local execution
                pass
        
        return {"status": "fallback_to_local"}

    def run_self_correction_batch(self, iterations: int = 5, samples_per_iteration: int = 10) -> Dict[str, Any]:
        """Trigger continuous self-correction training cycle."""
        if self.is_online():
            try:
                payload = {
                    "iterations": iterations,
                    "samples_per_iteration": samples_per_iteration
                }
                res = requests.post(f"{self.backend_url}/api/self_correction/run_batch", json=payload, timeout=60.0)
                if res.status_code == 200:
                    return res.json()
            except Exception:
                pass
        
        # Local fallback
        try:
            from models.continuous_self_correction import ContinuousSelfCorrectionEngine
            engine = ContinuousSelfCorrectionEngine()
            return engine.run_training_iteration(num_samples=samples_per_iteration, cycles=iterations)
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_self_correction_status(self) -> Dict[str, Any]:
        """Retrieve self-correction telemetry."""
        if self.is_online():
            try:
                res = requests.get(f"{self.backend_url}/api/self_correction/status", timeout=5.0)
                if res.status_code == 200:
                    return res.json()
            except Exception:
                pass
        try:
            from models.continuous_self_correction import ContinuousSelfCorrectionEngine
            return ContinuousSelfCorrectionEngine().get_telemetry()
        except Exception:
            return {}

    def generate_reports(self, telemetry: Dict[str, Any], detections: List[Dict[str, Any]], surveyor: str = "MoES/NIOT Chief Hydrographer") -> Dict[str, Any]:
        """Generate certified PDF, S-57 ENC, and JSON survey reports."""
        if self.is_online():
            try:
                payload = {
                    "mission_telemetry": telemetry,
                    "detections": detections,
                    "surveyor_name": surveyor
                }
                res = requests.post(f"{self.backend_url}/api/reports/generate", json=payload, timeout=20.0)
                if res.status_code == 200:
                    return res.json()
            except Exception:
                pass
        
        # Local fallback
        try:
            from reports.report_generator import HazardReportGenerator
            from reports.s57_enc_exporter import S57ENCExporter
            rg = HazardReportGenerator(output_dir="reports_output")
            enc = S57ENCExporter(output_dir="reports_output")
            pdf = rg.generate_pdf_report(detections, telemetry, surveyor_name=surveyor)
            s57 = enc.export_s57(detections, telemetry)
            return {"status": "success", "pdf_report": pdf, "s57_enc": s57}
        except Exception as e:
            return {"status": "error", "message": str(e)}
