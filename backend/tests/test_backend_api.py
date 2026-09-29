"""
Unit and Integration Tests for AquaProtect-AI FastAPI Backend REST Service
"""

import os
import sys
import unittest
from typing import Dict, Any, List, Optional
import numpy as np
import cv2
from fastapi.testclient import TestClient

# Ensure backend directory is in path
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from api import app

class TestBackendAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_root_and_health(self):
        # Root endpoint
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ONLINE")
        self.assertIn("endpoints", data)

        # Health endpoint
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        h_data = res.json()
        self.assertEqual(h_data["status"], "HEALTHY")
        self.assertTrue(h_data["models_loaded"])

    def test_02_missions_endpoints(self):
        # List missions
        res = self.client.get("/api/missions")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("missions", data)
        self.assertGreaterEqual(len(data["missions"]), 3)

        # Get specific mission detail
        res = self.client.get("/api/missions/mission_chennai_niot")
        self.assertEqual(res.status_code, 200)
        m_data = res.json()
        self.assertIn("nav_telemetry", m_data)
        self.assertIn("sonar_image_b64", m_data)
        self.assertGreater(len(m_data["sonar_image_b64"]), 100)

    def test_03_validate_image_ood(self):
        # Create acoustic-like synthetic grayscale image
        sonar_img = (np.random.normal(120, 25, (100, 200))).clip(0, 255).astype(np.uint8)
        sonar_bgr = cv2.cvtColor(sonar_img, cv2.COLOR_GRAY2BGR)
        _, buf = cv2.imencode(".png", sonar_bgr)
        
        res = self.client.post(
            "/api/validate_image",
            files={"file": ("sonar.png", buf.tobytes(), "image/png")}
        )
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["valid"])

        # Natural daylight photo (high green/blue saturated color)
        daylight = np.zeros((100, 200, 3), dtype=np.uint8)
        daylight[:, :, 1] = 220 # High green foliage
        _, buf_bad = cv2.imencode(".png", daylight)
        res_bad = self.client.post(
            "/api/validate_image",
            files={"file": ("photo.png", buf_bad.tobytes(), "image/png")}
        )
        self.assertEqual(res_bad.status_code, 200)
        self.assertFalse(res_bad.json()["valid"])

    def test_04_parse_pings(self):
        sample_csv = (
            "Ping_Number,Timestamp,Latitude,Longitude,Heading_Deg,Depth_m,Altitude_m,Speed_knots,Roll_Deg,Pitch_Deg\n"
            "1,2026-09-12T10:00:00.000Z,13.0827,80.3705,85.0,42.0,12.0,3.0,0.5,0.2\n"
            "2,2026-09-12T10:00:00.100Z,13.0828,80.3706,85.1,42.1,12.1,3.0,0.4,0.1\n"
        )
        res = self.client.post("/api/parse_pings", json={"raw_content": sample_csv})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["ping_count"], 2)
        self.assertAlmostEqual(data["telemetry"]["latitude"], 13.0827)

    def test_05_detect_pipeline(self):
        # Create acoustic sonar image
        sonar_img = (np.random.normal(120, 25, (256, 512))).clip(0, 255).astype(np.uint8)
        sonar_bgr = cv2.cvtColor(sonar_img, cv2.COLOR_GRAY2BGR)
        # Add high-contrast acoustic highlight and shadow
        sonar_bgr[50:70, 100:130] = 240
        sonar_bgr[50:70, 130:170] = 10
        _, buf = cv2.imencode(".png", sonar_bgr)

        res = self.client.post(
            "/api/detect",
            files={"file": ("sonar.png", buf.tobytes(), "image/png")},
            data={
                "filter_type": "lee",
                "cmap": "bronze",
                "confidence_thresh": "0.35",
                "enable_src": "true",
                "enable_motion_comp": "false",
                "include_benthos": "true"
            }
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("detections", data)
        self.assertIn("metrics", data)
        self.assertIn("annotated_image_b64", data)

    def test_06_self_correction_batch(self):
        res = self.client.post(
            "/api/self_correction/run_batch",
            json={"iterations": 2, "samples_per_iteration": 2}
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "COMPLETED")
        self.assertEqual(data["iterations_executed"], 2)

    def test_07_digital_twin_state(self):
        res = self.client.get("/api/digital_twin/state")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("digital_twin_id", data)
        self.assertIn("survey_epochs", data)

if __name__ == "__main__":
    unittest.main()
