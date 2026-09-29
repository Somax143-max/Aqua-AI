"""
AquaProtect-AI: Multi-Format Sonar & Navigation Dataset Ingestor
Supports:
- Image formats: TIFF, GeoTIFF, PNG, JPEG, BMP
- Acoustic telemetry: Sonar ping CSVs, NMEA navigation logs, Hydrographic survey metadata
- Standardizes raw acoustic backscatter to 8-bit uint8 grayscale
- Explicit Data Provenance Tagging:
  * REAL (Field operational data from NIOT / Sagar Nidhi surveys)
  * PUBLIC (Open oceanographic benchmark datasets: SeabedObjects-KGS, Turntable-Sonar)
  * USER_UPLOADED (Ad-hoc survey imagery uploaded via operator dashboard)
  * SYNTHETIC (Physics-grounded acoustic generator with Lloyd-Mirror speckle)
"""

import os
import csv
import json
import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

class SonarDatasetIngestor:
    """
    Unified Ingestor for heterogeneous hydroacoustic sonar files and navigation records.
    """

    SUPPORTED_IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
    SUPPORTED_NAV_EXTS = {".csv", ".txt", ".json", ".log"}

    @staticmethod
    def identify_data_source(file_path: str) -> str:
        """Determines the data source category based on file metadata and naming."""
        fname = os.path.basename(file_path).lower()
        if "synthetic" in fname or "sim" in fname:
            return "SYNTHETIC"
        elif "public" in fname or "kgs" in fname or "turntable" in fname:
            return "PUBLIC"
        elif "real" in fname or "niot" in fname or "sagar" in fname or "field" in fname:
            return "REAL"
        elif "upload" in fname:
            return "USER_UPLOADED"
        else:
            return "PUBLIC_OR_FIELD"

    @classmethod
    def load_sonar_image(
        cls,
        file_path: str,
        source_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Loads and standardizes any sonar image (TIFF, GeoTIFF, PNG, JPEG).
        Returns standardized single-channel uint8 array and hydrographic metadata.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Sonar file not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in cls.SUPPORTED_IMAGE_EXTS:
            raise ValueError(f"Unsupported sonar image format '{ext}'. Must be one of {cls.SUPPORTED_IMAGE_EXTS}")

        # Load with OpenCV supporting full dynamic range
        raw_img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if raw_img is None:
            raise IOError(f"Failed to decode sonar image from {file_path}")

        # Check GeoTIFF or multi-band metadata
        is_multiband = (len(raw_img.shape) == 3 and raw_img.shape[2] > 1)
        if is_multiband:
            gray_img = cv2.cvtColor(raw_img[:, :, :3], cv2.COLOR_BGR2GRAY)
        else:
            gray_img = raw_img

        # Normalize 16-bit or 32-bit floats down to 8-bit sonar uint8
        if gray_img.dtype == np.uint16:
            gray_u8 = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        elif gray_img.dtype == np.float32 or gray_img.dtype == np.float64:
            gray_u8 = np.clip(gray_img * 255.0 if np.max(gray_img) <= 1.0 else gray_img, 0, 255).astype(np.uint8)
        else:
            gray_u8 = gray_img.astype(np.uint8)

        h, w = gray_u8.shape
        source_type = source_override or cls.identify_data_source(file_path)

        # Estimate horizontal resolution (meters per pixel across swath)
        # Standard side-scan swath is typically 150m across dual channels (75m Port + 75m Starboard)
        m_per_px_across = round(150.0 / max(w, 1), 4)
        m_per_px_along = round(m_per_px_across * 1.2, 4) # typical along-track ping spacing

        return {
            "image": gray_u8,
            "dimensions": {"height_pings": h, "width_samples": w},
            "data_source": source_type,
            "file_name": os.path.basename(file_path),
            "file_path": os.path.abspath(file_path),
            "file_size_bytes": os.path.getsize(file_path),
            "bit_depth": raw_img.dtype.name,
            "is_multiband": is_multiband,
            "spatial_resolution_m_per_px": {
                "across_track": m_per_px_across,
                "along_track": m_per_px_along
            },
            "channels": {
                "port_swath_px": [0, w // 2],
                "nadir_center_px": w // 2,
                "starboard_swath_px": [w // 2, w]
            }
        }

    @classmethod
    def load_navigation_csv(cls, csv_path: str) -> List[Dict[str, Any]]:
        """
        Parses navigation and hydrographic telemetry logs (Lat, Lon, Heading, Speed, Depth, Altitude).
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Navigation file not found: {csv_path}")

        records = []
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                def get_float(keys: List[str], default: float) -> float:
                    for k in keys:
                        for rk in row.keys():
                            if k.lower() in rk.lower() and row[rk].strip():
                                try:
                                    return float(row[rk])
                                except ValueError:
                                    pass
                    return default

                rec = {
                    "ping_index": int(get_float(["ping", "index", "id"], float(i))),
                    "timestamp": row.get("timestamp", row.get("time", f"T+{i*0.1:.1f}s")),
                    "latitude": get_float(["lat", "latitude"], 13.0827),
                    "longitude": get_float(["lon", "long", "longitude"], 80.2707),
                    "heading_deg": get_float(["heading", "yaw", "course"], 90.0),
                    "speed_mps": get_float(["speed", "vel"], 1.5),
                    "depth_m": get_float(["depth", "water_depth"], 35.0),
                    "altitude_m": get_float(["altitude", "alt"], 12.0),
                    "roll_deg": get_float(["roll"], 0.0),
                    "pitch_deg": get_float(["pitch"], 0.0)
                }
                records.append(rec)
        return records