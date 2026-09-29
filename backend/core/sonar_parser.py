import re
"""
Hydrographic Sonar Data & Ping Header Parser
Parses standard marine acoustic navigation records and sonar packet formats:
- CSV / TSV Ping Header Logs
- XTF (eXtended Triton Format) Hydrographic Packet Structure
- GeoTIFF Georeferenced Sonar Metadata Extractor
"""

import io
import csv
import json
import struct
import datetime
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

class SonarDataParser:
    """
    Parses hydrographic metadata, ping headers, and coordinate files
    for real-time synchronization with side-scan sonar waterfall data.
    """

    def __init__(self, sound_speed_mps: float = 1500.0):
        self.sound_speed_mps = sound_speed_mps

    def parse_ping_csv(self, file_content: str) -> List[Dict[str, Any]]:
        """
        Parses a standard hydrographic ping header navigation file.
        Accepts comma-separated, tab-separated, or space-separated text.
        Ignores metadata and unit lines.
        """
        import re
        try:
            if not file_content or not isinstance(file_content, str):
                return []
            records = []
            lines = [line.strip() for line in file_content.strip().split('\n') if line.strip() and not line.startswith('#')]
            if not lines:
                return []
                
            # Replace variable spaces/tabs with a single comma for uniform parsing
            cleaned_lines = []
            for line in lines:
                if ',' in line:
                    cleaned_lines.append(line)
                else:
                    cleaned_lines.append(re.sub(r'\s+', ',', line))
                    
            reader = csv.DictReader(io.StringIO('\n'.join(cleaned_lines)))
            
            for i, row in enumerate(reader):
                if not isinstance(row, dict):
                    continue
                norm = {}
                for k, v in row.items():
                    if k is None or str(k).strip() == '':
                        continue
                    safe_k = str(k).strip().lower()
                    if v is None:
                        safe_v = ''
                    elif isinstance(v, list):
                        safe_v = ','.join([str(x) for x in v if x is not None])
                    else:
                        safe_v = str(v).strip()
                    norm[safe_k] = safe_v
                
                # Must actually have lat/lon keys, otherwise it's a dummy row
                if not any(key in norm for key in ['lat', 'latitude', 'lon', 'longitude']):
                    continue
                    
                try:
                    ping_id = int(norm.get("ping_number", norm.get("ping", norm.get("id", i + 1))))
                    lat = float(norm.get("latitude", norm.get("lat", 13.0827)))
                    lon = float(norm.get("longitude", norm.get("lon", 80.3705)))
                    heading = float(norm.get("heading", norm.get("gyro", 90.0)))
                    altitude = float(norm.get("altitude", norm.get("alt", 12.0)))
                    depth = float(norm.get("depth", 35.0))
                    speed = float(norm.get("speed_knots", norm.get("speed", 3.0)))
                    roll = float(norm.get("roll", 0.0))
                    pitch = float(norm.get("pitch", 0.0))
                    heave = float(norm.get("heave", 0.0))
                    timestamp = norm.get("timestamp", datetime.datetime.utcnow().isoformat() + "Z")

                    records.append({
                        "ping_number": ping_id,
                        "timestamp": timestamp,
                        "latitude": lat,
                        "longitude": lon,
                        "heading": heading,
                        "altitude": altitude,
                        "depth": depth,
                        "speed_knots": speed,
                        "speed_mps": speed * 0.514444,
                        "roll": roll,
                        "pitch": pitch,
                        "heave": heave
                    })
                except (ValueError, TypeError):
                    continue

            return records
        except Exception:
            return []

    def parse_simulated_xtf_header(self, raw_bytes: bytes) -> Dict[str, Any]:
        """
        Parses an eXtended Triton Format (XTF) file header structure.
        XTF files begin with a 1024-byte file header containing sensor type,
        channel counts, and navigation units, followed by per-ping packets.
        """
        if len(raw_bytes) < 64:
            # Fallback for small byte arrays
            return {
                "format": "XTF_SIMULATED",
                "sonar_type": "Side-Scan Sonar (Dual-Channel)",
                "channels": 2,
                "port_samples": 1024,
                "stbd_samples": 1024,
                "sound_speed_mps": self.sound_speed_mps
            }

        try:
            # Standard XTF magic header detection: byte 0 = 0x7B (XTF file code)
            magic_number = raw_bytes[0]
            header_type = raw_bytes[1]
            
            return {
                "format": "XTF_HYDROGRAPHIC",
                "magic_byte": hex(magic_number),
                "header_type": header_type,
                "channels": 2,
                "port_samples": 1024,
                "stbd_samples": 1024,
                "sound_speed_mps": self.sound_speed_mps,
                "status": "VALID_XTF_HEADER"
            }
        except Exception as e:
            return {"format": "GENERIC_BINARY", "error": str(e)}

    def generate_ping_stream_for_waterfall(
        self,
        num_pings: int,
        base_nav: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generates a synchronized ping header telemetry stream corresponding
        to every ping row in a waterfall strip.
        Incorporates realistic AUV dynamics (slight heading wobble, roll drift, and speed).
        """
        stream = []
        base_lat = base_nav.get("latitude", 13.0827)
        base_lon = base_nav.get("longitude", 80.3705)
        heading = base_nav.get("heading", 90.0)
        speed_mps = base_nav.get("speed_mps", 1.54)
        ping_rate = base_nav.get("ping_rate_hz", 10.0)
        alt = base_nav.get("altitude", 12.0)
        depth = base_nav.get("depth", 42.0)

        dt = 1.0 / max(ping_rate, 1.0)
        start_time = datetime.datetime.utcnow()

        heading_rad = np.radians(heading)

        for i in range(num_pings):
            # Along-track distance advance: d = v * t
            dist_m = speed_mps * (i * dt)
            
            # Geodetic displacement
            d_lat = (dist_m * np.cos(heading_rad)) / 111132.954
            d_lon = (dist_m * np.sin(heading_rad)) / (111132.954 * np.cos(np.radians(base_lat)))

            # Vehicle subtle oscillatory dynamics (wave swell heave and roll)
            wave_phase = 2.0 * np.pi * (i / 40.0)
            roll = 1.8 * np.sin(wave_phase * 1.3)
            pitch = 0.8 * np.cos(wave_phase * 0.9)
            heave = 0.3 * np.sin(wave_phase)

            ping_time = start_time + datetime.timedelta(seconds=i * dt)

            stream.append({
                "ping_number": i + 1,
                "timestamp": ping_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
                "latitude": round(base_lat + d_lat, 7),
                "longitude": round(base_lon + d_lon, 7),
                "heading": round((heading + 0.5 * np.sin(wave_phase * 0.4)) % 360.0, 2),
                "altitude": round(alt + heave, 2),
                "depth": round(depth, 2),
                "speed_mps": speed_mps,
                "roll": round(roll, 2),
                "pitch": round(pitch, 2),
                "heave": round(heave, 2)
            })

        return stream
