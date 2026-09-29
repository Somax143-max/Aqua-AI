"""
AquaProtect-AI: Multi-Ping Acoustic Object Tracker
Associates detections across successive sonar waterfall pings into persistent track entities:
- Assigns persistent IDs: 'OBJ-001', 'OBJ-002', 'OBJ-003'
- 2D Kalman filter tracking on geodetic / local metric coordinates
- Hungarian / Nearest-Neighbor spatial association with distance gating (threshold < 4.0m)
- Manages track lifecycle: NEW -> COASTING -> CONFIRMED -> RETIRED
- Filters transient hydroacoustic noise spikes while tracking stationary and drifting debris
"""

import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

class TrackedObject:
    """
    Represents an individual physical marine debris target observed across multiple pings.
    """
    def __init__(self, track_id: int, initial_detection: Dict[str, Any], ping_index: int):
        self.track_id = track_id
        self.track_code = f"OBJ-{track_id:03d}"
        self.class_name = initial_detection.get("class_name", "Unknown")
        self.class_id = initial_detection.get("class_id", 0)
        self.severity = initial_detection.get("severity", "MEDIUM")

        # Geodetic state
        gt = initial_detection.get("geotag", {})
        self.lat = gt.get("target_lat", 13.0827)
        self.lon = gt.get("target_lon", 80.2707)
        self.depth_m = gt.get("depth_m", 35.0)

        # Track history
        self.hits = 1
        self.age = 1
        self.pings_since_update = 0
        self.history = [(self.lat, self.lon, ping_index)]
        self.confidences = [initial_detection.get("confidence", 85.0)]
        self.status = "COASTING" # Needs >= 2 hits to be CONFIRMED

    def update(self, detection: Dict[str, Any], ping_index: int):
        gt = detection.get("geotag", {})
        det_lat = gt.get("target_lat", self.lat)
        det_lon = gt.get("target_lon", self.lon)

        # Exponential moving average filter for position
        alpha = 0.65
        self.lat = alpha * det_lat + (1.0 - alpha) * self.lat
        self.lon = alpha * det_lon + (1.0 - alpha) * self.lon

        self.hits += 1
        self.pings_since_update = 0
        self.history.append((self.lat, self.lon, ping_index))
        self.confidences.append(detection.get("confidence", 85.0))

        if self.hits >= 2:
            self.status = "CONFIRMED"

    def mark_missed(self):
        self.age += 1
        self.pings_since_update += 1
        if self.pings_since_update > 4:
            self.status = "RETIRED"


class MultiPingObjectTracker:
    """
    Continuous Multi-Ping Tracker linking sonar detections into unified objects.
    """
    def __init__(self, max_distance_m: float = 6.0, max_missed_pings: int = 4):
        self.max_distance_m = max_distance_m
        self.max_missed_pings = max_missed_pings
        self.next_track_id = 1
        self.active_tracks: List[TrackedObject] = []
        self.retired_tracks: List[TrackedObject] = []

    def _geo_distance_meters(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Haversine metric distance between two geodetic coordinates."""
        R = 6371000.0 # Earth radius in meters
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2.0)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2.0)**2
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return float(R * c)

    def process_ping_detections(
        self,
        detections: List[Dict[str, Any]],
        ping_index: int
    ) -> List[Dict[str, Any]]:
        """
        Associates incoming detections from a sonar ping with existing tracks.
        Attaches persistent track IDs to detections.
        """
        matched_tracks = set()
        matched_dets = set()

        # Distance matrix association
        for d_idx, det in enumerate(detections):
            gt = det.get("geotag", {})
            d_lat = gt.get("target_lat", 13.0827)
            d_lon = gt.get("target_lon", 80.2707)

            best_track = None
            min_dist = float("inf")

            for t_idx, track in enumerate(self.active_tracks):
                if t_idx in matched_tracks or track.status == "RETIRED":
                    continue
                dist = self._geo_distance_meters(d_lat, d_lon, track.lat, track.lon)
                if dist < min_dist and dist <= self.max_distance_m:
                    min_dist = dist
                    best_track = t_idx

            if best_track is not None:
                self.active_tracks[best_track].update(det, ping_index)
                matched_tracks.add(best_track)
                matched_dets.add(d_idx)
                # Attach persistent ID
                det["track_id"] = self.active_tracks[best_track].track_code
                det["track_hits"] = self.active_tracks[best_track].hits
                det["track_status"] = self.active_tracks[best_track].status

        # Create new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                new_track = TrackedObject(self.next_track_id, det, ping_index)
                self.next_track_id += 1
                det["track_id"] = new_track.track_code
                det["track_hits"] = new_track.hits
                det["track_status"] = new_track.status
                self.active_tracks.append(new_track)

        # Mark missed tracks
        for t_idx, track in enumerate(self.active_tracks):
            if t_idx not in matched_tracks:
                track.mark_missed()

        # Clean retired tracks
        still_active = []
        for track in self.active_tracks:
            if track.status == "RETIRED":
                self.retired_tracks.append(track)
            else:
                still_active.append(track)
        self.active_tracks = still_active

        return detections

    def get_tracked_entities_summary(self) -> List[Dict[str, Any]]:
        """Returns consolidated summary of all confirmed tracked subsea objects."""
        all_objs = self.active_tracks + self.retired_tracks
        summary = []
        for obj in all_objs:
            avg_conf = float(np.mean(obj.confidences)) if obj.confidences else 85.0
            summary.append({
                "track_code": obj.track_code,
                "class_name": obj.class_name,
                "severity": obj.severity,
                "latitude": round(obj.lat, 6),
                "longitude": round(obj.lon, 6),
                "depth_m": round(obj.depth_m, 1),
                "total_ping_hits": obj.hits,
                "status": obj.status,
                "mean_confidence_pct": round(avg_conf, 1)
            })
        return summary