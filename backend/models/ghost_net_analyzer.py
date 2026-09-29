"""
Ghost Net Filament Analysis & Entanglement Risk Index (ERI) Engine
Specialized ecological analysis module for abandoned, lost, or discarded fishing gear (ALDFG):
- Filament Mesh Skeletonization & Thinning
- Entanglement Risk Index (ERI) Calculation (0 - 100)
- Autonomous ROV Cutter Waypoint Generation
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

class GhostNetAnalyzer:
    """
    Analyzes acoustic signatures of discarded synthetic fishing gear (ghost nets).
    Quantifies threat to marine biodiversity and plans ROV cutting interventions.
    """

    def __init__(self, coral_proximity_weight: float = 1.3):
        self.coral_proximity_weight = coral_proximity_weight

    def skeletonize_net_filament(self, net_crop: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Applies morphological thinning to extract the fine filament network
        of the entangled netting.
        Returns the binary skeleton image and total filament length (pixels).
        """
        if net_crop is None or net_crop.size == 0:
            return np.zeros((10, 10), dtype=np.uint8), 0.0

        if len(net_crop.shape) == 3:
            gray = cv2.cvtColor(net_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = net_crop.copy()

        gray_u8 = np.ascontiguousarray(np.clip(gray, 0, 255), dtype=np.uint8)
        h, w = gray_u8.shape[:2]
        if h < 6 or w < 6:
            return np.zeros_like(gray_u8), 0.0

        try:
            # Otsu thresholding for filament specular returns
            blur = cv2.GaussianBlur(gray_u8, (3, 3), 0)
            _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Morphological skeletonization
            skeleton = np.zeros_like(binary)
            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            temp = np.zeros_like(binary)
            eroded = binary.copy()

            for _ in range(25):
                cv2.erode(eroded, element, eroded)
                cv2.dilate(eroded, element, temp)
                cv2.subtract(binary, temp, temp)
                cv2.bitwise_or(skeleton, temp, skeleton)
                binary = eroded.copy()
                if cv2.countNonZero(binary) == 0:
                    break

            filament_length_px = float(cv2.countNonZero(skeleton))
            return skeleton, filament_length_px
        except (cv2.error, ValueError, TypeError):
            binary = (gray_u8 > 140).astype(np.uint8) * 255
            return binary, float(np.count_nonzero(binary))

    def calculate_entanglement_risk_index(
        self,
        dimensions: Dict[str, float],
        filament_length_px: float,
        water_depth_m: float = 35.0,
        is_coral_habitat: bool = True
    ) -> Dict[str, Any]:
        """
        Computes the Entanglement Risk Index (ERI, 0 - 100):
            ERI = (0.35 * S_area + 0.25 * S_density + 0.25 * S_relief + 0.15 * S_depth) * HabitatFactor
        """
        area_m2 = dimensions.get("length_m", 3.0) * dimensions.get("width_m", 4.0)
        relief_h = dimensions.get("relief_height_m", 1.5)

        # 1. Surface Footprint Score [0 - 100]
        s_area = min(100.0, (area_m2 / 120.0) * 100.0)

        # 2. Filament Mesh Density Score [0 - 100]
        density_ratio = filament_length_px / max(dimensions.get("width_m", 1.0) * 10.0, 1.0)
        s_density = min(100.0, (density_ratio / 8.0) * 100.0)

        # 3. Vertical Water Column Relief Score [0 - 100]
        # Nets rising high into water column entrap dolphins, turtles, and pelagic sharks
        s_relief = min(100.0, (relief_h / 4.5) * 100.0)

        # 4. Shallow Water Phototrophic Zone Factor [0 - 100]
        # Shallow coral reef waters (<40m) carry highest ecological devastation
        s_depth = max(0.0, min(100.0, (1.0 - (water_depth_m / 80.0)) * 100.0))

        # Composite raw score
        raw_eri = (0.35 * s_area + 0.25 * s_density + 0.25 * s_relief + 0.15 * s_depth)

        # Habitat vulnerability multiplier
        habitat_mult = self.coral_proximity_weight if is_coral_habitat else 1.0
        final_eri = round(min(100.0, raw_eri * habitat_mult), 1)

        # Categorization
        if final_eri >= 75.0:
            tier = "EXTREME_ECOLOGICAL_THREAT"
            action = "Immediate ROV cutting intervention required within 24 hours"
        elif final_eri >= 50.0:
            tier = "HIGH_THREAT"
            action = "Scheduled diver/ROV recovery in next ocean survey cycle"
        else:
            tier = "MODERATE_THREAT"
            action = "Monitor benthic drift and plan mechanical winch recovery"

        return {
            "eri_score": final_eri,
            "threat_tier": tier,
            "estimated_footprint_m2": round(area_m2, 1),
            "filament_complexity_index": round(s_density, 1),
            "vertical_entanglement_hazard": round(s_relief, 1),
            "action_recommendation": action
        }

    def generate_rov_cutting_waypoints(
        self,
        skeleton: np.ndarray,
        target_lat: float,
        target_lon: float,
        relief_h_m: float,
        max_waypoints: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Computes 3D coordinates for ROV hydraulic cutter arms to sever net tension lines.
        Identifies high-curvature branching nodes in the skeletonized filament graph.
        """
        h, w = skeleton.shape
        if h < 4 or w < 4 or np.sum(skeleton) == 0:
            return []

        # Find junction/branch points using 3x3 convolution
        kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
        filtered = cv2.filter2D((skeleton > 0).astype(np.uint8), -1, kernel)
        
        # Branch points have value >= 13 (center pixel + at least 3 neighbors)
        branch_pts = np.argwhere(filtered >= 13)

        waypoints = []
        if len(branch_pts) > 0:
            step = max(1, len(branch_pts) // max_waypoints)
            selected_pts = branch_pts[::step][:max_waypoints]
        else:
            # Fallback to evenly spaced points along net perimeter
            pts = np.argwhere(skeleton > 0)
            if len(pts) > 0:
                step = max(1, len(pts) // max_waypoints)
                selected_pts = pts[::step][:max_waypoints]
            else:
                selected_pts = np.array([[h//2, w//2]])

        for i, (py, px) in enumerate(selected_pts):
            # Normalized offset
            dx_m = (px - w / 2.0) * 0.1
            dy_m = (py - h / 2.0) * 0.1
            # Elevation above bed: mid-height of net
            cutter_z_m = round(relief_h_m * 0.5, 2)

            # Geodetic small displacement
            d_lat = dy_m / 111132.954
            d_lon = dx_m / (111132.954 * np.cos(np.radians(target_lat)))

            waypoints.append({
                "waypoint_id": f"ROV-CUT-{i+1}",
                "cutter_lat": round(target_lat + d_lat, 7),
                "cutter_lon": round(target_lon + d_lon, 7),
                "depth_offset_above_bed_m": cutter_z_m,
                "cutting_angle_deg": round(float(np.arctan2(dy_m, dx_m) * 180.0 / np.pi), 1),
                "tool_type": "HYDRAULIC_SHEAR_JAW"
            })

        return waypoints

    def execute_specialist_pipeline(
        self,
        crop: np.ndarray,
        dimensions: Dict[str, float],
        target_lat: float,
        target_lon: float,
        water_depth_m: float = 35.0,
        is_coral_habitat: bool = True
    ) -> Dict[str, Any]:
        """
        Executes the dedicated Ghost Net Specialist Pipeline:
        Crop -> Skeletonization -> Entanglement Risk -> Severity Score -> ROV Shear Waypoints.
        """
        skeleton, fil_len = self.skeletonize_net_filament(crop)
        eri_info = self.calculate_entanglement_risk_index(
            dimensions, fil_len, water_depth_m=water_depth_m, is_coral_habitat=is_coral_habitat
        )
        waypoints = self.generate_rov_cutting_waypoints(
            skeleton, target_lat, target_lon, dimensions.get("relief_height_m", 1.5)
        )
        return {
            "specialist_pipeline": "GHOST_NET_SPECIALIST_V2",
            "filament_length_px": round(fil_len, 1),
            "entanglement_risk": eri_info,
            "rov_cutting_waypoints": waypoints,
            "hazard_tier": eri_info["threat_tier"],
            "action": eri_info["action_recommendation"]
        }

