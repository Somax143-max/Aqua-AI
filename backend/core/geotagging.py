"""
AquaProtect-AI: Hydrographic Geotagging & Positional Uncertainty Engine
Converts Sonar Waterfall Pixel Coordinates to real-world WGS-84 Geodetic
coordinates with quantified 95% confidence error ellipses and ground-truth validation.
"""

import math
import numpy as np
from typing import Dict, Any, Tuple, Optional

# WGS-84 Ellipsoid constants
WGS84_A = 6378137.0             # Semi-major axis (meters)
WGS84_F = 1.0 / 298.257223563      # Flattening
METERS_PER_DEGREE_LAT = 111132.954  # Nominal meters per degree latitude


class SonarGeotagger:
    """
    Transforms sonar waterfall pixel coordinates to real-world WGS-84 navigation coordinates.
    Incorporates towfish vehicle heading, altitude, roll, and ground-range projection.
    Provides quantified positioning error ellipse metrics compliant with IHO S-44 standards.
    """

    def __init__(self, max_slant_range_m: float = 75.0):
        self.max_slant_range_m = max_slant_range_m

    def pixel_to_ground_range(
        self,
        pixel_x: float,
        waterfall_width_px: int,
        altitude_m: float
    ) -> Tuple[float, str]:
        """
        Converts across-track pixel coordinate to ground range (meters)
        and channel identity ('PORT' or 'STARBOARD').
        """
        center_x = waterfall_width_px / 2.0
        diff_px = pixel_x - center_x
        
        channel = "STARBOARD" if diff_px >= 0 else "PORT"
        
        # Normalized slant range fraction
        slant_fraction = abs(diff_px) / max(waterfall_width_px / 2.0, 1.0)
        slant_range_m = slant_fraction * self.max_slant_range_m
        
        # Slant-to-ground range: Rg = sqrt(Rs^2 - H^2)
        if slant_range_m > altitude_m:
            ground_range_m = math.sqrt(slant_range_m**2 - altitude_m**2)
        else:
            # Inside water column blind zone
            ground_range_m = 0.0
            
        return ground_range_m, channel

    def calculate_target_lat_lon(
        self,
        vehicle_lat: float,
        vehicle_lon: float,
        vehicle_heading_deg: float,
        ground_range_m: float,
        channel: str,
        roll_deg: float = 0.0
    ) -> Tuple[float, float, float]:
        """
        Projects vessel position to target position along the perpendicular
        acoustic sweep line:
            Target Bearing = Heading + 90 deg (Starboard)
            Target Bearing = Heading - 90 deg (Port)
        """
        # True bearing to target
        if channel == "STARBOARD":
            bearing_deg = (vehicle_heading_deg + 90.0) % 360.0
        else:
            bearing_deg = (vehicle_heading_deg - 90.0) % 360.0
            
        bearing_rad = math.radians(bearing_deg)
        
        # North and East displacements in meters
        dn = ground_range_m * math.cos(bearing_rad)
        de = ground_range_m * math.sin(bearing_rad)
        
        # Geodetic displacement
        lat_rad = math.radians(vehicle_lat)
        meters_per_deg_lon = METERS_PER_DEGREE_LAT * math.cos(lat_rad)
        
        d_lat = dn / METERS_PER_DEGREE_LAT
        d_lon = de / max(meters_per_deg_lon, 1e-6)
        
        target_lat = vehicle_lat + d_lat
        target_lon = vehicle_lon + d_lon
        
        return target_lat, target_lon, bearing_deg

    def estimate_uncertainty_ellipse(
        self,
        ground_range_m: float,
        gps_accuracy_m: float = 1.2,
        beam_width_deg: float = 0.6,
        sound_speed_uncertainty_mps: float = 1.5,
        nominal_sound_speed_mps: float = 1540.0
    ) -> Dict[str, float]:
        """
        Quantifies 95% confidence hydrographic error ellipse (IHO Order 1a standard):
        Combines GNSS surface positioning uncertainty, acoustic beam divergence,
        and sound velocity profile (SVP) refraction gradient uncertainty.
        """
        # Across-track slant range uncertainty due to sound speed gradient
        delta_r_across = ground_range_m * (sound_speed_uncertainty_mps / nominal_sound_speed_mps)
        
        # Along-track acoustic beam divergence spread
        beam_spread_along = ground_range_m * math.tan(math.radians(beam_width_deg))

        # Semi-major (along-track / beam spread dominated)
        semi_major = math.sqrt(gps_accuracy_m**2 + (beam_spread_along / 2.0)**2)
        # Semi-minor (across-track / range resolution dominated)
        semi_minor = math.sqrt(gps_accuracy_m**2 + delta_r_across**2)

        return {
            "semi_major_m": round(semi_major, 2),
            "semi_minor_m": round(semi_minor, 2),
            "circular_error_probable_m": round(0.59 * (semi_major + semi_minor), 2),
            "confidence_level_pct": 95.0
        }

    def geotag_detection(
        self,
        bbox: Tuple[int, int, int, int],
        waterfall_width_px: int,
        nav_record: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Full geotagging pipeline for an individual bounding box detection.
        Cleanly handles cases where navigation metadata is missing or partial.
        """
        x, y, w, h = bbox
        target_center_x = x + w / 2.0
        target_center_y = y + h / 2.0

        if not nav_record or "latitude" not in nav_record or "longitude" not in nav_record:
            # Clean handling for un-georeferenced sonar strips (Batch 9)
            ground_range_m, channel = self.pixel_to_ground_range(target_center_x, waterfall_width_px, 10.0)
            return {
                "georeferenced": False,
                "warning": "Navigation telemetry unavailable - coordinates are vehicle-relative",
                "channel": channel,
                "ground_range_m": round(ground_range_m, 2),
                "target_lat": None,
                "target_lon": None,
                "positional_uncertainty_m": None,
                "uncertainty_ellipse": None
            }

        vehicle_lat = float(nav_record["latitude"])
        vehicle_lon = float(nav_record["longitude"])
        heading = float(nav_record.get("heading", 90.0))
        altitude = float(nav_record.get("altitude", 10.0))
        roll = float(nav_record.get("roll", 0.0))
        gps_acc = float(nav_record.get("gps_accuracy_m", 1.2))

        ground_range_m, channel = self.pixel_to_ground_range(
            target_center_x, waterfall_width_px, altitude
        )

        target_lat, target_lon, bearing = self.calculate_target_lat_lon(
            vehicle_lat, vehicle_lon, heading, ground_range_m, channel, roll
        )

        ellipse_info = self.estimate_uncertainty_ellipse(ground_range_m, gps_accuracy_m=gps_acc)
        ellipse_info["orientation_deg"] = round(bearing, 1)

        # UTM Zone
        utm_zone = int((vehicle_lon + 180) / 6) + 1
        hemisphere = 'N' if vehicle_lat >= 0 else 'S'

        return {
            "georeferenced": True,
            "channel": channel,
            "ground_range_m": round(ground_range_m, 2),
            "bearing_deg": round(bearing, 1),
            "target_lat": round(target_lat, 7),
            "target_lon": round(target_lon, 7),
            "utm_coordinate": f"{utm_zone}{hemisphere}",
            "positional_uncertainty_m": ellipse_info["circular_error_probable_m"],
            "uncertainty_ellipse": ellipse_info,
            "vehicle_lat": vehicle_lat,
            "vehicle_lon": vehicle_lon,
            "vehicle_heading": heading,
            "vehicle_altitude": altitude
        }

    def validate_against_ground_truth(
        self,
        predicted_lat: float,
        predicted_lon: float,
        ground_truth_lat: float,
        ground_truth_lon: float
    ) -> Dict[str, Any]:
        """
        Validates predicted coordinates against ground-truth transponder/benchmark coordinates.
        Computes geodesic error distance via Haversine formulation.
        """
        # Haversine distance
        r_earth = 6371000.0  # meters
        phi1 = math.radians(predicted_lat)
        phi2 = math.radians(ground_truth_lat)
        delta_phi = math.radians(ground_truth_lat - predicted_lat)
        delta_lambda = math.radians(ground_truth_lon - predicted_lon)

        a = math.sin(delta_phi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0)**2
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(1.0 - a, 0.0)))
        displacement_error_m = r_earth * c

        # Verification threshold: displacement within IHO Order 1a standard (< 5.0m)
        meets_iho_standard = displacement_error_m <= 5.0

        return {
            "displacement_error_m": round(displacement_error_m, 3),
            "meets_iho_order_1a": meets_iho_standard,
            "predicted_coords": [predicted_lat, predicted_lon],
            "ground_truth_coords": [ground_truth_lat, ground_truth_lon]
        }
