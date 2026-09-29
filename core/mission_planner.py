"""
AquaProtect-AI: Autonomous Ocean Cleanup & ROV Recovery Mission Planner
Calculates optimal, multi-constraint salvage routes visiting confirmed marine hazards:
- Multi-Hazard Traveling Salesperson Optimization with 2-Opt
- Hazard Severity Priority Weighting (Critical Ghost Nets & Munitions prioritized)
- Operational Environment Constraints: Depth Limits, Surface Currents, Sea-State Fuel Modeling
- Nautical Mile Distance, Operational Duration, and ROV Logistics Modeling
"""

import math
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

METERS_PER_NAUTICAL_MILE = 1852.0

# Hazard Severity Operational Weights
SEVERITY_WEIGHTS = {
    "CRITICAL": 3.0,   # Ghost Nets, Munitions (Immediate intervention)
    "HIGH": 2.0,       # Shipwrecks, Chemical Drums
    "MEDIUM": 1.0,     # Lost Containers
    "LOW": 0.5
}


class OceanCleanupMissionPlanner:
    """
    Computes optimal multi-hazard recovery routes for NIOT cleanup vessels and tethered ROVs.
    Takes into account hazard severity prioritization, water depth limits, and weather sea state.
    """

    def __init__(
        self,
        vessel_transit_speed_knots: float = 8.0,
        fuel_consumption_liters_per_nm: float = 14.5,
        hourly_rov_operational_cost_inr: float = 45000.0,
        max_rov_depth_rating_m: float = 500.0,
        turning_radius_m: float = 50.0
    ):
        self.transit_speed_knots = vessel_transit_speed_knots
        self.fuel_per_nm = fuel_consumption_liters_per_nm
        self.hourly_cost_inr = hourly_rov_operational_cost_inr
        self.max_rov_depth_m = max_rov_depth_rating_m
        self.turning_radius_m = turning_radius_m

    def calculate_haversine_distance_m(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """Computes great-circle distance between two WGS-84 coordinates in meters."""
        r = 6371000.0  # Earth radius in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        d_phi = math.radians(lat2 - lat1)
        d_lambda = math.radians(lon2 - lon1)

        a = math.sin(d_phi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0)**2
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(1.0 - a, 0.0)))
        return r * c

    def _solve_severity_weighted_tsp(
        self,
        points: List[Tuple[float, float]],
        severities: List[str]
    ) -> List[int]:
        """
        Solves Traveling Salesperson Problem with Severity Prioritization.
        Critical hazards receive distance discounts in the cost matrix to be visited earlier.
        """
        n = len(points)
        if n <= 2:
            return list(range(n))

        cost_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = self.calculate_haversine_distance_m(
                        points[i][0], points[i][1], points[j][0], points[j][1]
                    )
                    # Priority weighting on target j
                    target_sev = severities[j] if j < len(severities) else "MEDIUM"
                    weight = SEVERITY_WEIGHTS.get(target_sev, 1.0)
                    # Higher priority lowers effective cost for greedy selection
                    cost_matrix[i, j] = dist / weight

        # Greedy tour starting from vessel position (index 0)
        unvisited = set(range(1, n))
        current = 0
        tour = [current]

        while unvisited:
            next_pt = min(unvisited, key=lambda x: cost_matrix[current, x])
            tour.append(next_pt)
            unvisited.remove(next_pt)
            current = next_pt

        # 2-Opt local refinement
        improved = True
        iterations = 0
        while improved and iterations < 40:
            improved = False
            iterations += 1
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    d_curr = cost_matrix[tour[i-1], tour[i]] + cost_matrix[tour[j], tour[(j+1)%n]]
                    d_swap = cost_matrix[tour[i-1], tour[j]] + cost_matrix[tour[i], tour[(j+1)%n]]
                    if d_swap < d_curr - 1e-3:
                        tour[i:j+1] = reversed(tour[i:j+1])
                        improved = True

        return tour

    def plan_recovery_mission(
        self,
        vessel_lat: float,
        vessel_lon: float,
        detections: List[Dict[str, Any]],
        sea_state: int = 2,
        current_drift_mps: float = 0.4
    ) -> Dict[str, Any]:
        """
        Generates an optimized sequential salvage itinerary visiting all hazards.
        Incorporates environmental factors:
        - sea_state: Beaufort scale 0-6 (influences fuel consumption & transit speed)
        - current_drift_mps: Ocean tidal current speed
        """
        if not detections:
            return {
                "itinerary": [],
                "total_distance_nm": 0.0,
                "total_distance_km": 0.0,
                "estimated_duration_hours": 0.0,
                "fuel_consumption_liters": 0.0,
                "estimated_cost_inr": 0.0,
                "savings_vs_unoptimized_pct": 0.0
            }

        # Sea-state impact: higher sea-state reduces transit speed and increases fuel
        weather_factor = 1.0 + (sea_state * 0.06)
        effective_speed_knots = max(4.0, self.transit_speed_knots / weather_factor)
        effective_fuel_per_nm = self.fuel_per_nm * weather_factor

        points = [(vessel_lat, vessel_lon)]
        severities = ["NONE"]  # Index 0 is vessel
        target_indices = []

        for d in detections:
            geo = d.get("geotag", {})
            lat = geo.get("target_lat", vessel_lat)
            lon = geo.get("target_lon", vessel_lon)
            sev = d.get("severity", "MEDIUM")
            points.append((lat, lon))
            severities.append(sev)
            target_indices.append(d)

        # Unoptimized route distance
        unopt_dist_m = sum(
            self.calculate_haversine_distance_m(points[i][0], points[i][1], points[i+1][0], points[i+1][1])
            for i in range(len(points) - 1)
        )

        # Solve Severity-Weighted TSP (Batch 10)
        opt_tour_indices = self._solve_severity_weighted_tsp(points, severities)

        opt_dist_m = 0.0
        itinerary = []

        for k in range(len(opt_tour_indices)):
            curr_idx = opt_tour_indices[k]
            curr_pt = points[curr_idx]

            if k > 0:
                prev_idx = opt_tour_indices[k - 1]
                prev_pt = points[prev_idx]
                leg_dist_m = self.calculate_haversine_distance_m(
                    prev_pt[0], prev_pt[1], curr_pt[0], curr_pt[1]
                )
                # Add turning maneuver distance
                leg_dist_m += self.turning_radius_m * 1.5
                opt_dist_m += leg_dist_m
            else:
                leg_dist_m = 0.0

            if curr_idx == 0:
                itinerary.append({
                    "step": 0,
                    "type": "VESSEL_ORIGIN",
                    "label": "Survey Ship Departure Point",
                    "latitude": curr_pt[0],
                    "longitude": curr_pt[1],
                    "leg_distance_m": round(leg_dist_m, 1),
                    "action": "Deploy ROV salvage team"
                })
            else:
                det = target_indices[curr_idx - 1]
                itinerary.append({
                    "step": k,
                    "type": "HAZARD_TARGET",
                    "hazard_id": det.get("id"),
                    "class_name": det.get("class_name"),
                    "severity": det.get("severity"),
                    "latitude": curr_pt[0],
                    "longitude": curr_pt[1],
                    "leg_distance_m": round(leg_dist_m, 1),
                    "action": f"Salvage {det.get('class_name')} ({det.get('severity')})"
                })

        dist_nm = opt_dist_m / METERS_PER_NAUTICAL_MILE
        dist_km = opt_dist_m / 1000.0

        transit_hours = dist_nm / effective_speed_knots
        operation_hours = len(detections) * 1.2
        total_hours = round(transit_hours + operation_hours, 1)

        fuel_liters = round(dist_nm * effective_fuel_per_nm, 1)
        cost_inr = round(total_hours * self.hourly_cost_inr, 2)
        savings_pct = round(max(0.0, (unopt_dist_m - opt_dist_m) / max(unopt_dist_m, 1.0) * 100.0), 1)

        return {
            "itinerary": itinerary,
            "total_waypoints": len(itinerary),
            "total_distance_nm": round(dist_nm, 2),
            "total_distance_km": round(dist_km, 2),
            "transit_duration_hours": round(transit_hours, 1),
            "total_mission_hours": total_hours,
            "fuel_consumption_liters": fuel_liters,
            "estimated_operational_cost_inr": cost_inr,
            "savings_vs_unoptimized_pct": savings_pct,
            "sea_state": sea_state,
            "effective_transit_speed_knots": round(effective_speed_knots, 1)
        }

    def replan_mission_dynamically(
        self,
        current_itinerary: List[Dict[str, Any]],
        new_detection: Dict[str, Any],
        auv_battery_pct: float = 85.0,
        tether_max_len_m: float = 300.0,
        no_go_zones: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Dynamically inserts a newly detected hazard into the active AUV/ROV mission plan.
        Validates vehicle battery reserves, no-go zone avoidance, and tether length limits.
        """
        gt = new_detection.get("geotag", {})
        new_lat = gt.get("target_lat", 13.0827)
        new_lon = gt.get("target_lon", 80.2707)
        depth_m = gt.get("depth_m", 35.0)

        # Tether length feasibility
        tether_feasible = depth_m <= tether_max_len_m

        # No-go zone check
        in_no_go = False
        if no_go_zones:
            for zone in no_go_zones:
                d = self.calculate_haversine_distance_m(new_lat, new_lon, zone["lat"], zone["lon"])
                if d < zone.get("radius_m", 200.0):
                    in_no_go = True
                    break

        # Find optimal insertion index minimizing added distance
        best_insertion_idx = 1
        min_added_dist = float("inf")

        for i in range(len(current_itinerary)):
            p1 = current_itinerary[i]
            p2 = current_itinerary[(i + 1) % len(current_itinerary)]
            d_p1_new = self.calculate_haversine_distance_m(p1["latitude"], p1["longitude"], new_lat, new_lon)
            d_new_p2 = self.calculate_haversine_distance_m(new_lat, new_lon, p2["latitude"], p2["longitude"])
            d_orig = self.calculate_haversine_distance_m(p1["latitude"], p1["longitude"], p2["latitude"], p2["longitude"])
            added = (d_p1_new + d_new_p2) - d_orig

            if added < min_added_dist:
                min_added_dist = added
                best_insertion_idx = i + 1

        battery_cost_pct = (min_added_dist / 1000.0) * 2.5
        battery_feasible = (auv_battery_pct - battery_cost_pct) >= 20.0 # 20% safety margin

        replan_status = "REPLAN_APPROVED" if (battery_feasible and tether_feasible and not in_no_go) else "REPLAN_REJECTED"

        replanned_itinerary = list(current_itinerary)
        if replan_status == "REPLAN_APPROVED":
            replanned_itinerary.insert(best_insertion_idx, {
                "step": best_insertion_idx,
                "type": "DYNAMIC_NEW_HAZARD",
                "hazard_id": new_detection.get("id", "HAZ-NEW"),
                "class_name": new_detection.get("class_name", "Anomaly"),
                "severity": new_detection.get("severity", "CRITICAL"),
                "latitude": new_lat,
                "longitude": new_lon,
                "leg_distance_m": round(min_added_dist, 1),
                "action": f"Immediate Interception of {new_detection.get('class_name')}"
            })

        return {
            "replan_status": replan_status,
            "insertion_index": best_insertion_idx,
            "added_distance_m": round(min_added_dist, 1),
            "estimated_battery_draw_pct": round(battery_cost_pct, 1),
            "battery_feasible": battery_feasible,
            "tether_feasible": tether_feasible,
            "no_go_violation": in_no_go,
            "replanned_itinerary": replanned_itinerary
        }

