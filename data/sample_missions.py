"""
Pre-Configured Ocean Survey Missions & Indian Maritime Datasets
Tailored for MoES & NIOT operational scenarios:
- Bay of Bengal (Chennai Offshore - NIOT Deep Sea Trial)
- Arabian Sea (Mumbai High Logistics Corridor)
- Palk Strait & Gulf of Mannar (Coral Marine Biosphere)
"""

import numpy as np
from typing import Dict, Any, List
from data.sonar_generator import SonarDataGenerator

SAMPLE_MISSIONS: Dict[str, Dict[str, Any]] = {
    "mission_chennai_niot": {
        "title": "Mission Alpha: Bay of Bengal (Chennai Offshore - NIOT Mooring Testbed)",
        "location_name": "Bay of Bengal, 18 NM East of Chennai Harbour",
        "agency": "Ministry of Earth Sciences (MoES) / NIOT",
        "vessel_name": "ORV Sagar Nidhi (AUV Matsya-6000 Escort)",
        "seabed": "sand_ripples",
        "start_lat": 13.0827,
        "start_lon": 80.3705,
        "heading": 85.0,
        "depth_m": 42.0,
        "altitude_m": 12.0,
        "speed_knots": 3.0,
        "speed_mps": 1.54,
        "ping_rate_hz": 10.0,
        "max_range_m": 75.0,
        "targets": [
            {"type": "ghost_net", "y": 130, "x": 680, "scale": 1.15},
            {"type": "pipeline", "y": 280, "x": 340, "scale": 1.2},
            {"type": "drum", "y": 450, "x": 740, "scale": 1.0}
        ],
        "description": "High-priority acoustic search along Chennai continental shelf for discarded gillnets and subsea communication cable inspection."
    },
    "mission_mumbai_high": {
        "title": "Mission Bravo: Arabian Sea (Mumbai High Shipping & Energy Corridor)",
        "location_name": "Arabian Sea, Mumbai High Offshore Platform Zone",
        "agency": "MoES / National Institute of Ocean Technology",
        "vessel_name": "BTV Sagar Manjusha",
        "seabed": "rocky_reef",
        "start_lat": 19.4167,
        "start_lon": 71.3833,
        "heading": 178.0,
        "depth_m": 68.0,
        "altitude_m": 14.5,
        "speed_knots": 3.2,
        "speed_mps": 1.65,
        "ping_rate_hz": 10.0,
        "max_range_m": 85.0,
        "targets": [
            {"type": "container", "y": 140, "x": 310, "scale": 1.1},
            {"type": "shipwreck", "y": 360, "x": 750, "scale": 1.25},
            {"type": "mine", "y": 490, "x": 260, "scale": 0.95}
        ],
        "description": "Subsea hazard clearance for navigation safety; survey of jettisoned shipping containers and derelict wreckage."
    },
    "mission_gulf_mannar": {
        "title": "Mission Charlie: Palk Strait (Gulf of Mannar Marine Biosphere Reserve)",
        "location_name": "Gulf of Mannar, Mandapam Coastal Shelf",
        "agency": "MoES / NIOT Marine Protected Area Conservation Team",
        "vessel_name": "CRV Sagar Paschimi",
        "seabed": "muddy_flat",
        "start_lat": 9.2800,
        "start_lon": 79.1250,
        "heading": 215.0,
        "depth_m": 18.0,
        "altitude_m": 8.0,
        "speed_knots": 2.5,
        "speed_mps": 1.28,
        "ping_rate_hz": 12.0,
        "max_range_m": 50.0,
        "targets": [
            {"type": "ghost_net", "y": 160, "x": 330, "scale": 1.2},
            {"type": "ghost_net", "y": 380, "x": 670, "scale": 1.1},
            {"type": "drum", "y": 480, "x": 370, "scale": 1.0}
        ],
        "description": "Ecological crisis survey targeting lethal ghost fishing gear entangled on biogenic coral colonies."
    },
    "mission_andaman_trench": {
        "title": "Mission Delta: Andaman & Nicobar Deep Trench (Deep Ocean Mission)",
        "location_name": "Andaman Sea, Abyssal Trench Flank (12 NM off Port Blair)",
        "agency": "MoES / NIOT Deep Ocean Mission (Matsya-6000 Submersible Support)",
        "vessel_name": "ORV Sagar Kanya (Subsea Crane & Deep Tow SSS)",
        "seabed": "rocky_reef",
        "start_lat": 11.6234,
        "start_lon": 92.7265,
        "heading": 135.0,
        "depth_m": 85.0,
        "altitude_m": 16.0,
        "speed_knots": 3.0,
        "speed_mps": 1.54,
        "ping_rate_hz": 10.0,
        "max_range_m": 90.0,
        "targets": [
            {"type": "pipeline", "y": 170, "x": 720, "scale": 1.3},
            {"type": "mine", "y": 320, "x": 310, "scale": 1.1},
            {"type": "shipwreck", "y": 470, "x": 680, "scale": 1.3}
        ],
        "description": "Deep ocean survey for subsea optical telecommunication cables and WWII munitions clearance."
    },
    "mission_cochin_harbor": {
        "title": "Mission Echo: Cochin International Shipping Channel (Harbour Logistics)",
        "location_name": "Arabian Sea, Cochin Port Approach Channel & Anchorage",
        "agency": "MoES / NIOT Coastal & Estuarine Survey Division",
        "vessel_name": "CRV Sagar Tara",
        "seabed": "sand_ripples",
        "start_lat": 9.9674,
        "start_lon": 76.2215,
        "heading": 290.0,
        "depth_m": 24.0,
        "altitude_m": 9.5,
        "speed_knots": 2.8,
        "speed_mps": 1.44,
        "ping_rate_hz": 12.0,
        "max_range_m": 60.0,
        "targets": [
            {"type": "container", "y": 150, "x": 690, "scale": 1.2},
            {"type": "drum", "y": 340, "x": 280, "scale": 1.05},
            {"type": "ghost_net", "y": 460, "x": 730, "scale": 1.1}
        ],
        "description": "High-traffic navigational fairway inspection for dropped cargo containers and propeller-fouling debris."
    }
}

def load_mission_data(mission_key: str) -> Dict[str, Any]:
    """
    Generates and returns complete simulated mission imagery and telemetry.
    """
    if mission_key not in SAMPLE_MISSIONS:
        mission_key = "mission_chennai_niot"
        
    cfg = SAMPLE_MISSIONS[mission_key]
    gen = SonarDataGenerator(
        width=1000,
        height=600,
        altitude_m=cfg["altitude_m"],
        max_range_m=cfg["max_range_m"]
    )
    
    sonar_img, ground_truth = gen.generate_mission_waterfall(
        seabed=cfg["seabed"],
        targets=cfg["targets"]
    )
    
    nav_telemetry = {
        "mission_title": cfg["title"],
        "location_name": cfg["location_name"],
        "agency": cfg["agency"],
        "vessel_name": cfg["vessel_name"],
        "latitude": cfg["start_lat"],
        "longitude": cfg["start_lon"],
        "heading": cfg["heading"],
        "depth": cfg["depth_m"],
        "altitude": cfg["altitude_m"],
        "speed_mps": cfg["speed_mps"],
        "speed_knots": cfg["speed_knots"],
        "ping_rate_hz": cfg["ping_rate_hz"],
        "max_range_m": cfg["max_range_m"],
        "gps_accuracy_m": 1.1,
        "seabed_type": cfg["seabed"]
    }
    
    return {
        "config": cfg,
        "sonar_image": sonar_img,
        "ground_truth": ground_truth,
        "nav_telemetry": nav_telemetry
    }
