"""
Core Sonar Physics Engine
Implements hydroacoustic principles for Side-Scan Sonar (SSS):
- Slant-Range to Ground-Range Correction (SRC)
- Water Column Nadir Detection
- Acoustic Shadow Target Height Estimation
- Time-Varying Gain (TVG) Spherical Spreading Compensation
- Grazing Angle and Insonification Geometry
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional

SOUND_SPEED_SEAWATER = 1500.0  # m/s nominal acoustic propagation speed in seawater

class SonarPhysics:
    """
    Hydroacoustic physics modeling for Side-Scan Sonar systems.
    Supports dual-channel (Port/Starboard) geometry.
    """

    def __init__(
        self,
        sound_speed: float = SOUND_SPEED_SEAWATER,
        max_slant_range: float = 75.0,  # meters per channel
        absorption_coeff: float = 0.08,  # dB/m (~450 kHz)
        samples_per_channel: int = 1000
    ):
        self.sound_speed = sound_speed
        self.max_slant_range = max_slant_range
        self.absorption_coeff = absorption_coeff
        self.samples_per_channel = samples_per_channel
        self.range_res = max_slant_range / samples_per_channel

    def calculate_shadow_height(
        self,
        altitude: float,
        shadow_length_m: float,
        slant_range_m: float
    ) -> float:
        """
        Calculates the physical relief height (h) of a submerged object
        above the seafloor using acoustic shadow geometry:
            h = (H * L_s) / (R_s + L_s)  or  h = (H * L_s) / R_far
        Where:
            H = Towfish/AUV altitude above seafloor (m)
            L_s = Acoustic shadow length (m)
            R_s = Slant range from transducer to shadow origin (m)
        """
        if altitude <= 0 or shadow_length_m <= 0 or slant_range_m <= 0:
            return 0.0
        
        # Rigorous geometric formula: h = (H * Ls) / (Rs + Ls)
        # to account for the far boundary of the shadow
        height = (altitude * shadow_length_m) / (slant_range_m + shadow_length_m)
        return float(np.clip(height, 0.0, altitude * 0.95))

    def detect_water_column(
        self,
        ping_channel: np.ndarray,
        threshold_factor: float = 2.2
    ) -> Tuple[int, float]:
        """
        Detects the bottom return (first seabed echo) separating the
        uninsonified nadir water column from seafloor backscatter.
        
        Returns:
            first_bottom_idx: Sample index of first bottom return
            altitude_m: Estimated towfish altitude above seafloor (m)
        """
        if len(ping_channel) < 20:
            return 0, 0.0
            
        # Moving standard deviation / energy gradient
        window = 15
        smoothed = np.convolve(ping_channel.astype(float), np.ones(window)/window, mode='valid')
        diff = np.diff(smoothed)
        
        baseline_noise = np.mean(ping_channel[:min(len(ping_channel)//6, 50)])
        std_noise = np.std(ping_channel[:min(len(ping_channel)//6, 50)]) + 1e-6
        
        # Detection trigger when echo exceeds baseline noise by threshold
        detection_mask = np.where(smoothed > (baseline_noise + threshold_factor * std_noise))[0]
        
        if len(detection_mask) > 0:
            first_idx = int(detection_mask[0] + window // 2)
            first_idx = max(5, min(first_idx, len(ping_channel) - 1))
        else:
            # Fallback to default heuristic (approx 15% range)
            first_idx = int(len(ping_channel) * 0.15)
            
        altitude_m = first_idx * self.range_res
        return first_idx, altitude_m

    def slant_to_ground_range(
        self,
        sonar_strip: np.ndarray,
        altitude_m: float
    ) -> np.ndarray:
        """
        Performs Slant-Range Correction (SRC) on a 2D sonar waterfall strip.
        Rectifies geometric slant range compression:
            R_ground = sqrt(R_slant^2 - Altitude^2)
        Sonar strip shape: (num_pings, 2 * samples_per_channel)
        """
        orig_w = sonar_strip.shape[1]
        is_odd = (orig_w % 2 != 0)
        if is_odd:
            sonar_strip = sonar_strip[:, :-1]
            
        height, total_samples = sonar_strip.shape
        channel_samples = total_samples // 2
        
        # Prepare output grid
        corrected = np.zeros_like(sonar_strip)
        
        # Precompute slant range vector for one channel
        slant_ranges = np.linspace(0.0, self.max_slant_range, channel_samples)
        
        # Safe altitude check
        safe_alt = min(altitude_m, self.max_slant_range * 0.85)
        
        # Ground range calculation: Rg = sqrt(Rs^2 - H^2) for Rs >= H
        valid_mask = slant_ranges >= safe_alt
        ground_ranges = np.zeros_like(slant_ranges)
        ground_ranges[valid_mask] = np.sqrt(slant_ranges[valid_mask]**2 - safe_alt**2)
        
        # Rescale ground ranges to fill available channel width
        max_ground = np.max(ground_ranges) if np.max(ground_ranges) > 0 else 1.0
        target_ground_grid = np.linspace(0, max_ground, channel_samples)
        
        # Interpolate both Port (left) and Starboard (right) channels
        # Port: column index 0 to channel_samples - 1 (inverted direction from center)
        # Starboard: column index channel_samples to total_samples - 1 (outward from center)
        for i in range(height):
            # Starboard channel
            stbd_raw = sonar_strip[i, channel_samples:]
            stbd_valid = stbd_raw[valid_mask]
            stbd_rg = ground_ranges[valid_mask]
            if len(stbd_rg) > 1:
                corrected[i, channel_samples:] = np.interp(
                    target_ground_grid, stbd_rg, stbd_valid, left=0.0, right=stbd_raw[-1]
                )
                
            # Port channel (mirror geometry)
            port_raw = sonar_strip[i, :channel_samples][::-1] # center outward
            port_valid = port_raw[valid_mask]
            if len(stbd_rg) > 1:
                port_interp = np.interp(
                    target_ground_grid, stbd_rg, port_valid, left=0.0, right=port_raw[-1]
                )
                corrected[i, :channel_samples] = port_interp[::-1] # restore leftwards direction
                
        if is_odd:
            corrected = np.pad(corrected, ((0, 0), (0, 1)), mode='edge')
            
        return corrected

    def apply_tvg(
        self,
        ping_channel: np.ndarray,
        spreading_coeff: float = 20.0
    ) -> np.ndarray:
        """
        Applies Time-Varying Gain (TVG) to compensate for acoustic energy decay:
            TVG(R) = spreading_coeff * log10(R) + 2 * alpha * R
        """
        n = len(ping_channel)
        ranges = np.linspace(1.0, self.max_slant_range, n)
        # dB gain curve converted to linear gain
        gain_db = spreading_coeff * np.log10(ranges) + 2.0 * self.absorption_coeff * ranges
        gain_db = gain_db - gain_db[0]  # normalize relative to start
        gain_linear = 10.0 ** (gain_db / 40.0) # gentle acoustic compression
        
        compensated = ping_channel.astype(float) * gain_linear
        return np.clip(compensated, 0, 255).astype(np.uint8)

    def calculate_target_dimensions(
        self,
        bbox_xywh: Tuple[int, int, int, int],
        altitude_m: float,
        waterfall_width_px: int,
        ping_rate_hz: float = 10.0,
        vehicle_speed_mps: float = 1.5
    ) -> Dict[str, float]:
        """
        Calculates true physical dimensions of an acoustic target:
        - Across-track width (m)
        - Along-track length (m)
        - Estimated vertical relief height above seafloor (m)
        """
        x, y, w, h = bbox_xywh
        center_x = waterfall_width_px / 2.0
        
        # Across-track slant range to target
        dist_from_center_px = abs(x + w / 2.0 - center_x)
        slant_range_m = (dist_from_center_px / (waterfall_width_px / 2.0)) * self.max_slant_range
        slant_range_m = max(slant_range_m, altitude_m + 0.1)
        
        # Across-track physical width
        width_m = (w / (waterfall_width_px / 2.0)) * self.max_slant_range
        
        # Along-track physical length (driven by vehicle speed and ping count)
        # Each ping line corresponds to along-track advance: dy = v / ping_rate
        advance_per_ping_m = vehicle_speed_mps / max(ping_rate_hz, 0.1)
        length_m = h * advance_per_ping_m
        
        # Estimated acoustic shadow length (shadow usually extends away from center)
        # Average shadow extends roughly 40-70% of the target's across-track footprint
        shadow_length_m = width_m * 0.65
        relief_height_m = self.calculate_shadow_height(altitude_m, shadow_length_m, slant_range_m)
        
        return {
            "slant_range_m": round(slant_range_m, 2),
            "width_m": round(max(width_m, 0.2), 2),
            "length_m": round(max(length_m, 0.2), 2),
            "relief_height_m": round(max(relief_height_m, 0.1), 2),
            "aspect_ratio": round(length_m / max(width_m, 0.01), 2)
        }
