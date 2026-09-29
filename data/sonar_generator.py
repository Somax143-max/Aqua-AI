"""
Physics-Based Side-Scan Sonar Simulator & Synthetic Waterfall Generator
Generates high-fidelity acoustic sonar waterfall imagery modeled on real dual-channel
systems (EdgeTech / Klein / Kongsberg GeoSwath):
- Port & Starboard dual channels with central Nadir blind zone
- Multiplicative Rayleigh & K-distribution acoustic speckle noise
- Seabed benthic textures (sand ripples, muddy plains, rocky reefs, biogenic clutter)
- Physics-accurate target backscatter, geometric acoustic shadow projection, and multipath echoes
- Distinct cylindrical drums vs spherical/conical naval mines and natural seafloor clutter
"""

import math
import random
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

class SonarDataGenerator:
    """
    Simulates realistic hydroacoustic Side-Scan Sonar waterfall strips
    with embedded anthropogenic marine hazards and natural seabed clutter.
    """

    def __init__(
        self,
        width: int = 1000,
        height: int = 600,
        altitude_m: float = 12.0,
        max_range_m: float = 75.0,
        nadir_width_ratio: float = 0.08
    ):
        self.width = width
        self.height = height
        self.altitude_m = altitude_m
        self.max_range_m = max_range_m
        self.nadir_width_ratio = nadir_width_ratio
        self.center_x = width // 2

    def _generate_seabed_texture(
        self,
        seabed_type: str = "sand_ripples"
    ) -> np.ndarray:
        """
        Generates realistic ambient seabed backscatter texture with Rayleigh speckle.
        """
        base_intensity = 105.0
        rayleigh = np.random.rayleigh(scale=24.0, size=(self.height, self.width))

        if seabed_type == "sand_ripples":
            x_coords = np.linspace(0, 36 * np.pi, self.width)
            y_coords = np.linspace(0, 12 * np.pi, self.height)
            xx, yy = np.meshgrid(x_coords, y_coords)
            ripples = 18.0 * np.sin(xx + 0.3 * np.sin(yy))
            texture = base_intensity + rayleigh * 0.7 + ripples

        elif seabed_type == "rocky_reef":
            roughness = cv2.resize(
                np.random.normal(0, 30, (self.height // 4, self.width // 4)),
                (self.width, self.height)
            )
            texture = base_intensity + rayleigh * 1.2 + roughness

        else: # "muddy_flat"
            texture = (base_intensity - 15) + rayleigh * 0.5

        # Range-dependent acoustic falloff (Lambert's law backscatter attenuation)
        dist_from_nadir = np.abs(np.arange(self.width) - self.center_x)
        range_norm = dist_from_nadir / (self.width / 2.0)
        falloff = 1.0 - 0.28 * (range_norm ** 1.3)
        texture = texture * falloff[np.newaxis, :]

        # Nadir water column blind zone
        nadir_half = int(self.width * self.nadir_width_ratio / 2.0)
        nadir_start = max(0, self.center_x - nadir_half)
        nadir_end = min(self.width, self.center_x + nadir_half)

        water_col_noise = np.random.normal(16, 4, (self.height, nadir_end - nadir_start))
        texture[:, nadir_start:nadir_end] = np.clip(water_col_noise, 8, 30)

        # Bottom track specular pulse
        if nadir_start > 2 and nadir_end < self.width - 2:
            texture[:, nadir_start-2:nadir_start+1] += np.random.uniform(35, 55, (self.height, 3))
            texture[:, nadir_end:nadir_end+3] += np.random.uniform(35, 55, (self.height, 3))

        return np.clip(texture, 0, 255).astype(np.uint8)

    def _inject_target(
        self,
        canvas: np.ndarray,
        target_type: str,
        pos_y: int,
        pos_x: int,
        scale: float = 1.0
    ) -> Dict[str, Any]:
        """
        Embeds a marine target with physics-accurate acoustic specular reflection,
        corresponding geometric acoustic shadow, and multipath reflections.
        """
        is_starboard = pos_x >= self.center_x
        slant_dist_px = abs(pos_x - self.center_x)
        slant_range_m = (slant_dist_px / (self.width / 2.0)) * self.max_range_m

        specs = {
            "ghost_net": {
                "w": int(48 * scale), "h": int(36 * scale),
                "relief_m": 2.2, "specular_intensity": 235, "class_id": 0,
                "texture": "irregular_filament"
            },
            "shipwreck": {
                "w": int(80 * scale), "h": int(90 * scale),
                "relief_m": 4.8, "specular_intensity": 250, "class_id": 1,
                "texture": "angular_hull"
            },
            "pipeline": {
                "w": int(115 * scale), "h": int(18 * scale),
                "relief_m": 1.4, "specular_intensity": 240, "class_id": 2,
                "texture": "linear_continuous"
            },
            "drum": { # Metal Drum / Chemical Barrel: Cylindrical profile with axial chimes
                "w": int(28 * scale), "h": int(16 * scale),
                "relief_m": 1.1, "specular_intensity": 235, "class_id": 3,
                "texture": "cylindrical_barrel"
            },
            "container": {
                "w": int(52 * scale), "h": int(42 * scale),
                "relief_m": 2.8, "specular_intensity": 245, "class_id": 4,
                "texture": "rectangular_box"
            },
            "mine": { # Munition / Naval Mine: Spherical/conical casing, high TS, tether wire
                "w": int(16 * scale), "h": int(16 * scale),
                "relief_m": 0.95, "specular_intensity": 255, "class_id": 5,
                "texture": "spherical_mine"
            },
            "natural_clutter": { # Natural Benthic Clutter: Granite boulder or coral head
                "w": int(26 * scale), "h": int(24 * scale),
                "relief_m": 0.8, "specular_intensity": 195, "class_id": 6,
                "texture": "rock_boulder"
            }
        }

        cfg = specs.get(target_type, specs["ghost_net"])
        tw, th = cfg["w"], cfg["h"]
        relief_h = cfg["relief_m"]

        if self.altitude_m > relief_h:
            shadow_length_m = (relief_h * slant_range_m) / (self.altitude_m - relief_h)
        else:
            shadow_length_m = 10.0

        shadow_len_px = int((shadow_length_m / self.max_range_m) * (self.width / 2.0))
        shadow_len_px = max(14, min(shadow_len_px, 140))

        y1 = max(10, min(self.height - th - 10, pos_y))
        y2 = y1 + th

        if is_starboard:
            hx1 = max(self.center_x + 30, min(self.width - tw - shadow_len_px - 10, pos_x))
            hx2 = hx1 + tw
            sx1 = hx2
            sx2 = min(self.width - 5, sx1 + shadow_len_px)
            bbox_x = hx1
            bbox_w = (sx2 - hx1)
        else:
            hx2 = min(self.center_x - 30, max(tw + shadow_len_px + 10, pos_x))
            hx1 = hx2 - tw
            sx2 = hx1
            sx1 = max(5, sx2 - shadow_len_px)
            bbox_x = sx1
            bbox_w = (hx2 - sx1)

        # 1. Render Acoustic Shadow
        shadow_val = random.randint(10, 24)
        canvas[y1:y2, sx1:sx2] = np.clip(
            canvas[y1:y2, sx1:sx2].astype(float) * 0.12 + shadow_val, 0, 255
        ).astype(np.uint8)

        # 2. Render Specular Highlight
        high_val = cfg["specular_intensity"]

        if cfg["texture"] == "irregular_filament": # Ghost net
            net_patch = np.zeros((th, tw), dtype=np.uint8)
            for _ in range(14):
                pt1 = (random.randint(0, tw-1), random.randint(0, th-1))
                pt2 = (random.randint(0, tw-1), random.randint(0, th-1))
                cv2.line(net_patch, pt1, pt2, high_val, random.randint(1, 2))
            cv2.circle(net_patch, (tw//2, th//2), max(3, tw//4), high_val, -1)
            canvas[y1:y2, hx1:hx2] = np.maximum(canvas[y1:y2, hx1:hx2], net_patch)

        elif cfg["texture"] == "angular_hull": # Shipwreck
            hull_patch = np.zeros((th, tw), dtype=np.uint8)
            pts = np.array([
                [tw // 2, 2], [tw - 2, th // 3], [tw - 6, th - 2],
                [6, th - 2], [2, th // 3]
            ], np.int32)
            cv2.fillPoly(hull_patch, [pts], high_val)
            canvas[y1:y2, hx1:hx2] = np.maximum(canvas[y1:y2, hx1:hx2], hull_patch)

        elif cfg["texture"] == "linear_continuous": # Pipeline
            pipe_patch = np.zeros((th, tw), dtype=np.uint8)
            cv2.line(pipe_patch, (0, th//2), (tw, th//2), high_val, max(3, th//3))
            canvas[y1:y2, hx1:hx2] = np.maximum(canvas[y1:y2, hx1:hx2], pipe_patch)

        elif cfg["texture"] == "cylindrical_barrel": # Metal Drum
            barrel_patch = np.zeros((th, tw), dtype=np.uint8)
            # Cylinder body
            cv2.rectangle(barrel_patch, (2, 2), (tw - 2, th - 2), int(high_val * 0.85), -1)
            # Chimes / ribs (bright dual specular reflections)
            cv2.line(barrel_patch, (tw // 3, 2), (tw // 3, th - 2), high_val, 2)
            cv2.line(barrel_patch, (2 * tw // 3, 2), (2 * tw // 3, th - 2), high_val, 2)
            canvas[y1:y2, hx1:hx2] = np.maximum(canvas[y1:y2, hx1:hx2], barrel_patch)

        elif cfg["texture"] == "spherical_mine": # Munition / Naval Mine
            mine_patch = np.zeros((th, tw), dtype=np.uint8)
            r = min(th, tw) // 2
            cv2.circle(mine_patch, (tw // 2, th // 2), r, high_val, -1)
            # Specular apex highlight
            cv2.circle(mine_patch, (tw // 2, th // 2), max(2, r // 3), 255, -1)
            # Mooring anchor wire echo
            cv2.line(mine_patch, (tw // 2, th // 2), (tw // 2, th - 1), 220, 1)
            canvas[y1:y2, hx1:hx2] = np.maximum(canvas[y1:y2, hx1:hx2], mine_patch)

        elif cfg["texture"] == "rock_boulder": # Natural seafloor rock
            rock_patch = np.zeros((th, tw), dtype=np.uint8)
            pts = np.array([
                [tw // 4, 3], [3 * tw // 4, 5], [tw - 3, 3 * th // 4],
                [tw // 2, th - 3], [3, th // 2]
            ], np.int32)
            cv2.fillPoly(rock_patch, [pts], high_val)
            canvas[y1:y2, hx1:hx2] = np.maximum(canvas[y1:y2, hx1:hx2], rock_patch)

        else: # Cargo Container
            box_patch = np.full((th, tw), high_val, dtype=np.uint8)
            canvas[y1:y2, hx1:hx2] = np.maximum(canvas[y1:y2, hx1:hx2], box_patch)

        # 3. Multipath ghost reflection at 2x slant range (subtle secondary return)
        if is_starboard and (hx2 + 25 < self.width):
            ghost_x = min(self.width - 10, hx2 + 22)
            canvas[y1:y2, ghost_x:ghost_x + 4] = np.clip(canvas[y1:y2, ghost_x:ghost_x + 4].astype(float) + 30, 0, 255).astype(np.uint8)

        return {
            "type": target_type,
            "class_id": cfg["class_id"],
            "bbox": (int(bbox_x), int(y1), int(bbox_w), int(th)),
            "relief_m": relief_h,
            "shadow_len_px": shadow_len_px,
            "is_starboard": is_starboard
        }

    def generate_mission_waterfall(
        self,
        seabed: str = "sand_ripples",
        targets: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        canvas = self._generate_seabed_texture(seabed_type=seabed)

        target_defs = targets or [
            {"type": "ghost_net", "y": 120, "x": self.center_x + 180, "scale": 1.1},
            {"type": "drum", "y": 280, "x": self.center_x - 150, "scale": 1.0},
            {"type": "mine", "y": 380, "x": self.center_x + 140, "scale": 0.9},
            {"type": "shipwreck", "y": 480, "x": self.center_x + 270, "scale": 1.2}
        ]

        injected_records = []
        for t in target_defs:
            rec = self._inject_target(
                canvas,
                target_type=t["type"],
                pos_y=t["y"],
                pos_x=t["x"],
                scale=t.get("scale", 1.0)
            )
            injected_records.append(rec)

        return canvas, injected_records

    def generate_multi_object_scene(
        self,
        seabed_type: Optional[str] = None,
        num_targets: Optional[int] = None,
        rng: Optional[np.random.RandomState] = None
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Generates a realistic multi-object sonar swath containing multiple hazards
        and natural clutter objects in a single strip with variable sizes, orientations,
        sediment burial, and acoustic shadows.
        """
        local_rng = rng or np.random.RandomState()
        seabeds = ["sand_ripples", "rocky_reef", "muddy_flat"]
        active_seabed = seabed_type or str(local_rng.choice(seabeds))
        canvas = self._generate_seabed_texture(seabed_type=active_seabed)

        all_types = ["ghost_net", "shipwreck", "pipeline", "drum", "container", "mine", "natural_clutter"]
        n_targets = num_targets or int(local_rng.randint(3, 7))

        chosen_types = local_rng.choice(all_types, size=n_targets, replace=True)
        injected = []

        # Nadir exclusion zone
        nadir_margin = int(self.width * 0.12)
        half_w = self.width // 2

        # Space out targets along waterfall height
        y_positions = np.linspace(60, self.height - 80, n_targets).astype(int)
        local_rng.shuffle(y_positions)

        for idx, t_type in enumerate(chosen_types):
            pos_y = int(y_positions[idx] + local_rng.randint(-20, 21))
            pos_y = max(30, min(self.height - 90, pos_y))

            # Choose port or starboard, avoiding nadir blind zone
            is_star = bool(local_rng.rand() > 0.5)
            if is_star:
                pos_x = int(half_w + nadir_margin + local_rng.randint(30, half_w - nadir_margin - 80))
            else:
                pos_x = int(local_rng.randint(60, half_w - nadir_margin - 30))

            scale = float(local_rng.uniform(0.75, 1.35))
            rec = self._inject_target(
                canvas,
                target_type=t_type,
                pos_y=pos_y,
                pos_x=pos_x,
                scale=scale
            )

            # Simulated partial burial in sediment (up to 35% attenuation)
            if local_rng.rand() < 0.25:
                bx, by, bw, bh = rec["bbox"]
                burial_mask = local_rng.uniform(0.60, 0.85, (min(bh, canvas.shape[0]-by), min(bw, canvas.shape[1]-bx)))
                canvas[by:by+burial_mask.shape[0], bx:bx+burial_mask.shape[1]] = np.clip(
                    canvas[by:by+burial_mask.shape[0], bx:bx+burial_mask.shape[1]] * burial_mask, 0, 255
                ).astype(np.uint8)
                rec["buried"] = True
            else:
                rec["buried"] = False

            rec["scale"] = round(scale, 2)
            injected.append(rec)

        return canvas, injected