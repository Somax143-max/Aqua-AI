"""
Sonar Preprocessing & Speckle Noise Filtering Module
Implements acoustic signal enhancement:
- Speckle noise suppression (Lee Filter, Frost Filter)
- Sonar-CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Bilateral edge-preserving filtering
- Colormap conversion for hydrographic waterfall displays
"""

import cv2
import numpy as np
from typing import Tuple, Optional

class SonarPreprocessor:
    """
    Advanced preprocessor designed specifically for Side-Scan Sonar imagery.
    Suppresses acoustic speckle noise while preserving acoustic shadow edges.
    """

    def __init__(
        self,
        clahe_clip_limit: float = 2.5,
        clahe_grid_size: Tuple[int, int] = (8, 8),
        default_filter: str = "lee"
    ):
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_grid_size)
        self.default_filter = default_filter

    def apply_lee_filter(
        self,
        img: np.ndarray,
        window_size: int = 5,
        noise_variance_factor: float = 0.25
    ) -> np.ndarray:
        """
        Implements the classic Lee Speckle Filter for coherent acoustic radar/sonar:
            R_hat = I_bar + W * (I - I_bar)
            W = max(0, (sigma^2 - sigma_v^2) / sigma^2)
        Preserves sharp acoustic specular returns and shadow boundaries while
        smoothing diffuse seabed speckle noise.
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        img_float = gray.astype(np.float32)
        
        # Local mean and local squared mean using box filter
        ksize = (window_size, window_size)
        mean = cv2.boxFilter(img_float, -1, ksize)
        sqr_mean = cv2.boxFilter(img_float ** 2, -1, ksize)
        
        # Local variance: Var(X) = E[X^2] - (E[X])^2
        variance = np.maximum(sqr_mean - mean ** 2, 0.0)
        
        # Noise variance estimation proportional to local mean (multiplicative speckle model)
        noise_var = (noise_variance_factor * mean) ** 2
        
        # Weight factor W
        weight = np.maximum(0.0, (variance - noise_var) / (variance + 1e-6))
        weight = np.clip(weight, 0.0, 1.0)
        
        # Filtered output
        filtered = mean + weight * (img_float - mean)
        return np.clip(filtered, 0, 255).astype(np.uint8)

    def apply_frost_filter(
        self,
        img: np.ndarray,
        window_size: int = 5,
        damping_factor: float = 1.0
    ) -> np.ndarray:
        """
        Implements Frost Filter: an exponentially damped convolutional kernel
        where weights are adapted based on the local coefficient of variation (sigma/mean).
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
            
        img_float = gray.astype(np.float32)
        ksize = (window_size, window_size)
        
        mean = cv2.boxFilter(img_float, -1, ksize)
        sqr_mean = cv2.boxFilter(img_float ** 2, -1, ksize)
        variance = np.maximum(sqr_mean - mean ** 2, 0.0)
        
        # Coefficient of variation B = sigma / mean
        b = np.sqrt(variance) / (mean + 1e-5)
        
        # Approximation via fast adaptive bilateral-like blur
        sigma_color = float(np.mean(b) * 40.0 * damping_factor + 10.0)
        filtered = cv2.bilateralFilter(gray, window_size, sigma_color, window_size * 2)
        return filtered

    def apply_clahe(self, gray_img: np.ndarray) -> np.ndarray:
        """
        Applies Contrast-Limited Adaptive Histogram Equalization (CLAHE).
        Prevents over-amplification of acoustic noise while revealing subtle
        debris contours in dark seabed basins and shadow margins.
        """
        if len(gray_img.shape) == 3:
            gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
        return self.clahe.apply(gray_img)

    def denoise_and_enhance(
        self,
        sonar_img: np.ndarray,
        filter_type: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full two-stage acoustic pipeline:
        Stage 1: Physics-based Speckle Denoising (Lee or Frost)
        Stage 2: Sonar-CLAHE Dynamic Range Balancing
        
        Returns:
            denoised_raw: Denoised grayscale image
            enhanced: Denoised + CLAHE enhanced image
        """
        choice = filter_type or self.default_filter
        
        if len(sonar_img.shape) == 3:
            gray = cv2.cvtColor(sonar_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = sonar_img.copy()

        # Step 1: Denoise
        if choice.lower() == "frost":
            denoised = self.apply_frost_filter(gray)
        elif choice.lower() == "bilateral":
            denoised = cv2.bilateralFilter(gray, 7, 50, 50)
        else:
            denoised = self.apply_lee_filter(gray)
            
        # Step 2: Adaptive Enhancement
        enhanced = self.apply_clahe(denoised)
        
        return denoised, enhanced

    @staticmethod
    def apply_sonar_colormap(gray_img: np.ndarray, colormap_name: str = "bronze") -> np.ndarray:
        """
        Renders sonar backscatter with authentic hydrographic colormaps:
        - 'bronze' / 'sepia': Classic Klein/EdgeTech marine acoustic standard
        - 'jet': Scientific multibeam / acoustic intensity standard
        - 'viridis': Modern perceptual oceanographic standard
        - 'copper': Deep trench high-relief acoustic standard
        - 'gray': Monochromatic acoustic intensity
        """
        if len(gray_img.shape) == 3:
            gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
            
        cmap_lower = colormap_name.lower()
        
        if cmap_lower in ["bronze", "sepia", "amber"]:
            # Custom Bronze/Amber Sonar lookup table
            # Low intensity: dark navy/black (shadows)
            # Medium intensity: golden amber / bronze (seabed backscatter)
            # High intensity: bright pale yellow / white (specular highlight reflections)
            lut = np.zeros((256, 1, 3), dtype=np.uint8)
            for i in range(256):
                t = i / 255.0
                r = int(min(255, 255 * (t ** 0.8) * 1.1))
                g = int(min(255, 180 * (t ** 1.1)))
                b = int(min(255, 80 * (t ** 2.0)))
                lut[i, 0] = [b, g, r] # OpenCV BGR
            return cv2.LUT(cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR), lut)
            
        elif cmap_lower == "jet":
            return cv2.applyColorMap(gray_img, cv2.COLORMAP_JET)
        elif cmap_lower == "viridis":
            return cv2.applyColorMap(gray_img, cv2.COLORMAP_VIRIDIS)
        elif cmap_lower == "copper":
            return cv2.applyColorMap(gray_img, cv2.COLORMAP_COPPER)
        else:
            return cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
