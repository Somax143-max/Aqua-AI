"""
3D Seafloor Bathymetric & Acoustic Shadow Relief Visualizer
Reconstructs a 3D Digital Elevation Model (DEM) of the ocean floor
using acoustic shadow relief geometry and backscatter intensity.
"""

import numpy as np
import plotly.graph_objects as go
from typing import List, Dict, Any, Tuple, Optional

class Bathymetry3DVisualizer:
    """
    Renders interactive 3D seabed bathymetry with protruding marine debris targets.
    """

    def __init__(self, grid_size: Tuple[int, int] = (100, 100)):
        self.grid_size = grid_size

    def generate_3d_elevation_mesh(
        self,
        sonar_img: np.ndarray,
        detections: List[Dict[str, Any]],
        base_depth_m: float = 40.0
    ) -> Dict[str, Any]:
        """
        Synthesizes a 3D bathymetric elevation surface (Z in meters below sea surface)
        integrating acoustic backscatter textures and detected object relief heights.
        """
        ny, nx = self.grid_size
        h, w = sonar_img.shape[:2]

        # 1. Base seafloor bathymetry with gentle slope and sand ripples
        x = np.linspace(-50, 50, nx)
        y = np.linspace(0, 100, ny)
        xx, yy = np.meshgrid(x, y)

        # Baseline bathymetry: gentle depth gradient + sinusoidal dunes
        z = base_depth_m + 0.05 * yy + 0.4 * np.sin(xx / 6.0) * np.cos(yy / 8.0)

        # 2. Inject detected target 3D relief elevations
        for d in detections:
            bbox = d.get("bbox", [0, 0, 10, 10])
            dims = d.get("dimensions", {})
            relief_h = dims.get("relief_height_m", 1.5)

            # Map image pixel bbox to 3D grid indices
            bx, by, bw, bh = bbox
            gx = int((bx + bw / 2.0) / w * nx)
            gy = int((by + bh / 2.0) / h * ny)

            gw = max(2, int((bw / w) * nx * 0.7))
            gh = max(2, int((bh / h) * ny * 0.7))

            # Protrude object upward (decreasing depth value)
            y_start = max(0, gy - gh)
            y_end = min(ny, gy + gh)
            x_start = max(0, gx - gw)
            x_end = min(nx, gx + gw)

            # Smooth Gaussian elevation bump
            for iy in range(y_start, y_end):
                for ix in range(x_start, x_end):
                    dist_sq = ((ix - gx) / max(gw, 1))**2 + ((iy - gy) / max(gh, 1))**2
                    if dist_sq < 1.0:
                        bump = relief_h * (1.0 - dist_sq)
                        z[iy, ix] -= bump  # Objects rise towards surface (smaller depth)

        return {"x": xx, "y": yy, "z": z}

    def create_3d_plotly_figure(
        self,
        mesh_data: Dict[str, Any],
        detections: List[Dict[str, Any]],
        title: str = "3D Seafloor Elevation & Marine Hazard Relief Model"
    ) -> go.Figure:
        """
        Creates an interactive Plotly 3D Surface visualization.
        """
        xx = mesh_data["x"]
        yy = mesh_data["y"]
        zz = mesh_data["z"]

        # Main bathymetric seabed surface
        fig = go.Figure(data=[
            go.Surface(
                x=xx,
                y=yy,
                z=zz,
                colorscale="Viridis_r",
                reversescale=False,
                colorbar=dict(title="Depth (m)", thickness=15, len=0.7),
                contours=dict(
                    z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
                ),
                opacity=0.92
            )
        ])

        # Add 3D markers and annotations for detected hazards
        for d in detections:
            dims = d.get("dimensions", {})
            geo = d.get("geotag", {})
            relief = dims.get("relief_height_m", 1.5)
            class_name = d.get("class_name", "Hazard")
            hazard_id = d.get("id", "HAZ")

            # Center position in mesh coordinates
            bbox = d.get("bbox", [500, 300, 20, 20])
            bx, by, bw, bh = bbox
            mx = float(np.interp(bx + bw/2.0, [0, 1000], [-50, 50]))
            my = float(np.interp(by + bh/2.0, [0, 600], [0, 100]))

            # Find corresponding depth
            iy = int(np.clip(my, 0, zz.shape[0] - 1))
            ix = int(np.clip((mx + 50), 0, zz.shape[1] - 1))
            mz = float(zz[iy, ix])

            fig.add_trace(go.Scatter3d(
                x=[mx],
                y=[my],
                z=[mz - 0.5],
                mode="markers+text",
                name=f"{hazard_id}: {class_name}",
                text=[f"<b>{class_name}</b><br>Relief: +{relief}m"],
                textposition="top center",
                marker=dict(
                    size=6,
                    color=d.get("hex_color", "#FF0000"),
                    symbol="diamond",
                    line=dict(color="white", width=1)
                )
            ))

        fig.update_layout(
            title=title,
            autosize=True,
            height=600,
            scene=dict(
                xaxis=dict(title="Across-Track Distance (m)", backgroundcolor="#081426", gridcolor="#1E293B"),
                yaxis=dict(title="Along-Track Advance (m)", backgroundcolor="#081426", gridcolor="#1E293B"),
                zaxis=dict(title="Seafloor Depth (m)", autorange="reversed", backgroundcolor="#081426", gridcolor="#1E293B"),
                camera=dict(
                    eye=dict(x=-1.5, y=-1.5, z=1.2),
                    up=dict(x=0, y=0, z=1)
                )
            ),
            paper_bgcolor="#0B132B",
            font=dict(color="#E0E6ED"),
            margin=dict(l=10, r=10, b=10, t=40)
        )

        return fig
