"""
AquaProtect-AI: 3D Subsea ROV Manipulator Arm Digital Twin & Cutting Simulator
Simulates:
- 5-Degree-of-Freedom (5-DOF) hydraulic robotic manipulator arm
- Kinematic positioning of hydraulic cutter shears at calculated 3D filament waypoints
- Real-time cutting sequence telemetry (Pressure PSI, severed tension lines)
- Interactive 3D Plotly visualizer for SIH / MoES / NIOT presentation
"""

import math
from typing import Dict, Any, List
import numpy as np
import plotly.graph_objects as go

class ROVManipulator3DSimulator:
    """
    Simulates subsea robotic manipulation and hydraulic shearing operations on ghost nets.
    """

    @staticmethod
    def generate_rov_arm_kinematics(
        target_xyz: List[float] = [2.2, 1.4, -1.8]
    ) -> Dict[str, Any]:
        """
        Inverse kinematics approximation for a 5-DOF hydraulic manipulator arm
        reaching toward a subsea target waypoint (x, y, z).
        """
        tx, ty, tz = target_xyz

        # Base origin (ROV chassis mount point)
        base = np.array([0.0, 0.0, 0.0])
        # Shoulder joint (raised 0.4m)
        shoulder = np.array([0.0, 0.0, 0.4])

        # Direction vector toward target
        target = np.array([tx, ty, tz])
        arm_dir = target - shoulder
        reach = np.linalg.norm(arm_dir)

        # Elbow and Wrist interpolation
        elbow = shoulder + arm_dir * 0.45 + np.array([0.0, 0.0, 0.35])
        wrist = shoulder + arm_dir * 0.85
        end_effector = target

        # Hydraulic cutter jaws (two shear blades)
        blade_left = end_effector + np.array([-0.12, 0.08, 0.1])
        blade_right = end_effector + np.array([0.12, -0.08, 0.1])

        return {
            "joints": {
                "base": base.tolist(),
                "shoulder": shoulder.tolist(),
                "elbow": elbow.tolist(),
                "wrist": wrist.tolist(),
                "cutter_tool": end_effector.tolist()
            },
            "cutter_jaws": {
                "left_blade": blade_left.tolist(),
                "right_blade": blade_right.tolist()
            },
            "hydraulic_pressure_psi": 2850.0,
            "shear_force_kn": 42.5,
            "reach_distance_m": round(float(reach), 2)
        }

    @classmethod
    def create_3d_manipulator_figure(
        cls,
        waypoints: List[Dict[str, Any]]
    ) -> go.Figure:
        """
        Creates an interactive 3D Plotly figure of the ROV Manipulator Arm
        engaged in cutting a ghost fishing net filament mesh.
        """
        fig = go.Figure()

        # If waypoints exist, target the first active cutting point
        if waypoints:
            target_pt = [2.0, 1.2, -waypoints[0].get("depth_relief_m", 1.8)]
        else:
            target_pt = [2.2, 1.4, -1.8]

        kinematics = cls.generate_rov_arm_kinematics(target_pt)
        j = kinematics["joints"]

        # Plot Manipulator Arm Links
        arm_x = [j["base"][0], j["shoulder"][0], j["elbow"][0], j["wrist"][0], j["cutter_tool"][0]]
        arm_y = [j["base"][1], j["shoulder"][1], j["elbow"][1], j["wrist"][1], j["cutter_tool"][1]]
        arm_z = [j["base"][2], j["shoulder"][2], j["elbow"][2], j["wrist"][2], j["cutter_tool"][2]]

        fig.add_trace(go.Scatter3d(
            x=arm_x, y=arm_y, z=arm_z,
            mode="lines+markers",
            line=dict(color="#FFD166", width=12),
            marker=dict(size=8, color="#073B4C", symbol="circle"),
            name="ROV Hydraulic Arm"
        ))

        # Plot Hydraulic Cutter Blades
        bl = kinematics["cutter_jaws"]["left_blade"]
        br = kinematics["cutter_jaws"]["right_blade"]
        ct = j["cutter_tool"]

        fig.add_trace(go.Scatter3d(
            x=[bl[0], ct[0], br[0]],
            y=[bl[1], ct[1], br[1]],
            z=[bl[2], ct[2], br[2]],
            mode="lines+markers",
            line=dict(color="#FF5964", width=8),
            marker=dict(size=6, color="#FF0000"),
            name="Hydraulic Cutter Shears (42.5 kN)"
        ))

        # Plot Ghost Net Webbing Structure
        net_x, net_y, net_z = [], [], []
        np.random.seed(42)
        cx, cy, cz = target_pt
        for _ in range(16):
            p1 = [cx + np.random.uniform(-1.0, 1.0), cy + np.random.uniform(-1.0, 1.0), cz + np.random.uniform(-0.5, 0.5)]
            p2 = [cx + np.random.uniform(-1.0, 1.0), cy + np.random.uniform(-1.0, 1.0), cz + np.random.uniform(-0.5, 0.5)]
            net_x.extend([p1[0], p2[0], None])
            net_y.extend([p1[1], p2[1], None])
            net_z.extend([p1[2], p2[2], None])

        fig.add_trace(go.Scatter3d(
            x=net_x, y=net_y, z=net_z,
            mode="lines",
            line=dict(color="#06D6A0", width=3, dash="dot"),
            name="Ghost Net Filament Mesh"
        ))

        # Plot Cutting Waypoints
        if waypoints:
            wp_x = [target_pt[0] + (w.get("step_index", 0)*0.35) for w in waypoints[:6]]
            wp_y = [target_pt[1] + (w.get("step_index", 0)*0.25) for w in waypoints[:6]]
            wp_z = [-w.get("depth_relief_m", 1.8) for w in waypoints[:6]]

            fig.add_trace(go.Scatter3d(
                x=wp_x, y=wp_y, z=wp_z,
                mode="markers+text",
                marker=dict(size=9, color="#FF5964", symbol="diamond"),
                text=[f"Cut #{w.get('step_index', i+1)}" for i, w in enumerate(waypoints[:6])],
                textposition="top center",
                name="3D Severing Waypoints"
            ))

        fig.update_layout(
            title="Interactive 3D Subsea ROV Manipulator Arm & Ghost Net Cutting Toolpath",
            scene=dict(
                xaxis_title="Surge X (meters forward)",
                yaxis_title="Sway Y (meters lateral)",
                zaxis_title="Heave Z (meters depth)",
                bgcolor="#050B18",
                xaxis=dict(backgroundcolor="#050B18", gridcolor="#1B2A4A"),
                yaxis=dict(backgroundcolor="#050B18", gridcolor="#1B2A4A"),
                zaxis=dict(backgroundcolor="#050B18", gridcolor="#1B2A4A"),
                camera=dict(eye=dict(x=1.8, y=1.8, z=1.2))
            ),
            paper_bgcolor="#050B18",
            font=dict(color="#E2E8F0"),
            margin=dict(l=0, r=0, b=0, t=40)
        )

        return fig
