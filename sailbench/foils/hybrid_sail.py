"""Hybrid aerodynamic sail model using rotation matrix for force conversion."""

from typing import Any

import numpy as np

from sailbench.models.model import Model
from sailbench.models.model import State
from sailbench.tf.tf_tree import TFTree2D

# Luffing threshold: sails stop generating lift when they flap
LUFF_DEG = 5.0


class HybridSail(Model):
    """Hybrid aerodynamic sail model.

    Behaves correctly both when the sail acts like a wing (small AoA)
    and when it acts like a parachute (large AoA, downwind).
    Uses the standard simplified model: CL = CL_max * sin(2*alpha),
    CD = CD0 + CD1 * (1 - cos(2*alpha)).

    Converts forces from fluid frame to boat frame via rotation matrix
    (fluid → sail) then tf_tree (sail → boat).
    """

    def __init__(self, params: dict[str, Any]) -> None:
        """Initialize the HybridSail model.

        Args:
            params (dict): Dictionary of parameters for the sail model.
                - wind_speed: [m/s]
                - wind_dir_deg: direction wind blows TO, in world frame (deg, 0=east)
                - area: sail area [m²]
                - CL_max: max lift coefficient (default 1.2)
                - CD0: zero-AoA drag (default 0.1)
                - CD1: drag shape factor (default 1.0)
                - rho_air: kg/m³ (default 1.225), usually from `environment`
        """
        super().__init__(params)

    def compute(self, state: State, tf_tree: TFTree2D) -> np.ndarray:
        """Compute lift and drag forces for the sail.

        Uses hybrid aerodynamic model with analytic CL/CD.
        Returns forces in boat frame (Fx, Fy).

        Args:
            state (State): Current boat state.
            tf_tree (TFTree2D): Transform tree with boat and sail frames.

        Returns:
            np.ndarray: [Fx, Fy] forces in newtons (boat frame).
        """
        wind_speed = float(self.p.get("wind_speed", 0.0))
        wind_angle_deg = float(self.p.get("wind_dir_deg", 0.0))
        area = float(self.p.get("area", 1.0))
        rho = float(self.p.get("rho_air", 1.225))
        cl_max = float(self.p.get("CL_max", 1.2))
        cd0 = float(self.p.get("CD0", 0.1))
        cd1 = float(self.p.get("CD1", 1.0))

        # 1. Wind vector in world frame
        # wind_dir_deg = direction wind is blowing TO (0° = east)
        wind_rad = np.radians(wind_angle_deg)
        wind_world = wind_speed * np.array([np.cos(wind_rad), np.sin(wind_rad)])

        # Boat velocity in world frame
        v_boat_world = tf_tree.vector_to_frame(
            np.array([state.u, state.v]), "boat", "world"
        )

        # Apparent wind: V_aw = V_wind - V_boat
        apparent_wind_world = wind_world - v_boat_world

        # Apparent wind in sail frame (sail chord along sail +x)
        apparent_wind_sail = tf_tree.vector_to_frame(
            apparent_wind_world, "world", "sail"
        )

        V = float(np.linalg.norm(apparent_wind_sail))
        if V < 1e-6:
            return np.array([0.0, 0.0])

        # 2. Angle of attack from apparent wind in sail frame
        # alpha = angle from sail chord (+x) to wind direction
        alpha = float(np.arctan2(apparent_wind_sail[1], apparent_wind_sail[0]))

        # 3. Lift and drag coefficients (hybrid model)
        if np.degrees(np.abs(alpha)) < LUFF_DEG:
            cl = 0.0  # Sails stop generating lift when they luff
        else:
            cl = cl_max * np.sin(2 * alpha)

        cd = cd0 + cd1 * (1.0 - np.cos(2 * alpha))

        # 4. Aerodynamic forces
        q = 0.5 * rho * V**2
        lift = q * area * cl
        drag = q * area * cd

        # 5. Convert fluid frame → sail frame → boat frame
        # Fluid frame: x = flow direction. f_fluid = [drag, lift]
        f_fluid = np.array([drag, lift], dtype=float)

        # Rotation fluid → sail: alpha = angle from sail chord to flow
        c, s = np.cos(alpha), np.sin(alpha)
        R_fluid_to_sail = np.array([[c, -s], [s, c]])
        f_sail = R_fluid_to_sail @ f_fluid

        return tf_tree.vector_to_frame(f_sail, "sail", "boat")
