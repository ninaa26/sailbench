"""Basic sail foil model."""

from typing import Any

import numpy as np

from sailbench.models.foil import Foil
from sailbench.models.model import State
from sailbench.tf.tf_tree import TFTree2D


class BasicSail(Foil):
    """Basic sail foil model."""

    def __init__(self, params: dict[str, Any]) -> None:
        """Initialize the BasicSail model.

        Args:
            params (dict): Dictionary of parameters for the sail model.

        """
        super().__init__(params)

    def compute(self, state: State, tf_tree: TFTree2D) -> np.ndarray:
        """Compute lift and drag forces for the sail.

        Uses tf_tree for all coordinate transforms. Sail angle comes from the
        "sail" frame. Rotates fluid frame → sail frame, then sail → boat via tf_tree.

        Args:
            state (State): Current boat state.
            tf_tree (TFTree2D): Transform tree with boat and sail frames.

        Returns:
            np.ndarray: X and Y forces in newtons (boat frame).

        """
        wind_speed = float(self.p.get("wind_speed", 0.0))
        wind_angle_deg = float(self.p.get("wind_dir_deg", 0.0))

        # Wind vector in world frame
        wind_rad = float(np.radians(wind_angle_deg))
        wind_global = np.array([
            wind_speed * np.cos(wind_rad),
            wind_speed * np.sin(wind_rad),
        ], dtype=float)
        # Boat velocity: state.u, state.v are body-frame; convert to world
        v_world = tf_tree.vector_to_frame(np.array([state.u, state.v], dtype=float), "boat", "world")

        # Compute apparent wind in sail frame
        apparent_wind_global = wind_global - v_world
        apparent_wind_sail = tf_tree.vector_to_frame(apparent_wind_global, "world", "sail")
        speed = float(np.hypot(apparent_wind_sail[0], apparent_wind_sail[1]))
        if speed < 1e-6:
            return np.array([0.0, 0.0], dtype=float)

        # alpha is angle from sail +x (chord) to local apparent-wind direction.
        aoa_rad = float(np.arctan2(apparent_wind_sail[1], apparent_wind_sail[0]))

        cl, cd = self.cl_cd(aoa_rad, re=self.get_reynolds())

        # Luffing model:
        # - near centerline apparent flow, sail flaps and loses lift authority
        # - use a smooth ramp to avoid discontinuous force jumps
        aoa_deg_abs = float(np.degrees(np.abs(aoa_rad)))
        luff_deg = float(self.p.get("luff_deg", 7.5))
        luff_ramp_deg = max(float(self.p.get("luff_ramp_deg", 4.0)), 1e-6)
        cl_scale = float(np.clip((aoa_deg_abs - luff_deg) / luff_ramp_deg, 0.0, 1.0))
        cl *= cl_scale

        rho = float(self.p.get("rho_air", 1.225))  # kg/m³
        q = 0.5 * rho * speed**2
        s = float(self.p.get("area", 1.0))  # m²
        lift = cl * q * s
        drag = cd * q * s

        # Force in fluid frame (x = wind direction; drag opposes motion => +drag along flow)
        f_fluid = np.array([drag, lift], dtype=float)

        # Rotate fluid → sail
        c, s = np.cos(aoa_rad), np.sin(aoa_rad)
        r_fluid_to_sail = np.array([[c, -s], [s, c]], dtype=float)

        f_sail = r_fluid_to_sail @ f_fluid
        # Rotate sail -> boat using tf_tree
        f_boat = tf_tree.vector_to_frame(f_sail, "sail", "boat")

        return f_boat
