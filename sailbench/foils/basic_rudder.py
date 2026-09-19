"""Basic rudder foil model."""

from typing import Any

import numpy as np

from sailbench.models.foil import Foil
from sailbench.models.model import State
from sailbench.tf.tf_tree import TFTree2D


class BasicRudder(Foil):
    """Basic rudder foil model."""

    def __init__(self, params: dict[str, Any]) -> None:
        """Initialize the BasicRudder model.

        Args:
            params (dict): Dictionary of parameters for the rudder model.

        """
        super().__init__(params)

    def compute(self, state: State, tf_tree: TFTree2D) -> np.ndarray:
        """Compute the lift and drag coefficients for the rudder.

        Args:
            state (np.ndarray): Current boat state.
            tf_tree (TFTree2D): Current transform tree of the boat, used to get component positions.

        Returns:
            np.ndarray: Returns X and Y forces in newtons (within rudder frame)


        """
        # Use local inflow at rudder position, including yaw-rate contribution.
        # Body velocity at point (x, y): [u - r*y, v + r*x].
        x_pos = float(self.p.get("x_pos", 0.0))
        y_pos = float(self.p.get("y_pos", 0.0))
        u_local = float(state.u - state.r * y_pos)
        v_local = float(state.v + state.r * x_pos)

        v_local_boat = np.array([u_local, v_local], dtype=float)
        speed = float(np.hypot(u_local, v_local))
        if speed < 1e-6:
            return np.array([0.0, 0.0], dtype=float)

        # Resolve local velocity in rudder frame; AoA is opposite of local track angle.
        v_local_rudder = tf_tree.vector_to_frame(v_local_boat, "boat", "rudder")
        track = float(np.arctan2(v_local_rudder[1], v_local_rudder[0]))
        aoa = -track

        # Prevent unrealistically large coefficients at extreme deflection/stall.
        # Blending supersedes the aoa_limit_deg / cl_max / cd_max clamps: those
        # pin the coefficients at a constant past stall and leave a kink in the
        # derivative right where a stalled rudder operates. A config that has not
        # set alpha_sep_deg keeps the old clamps and the old behaviour.
        if not self.stall_blending:
            aoa_limit_deg = float(self.p.get("aoa_limit_deg", 25.0))
            aoa = float(np.clip(aoa, -np.radians(aoa_limit_deg), np.radians(aoa_limit_deg)))
        cl, cd = self.cl_cd(aoa, re=self.get_reynolds())
        # 2-D section -> finite span. A no-op unless the config states a span.
        cl, cd = self.apply_finite_span(cl, cd)
        if self.stall_blending:
            cl, cd = self.blend_stall(aoa, cl, cd)
        else:
            cl = float(np.clip(cl, -float(self.p.get("cl_max", 1.0)), float(self.p.get("cl_max", 1.0))))
            cd = float(np.clip(cd, 0.0, float(self.p.get("cd_max", 1.2))))

        # Dynamic pressure and net foil forces.
        rho = float(self.p.get("rho_water", 1000.0))  # kg/m^3
        q = 0.5 * rho * speed**2
        area = float(self.p.get("area", 1.0))  # m^2
        effectiveness = float(self.p.get("effectiveness", 0.25))

        drag = cd * q * area * effectiveness
        lift = cl * q * area * effectiveness

        # Resolve in the flow frame, whose +x axis is the rudder's direction of travel:
        # drag opposes that motion, lift acts perpendicular to it.
        f_flow = np.array([-drag, lift], dtype=float)
        c, s = np.cos(track), np.sin(track)
        r_flow_to_rudder = np.array([[c, -s], [s, c]], dtype=float)
        f_rudder = r_flow_to_rudder @ f_flow
        return tf_tree.vector_to_frame(f_rudder, "rudder", "boat")
