"""Basic hull drag model."""

import numpy as np

from sailbench.dynamics.friction import friction_law
from sailbench.models.model import Model, State
from sailbench.tf.tf_tree import TFTree2D


class BasicHullModel(Model):
    """Quadratic hull drag from simple geometry-based coefficients."""

    def compute(self, state: State, tf_tree: TFTree2D) -> np.ndarray:
        """Compute forces on hull model."""
        del tf_tree

        u, v, r = state.u, state.v, state.r

        l = float(self.p["L"])
        b = float(self.p["B"])
        t = float(self.p["T"])
        rho = float(self.p.get("rho_water", 1000.0))

        s = 1.7 * l * (b + t)
        aside = l * t

        # Skin friction. `flat` is the constant 0.004 this used to hardcode;
        # `hughes` varies it with Reynolds number, which is what friction does.
        cf = friction_law(self.p)(float(u), l, self.p)
        k_u = 0.5 * rho * s * cf
        k_v = 0.5 * rho * aside
        k_r = (1.0 / 8.0) * rho * t * (l**4)

        fx = -k_u * u * abs(u)
        fy = -k_v * v * abs(v)
        mz = -k_r * r * abs(r)

        return np.array([fx, fy, mz], dtype=float)
