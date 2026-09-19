"""Basic hull drag model."""

import numpy as np

from sailbench.dynamics.friction import friction_law
from sailbench.models.model import Model, State
from sailbench.tf.tf_tree import TFTree2D

# Fewer stations than this and there is nothing to integrate between.
MIN_SECTIONS = 2


class BasicHullModel(Model):
    """Quadratic hull drag from simple geometry-based coefficients."""

    def added_mass(self) -> tuple[float, float, float]:
        """Surge, sway and yaw added mass, by strip theory over the measured hull.

        Water has to be pushed aside for the hull to accelerate, and the boat
        carries that water with it. For a hull this beamy relative to its length
        that is not a correction: sway added mass comes out roughly equal to the
        boat's own mass, so leaving it out makes the hull slide sideways and spin
        up about twice as readily as it should.

        Each station contributes a 2-D sway added mass of `rho * pi * T^2` per
        unit length, the flat-plate result, and the yaw term is the same
        integrand weighted by x^2:

            A22 = integral rho pi T(x)^2 dx
            A66 = integral rho pi T(x)^2 x^2 dx

        Surge is not a strip-theory quantity. A slender hull accelerating along
        its own axis disturbs very little water, and the usual estimate is a
        small fraction of the displacement, which `added_mass_surge_fraction`
        sets.

        Returns zeros when no `sections` table is configured, so a hull that has
        not opted in behaves exactly as before.
        """
        sections = self.p.get("sections") or []
        if len(sections) < MIN_SECTIONS:
            return 0.0, 0.0, 0.0
        rho = float(self.p.get("rho_water", 1000.0))
        coeff = float(self.p.get("added_mass_section_coeff", 1.0))
        xs = np.array([float(sec["x_m"]) for sec in sections], dtype=float)
        drafts = np.array([float(sec["draft_m"]) for sec in sections], dtype=float)

        order = np.argsort(xs)
        x_sorted = xs[order]
        strip = (rho * np.pi * coeff * drafts * drafts)[order]

        a22 = float(np.trapezoid(strip, x_sorted))
        a66 = float(np.trapezoid(strip * x_sorted**2, x_sorted))
        a11 = float(self.p.get("added_mass_surge_fraction", 0.05)) * float(self.p.get("mass", 0.0))
        return a11, a22, a66

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
