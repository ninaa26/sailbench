"""Basic hull drag model."""

import numpy as np

from sailbench.models.model import Model, State
from sailbench.tf.tf_tree import TFTree2D

GRAVITY = 9.81  # [m/s^2]


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

        # 1.7*L*(B+T) is a rough stand-in; prefer a measured hull area when the
        # config carries one. On flingo the approximation is about 2.5x high.
        s = float(self.p.get("wetted_surface_m2") or 1.7 * l * (b + t))
        aside = l * t

        k_u = 0.5 * rho * s * self._friction_coefficient(u, l)
        k_v = 0.5 * rho * aside
        k_r = (1.0 / 8.0) * rho * t * (l**4)

        fx = -k_u * u * abs(u)
        fy = -k_v * v * abs(v)
        mz = -k_r * r * abs(r)

        fx += self._residuary_resistance(u, rho, l, t)

        return np.array([fx, fy, mz], dtype=float)

    def _friction_coefficient(self, u: float, l: float) -> float:
        """Skin-friction coefficient, times a form factor.

        The flat 0.004 this replaces is a plausible mid-range number but it does
        not vary with speed, and skin friction is the one term here that has a
        well-established empirical line. `friction_model: hughes` opts in to it:

            Re = 0.85 * |u| * L / nu     (0.85 accounts for the boundary layer
                                          not running the full waterline)
            Cf = 0.066 / (log10(Re) - 2.03)^2
            ff = 1.05                     (form factor: a hull is not a flat plate)

        Left unset, the coefficient stays at the previous constant so nothing
        changes for a config that has not opted in.
        """
        if str(self.p.get("friction_model", "flat")).lower() != "hughes":
            return 0.004

        nu = float(self.p.get("nu_water", 1.19e-6))  # [m^2/s] fresh water, ~15 C
        re = 0.85 * abs(u) * l / nu
        if re < 1.0e4:  # below this the line is not valid and Cf is not the story
            return 0.004
        cf = 0.066 / (np.log10(re) - 2.03) ** 2
        return cf * float(self.p.get("form_factor", 1.05))

    def _residuary_resistance(self, u: float, rho: float, l: float, t: float) -> float:
        """Wave-making resistance, the term that makes a displacement hull have a top speed.

        The model above is skin friction only, which grows as u^2 and so never
        stops the boat: nothing here resisted it past hull speed. For a hull of
        this size wave-making dominates above roughly Froude 0.3, and its absence
        is why the boat reached Froude 0.77 against a hull-speed scale of 0.4.

        Buehler et al. (Robotic Sailing, 2018) use a quartic in the ratio of speed
        to hull speed, which is a hard enough wall to cap the boat near
        v_hull = 0.4*sqrt(g*L) without a discontinuity. `c_wave` sets how hard.

        Absent from the config, `c_wave` is zero and this term does nothing, so
        configs that have not opted in keep their previous behaviour exactly.
        """
        c_wave = float(self.p.get("c_wave", 0.0))
        if c_wave <= 0.0 or abs(u) < 1e-9:
            return 0.0

        v_hull = 0.4 * np.sqrt(GRAVITY * l)
        q = 0.5 * rho * u * u
        return float(-np.sign(u) * c_wave * q * (l * t) * (abs(u) / v_hull) ** 4)
