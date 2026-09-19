"""Foil model backed by tabulated NeuralFoil polars."""

from typing import Any

import numpy as np
from aerosandbox import Airfoil

from sailbench.models.model import Model

# Default angle-of-attack resolution of a tabulated polar, in degrees.
# 0.25 deg keeps the worst interpolation error near 3e-3 in CL, which is far
# below the uncertainty of every other term in the force model.
DEFAULT_POLAR_STEP_DEG = 0.25

# Tabulated polars, shared across every Foil instance in the process.
# Key: (airfoil name, Reynolds number, alpha_min, alpha_max, step).
_POLAR_CACHE: dict[tuple[str, int, float, float, float], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def clear_polar_cache() -> None:
    """Drop every tabulated polar. Intended for tests and benchmarks."""
    _POLAR_CACHE.clear()


class Foil(Model):
    """Foil model backed by a tabulated NeuralFoil polar.

    NeuralFoil is evaluated once per (airfoil, Reynolds number, alpha range) over
    a dense sweep of angles of attack; afterwards ``cl_cd`` is a linear
    interpolation into that table. The simulation asks for coefficients twelve
    times per RK4 step, so calling NeuralFoil directly dominates runtime.
    """

    def __init__(self, params: dict[str, Any]) -> None:
        """Init foil parameters."""
        self.p = params
        self.airfoil_name = self.p.get("airfoil_name", "NACA0012")

        self.foil = Airfoil(name=self.airfoil_name)

        # Alpha limits (no cache dependency anymore)
        self.alpha_min = float(self.p.get("alpha_min", -20))
        self.alpha_max = float(self.p.get("alpha_max", 20))
        self.polar_step_deg = float(self.p.get("polar_step_deg", DEFAULT_POLAR_STEP_DEG))

    def get_reynolds(self) -> float:
        """Get Reynolds number from config (supports 're' or 'res' list)."""
        re = self.p.get("re")
        if re is not None:
            return float(re)
        res = self.p.get("res")
        if res is not None:
            return float(res[0]) if isinstance(res, (list, tuple)) else float(res)
        return 1e5


    # ------------------------
    # Finite-span correction
    # ------------------------
    def effective_aspect_ratio(self) -> float:
        """Effective aspect ratio of this foil, or 0.0 when it is not configured.

        Taken from `effective_aspect_ratio` if given, otherwise derived from
        `span` and `area`. `end_plate_factor` is 2.0 for a foil whose root is
        sealed against the hull, which mirrors the flow and doubles the effective
        aspect ratio, and 1.0 for a surface-piercing foil with no image.
        """
        explicit = self.p.get("effective_aspect_ratio")
        if explicit is not None:
            return float(explicit)
        span = float(self.p.get("span", 0.0))
        area = float(self.p.get("area", 0.0))
        if span <= 0.0 or area <= 0.0:
            return 0.0
        return float(self.p.get("end_plate_factor", 1.0)) * span * span / area

    def apply_finite_span(self, cl: float, cd: float) -> tuple[float, float]:
        """Turn 2-D section coefficients into finite-span ones.

        NeuralFoil returns the coefficients of an infinite-span section. A real
        foil sheds tip vortices, which does two things this model was missing
        entirely: it flattens the lift-curve slope, and it charges induced drag
        proportional to the square of lift. Without them a keel makes its side
        force almost for free, which is why leeway came out near zero.

        Lifting-line theory, in the form Buehler et al. (2018) use:

            CL_3d = CL_2d * AR / (AR + 2)
            CD_3d = CD_2d + CL_3d^2 / (pi * AR * e)

        Returns the inputs untouched when no aspect ratio is configured, so a
        foil that has not opted in behaves exactly as before.
        """
        ar = self.effective_aspect_ratio()
        if ar <= 0.0:
            return cl, cd
        e = float(self.p.get("oswald_efficiency", 0.9))
        cl_3d = cl * ar / (ar + 2.0)
        return cl_3d, cd + (cl_3d * cl_3d) / (np.pi * ar * e)

    # ------------------------
    # CL/CD Interface
    # ------------------------
    def _polar(self, re: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (alpha_deg, cl, cd) for this foil, building the table on first use."""
        key = (
            self.airfoil_name,
            round(re),
            self.alpha_min,
            self.alpha_max,
            self.polar_step_deg,
        )
        cached = _POLAR_CACHE.get(key)
        if cached is not None:
            return cached

        # endpoint=True so alpha_max is always sampled exactly, whatever the step.
        n = max(round((self.alpha_max - self.alpha_min) / self.polar_step_deg) + 1, 2)
        alpha_deg = np.linspace(self.alpha_min, self.alpha_max, n)
        result = self.foil.get_aero_from_neuralfoil(alpha=alpha_deg, Re=re, mach=0.0)
        table = (
            alpha_deg,
            np.asarray(result["CL"], dtype=float),
            np.asarray(result["CD"], dtype=float),
        )
        _POLAR_CACHE[key] = table
        return table

    def cl_cd(self, alpha_rad: float, re: float) -> tuple[float, float]:
        """Get CL and CD for a given angle of attack (in radians) and Reynolds number."""
        alpha_deg, cl, cd = self._polar(re)
        a_deg = float(np.clip(np.degrees(alpha_rad), self.alpha_min, self.alpha_max))
        return float(np.interp(a_deg, alpha_deg, cl)), float(np.interp(a_deg, alpha_deg, cd))
