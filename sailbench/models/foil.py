from typing import Any

import numpy as np
from aerosandbox import Airfoil

from sailbench.models.model import Model


class Foil(Model):
    """Foil model with selectable backend (xfoil | neuralfoil)."""

    def __init__(self, params: dict[str, Any]) -> None:
        """Init neuralfoil parameters."""
        self.p = params
        self.airfoil_name = self.p.get("airfoil_name", "NACA0012")

        self.foil = Airfoil(name=self.airfoil_name)

        # Alpha limits (no cache dependency anymore)
        self.alpha_min = self.p.get("alpha_min", -20)
        self.alpha_max = self.p.get("alpha_max", 20)

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
    # CL/CD Interface
    # ------------------------

    def effective_aspect_ratio(self) -> float:
        """Effective aspect ratio, or 0.0 when the config has not stated one.

        Taken from `effective_aspect_ratio` if given, otherwise from `span` and
        `area`. `end_plate_factor` is 2.0 for a foil whose root is sealed
        against the hull, which mirrors the flow and doubles the effective
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
        foil sheds tip vortices, which does two things missing here entirely: it
        flattens the lift-curve slope, and it charges induced drag proportional
        to the square of lift. Without them a keel makes its side force almost
        for free, which is why leeway comes out near zero.

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

    def cl_cd(self, alpha_rad: float, re: float) -> tuple[float, float]:
        """Get CL and CD for a given angle of attack (in radians) and Reynolds number."""
        a_deg = float(np.clip(np.degrees(alpha_rad), self.alpha_min, self.alpha_max))

        result = self.foil.get_aero_from_neuralfoil(
            alpha=a_deg,
            Re=re,
            mach=0.0,
        )
        return float(result["CL"]), float(result["CD"])
