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

    # ------------------------
    # Post-stall blending
    # ------------------------
    @property
    def stall_blending(self) -> bool:
        """Whether a separation angle is configured."""
        return float(self.p.get("alpha_sep_deg", 0.0)) > 0.0

    def plate_normal_force(self) -> float:
        """Normal-force coefficient of the fully separated foil, i.e. CN at 90 degrees.

        `cn_plate` if the config states one. Otherwise derived from the aspect
        ratio, because a flat plate's normal force is not a constant: 2.0 is the
        two-dimensional value, approached only as the span grows without limit,
        and a real plate lets flow escape round its tips. Hoerner
        (*Fluid-Dynamic Drag*, ch. 3) measures 1.18 at AR 1, 1.20 at AR 5, 1.29
        at AR 10 and 1.50 at AR 20; the standard fit to that data,

            CN_max = 1.11 + 0.018 * AR

        is the one Viterna and Janetzke (1982) use for post-stall extrapolation.
        On Flingo it gives 1.25 for the keel (AR 8) and 1.20 for the rudder
        (AR 4.95), against the 2.0 both were charged before.

        Uses the *effective* aspect ratio, so a foil sealed against the hull gets
        the mirrored value: the image plane works on separated flow too.

        Falls back to the 2-D value when no aspect ratio is configured, which is
        the only defensible default for a foil that has not stated its span.
        """
        stated = self.p.get("cn_plate")
        if stated is not None:
            return float(stated)
        ar = self.effective_aspect_ratio()
        if ar <= 0.0:
            return 2.0
        return float(min(1.11 + 0.018 * ar, 2.0))

    def blend_stall(self, alpha_rad: float, cl: float, cd: float) -> tuple[float, float]:
        """Blend attached-flow coefficients towards a flat plate past stall.

        A 2-D section polar is only meaningful while the flow stays attached,
        roughly the first 15 degrees. Past that, clamping the angle and the
        coefficients pins them at a constant and leaves a kink in the derivative
        exactly where a stalled rudder lives. The table underneath is no better:
        this NACA0012 at the rudder's Reynolds number peaks at CL 0.999 at 10
        degrees and then climbs back to 1.116 at 45, which a stalled foil cannot
        do.

        Buehler et al. (2018) blend with a separation fraction

            s(alpha) = 1 - exp(-(alpha / alpha_sep)^2)

        which is 0 at zero incidence, approaches 1 well past stall, and is smooth
        throughout. The separated end is a flat plate carrying a normal force of
        `cn_plate sin(alpha)`, resolved into

            CL_plate = cn_plate sin(alpha) cos(alpha)
            CD_plate = cn_plate sin^2(alpha)

        Returns the inputs untouched when no separation angle is configured.
        """
        if not self.stall_blending:
            return cl, cd
        alpha_sep = np.radians(float(self.p.get("alpha_sep_deg", 25.0)))
        cn_plate = self.plate_normal_force()
        ratio = float(alpha_rad) / alpha_sep
        separated = 1.0 - float(np.exp(-ratio * ratio))
        sin_a, cos_a = np.sin(float(alpha_rad)), np.cos(float(alpha_rad))
        return (
            float((1.0 - separated) * cl + separated * cn_plate * sin_a * cos_a),
            float((1.0 - separated) * cd + separated * cn_plate * sin_a * sin_a),
        )

    def cl_cd(self, alpha_rad: float, re: float) -> tuple[float, float]:
        """Get CL and CD for a given angle of attack (in radians) and Reynolds number."""
        a_deg = float(np.clip(np.degrees(alpha_rad), self.alpha_min, self.alpha_max))

        result = self.foil.get_aero_from_neuralfoil(
            alpha=a_deg,
            Re=re,
            mach=0.0,
        )
        return float(result["CL"]), float(result["CD"])
