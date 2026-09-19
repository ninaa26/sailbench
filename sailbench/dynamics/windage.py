"""Above-water drag: mast, rigging and hull topsides.

Separate from the sail on purpose. ORC VPP tabulates *sail* coefficients (Table
5.1) and adds windage as its own term, because the two are different objects: the
sail is a lifting surface that can be trimmed and depowered, while the mast,
rigging and freeboard are bluff bodies that always cost drag and never make
drive.

Leaving it out is not a small omission. Measured on Flingo with the ORC sail and
no windage, the rig's drag angle came out at 7.7 degrees against a published
12-15, and that shortfall is most of the reason the boat pointed about 8 degrees
higher than a real monohull.

The projected area depends on where the wind is coming from -- on Flingo it is
0.1818 m^2 bow-on against 0.3678 m^2 abeam, a factor of two -- so a single number
cannot be right at every angle. The two are blended with the usual elliptical
interpolation::

    A(beta) = A_frontal cos^2(beta) + A_lateral sin^2(beta)

There is no lift, because a bluff body at arbitrary heading has none worth
modelling at this fidelity, and no heeling moment, because the simulator is
3-DOF. The measured centroid height is recorded in the config for when it gains
a heel degree of freedom.
"""

from typing import Any

import numpy as np

import sailbench.utils.coordinate_helper as utils
from sailbench.models.model import Model, State
from sailbench.tf.tf_tree import TFTree2D


class Windage(Model):
    """Parasitic aerodynamic drag on everything above the waterline.

    Config keys:
        frontal_area_m2: projected silhouette seen bow-on, above the waterline.
        lateral_area_m2: projected silhouette seen abeam. Defaults to the frontal
            area, which makes the model angle-independent.
        drag_coefficient: bluff-body CD, default 0.8.
        drag_area_m2: alternative to the above -- a CD*A product used directly at
            every angle. Ignored when frontal_area_m2 is given.
        wind_speed, wind_dir_deg: true wind. The hub keeps these in step with
            the sail's, so both see the same wind.
        air_density: [kg/m^3], default 1.225. The key the sail models use.
        x_pos, y_pos: position relative to the centre of rotation, used by the
            hub to turn this force into a yaw moment.
    """

    def __init__(self, params: dict[str, Any]) -> None:
        """Initialize the windage model."""
        super().__init__(params)
        self.frontal_area = float(self.p.get("frontal_area_m2", 0.0))
        self.lateral_area = float(self.p.get("lateral_area_m2", self.frontal_area))
        self.drag_coefficient = float(self.p.get("drag_coefficient", 0.8))
        # Fallback for a config that states a CD*A product directly.
        self.drag_area = float(self.p.get("drag_area_m2", 0.0))

    def drag_area_at(self, beta_rad: float) -> float:
        """CD*A for wind `beta_rad` off the bow, blending frontal and lateral."""
        if self.frontal_area <= 0.0:
            return self.drag_area
        c, s = np.cos(beta_rad), np.sin(beta_rad)
        return self.drag_coefficient * (self.frontal_area * c * c + self.lateral_area * s * s)

    def compute(self, state: State, tf_tree: TFTree2D) -> np.ndarray:
        """Return ``[Fx, Fy]`` in the boat frame [N]."""
        if self.frontal_area <= 0.0 and self.drag_area <= 0.0:
            return np.zeros(2, dtype=float)

        aw = utils.apparent_wind_boat(
            state,
            tf_tree,
            float(self.p.get("wind_speed", 0.0)),
            float(self.p.get("wind_dir_deg", 0.0)),
        )
        speed = float(np.hypot(aw[0], aw[1]))
        if speed < 1e-6:
            return np.zeros(2, dtype=float)

        # Apparent wind angle off the bow: the wind blows FROM -aw.
        beta = float(np.arctan2(-aw[1], -aw[0]))

        # Drag acts along the apparent wind, i.e. the air pushes the boat the way
        # it is travelling. Magnitude 0.5 * rho * V^2 * CdA.
        rho = float(self.p.get("air_density", 1.225))
        return np.asarray(0.5 * rho * speed * self.drag_area_at(beta) * aw, dtype=float)
