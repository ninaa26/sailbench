"""Ballast: dead weight the boat carries, in a keel bulb or as plates in the hull.

Its real job is righting moment, and this simulator cannot see that: it is 3-DOF
(surge, sway, yaw) with no heel. What it can see is the weight. Ballast is more
mass to accelerate, and more yaw inertia the further it sits from the centre of
rotation, and that is all this model contributes. It makes no force.

With a ``ballast`` section, the ``boat`` section's mass and inertia_z are the
boat without it, and the hub adds this on top. A config whose ``boat.mass`` was
weighed with the ballast aboard should not list it here as well, or it is
counted twice.

Righting moment has somewhere to go already: the ORC sails depower against
``max_heeling_moment_nm``. It is set there from a measurement rather than
derived from this, because hull form stability contributes to it too.
"""

from typing import Any

import numpy as np

from sailbench.models.model import Model, State
from sailbench.tf.tf_tree import TFTree2D


class Ballast(Model):
    """Weight added to the boat's rigid body.

    Config keys:
        mass: [kg] required, positive.
        x_pos, y_pos: [m] where it sits relative to the centre of rotation,
            default 0. Off-centre ballast adds its parallel-axis inertia,
            ``mass * (x_pos^2 + y_pos^2)``. The hub writes its equations of
            motion about the centre of rotation as if that were the centre of
            gravity, so the CG shift it would also cause is not modelled: keep
            it near the origin, or re-derive the boat's own numbers with it
            aboard.
        inertia_z: [kg m^2] its own yaw inertia about its centroid, default 0
            (a point mass). Plates spread along the hull have some.
        enabled: set false to sail without it, keeping the numbers.
    """

    def __init__(self, params: dict[str, Any]) -> None:
        """Initialize the ballast model."""
        super().__init__(params)
        if "mass" not in self.p:
            msg = "ballast needs a mass in kg"
            raise ValueError(msg)
        self.mass = float(self.p["mass"])
        self.own_inertia = float(self.p.get("inertia_z", 0.0))
        if self.mass <= 0.0 or self.own_inertia < 0.0:
            msg = f"ballast mass must be positive and inertia_z non-negative: got {self.mass}, {self.own_inertia}"
            raise ValueError(msg)

    def mass_properties(self) -> tuple[float, float]:
        """Return the mass and yaw inertia about the centre of rotation."""
        x = float(self.p.get("x_pos", 0.0))
        y = float(self.p.get("y_pos", 0.0))
        return self.mass, self.own_inertia + self.mass * (x * x + y * y)

    def compute(self, state: State, tf_tree: TFTree2D) -> np.ndarray:
        """Return ``[Fx, Fy]``: nothing, since weight acts out of the plane."""
        del state, tf_tree
        return np.zeros(2, dtype=float)
