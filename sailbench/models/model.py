"""Base physics model."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from sailbench.tf.tf_tree import TFTree2D


@dataclass
class State:
    """State class."""

    x: float
    y: float
    psi: tuple[float, float]  # angle is cosine (tuple[0]) + i*sine (tuple[1])
    u: float  # surge velocity (body x)
    v: float  # sway velocity (body y)
    r: float  # angular velocity

    @property
    def get_heading(self) -> float:
        """Returns angle in degress."""
        return float(np.degrees(np.arctan2(self.psi[1], self.psi[0])))

    @property
    def get_heading_rad(self) -> float:
        """Returns angle in radians."""
        return float(np.arctan2(self.psi[1], self.psi[0]))

    def to_array(self) -> np.ndarray:
        """Convert State to array."""
        return np.array(
            [
                self.x,
                self.y,
                self.psi[0],
                self.psi[1],
                self.u,
                self.v,
                self.r,
            ],
            dtype=float,
        )

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "State":
        """Create State from array."""
        x, y, c, s, u, v, r = arr

        # normalize heading pair (important for numerical drift)
        norm = float(np.hypot(c, s))
        if norm == 0:
            c, s = 1.0, 0.0
        else:
            c /= norm
            s /= norm

        return cls(
            x=float(x),
            y=float(y),
            psi=(float(c), float(s)),
            u=float(u),
            v=float(v),
            r=float(r),
        )

    def __str__(self) -> str:
        """Return string representation of State."""
        return f"State(x={self.x:.2f}, y={self.y:.2f}, psi=({self.psi[0]:.2f}, {self.psi[1]:.2f}), u={self.u:.2f}, v={self.v:.2f}, r={self.r:.2f})"


class Model(ABC):
    """Base class for physics models."""

    # Whether compute() reads wind_speed and wind_dir_deg from its parameters.
    # Wind lives in the sail section; the hub copies it into every part that
    # sets this, so a windage model or a jib cannot sail on a stale wind.
    reads_wind = False

    def __init__(self, params: dict) -> None:
        """Store a dict of parameters (usually from YAML)."""
        self.p = params

    def mass_properties(self) -> tuple[float, float]:
        """Return the mass [kg] and yaw inertia [kg m^2] this part adds to the boat.

        The ``boat`` section is the boat as weighed, so most parts are already
        in it and add nothing. A part that is not, like ballast, overrides this
        and the hub adds it on top.
        """
        return 0.0, 0.0

    @abstractmethod
    def compute(self, state: State, tf_tree: TFTree2D) -> np.ndarray:
        """Compute the outputted force vector produced by the component.

        Args:
            state (State): Current body state of the sailboat -> [x, y, psi, u, v, r]
            tf_tree (TFTree2D): The transform tree for the sailboat

        Returns:
            np.ndarray: Returns X and Y forces in newtons (within component frame)

        """
        ...
