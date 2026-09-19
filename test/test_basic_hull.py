"""BasicHullModel resistance tests: skin friction, wave-making, wetted surface."""

import math

import numpy as np

from sailbench.dynamics.basic_hull_model import GRAVITY, BasicHullModel
from sailbench.models.model import State
from sailbench.tf.tf_tree import TFTree2D

BASE = {"L": 1.372, "B": 0.492, "T": 0.110, "rho_water": 1000.0}


def make_state(u: float = 0.0, v: float = 0.0, r: float = 0.0) -> State:
    """Generate a test state."""
    return State.from_array(np.array([0.0, 0.0, 1.0, 0.0, u, v, r]))


def surge(params: dict, u: float) -> float:
    """Surge force from a hull built with `params` at forward speed `u`."""
    return float(BasicHullModel({**BASE, **params}).compute(make_state(u=u), TFTree2D())[0])


class TestResiduaryResistance:
    """Wave-making resistance: the term that gives the hull a top speed."""

    def test_absent_by_default(self) -> None:
        """A config without c_wave keeps its previous resistance exactly."""
        assert surge({}, 2.0) == surge({"c_wave": 0.0}, 2.0)

    def test_opposes_motion_in_both_directions(self) -> None:
        """Wave drag resists travel whichever way the hull is moving."""
        assert surge({"c_wave": 0.01}, 2.0) < 0.0
        assert surge({"c_wave": 0.01}, -2.0) > 0.0

    def test_grows_faster_than_skin_friction(self) -> None:
        """Doubling speed must cost more than the quadratic friction term alone.

        This is the whole point of the term: friction alone scales as u^2 and so
        never stops the boat, which is why the hull used to pass hull speed.
        """
        friction_only = abs(surge({}, 2.0)) / abs(surge({}, 1.0))
        with_waves = abs(surge({"c_wave": 0.01}, 2.0)) / abs(surge({"c_wave": 0.01}, 1.0))
        assert with_waves > friction_only

    def test_negligible_well_below_hull_speed(self) -> None:
        """Below hull speed the quartic should barely matter."""
        v_hull = 0.4 * math.sqrt(GRAVITY * BASE["L"])
        slow = 0.3 * v_hull
        assert abs(surge({"c_wave": 0.01}, slow) - surge({}, slow)) < 0.02 * abs(surge({}, slow))

    def test_dominant_above_hull_speed(self) -> None:
        """Well past hull speed it should be the larger part of the resistance."""
        v_hull = 0.4 * math.sqrt(GRAVITY * BASE["L"])
        fast = 2.0 * v_hull
        assert abs(surge({"c_wave": 0.01}, fast)) > 2.0 * abs(surge({}, fast))


class TestFrictionLine:
    """Hughes skin friction versus the flat coefficient it replaces."""

    def test_flat_by_default(self) -> None:
        """Unset, the friction coefficient stays at the old constant."""
        assert BasicHullModel(BASE)._friction_coefficient(1.5, BASE["L"]) == 0.004

    def test_falls_with_speed(self) -> None:
        """Cf drops as Reynolds number rises; a constant cannot do that."""
        hull = BasicHullModel({**BASE, "friction_model": "hughes"})
        coeffs = [hull._friction_coefficient(u, BASE["L"]) for u in (0.5, 1.0, 2.0)]
        assert coeffs[0] > coeffs[1] > coeffs[2]

    def test_falls_back_when_barely_moving(self) -> None:
        """The line is invalid at tiny Reynolds numbers, so it must not be used."""
        hull = BasicHullModel({**BASE, "friction_model": "hughes"})
        assert hull._friction_coefficient(1e-6, BASE["L"]) == 0.004


class TestWettedSurface:
    """A measured hull area should override the 1.7*L*(B+T) approximation."""

    def test_measured_area_scales_friction(self) -> None:
        """Friction drag is proportional to wetted area."""
        approx = 1.7 * BASE["L"] * (BASE["B"] + BASE["T"])
        half = surge({"wetted_surface_m2": approx / 2.0}, 1.5)
        assert math.isclose(half, surge({}, 1.5) / 2.0, rel_tol=1e-9)

    def test_absent_key_uses_the_approximation(self) -> None:
        """Without the key nothing changes."""
        approx = 1.7 * BASE["L"] * (BASE["B"] + BASE["T"])
        assert math.isclose(surge({"wetted_surface_m2": approx}, 1.5), surge({}, 1.5), rel_tol=1e-9)
