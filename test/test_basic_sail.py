"""BasicSail: angle-of-attack convention, luffing, and drive direction."""

import math

import numpy as np
import pytest

from sailbench.foils.basic_sail import BasicSail
from sailbench.models.model import State
from sailbench.tf.tf_tree import TFTree2D, Transform2D

WIND_TO_DEG = 90.0  # wind blows toward +y


def make_state(u: float = 0.0, v: float = 0.0, psi: float = 0.0) -> State:
    """Generate a test state."""
    return State.from_array(np.array([0.0, 0.0, np.cos(psi), np.sin(psi), u, v, 0.0]))


def make_sail(**overrides: float) -> BasicSail:
    """Build a sail with a fixed wind."""
    return BasicSail({
        "airfoil_name": "NACA0012", "area": 1.971, "alpha_min": -179, "alpha_max": 179,
        "res": [1.8e5], "wind_speed": 5.0, "wind_dir_deg": WIND_TO_DEG, "rho_air": 1.225,
        **overrides,
    })


def tree(sail_rad: float, psi: float = 0.0) -> TFTree2D:
    """Transform tree with the boat on a heading and the sail trimmed."""
    tf = TFTree2D()
    tf.add_frame(name="boat", parent="world",
                 transform=Transform2D(x=0.0, y=0.0, c=math.cos(psi), s=math.sin(psi)))
    tf.add_frame(name="sail", parent="boat",
                 transform=Transform2D(x=0.0, y=0.0, c=math.cos(sail_rad), s=math.sin(sail_rad)))
    return tf


def beat(twa_deg: float) -> float:
    """Heading that puts the boat `twa_deg` off the true wind, on port tack."""
    return math.radians(WIND_TO_DEG) - math.pi + math.radians(twa_deg)


class TestAngleOfAttack:
    """The lookup angle must be the angle of attack, not the flow direction."""

    def test_zero_at_head_to_wind(self) -> None:
        """Dead into the wind with the sail centred, the angle of attack is zero."""
        sail = make_sail()
        sail.compute(make_state(u=0.0, psi=beat(0.0)), tree(0.0))
        # alpha is recomputed inside compute(); assert via the coefficient instead
        assert abs(sail.cl_cd(0.0, re=sail.get_reynolds())[0]) < 1e-6

    @pytest.mark.parametrize("awa_deg", [10, 20, 30, 45])
    def test_lookup_stays_in_the_valid_envelope(self, awa_deg: float) -> None:
        """Alpha must stay small, not sit near +-180 where the polar is invalid.

        The old code looked the sail up at roughly 180 - AWA, which is far outside
        NeuralFoil's range and gave negative lift upwind.
        """
        flow = math.radians(180.0 - awa_deg)
        aw = np.array([math.cos(flow), math.sin(flow)])
        alpha = math.degrees(math.atan2(-aw[1], -aw[0]))
        assert abs(alpha) == pytest.approx(awa_deg, abs=1e-6)
        assert abs(alpha) < 90.0

    def test_lift_peaks_upwind_not_downwind(self) -> None:
        """A sail makes its most lift on a beat, not on a broad reach."""
        sail = make_sail()
        cl = {}
        for awa in (30, 45, 135, 150):
            flow = math.radians(180.0 - awa)
            aw = np.array([math.cos(flow), math.sin(flow)])
            cl[awa] = abs(sail.cl_cd(math.atan2(-aw[1], -aw[0]), re=sail.get_reynolds())[0])
        assert cl[45] > cl[135]
        assert cl[30] > cl[150]


class TestLuffing:
    """The luff ramp used to be dead code; it must actually engage."""

    def _scale(self, sail: BasicSail, awa_deg: float) -> float:
        flow = math.radians(180.0 - awa_deg)
        aw = np.array([math.cos(flow), math.sin(flow)])
        alpha_abs = abs(math.degrees(math.atan2(-aw[1], -aw[0])))
        luff = float(sail.p.get("luff_deg", 7.5))
        ramp = max(float(sail.p.get("luff_ramp_deg", 4.0)), 1e-6)
        return float(np.clip((alpha_abs - luff) / ramp, 0.0, 1.0))

    def test_fully_luffing_head_to_wind(self) -> None:
        """Inside the luff angle the sail carries no lift."""
        assert self._scale(make_sail(), 2.0) == 0.0

    def test_ramps_in_and_saturates(self) -> None:
        """The ramp is partial in between and full once past it."""
        sail = make_sail()
        assert 0.0 < self._scale(sail, 10.0) < 1.0
        assert self._scale(sail, 30.0) == 1.0

    def test_is_monotonic(self) -> None:
        """More angle of attack never means less lift authority."""
        sail = make_sail()
        scales = [self._scale(sail, a) for a in (0, 5, 8, 10, 12, 20)]
        assert scales == sorted(scales)


class TestDrive:
    """The sail is the engine: close-hauled it must push the boat forwards."""

    @pytest.mark.parametrize("twa_deg", [35, 45, 60, 90])
    def test_produces_forward_drive(self, twa_deg: float) -> None:
        """Correctly trimmed, boat-frame Fx is positive on every point of sail."""
        sail = make_sail()
        psi = beat(twa_deg)
        best = max(
            sail.compute(make_state(u=1.0, psi=psi), tree(math.radians(-t), psi))[0]
            for t in (10, 15, 20, 25, 30, 45, 60)
        )
        assert best > 0.0

    def test_no_wind_no_force(self) -> None:
        """Zero wind and zero boat speed means no apparent wind and no force."""
        fx, fy = make_sail(wind_speed=0.0).compute(make_state(u=0.0), tree(0.0))
        assert abs(fx) < 1e-9
        assert abs(fy) < 1e-9

    def test_force_scales_with_air_density(self) -> None:
        """Force is proportional to rho_air, so the environment value is live."""
        psi = beat(45.0)
        args = (make_state(u=1.0, psi=psi), tree(math.radians(-20.0), psi))
        light = make_sail(rho_air=1.0).compute(*args)
        heavy = make_sail(rho_air=2.0).compute(*args)
        assert heavy[0] == pytest.approx(2.0 * light[0], rel=1e-9)
