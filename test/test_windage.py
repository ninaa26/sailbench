"""Windage: above-water drag on mast, rigging, topsides and deck gear."""

import math

import numpy as np
import pytest

from sailbench.dynamics.windage import Windage
from sailbench.models.model import State
from sailbench.tf.tf_tree import TFTree2D, Transform2D

WIND_TO_DEG = 90.0
# Flingo, measured by silhouette projection above the waterline.
MEASURED = {"frontal_area_m2": 0.1818, "lateral_area_m2": 0.3678, "drag_coefficient": 0.8}


def make_windage(**overrides: float) -> Windage:
    """Build a windage model with a fixed wind."""
    return Windage(
        {
            "wind_speed": 5.0,
            "wind_dir_deg": WIND_TO_DEG,
            "rho_air": 1.225,
            **MEASURED,
            **overrides,
        }
    )


def make_state(u: float = 0.0, psi: float = 0.0) -> State:
    """Generate a test state."""
    return State.from_array(np.array([0.0, 0.0, np.cos(psi), np.sin(psi), u, 0.0, 0.0]))


def tree(psi: float = 0.0) -> TFTree2D:
    """Transform tree with the boat on a heading."""
    tf = TFTree2D()
    tf.add_frame(name="boat", parent="world", transform=Transform2D(x=0.0, y=0.0, c=math.cos(psi), s=math.sin(psi)))
    return tf


def beat(twa_deg: float) -> float:
    """Heading `twa_deg` off the true wind."""
    return math.radians(WIND_TO_DEG) - math.pi + math.radians(twa_deg)


class TestDragArea:
    """Projected area blends between bow-on and abeam."""

    def test_bow_on_uses_frontal_area(self) -> None:
        """Dead upwind the boat presents its frontal silhouette."""
        assert make_windage().drag_area_at(0.0) == pytest.approx(0.8 * 0.1818)

    def test_abeam_uses_lateral_area(self) -> None:
        """At 90 degrees it presents its side."""
        assert make_windage().drag_area_at(math.pi / 2) == pytest.approx(0.8 * 0.3678)

    def test_dead_downwind_matches_bow_on(self) -> None:
        """A silhouette from astern is the frontal one again."""
        w = make_windage()
        assert w.drag_area_at(math.pi) == pytest.approx(w.drag_area_at(0.0))

    def test_blends_monotonically_towards_abeam(self) -> None:
        """Area grows steadily from bow-on to abeam."""
        w = make_windage()
        areas = [w.drag_area_at(math.radians(b)) for b in (0, 30, 45, 60, 90)]
        assert areas == sorted(areas)

    def test_symmetric_in_wind_side(self) -> None:
        """Which side the wind comes from cannot change the silhouette."""
        w = make_windage()
        assert w.drag_area_at(math.radians(-45)) == pytest.approx(w.drag_area_at(math.radians(45)))

    def test_lateral_defaults_to_frontal(self) -> None:
        """Given only a frontal area, the model is angle-independent."""
        w = Windage({"frontal_area_m2": 0.2, "drag_coefficient": 1.0})
        assert w.drag_area_at(0.0) == pytest.approx(w.drag_area_at(math.pi / 2))

    def test_explicit_drag_area_is_used_directly(self) -> None:
        """A CD*A product given directly applies at every angle."""
        w = Windage({"drag_area_m2": 0.33})
        assert w.drag_area_at(0.0) == pytest.approx(0.33)
        assert w.drag_area_at(math.pi / 2) == pytest.approx(0.33)


class TestForce:
    """Force direction and magnitude."""

    def test_inert_without_any_area(self) -> None:
        """A windage model with nothing configured produces nothing."""
        assert np.allclose(Windage({"wind_speed": 5.0}).compute(make_state(), tree()), 0.0)

    def test_no_apparent_wind_no_force(self) -> None:
        """Becalmed and stopped means no drag."""
        assert np.allclose(make_windage(wind_speed=0.0).compute(make_state(), tree()), 0.0)

    def test_matches_the_closed_form_bow_on(self) -> None:
        """Head to wind, drag is 0.5 rho V^2 CdA and points dead aft."""
        psi = beat(0.0)
        f = make_windage().compute(make_state(u=0.0, psi=psi), tree(psi))
        assert f[0] == pytest.approx(-0.5 * 1.225 * 25.0 * 0.8 * 0.1818, rel=1e-6)
        assert f[1] == pytest.approx(0.0, abs=1e-9)

    def test_always_opposes_progress_upwind(self) -> None:
        """Beating, windage can only ever slow the boat."""
        for twa in (30, 45, 60, 90):
            psi = beat(twa)
            assert make_windage().compute(make_state(u=1.8, psi=psi), tree(psi))[0] < 0.0

    def test_pushes_the_boat_along_downwind(self) -> None:
        """Running, the same drag helps rather than hinders."""
        psi = beat(180.0)
        assert make_windage().compute(make_state(u=1.5, psi=psi), tree(psi))[0] > 0.0

    def test_grows_with_the_square_of_apparent_wind(self) -> None:
        """Doubling apparent wind quadruples the force."""
        psi = beat(0.0)
        args = (make_state(u=0.0, psi=psi), tree(psi))
        one = make_windage(wind_speed=4.0).compute(*args)
        two = make_windage(wind_speed=8.0).compute(*args)
        assert abs(two[0]) == pytest.approx(4.0 * abs(one[0]), rel=1e-6)

    def test_scales_with_rho_air(self) -> None:
        """Force is proportional to rho_air."""
        psi = beat(45.0)
        args = (make_state(u=1.5, psi=psi), tree(psi))
        light = make_windage(rho_air=1.0).compute(*args)
        heavy = make_windage(rho_air=2.0).compute(*args)
        assert heavy[0] == pytest.approx(2.0 * light[0], rel=1e-9)

    def test_boat_speed_raises_upwind_drag(self) -> None:
        """Sailing faster upwind raises apparent wind and so raises windage."""
        psi = beat(40.0)
        slow = make_windage().compute(make_state(u=0.5, psi=psi), tree(psi))
        fast = make_windage().compute(make_state(u=2.0, psi=psi), tree(psi))
        assert abs(fast[0]) > abs(slow[0])


class TestHubIntegration:
    """The hub must hand windage the same wind the sail is using."""

    def test_wind_is_live_before_any_step(self) -> None:
        """Windage carries wind from construction, not from the first step().

        Regression: it used to be populated only inside step(), so calling
        compute() directly returned a force built on zero wind.
        """
        from sailbench.sim.sailboat_hub import SailboatHub

        hub = SailboatHub("flingo_floty.yaml")
        assert hub.windage is not None
        assert hub.windage_cfg.get("wind_speed") == hub.sail_cfg.get("wind_speed")
        assert hub.windage_cfg.get("wind_dir_deg") == hub.sail_cfg.get("wind_dir_deg")

    def test_force_is_along_the_apparent_wind_without_stepping(self) -> None:
        """The force must be parallel to the apparent wind straight away."""
        import sailbench.utils.coordinate_helper as utils
        from sailbench.sim.sailboat_hub import SailboatHub

        hub = SailboatHub("flingo_floty.yaml")
        psi = beat(60.0)
        st = make_state(u=1.8, psi=psi)
        hub._update_dynamic_frames(st, math.radians(20.0), 0.0, 0.02)
        aw = utils.apparent_wind_boat(st, hub.tf, 5.0, WIND_TO_DEG)
        f = np.asarray(hub.windage.compute(st, hub.tf), dtype=float)
        cos = float(np.dot(f, aw) / (np.linalg.norm(f) * np.linalg.norm(aw)))
        assert cos == pytest.approx(1.0, abs=1e-9)

    def test_changing_the_sails_wind_reaches_windage(self) -> None:
        """The web runner writes wind into sail_cfg; windage must follow."""
        import numpy as np_

        from sailbench.models.model import State as S
        from sailbench.sim.sailboat_hub import SailboatHub
        from sailbench.solvers.rk4 import rk4_step

        hub = SailboatHub("flingo_floty.yaml")
        hub.sail_cfg["wind_speed"] = 11.0
        hub.step(S.from_array(np_.array([0, 0, 1, 0, 1.0, 0, 0], float)), 0.02, rk4_step, 0.3, 0.0)
        assert hub.windage_cfg["wind_speed"] == 11.0
