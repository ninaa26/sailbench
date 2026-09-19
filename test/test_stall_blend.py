"""Past stall a foil is a flat plate, not a section polar read out of range."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sailbench.models.foil import Foil
from sailbench.sim.sailboat_hub import SailboatHub


class _BareFoil(Foil):
    """Foil is abstract; the blending maths lives on it and needs no compute()."""

    def compute(self, state: object, tf_tree: object) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


def _foil(**params: float) -> Foil:
    return _BareFoil({"airfoil_name": "NACA0012", "area": 0.1225, **params})


def test_off_unless_a_separation_angle_is_configured() -> None:
    assert _foil().stall_blending is False
    assert _foil(alpha_sep_deg=25.0).stall_blending is True


def test_not_configured_means_coefficients_pass_through() -> None:
    """The opt-in guarantee, same as the finite-span correction."""
    assert _foil().blend_stall(math.radians(45.0), 1.116, 1.228) == (1.116, 1.228)


def test_plate_normal_force_falls_back_to_the_two_dimensional_value() -> None:
    """Without a span there is nothing to derive it from, so 2.0 is the only honest default."""
    assert _foil(alpha_sep_deg=25.0).plate_normal_force() == pytest.approx(2.0)


@pytest.mark.parametrize(("span", "end_plate", "expected_ar"), [(0.700, 2.0, 8.0), (0.478, 1.0, 1.865)])
def test_plate_normal_force_follows_aspect_ratio(span: float, end_plate: float, expected_ar: float) -> None:
    """CN_max = 1.11 + 0.018 AR. A real plate lets flow escape round its tips."""
    foil = _foil(alpha_sep_deg=25.0, span=span, end_plate_factor=end_plate)
    assert foil.effective_aspect_ratio() == pytest.approx(expected_ar, abs=0.01)
    assert foil.plate_normal_force() == pytest.approx(1.11 + 0.018 * expected_ar, abs=0.001)
    assert foil.plate_normal_force() < 2.0, "a finite plate cannot reach the 2-D value"


def test_an_explicit_cn_plate_wins() -> None:
    assert _foil(alpha_sep_deg=25.0, span=0.7, cn_plate=1.4).plate_normal_force() == pytest.approx(1.4)


def test_attached_flow_is_barely_touched() -> None:
    """At small incidence the separation fraction is near zero, so the polar stands."""
    foil = _foil(alpha_sep_deg=25.0, span=0.700, end_plate_factor=2.0)
    cl, cd = foil.blend_stall(math.radians(2.0), 0.25, 0.015)
    assert cl == pytest.approx(0.25, rel=0.02)
    assert cd == pytest.approx(0.015, rel=0.5)


def test_fully_separated_is_a_flat_plate() -> None:
    """At 90 degrees the foil carries cn_plate in drag and nothing in lift."""
    foil = _foil(alpha_sep_deg=25.0, span=0.700, end_plate_factor=2.0)
    cl, cd = foil.blend_stall(math.pi / 2, 0.09, 2.087)
    assert cl == pytest.approx(0.0, abs=1e-6)
    assert cd == pytest.approx(foil.plate_normal_force(), rel=1e-3)


def test_no_kink_across_stall() -> None:
    """The clamps this replaces left a discontinuous derivative exactly at the limit.

    Step through stall and check the second difference stays bounded: a clamp
    shows up as a spike where the curve suddenly flattens.
    """
    foil = _foil(alpha_sep_deg=25.0, span=0.478, end_plate_factor=1.0)
    angles = np.radians(np.arange(0.0, 60.0, 0.5))
    cl = np.array([foil.blend_stall(a, 0.8, 0.2)[0] for a in angles])
    second = np.abs(np.diff(cl, n=2))
    assert second.max() < 1e-3, f"kink in the lift curve: {second.max()}"


def test_flingo_uses_it_and_other_boats_do_not() -> None:
    flingo = SailboatHub(config_file="flingo_floty.yaml")
    assert flingo.keel.stall_blending
    assert flingo.rudder.stall_blending
    assert flingo.rudder.plate_normal_force() < 2.0

    plain = SailboatHub(config_file="basic_sailbot.yaml")
    assert not plain.keel.stall_blending
    assert not plain.rudder.stall_blending


def test_the_superseded_clamps_are_gone_from_flingo() -> None:
    """Leaving them in a config that blends would be a number doing nothing."""
    rudder = SailboatHub(config_file="flingo_floty.yaml").rudder_cfg
    for key in ("aoa_limit_deg", "cl_max", "cd_max"):
        assert key not in rudder
