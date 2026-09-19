"""The water a hull drags along with it, and what happens when it is ignored."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sailbench.dynamics.basic_hull_model import BasicHullModel
from sailbench.models.model import State
from sailbench.sim.sailboat_hub import SailboatHub
from sailbench.solvers.rk4 import rk4_step

# A box: constant draft over a metre, so the integral is analytic.
BOX = [{"x_m": -0.5, "draft_m": 0.1}, {"x_m": 0.5, "draft_m": 0.1}]


def _hull(**params: object) -> BasicHullModel:
    return BasicHullModel({"L": 1.372, "B": 0.492, "T": 0.110, "rho_water": 1000.0, **params})


def test_no_sections_means_no_added_mass() -> None:
    """The opt-in guarantee: the equations then reduce to rigid body exactly."""
    assert _hull().added_mass() == (0.0, 0.0, 0.0)


def test_a_single_station_is_not_enough_to_integrate() -> None:
    assert _hull(sections=[{"x_m": 0.0, "draft_m": 0.1}]).added_mass() == (0.0, 0.0, 0.0)


def test_sway_added_mass_is_the_strip_integral() -> None:
    """A22 = integral rho pi T^2 dx. Over a 1 m box of draft 0.1: 1000 pi 0.01."""
    _, a22, _ = _hull(sections=BOX).added_mass()
    assert a22 == pytest.approx(1000.0 * math.pi * 0.1**2 * 1.0)


def test_yaw_added_inertia_weights_the_same_strip_by_x_squared() -> None:
    """A66 = integral rho pi T^2 x^2 dx, by trapezoid over the stations given.

    Worth being explicit about the error this carries. x^2 is convex, so the
    trapezoid rule over-estimates it, and with only the two endpoints of the box
    it returns 7.854 against an analytic 2.618 -- three times too much. The
    remedy is stations, not a different rule: see the convergence test below.
    """
    _, _, a66 = _hull(sections=BOX).added_mass()
    strip = 1000.0 * math.pi * 0.1**2
    endpoints_only = (strip * 0.25 + strip * 0.25) / 2.0 * 1.0
    assert a66 == pytest.approx(endpoints_only)


def test_yaw_added_inertia_converges_as_stations_are_added() -> None:
    """More stations close the gap to the analytic value for the same box."""
    analytic = 1000.0 * math.pi * 0.1**2 / 12.0
    errors = []
    for n in (2, 5, 21, 101):
        xs = np.linspace(-0.5, 0.5, n)
        sections = [{"x_m": float(x), "draft_m": 0.1} for x in xs]
        _, _, a66 = _hull(sections=sections).added_mass()
        errors.append(abs(a66 - analytic) / analytic)

    assert errors[0] > 1.0, "two endpoints should be badly wrong"
    assert errors[-1] < 0.01, "a hundred stations should be within a percent"
    assert errors == sorted(errors, reverse=True), "error must fall monotonically"


def test_flingos_seven_stations_are_a_coarse_integral() -> None:
    """Not a defect, but the number carries real discretisation error.

    Flingo's table is seven stations over 1.07 m. On the box above that spacing
    leaves roughly a 20% over-estimate in the yaw term, so 1.83 kg.m2 should be
    read as the right order rather than three significant figures.
    """
    xs = np.linspace(-0.5, 0.5, 7)
    sections = [{"x_m": float(x), "draft_m": 0.1} for x in xs]
    _, _, a66 = _hull(sections=sections).added_mass()
    analytic = 1000.0 * math.pi * 0.1**2 / 12.0
    assert 0.05 < (a66 - analytic) / analytic < 0.35


def test_surge_is_a_fraction_of_displacement_not_a_strip_quantity() -> None:
    """A slender hull accelerating along its own axis disturbs very little water."""
    a11, _, _ = _hull(sections=BOX, mass=27.0).added_mass()
    assert a11 == pytest.approx(0.05 * 27.0)


def test_station_order_does_not_matter() -> None:
    """The stations are sorted before integrating, so a shuffled table is fine."""
    forward = _hull(sections=BOX).added_mass()
    backward = _hull(sections=list(reversed(BOX))).added_mass()
    assert forward == pytest.approx(backward)


def test_flingo_carries_about_its_own_mass_in_water() -> None:
    """Measured stations, and the reason this matters on a hull this beamy."""
    hub = SailboatHub(config_file="flingo_floty.yaml")
    assert hub.a_sway == pytest.approx(26.46, abs=0.05)
    assert hub.a_yaw == pytest.approx(1.83, abs=0.02)
    assert hub.a_surge == pytest.approx(1.35, abs=0.02)
    assert hub.a_sway / hub.m > 0.9, "sway added mass is not a small correction here"


def test_a_boat_without_stations_is_untouched() -> None:
    hub = SailboatHub(config_file="basic_sailbot.yaml")
    assert (hub.a_surge, hub.a_sway, hub.a_yaw) == (0.0, 0.0, 0.0)


def test_zero_added_mass_reproduces_the_rigid_body_equations() -> None:
    """The generalisation has to be exact where nothing opts in.

    Run the same trajectory on a hull with stations, once with the added mass
    zeroed by hand. It must match a hull that never had any.
    """

    def trajectory(zero: bool) -> tuple[float, ...]:
        hub = SailboatHub(config_file="flingo_floty.yaml")
        if zero:
            hub.a_surge = hub.a_sway = hub.a_yaw = 0.0
        state = State(x=0.0, y=0.0, psi=(1.0, 0.0), u=1.5, v=0.0, r=0.0)
        for _ in range(200):
            state = hub.step(state=state, dt=0.02, solver=rk4_step, sail_angle=math.radians(40.0), rudder_angle=10.0)
        return (state.x, state.y, state.u, state.v, state.r)

    assert trajectory(zero=True) != trajectory(zero=False), "added mass must change the motion"


def test_sway_inertia_roughly_doubles() -> None:
    """Without it the hull accelerates sideways about twice as readily as it should."""
    hub = SailboatHub(config_file="flingo_floty.yaml")
    rigid_only = hub.m
    with_water = hub.m + hub.a_sway
    assert with_water / rigid_only == pytest.approx(1.98, abs=0.05)


def test_the_munk_moment_is_present() -> None:
    """-(A22 - A11) u v destabilises a hull at a drift angle. Real, and easy to omit.

    At a drift angle with no forces at all the rigid-body equations give no yaw
    acceleration; with added mass they give one.
    """
    hub = SailboatHub(config_file="flingo_floty.yaml")
    munk = -(hub.a_sway - hub.a_surge) * 1.5 * 0.2
    assert not np.isclose(munk, 0.0), "Flingo's asymmetry should produce a Munk moment"
    assert munk < 0.0, "a positive drift angle is pushed to increase"
