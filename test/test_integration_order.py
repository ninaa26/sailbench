"""RK4 has to actually be fourth order, or the four force evaluations are wasted."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sailbench.models.model import State
from sailbench.sim.sailboat_hub import SailboatHub
from sailbench.solvers.rk4 import rk4_step

CONFIG = "flingo_floty.yaml"


def _run(dt: float, rudder_deg: float, duration: float = 1.5) -> np.ndarray:
    hub = SailboatHub(config_file=CONFIG)
    state = State(x=0.0, y=0.0, psi=(1.0, 0.0), u=1.5, v=0.0, r=0.0)
    for _ in range(int(duration / dt)):
        state = hub.step(state=state, dt=dt, solver=rk4_step, sail_angle=math.radians(35.0), rudder_angle=rudder_deg)
    return np.array([state.x, state.y, state.u, state.v, state.r])


def _order(rudder_deg: float) -> float:
    reference = _run(0.0025, rudder_deg)
    errors = [float(np.linalg.norm(_run(dt, rudder_deg) - reference)) for dt in (0.04, 0.02, 0.01)]
    pairs = zip(errors[:-1], errors[1:], strict=True)
    return float(np.mean([math.log2(a / b) for a, b in pairs]))


def test_the_integrator_is_fourth_order_with_the_helm_still() -> None:
    """Halving dt must cut the error by about sixteen, not two.

    It used to converge at 1.1: the frames held the heading from the end of the
    previous step, so all four stages evaluated forces against one stale
    attitude.
    """
    assert _order(0.0) > 3.5


def test_a_slewing_rudder_still_limits_the_order() -> None:
    """The half of the problem this change does not fix, pinned so it is not lost.

    The rudder slews toward its command at a fixed rate with forward Euler,
    which is first order in dt whatever the integrator around it does. With the
    frames fixed the helm is now the bottleneck: order 4.1 with the rudder
    still, about 1.9 while it moves.

    An exact first-order lag has a closed-form solution for a command held
    across the step, so it contributes no integration error at any step size.
    That is the actuator change, and this test should be raised to 3.5 when it
    lands.
    """
    steering = _order(12.0)
    assert steering < 3.0, "if this now passes 3.5, the actuator work is done -- tighten it"
    assert steering > 1.0


def test_the_frames_follow_the_state_inside_a_stage() -> None:
    """The mechanism, directly: placing frames must move the boat frame."""
    hub = SailboatHub(config_file=CONFIG)
    hub._set_kinematic_frames(State(x=0.0, y=0.0, psi=(1.0, 0.0), u=1.0, v=0.0, r=0.0), 0.5)  # noqa: SLF001
    heading_before = (hub.tf.transforms["boat"].c, hub.tf.transforms["boat"].s)

    turned = State(x=3.0, y=4.0, psi=(0.0, 1.0), u=1.0, v=0.0, r=0.0)
    hub._set_kinematic_frames(turned, 0.5)  # noqa: SLF001
    heading_after = (hub.tf.transforms["boat"].c, hub.tf.transforms["boat"].s)

    assert heading_before != heading_after


def test_placing_frames_does_not_move_the_rudder() -> None:
    """The split this rests on: a stage may sample the rudder, never advance it.

    Advancing per stage would apply one step of slew four times over.
    """
    hub = SailboatHub(config_file=CONFIG)
    hub._rudder_angle_deg = 5.0  # noqa: SLF001
    state = State(x=0.0, y=0.0, psi=(1.0, 0.0), u=1.5, v=0.0, r=0.0)
    for _ in range(4):
        hub._set_kinematic_frames(state, 0.5)  # noqa: SLF001
    assert hub._rudder_angle_deg == 5.0  # noqa: SLF001


def test_a_step_advances_the_rudder_exactly_once() -> None:
    """One step's worth of slew, not four."""
    hub = SailboatHub(config_file=CONFIG)
    hub._rudder_angle_deg = 0.0  # noqa: SLF001
    state = State(x=0.0, y=0.0, psi=(1.0, 0.0), u=1.5, v=0.0, r=0.0)
    hub.step(state=state, dt=0.02, solver=rk4_step, sail_angle=0.5, rudder_angle=35.0)

    max_rate = float(hub.rudder_cfg.get("max_rate_deg_s", 120.0))
    assert abs(hub._rudder_angle_deg) <= max_rate * 0.02 + 1e-9  # noqa: SLF001


def test_the_rudder_actually_steers() -> None:
    """Guard against the whole command path going missing.

    While writing this change the call that advances the rudder was dropped from
    step(), so `rudder_angle` was ignored entirely and the boat had no steering.
    The suite went green anyway: 213 tests passed with a boat that could not
    turn. Nothing was checking the one thing a rudder is for.
    """

    def yaw_after(rudder_deg: float) -> float:
        hub = SailboatHub(config_file=CONFIG)
        state = State(x=0.0, y=0.0, psi=(1.0, 0.0), u=1.5, v=0.0, r=0.0)
        for _ in range(100):
            state = hub.step(
                state=state, dt=0.02, solver=rk4_step, sail_angle=math.radians(35.0), rudder_angle=rudder_deg
            )
        return state.r

    straight = yaw_after(0.0)
    hard_over = yaw_after(35.0)
    assert abs(hard_over - straight) > math.radians(5.0), "35 degrees of rudder must change the yaw rate"


def test_the_rudder_reaches_its_commanded_angle() -> None:
    """And the command must actually arrive, not just perturb something."""
    hub = SailboatHub(config_file=CONFIG)
    state = State(x=0.0, y=0.0, psi=(1.0, 0.0), u=1.5, v=0.0, r=0.0)
    for _ in range(50):
        state = hub.step(state=state, dt=0.02, solver=rk4_step, sail_angle=0.5, rudder_angle=20.0)
    assert hub._rudder_angle_deg == pytest.approx(20.0, abs=0.5)  # noqa: SLF001


def test_opposite_rudder_turns_the_other_way() -> None:
    """A sign error would pass both tests above."""

    def yaw_after(rudder_deg: float) -> float:
        hub = SailboatHub(config_file=CONFIG)
        state = State(x=0.0, y=0.0, psi=(1.0, 0.0), u=1.5, v=0.0, r=0.0)
        for _ in range(100):
            state = hub.step(
                state=state, dt=0.02, solver=rk4_step, sail_angle=math.radians(35.0), rudder_angle=rudder_deg
            )
        return state.r

    assert yaw_after(25.0) < yaw_after(-25.0)
