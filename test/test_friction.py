"""Skin friction varies with speed, and a config picks which law says how."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sailbench.dynamics.friction import (
    DEFAULT_FRICTION_LAW,
    FRICTION_LAWS,
    LAMINAR_RE,
    RE_LENGTH_FRACTION,
    flat_plate,
    friction_law,
    hughes,
)
from sailbench.models.model import State
from sailbench.sim.sailboat_hub import SailboatHub

FLINGO_L = 1.372
NU = 1.19e-6


def test_a_hull_naming_no_law_gets_the_constant() -> None:
    """The previous behaviour, so existing configs do not move."""
    assert DEFAULT_FRICTION_LAW == "flat"
    assert friction_law({}) is flat_plate


@pytest.mark.parametrize("name", sorted(FRICTION_LAWS))
def test_every_named_law_resolves(name: str) -> None:
    assert friction_law({"friction_model": name}) is FRICTION_LAWS[name]


def test_a_typo_is_an_error_not_a_fallback() -> None:
    """`hugues` used to silently mean the flat coefficient."""
    with pytest.raises(ValueError, match="does not exist") as excinfo:
        friction_law({"friction_model": "hugues"})
    for known in FRICTION_LAWS:
        assert known in str(excinfo.value)


def test_the_flat_law_ignores_speed() -> None:
    """Which is the one thing skin friction is known not to do."""
    assert flat_plate(0.5, FLINGO_L, {}) == flat_plate(3.0, FLINGO_L, {}) == 0.004


def test_hughes_falls_with_reynolds_number() -> None:
    slow = hughes(0.5, FLINGO_L, {"nu_water": NU})
    fast = hughes(3.0, FLINGO_L, {"nu_water": NU})
    assert slow > fast, "the coefficient must fall as Re rises"


def test_hughes_matches_the_ittc_line() -> None:
    """Cf = 0.066 / (log10(Re) - 2.03)^2, times the form factor."""
    u = 1.5
    re = RE_LENGTH_FRACTION * u * FLINGO_L / NU
    expected = 0.066 / (math.log10(re) - 2.03) ** 2 * 1.05
    assert hughes(u, FLINGO_L, {"nu_water": NU}) == pytest.approx(expected)


def test_hughes_falls_back_below_the_laminar_limit() -> None:
    """The line does not hold there, and the flow is not turbulent anyway."""
    crawl = LAMINAR_RE * NU / (RE_LENGTH_FRACTION * FLINGO_L) * 0.5
    assert hughes(crawl, FLINGO_L, {"nu_water": NU}) == flat_plate(crawl, FLINGO_L, {})


def test_the_two_laws_cross_in_the_middle_of_the_range() -> None:
    """Hughes is the higher number at low speed and the lower one at high speed.

    That is the whole point: the flat 0.004 is a mid-range compromise, so it
    understates drag accelerating out of a tack and overstates it at full speed.
    """
    params = {"nu_water": NU}
    assert hughes(0.25, FLINGO_L, params) > flat_plate(0.25, FLINGO_L, params)
    assert hughes(3.0, FLINGO_L, params) < flat_plate(3.0, FLINGO_L, params)


def test_flingo_uses_it_and_the_others_do_not() -> None:
    assert SailboatHub(config_file="flingo_floty.yaml").hull_cfg["friction_model"] == "hughes"
    assert "friction_model" not in SailboatHub(config_file="basic_sailbot.yaml").hull_cfg


def test_the_hull_force_follows_the_law() -> None:
    """At boat level, not just in the coefficient."""
    hub = SailboatHub(config_file="flingo_floty.yaml")
    state = State(x=0.0, y=0.0, psi=(1.0, 0.0), u=0.25, v=0.0, r=0.0)
    with_hughes = float(np.asarray(hub.hull.compute(state, hub.tf), dtype=float)[0])

    hub.hull_cfg["friction_model"] = "flat"
    with_flat = float(np.asarray(hub.hull.compute(state, hub.tf), dtype=float)[0])

    assert abs(with_hughes) > abs(with_flat), "crawling, Hughes must resist more"
