"""A foil with a span sheds tip vortices; one without behaves as it always did."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sailbench.models.foil import Foil
from sailbench.sim.sailboat_hub import SailboatHub


class _BareFoil(Foil):
    """Foil is abstract; the finite-span maths lives on it and needs no compute()."""

    def compute(self, state: object, tf_tree: object) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


def _foil(**params: float) -> Foil:
    return _BareFoil({"airfoil_name": "NACA0012", "area": 0.1225, **params})


def test_no_span_means_no_aspect_ratio() -> None:
    """A foil that has not opted in reports 0.0, which is the switch for everything else."""
    assert _foil().effective_aspect_ratio() == 0.0


def test_aspect_ratio_from_span_and_area() -> None:
    """Geometric AR is span^2 / area: 0.700^2 / 0.1225 = 4.0."""
    assert _foil(span=0.700).effective_aspect_ratio() == pytest.approx(4.0)


def test_an_end_plate_mirrors_the_flow() -> None:
    """A root sealed against the hull doubles the effective aspect ratio."""
    assert _foil(span=0.700, end_plate_factor=2.0).effective_aspect_ratio() == pytest.approx(8.0)


def test_an_explicit_aspect_ratio_wins() -> None:
    """Stating it directly beats deriving it, for a foil whose span is awkward to define."""
    assert _foil(span=0.700, effective_aspect_ratio=3.0).effective_aspect_ratio() == pytest.approx(3.0)


def test_without_an_aspect_ratio_the_coefficients_pass_through() -> None:
    """The opt-in guarantee: no span configured, no change to any force."""
    foil = _foil()
    assert foil.apply_finite_span(1.2, 0.02) == (1.2, 0.02)


def test_finite_span_flattens_lift_and_charges_induced_drag() -> None:
    """Lifting line: CL_3d = CL_2d * AR/(AR+2), CD_3d = CD_2d + CL_3d^2/(pi AR e)."""
    foil = _foil(span=0.700, end_plate_factor=2.0)  # AR 8
    cl, cd = foil.apply_finite_span(1.2, 0.02)

    expected_cl = 1.2 * 8.0 / 10.0
    expected_cd = 0.02 + expected_cl**2 / (math.pi * 8.0 * 0.9)
    assert cl == pytest.approx(expected_cl)
    assert cd == pytest.approx(expected_cd)
    assert cl < 1.2, "lift-curve slope must flatten"
    assert cd > 0.02, "induced drag must be charged"


def test_a_stubbier_foil_pays_more() -> None:
    """Induced drag goes as 1/AR, so the short foil loses more of its lift to drag."""
    long_foil = _foil(span=0.700, end_plate_factor=2.0)  # AR 8
    stub = _foil(span=0.350, end_plate_factor=1.0)  # AR 1
    cl_long, cd_long = long_foil.apply_finite_span(1.2, 0.02)
    cl_stub, cd_stub = stub.apply_finite_span(1.2, 0.02)

    assert cl_stub < cl_long
    assert cd_stub > cd_long


def test_flingo_keel_and_rudder_are_configured_for_it() -> None:
    """The measured spans, and the end plate only where the root is sealed."""
    hub = SailboatHub(config_file="flingo_floty.yaml")
    assert hub.keel.effective_aspect_ratio() == pytest.approx(8.0, abs=0.01)
    assert hub.rudder.effective_aspect_ratio() == pytest.approx(4.95, abs=0.01)


def test_a_boat_without_spans_is_untouched() -> None:
    """basic_sailbot states no span, so its foils must still report nothing."""
    hub = SailboatHub(config_file="basic_sailbot.yaml")
    assert hub.keel.effective_aspect_ratio() == 0.0
    assert hub.rudder.effective_aspect_ratio() == 0.0


def test_leeway_rises_once_the_keel_pays_for_its_lift() -> None:
    """The point of the change, at boat level.

    A 2-D keel makes side force almost for free, so the boat barely drifts. With
    induced drag charged it has to work at a real angle of attack.
    """
    hub = SailboatHub(config_file="flingo_floty.yaml")
    from sailbench.models.model import State

    state = State(x=0.0, y=0.0, psi=(1.0, 0.0), u=1.5, v=0.12, r=0.0)
    hub._update_dynamic_frames(state, math.radians(30.0), 0.0, 0.02)  # noqa: SLF001
    with_span = np.asarray(hub.keel.compute(state, hub.tf), dtype=float)

    hub.keel.p.pop("span")
    without_span = np.asarray(hub.keel.compute(state, hub.tf), dtype=float)

    assert abs(with_span[1]) < abs(without_span[1]), "side force should cost more lift"
    assert with_span[0] < without_span[0], "and more drag"
