"""Finite-span correction shared by every Foil (keel, rudder and sail)."""

import math

import numpy as np
import pytest

from sailbench.models.foil import Foil
from sailbench.models.model import State
from sailbench.tf.tf_tree import TFTree2D


class _Foil(Foil):
    """Concrete Foil; the correction under test is independent of compute()."""

    def compute(self, state: State, tf_tree: TFTree2D) -> np.ndarray:
        """Not exercised: the correction under test is independent of compute()."""
        raise NotImplementedError


def foil(**params: float) -> _Foil:
    """Build a foil with the given geometry."""
    return _Foil({"airfoil_name": "NACA0012", **params})


class TestEffectiveAspectRatio:
    """Where the aspect ratio comes from."""

    def test_derived_from_span_and_area(self) -> None:
        """AR is span^2 / area."""
        assert math.isclose(foil(span=0.7, area=0.1225).effective_aspect_ratio(), 4.0)

    def test_end_plate_factor_multiplies(self) -> None:
        """A root sealed against the hull mirrors the flow and doubles AR."""
        assert math.isclose(foil(span=0.7, area=0.1225, end_plate_factor=2.0).effective_aspect_ratio(), 8.0)

    def test_explicit_value_wins(self) -> None:
        """An explicit effective_aspect_ratio overrides the geometry."""
        assert foil(span=0.7, area=0.1225, effective_aspect_ratio=3.0).effective_aspect_ratio() == 3.0

    @pytest.mark.parametrize("params", [{}, {"span": 0.7}, {"area": 0.1225}, {"span": 0.0, "area": 0.1225}])
    def test_unconfigured_is_zero(self, params: dict) -> None:
        """Without both span and area there is no aspect ratio to apply."""
        assert foil(**params).effective_aspect_ratio() == 0.0


class TestFiniteSpanCorrection:
    """What the correction does to 2-D section coefficients."""

    def test_inert_when_unconfigured(self) -> None:
        """A foil that has not opted in is left exactly as it was."""
        assert foil().apply_finite_span(0.8, 0.02) == (0.8, 0.02)

    def test_reduces_lift(self) -> None:
        """Finite span flattens the lift-curve slope."""
        cl, _ = foil(span=0.7, area=0.1225).apply_finite_span(1.0, 0.02)
        assert cl == pytest.approx(1.0 * 4.0 / 6.0)

    def test_adds_induced_drag(self) -> None:
        """Drag rises by CL^2 / (pi * AR * e), never falls."""
        f = foil(span=0.7, area=0.1225)
        cl3, cd3 = f.apply_finite_span(1.0, 0.02)
        assert cd3 == pytest.approx(0.02 + cl3**2 / (math.pi * 4.0 * 0.9))
        assert cd3 > 0.02

    def test_no_induced_drag_without_lift(self) -> None:
        """At zero lift there is no induced drag to charge."""
        assert foil(span=0.7, area=0.1225).apply_finite_span(0.0, 0.02) == (0.0, 0.02)

    def test_induced_drag_is_quadratic_in_lift(self) -> None:
        """Doubling lift quadruples the induced part."""
        f = foil(span=0.7, area=0.1225)
        one = f.apply_finite_span(0.5, 0.0)[1]
        two = f.apply_finite_span(1.0, 0.0)[1]
        assert two == pytest.approx(4.0 * one)

    def test_higher_aspect_ratio_costs_less_for_the_same_side_force(self) -> None:
        """A deeper foil of the same area buys the same lift for less drag.

        The comparison has to be made at equal *delivered* lift. At equal 2-D CL
        the two come out identical, because AR/(AR+2)^2 is symmetric about AR = 2
        -- which says nothing about efficiency, only that the shallower foil is
        also delivering less side force.
        """
        target_cl3d = 0.3
        drags = []
        for span in (0.35, 0.70):
            f = foil(span=span, area=0.1225)
            ar = f.effective_aspect_ratio()
            cl2d = target_cl3d * (ar + 2.0) / ar  # 2-D CL that delivers target_cl3d
            cl3d, cd = f.apply_finite_span(cl2d, 0.0)
            assert cl3d == pytest.approx(target_cl3d)
            drags.append(cd)
        assert drags[1] < drags[0]

    def test_lift_is_never_amplified(self) -> None:
        """AR/(AR+2) < 1 for any real foil, so 3-D lift cannot exceed 2-D."""
        for span in (0.2, 0.5, 1.0, 2.6):
            cl, _ = foil(span=span, area=1.0).apply_finite_span(1.0, 0.01)
            assert 0.0 < cl < 1.0

    def test_sign_of_lift_is_preserved(self) -> None:
        """Negative lift stays negative and still costs drag."""
        cl, cd = foil(span=0.7, area=0.1225).apply_finite_span(-1.0, 0.02)
        assert cl < 0.0
        assert cd > 0.02
