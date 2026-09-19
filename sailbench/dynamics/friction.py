"""Skin-friction laws a hull can be built on.

A law is a function of speed, waterline length and the hull's own parameters,
and a config names the one it wants with `hull.friction_model`. Adding a law is
writing it here and adding a row; the hull model does not change.

Kept out of BasicHullModel on purpose. The alternative is an `if` inside a
method that is otherwise about hull geometry, and the usual failure mode of that
shape is that anything not matching the string silently means the default, so a
typo runs a model the config did not ask for and says nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

# Below this the Hughes line is not valid and skin friction is not the story.
LAMINAR_RE = 1.0e4

# The boundary layer does not run the full waterline.
RE_LENGTH_FRACTION = 0.85


def flat_plate(u: float, length: float, params: Mapping[str, Any]) -> float:
    """Return a constant coefficient, independent of speed.

    A plausible mid-range number for a hull this size, and what every config got
    before a law could be named. It does not vary with speed, which is the one
    thing skin friction is known to do.
    """
    del u, length, params
    return 0.004


def hughes(u: float, length: float, params: Mapping[str, Any]) -> float:
    """Return an ITTC-style Reynolds-dependent coefficient, times a form factor.

        Re = 0.85 * |u| * L / nu
        Cf = 0.066 / (log10(Re) - 2.03)^2
        ff = 1.05                        (a hull is not a flat plate)

    Falls back to the flat coefficient below Re 1e4, where the line does not
    hold and the flow is not turbulent anyway.
    """
    nu = float(params.get("nu_water", 1.19e-6))  # [m^2/s] fresh water, ~15 C
    re = RE_LENGTH_FRACTION * abs(float(u)) * float(length) / nu
    if re < LAMINAR_RE:
        return flat_plate(u, length, params)
    cf = 0.066 / (np.log10(re) - 2.03) ** 2
    return float(cf * float(params.get("form_factor", 1.05)))


# Laws a config may select with `hull.friction_model`.
FRICTION_LAWS: dict[str, Callable[[float, float, Mapping[str, Any]], float]] = {
    "flat": flat_plate,  # constant 0.004
    "hughes": hughes,  # ITTC-style, Reynolds-dependent, with a form factor
}

# What a hull that names no law gets: the constant, i.e. the previous behaviour.
DEFAULT_FRICTION_LAW = "flat"


def friction_law(hull_cfg: Mapping[str, Any]) -> Callable[[float, float, Mapping[str, Any]], float]:
    """Return the law named by `hull_cfg["friction_model"]`.

    Raises:
        ValueError: If the name is not a law. That is a typo, not a request for
            the default, and falling back would run a model the config did not
            ask for without saying so.

    """
    name = str(hull_cfg.get("friction_model", DEFAULT_FRICTION_LAW)).lower()
    law = FRICTION_LAWS.get(name)
    if law is None:
        known = ", ".join(sorted(FRICTION_LAWS))
        msg = f"hull friction_model {name!r} does not exist; pick one of: {known}"
        raise ValueError(msg)
    return law
