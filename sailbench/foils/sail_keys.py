"""Which config keys belong to which family of sail model.

The sail models read disjoint parameters. A section-polar model wants an airfoil
and a Reynolds number; the analytic model wants three coefficients; the ORC
envelope wants none of that and takes its coefficients from apparent wind angle
instead. A config that names one model and carries another's parameters is
describing a boat the simulator is not sailing.

The lists live here rather than in any one model because they describe the
*boundary* between them and belong to none of them. (In the sim-overhaul line,
where ``basic_sail`` imported its list from ``orc_sail``, that import decided
model registration order as a side effect. A module neither model owns removes
the coupling.)

What is policed, and what is not
--------------------------------
:func:`unread_keys` reports across the ORC boundary only: ORC keys on a
section/analytic model, or section/analytic keys on an ORC model. That is the
boundary this repo's configs have never crossed, so nothing already in
``configs/`` is affected.

It deliberately says nothing about a block holding both ``FOIL_ONLY_KEYS`` and
``HYBRID_ONLY_KEYS``. Every config in ``configs/`` does exactly that on purpose,
with a comment naming the model each group belongs to, so that switching between
those two is a one-line edit. Reporting it would fire on every boat in the tree
on every load, and a report that always fires is worth nothing within a day.
"""

from __future__ import annotations

# Read only by BasicSail, via the NeuralFoil section polar.
FOIL_ONLY_KEYS: tuple[str, ...] = (
    "airfoil_name",
    "alpha_max",
    "alpha_min",
    "luff_deg",
    "luff_ramp_deg",
    "re",
    "res",
)

# Read only by HybridSail's analytic CL/CD curve.
HYBRID_ONLY_KEYS: tuple[str, ...] = (
    "CD0",
    "CD1",
    "CL_max",
)

# Read only by the ORC envelope models. `jib_area` is on this list and is read
# by both of them: orc_w_jib blends the jib in, orc_main strikes it.
ORC_ONLY_KEYS: tuple[str, ...] = (
    "alpha_opt_deg",
    "eff_span_corr",
    "flat_stall_floor",
    "heel_arm_m",
    "heff",
    "heff_model",
    "jib_area",
    "max_heeling_moment_nm",
)


def unread_keys(sail_cfg: dict[str, object], *, is_orc: bool) -> list[str]:
    """Return the keys in a sail block that the selected model cannot read.

    Reports across the ORC boundary only; see the module docstring for why the
    section/analytic overlap is left alone.

    Args:
        sail_cfg (dict): The config's ``sail`` block.
        is_orc (bool): Whether the selected model is one of the ORC envelopes.

    Returns:
        list[str]: The unread keys, sorted, or empty if there are none.

    """
    other_side = FOIL_ONLY_KEYS + HYBRID_ONLY_KEYS if is_orc else ORC_ONLY_KEYS
    return sorted(k for k in other_side if k in sail_cfg)
