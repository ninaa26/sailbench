"""Which config keys belong to which family of sail model.

The sail models read disjoint parameters. A section-polar model wants an
airfoil and a Reynolds number, the analytic model wants three coefficients, and
the ORC envelope wants neither because it works off apparent wind angle. A
config naming one model while carrying another's parameters describes a boat
the simulator is not sailing.

The lists are here rather than in a model because they describe the boundary
between models and belong to none of them. (In the sim-overhaul line
basic_sail imported its list from orc_sail, which also fixed model
registration order as a side effect.)

:func:`unread_keys` only looks across the ORC boundary: ORC keys on a
section/analytic model, or section/analytic keys on an ORC model. No config in
configs/ crosses that boundary, so none of them are affected.

It says nothing about a block holding both FOIL_ONLY_KEYS and
HYBRID_ONLY_KEYS. Every config does that, with a comment naming the model
each group belongs to, so switching between those two stays a one-line edit.
Flagging it would fire on every boat on every load and get ignored.
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

# Read only by the ORC envelope models. Both of them read jib_area: orc_w_jib
# blends the jib in, orc_main strikes it.
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

    Looks across the ORC boundary only. See the module docstring for why the
    section/analytic overlap is left alone.

    Args:
        sail_cfg (dict): The config's ``sail`` block.
        is_orc (bool): Whether the selected model is one of the ORC envelopes.

    Returns:
        list[str]: The unread keys, sorted, or empty if there are none.

    """
    other_side = FOIL_ONLY_KEYS + HYBRID_ONLY_KEYS if is_orc else ORC_ONLY_KEYS
    return sorted(k for k in other_side if k in sail_cfg)
