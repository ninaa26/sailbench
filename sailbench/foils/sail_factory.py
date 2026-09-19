"""Build the sail model a config's ``sail`` block asks for.

The hub composes a boat from one hull, one keel, one rudder and one sail, and
until now the sail was hard-wired to :class:`~sailbench.foils.basic_sail.BasicSail`
while every config already carried an unread ``model_type`` key beside it. This
module gives that key its meaning, so adding a sail model is a change in
``sailbench/foils/`` alone -- one new class and one new row in :data:`SAIL_MODELS`
-- rather than a change to the hub.

The names match the ones the sim-overhaul line registers its sail models under,
so a config written against either reads the same: ``basic``, ``hybrid``,
``orc_main``, ``orc_w_jib``. ``sail`` is kept as an alias for ``basic`` because
the four configs already in ``configs/`` spell it that way, and rewriting them
here would only create conflicts for the branches that also touch them.

Selecting a model here is also the moment at which a key belonging to a
*different* model becomes a mistake, so this is where that is reported. See
:mod:`sailbench.foils.sail_keys` for what is and is not policed.
"""

import warnings
from typing import Any

from sailbench.foils.basic_sail import BasicSail
from sailbench.foils.hybrid_sail import HybridSail
from sailbench.foils.orc_sail import ORCMainSail, ORCWithJibSail
from sailbench.foils.sail_keys import unread_keys
from sailbench.models.model import Model

# Sail models a config may select with `sail.model_type`.
SAIL_MODELS: dict[str, type[Model]] = {
    "basic": BasicSail,  # NeuralFoil section polar
    "sail": BasicSail,  # legacy alias for `basic`; what configs/ already says
    "hybrid": HybridSail,  # analytic CL = CL_max sin(2a)
    "orc_main": ORCMainSail,  # ORC VPP envelope, single mainsail
    "orc_w_jib": ORCWithJibSail,  # ORC VPP envelope, main + jib
}

# What a config gets when its sail block names no model: the model the hub built
# unconditionally before `model_type` was read, so existing configs are unmoved.
DEFAULT_SAIL_MODEL = "basic"

# The models whose coefficients come from the ORC envelope rather than from a
# section polar or an analytic curve. Which side of that boundary a model sits
# on is what decides whether a given config key is read; see `sail_keys`.
ORC_SAIL_MODELS = frozenset({"orc_main", "orc_w_jib"})


class UnreadSailKeyWarning(UserWarning):
    """A sail block carries keys the model it selected does not read.

    A warning and not an error: this repo's configs deliberately keep every
    model's parameters in one flat block so that switching model is a one-line
    edit, and refusing would make that impossible. But a key that is read by
    nothing is a number someone wrote down and believes is in effect, so it
    should not pass in silence either.
    """


def build_sail(sail_cfg: dict[str, Any]) -> Model:
    """Instantiate the sail model named by ``sail_cfg["model_type"]``.

    Args:
        sail_cfg (dict): The config's ``sail`` block.

    Returns:
        Model: The sail model, constructed from the same block.

    Raises:
        ValueError: If ``model_type`` names a model that does not exist. An
            unknown name is a typo, not a request for the default: silently
            falling back would sail a different boat than the config describes.

    Warns:
        UnreadSailKeyWarning: If the block carries keys belonging to a sail
            model on the other side of the ORC boundary from the one selected.

    """
    name = str(sail_cfg.get("model_type", DEFAULT_SAIL_MODEL))
    model = SAIL_MODELS.get(name)
    if model is None:
        known = ", ".join(sorted(SAIL_MODELS))
        msg = f"sail model_type must be one of {known}: got {name!r}"
        raise ValueError(msg)

    stray = unread_keys(sail_cfg, is_orc=name in ORC_SAIL_MODELS)
    if stray:
        other = "a section polar or the analytic curve" if name in ORC_SAIL_MODELS else "the ORC envelope"
        warnings.warn(
            f"sail model_type: {name} does not read {', '.join(stray)} -- "
            f"{'that key belongs' if len(stray) == 1 else 'those keys belong'} to {other}. "
            f"They are being ignored, so the boat is not sailing the numbers the config gives it.",
            UnreadSailKeyWarning,
            stacklevel=2,
        )
    return model(sail_cfg)
