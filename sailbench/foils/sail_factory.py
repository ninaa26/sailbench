"""Build the sail model a config's ``sail`` block asks for.

The hub builds one hull, one keel, one rudder and one sail. The sail used to be
hard-wired to BasicSail even though every config already had a ``model_type``
key sitting next to it that nothing read. This gives that key its meaning, so a
new sail model is one class plus one row in :data:`SAIL_MODELS` and the hub
stays as it is.

Names are the same ones the sim-overhaul line registers under, so a config
reads the same against either: ``basic``, ``hybrid``, ``orc_main``,
``orc_w_jib``. ``sail`` is an alias for ``basic`` because the four existing
configs spell it that way and renaming them would conflict with the other
branches that touch those files.

Picking a model is also when a key belonging to a different model becomes a
mistake, so the check for that lives here. See :mod:`sailbench.foils.sail_keys`
for what it does and does not look at.
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
    "sail": BasicSail,  # older name for `basic`, used by the existing configs
    "hybrid": HybridSail,  # analytic CL = CL_max sin(2a)
    "orc_main": ORCMainSail,  # ORC VPP envelope, single mainsail
    "orc_w_jib": ORCWithJibSail,  # ORC VPP envelope, main + jib
}

# What a sail block with no model_type gets. This is what the hub built
# unconditionally before model_type was read, so existing configs do not move.
DEFAULT_SAIL_MODEL = "basic"

# Models taking their coefficients from the ORC envelope rather than a section
# polar or an analytic curve. Which side of that line a model is on decides
# which config keys it reads. See `sail_keys`.
ORC_SAIL_MODELS = frozenset({"orc_main", "orc_w_jib"})


class UnreadSailKeyWarning(UserWarning):
    """A sail block carries keys the model it selected does not read.

    Warning rather than error: the configs here keep every model's parameters
    in one flat block so that switching model is a one-line edit, and refusing
    would end that. An unread key is still someone's number that is not doing
    anything, so it should not pass silently either.
    """


def build_sail(sail_cfg: dict[str, Any]) -> Model:
    """Instantiate the sail model named by ``sail_cfg["model_type"]``.

    Args:
        sail_cfg (dict): The config's ``sail`` block.

    Returns:
        Model: The sail model, constructed from the same block.

    Raises:
        ValueError: If ``model_type`` names a model that does not exist. That
            is a typo, not a request for the default, and falling back would
            sail a different boat from the one the config describes.

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
            f"sail model_type: {name} does not read {', '.join(stray)}; "
            f"{'that key belongs' if len(stray) == 1 else 'those keys belong'} to {other}. "
            f"They are ignored, so the boat is not sailing the numbers in the config.",
            UnreadSailKeyWarning,
            stacklevel=2,
        )
    return model(sail_cfg)
