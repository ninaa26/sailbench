"""Build the sail model a config's ``sail`` block asks for.

The hub builds one hull, one keel, one rudder and one sail. The sail was
hard-wired to BasicSail even though every config already had a ``model_type``
key sitting next to it that nothing read. This gives that key its meaning, so
adding a sail model is one class plus one row in :data:`SAIL_MODELS`, and the
hub stays as it is.

``sail`` is an alias for ``basic`` because the existing configs spell it that
way. Renaming them would touch four files for no gain and conflict with other
branches.
"""

from typing import Any

from sailbench.foils.basic_sail import BasicSail
from sailbench.foils.hybrid_sail import HybridSail
from sailbench.models.model import Model

# Sail models a config may select with `sail.model_type`.
SAIL_MODELS: dict[str, type[Model]] = {
    "basic": BasicSail,  # NeuralFoil section polar
    "sail": BasicSail,  # older name for `basic`, used by the existing configs
    "hybrid": HybridSail,  # analytic CL = CL_max sin(2a)
}

# What a sail block with no model_type gets. This is what the hub built
# unconditionally before model_type was read, so existing configs do not move.
DEFAULT_SAIL_MODEL = "basic"


def build_sail(sail_cfg: dict[str, Any]) -> Model:
    """Instantiate the sail model named by ``sail_cfg["model_type"]``.

    Args:
        sail_cfg (dict): The config's ``sail`` block.

    Returns:
        Model: The sail model, built from the same block.

    Raises:
        ValueError: If ``model_type`` names a model that does not exist. That
            is a typo, not a request for the default. Falling back would sail a
            different boat from the one the config describes, without saying so.

    """
    name = str(sail_cfg.get("model_type", DEFAULT_SAIL_MODEL)).lower()
    model = SAIL_MODELS.get(name)
    if model is None:
        known = ", ".join(sorted(set(SAIL_MODELS)))
        msg = f"sail model_type {name!r} does not exist; pick one of: {known}"
        raise ValueError(msg)
    return model(sail_cfg)
