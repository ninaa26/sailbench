"""Pick a model class from a config section's ``model_type``.

Every part of the boat can have more than one model -- three hulls, five sails
-- and the config chooses between them. This is the one place that choice is
made, so every section reads ``model_type`` the same way and a bad name fails
the same way.
"""

from typing import Any

from sailbench.models.model import Model


def build_model(section: str, block: dict[str, Any], models: dict[str, type[Model]], default: str) -> Model:
    """Instantiate the model ``block["model_type"]`` names, built from ``block``.

    Args:
        section (str): The config section the block came from. Only used to
            say where a bad name was found.
        block (dict): The section's parameters. The model is built from these.
        models (dict): The names a config may use, mapped to the class each
            selects. Matching is case-insensitive, so keys are lower case.
        default (str): The name used when the block has no ``model_type``.

    Returns:
        Model: The selected model, built from ``block``.

    Raises:
        ValueError: If ``model_type`` names a model that does not exist. That
            is a typo, not a request for the default. Falling back would sail a
            different boat from the one the config describes, without saying so.

    """
    name = str(block.get("model_type", default)).lower()
    model = models.get(name)
    if model is None:
        known = ", ".join(sorted(models))
        msg = f"{section} model_type {name!r} does not exist; pick one of: {known}"
        raise ValueError(msg)
    return model(block)
