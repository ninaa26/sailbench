"""Build the parts of the boat a config describes.

There are four required parts: HULL, KEEL, SAIL AND RUDDER
optional components include: JIB, WINDAGE, BALLAST

Each part of the boat has it's own config section that picks its model correlating to
''model_type''. Add required components to REQUIRED_MODELS and optional components to
OPTIONAL_MODELS. If adding alternate models to a component, add to corresponding section
eg. Add LinearHydroModel to HULL MODELS.
"""

import warnings
from typing import Any

from sailbench.dynamics.ballast import Ballast
from sailbench.dynamics.basic_hull_model import BasicHullModel
from sailbench.dynamics.linear_hydro import LinearHydroModel
from sailbench.dynamics.quadratic_drag_hydro import QuadraticHydroModel
from sailbench.dynamics.windage import Windage
from sailbench.foils.basic_keel import BasicKeel
from sailbench.foils.basic_rudder import BasicRudder
from sailbench.foils.basic_sail import BasicSail
from sailbench.foils.hybrid_sail import HybridSail
from sailbench.foils.orc_sail import ORCMainSail
from sailbench.foils.sail_factory import DEFAULT_SAIL_MODEL, SAIL_MODELS
from sailbench.models.model import Model
from sailbench.models.registry import build_model

ModelTable = tuple[dict[str, type[Model]], str]

# Hull models a config may select with `hull.model_type`.
HULL_MODELS: dict[str, type[Model]] = {
    "basic": BasicHullModel,  # quadratic drag from L, B and T
    "linear": LinearHydroModel,  # linear damping from xu1, yv1, nr1
    "quadratic": QuadraticHydroModel,  # quadratic damping from xu2, yv2, nr2
}

# Keel models a config may select with `keel.model_type`.
KEEL_MODELS: dict[str, type[Model]] = {
    "basic": BasicKeel,  # NeuralFoil section polar
    "keel": BasicKeel,  # older name for `basic`, used by the existing configs
}

# Rudder models a config may select with `rudder.model_type`.
RUDDER_MODELS: dict[str, type[Model]] = {
    "basic": BasicRudder,  # NeuralFoil section polar
    "keel": BasicRudder,  # what every existing config's rudder says, copied from its keel section
}

# Jib models a config may select with `jib.model_type`. The main's section-polar
# and analytic sails, not the ORC ones: ORC's jib coefficients only hold blended
# into its collective rig, which the sail section selects as orc_w_jib.
JIB_MODELS: dict[str, type[Model]] = {
    "basic": BasicSail,  # NeuralFoil section polar
    "hybrid": HybridSail,  # analytic CL = CL_max sin(2a)
}

# Every boat has atleast one of each. Each default is the class the hub built
# unconditionally before the section's model_type was read, so existing configs
# do not move.
REQUIRED_COMPONENTS: dict[str, ModelTable] = {
    "hull": (HULL_MODELS, "basic"),
    "keel": (KEEL_MODELS, "basic"),
    "sail": (SAIL_MODELS, DEFAULT_SAIL_MODEL),
    "rudder": (RUDDER_MODELS, "basic"),
}

# Built only when the config has the section and has not switched it off.
OPTIONAL_COMPONENTS: dict[str, ModelTable] = {
    "windage": ({"basic": Windage}, "basic"),  # above-water drag on mast, rigging and topsides
    "ballast": ({"basic": Ballast}, "basic"),  # dead weight: mass and yaw inertia, no force
    "jib": (JIB_MODELS, "basic"),  # headsail on the main's sheet
}

# Sections the hub reads itself rather than building a part from.
HUB_SECTIONS = frozenset({"simulation", "boat"})


def build_components(cfg: dict[str, Any]) -> dict[str, Model]:
    """Instantiate every part of the boat the config describes.

    Args:
        cfg (dict): The whole parsed config.

    Returns:
        dict[str, Model]: Each part keyed by its config section: the four
            required parts first, in the order the hub has always summed them,
            then the enabled optional ones in table order.

    Raises:
        ValueError: If a ``jib`` section sits beside an ORC sail. ORC's
            coefficients describe its own rig, main alone or main and jib
            blended, so a jib modelled beside it would be counted twice.

    """
    parts = {section: build_required(section, cfg[section]) for section in REQUIRED_COMPONENTS}
    optional = build_optional(cfg)
    if "jib" in optional and isinstance(parts["sail"], ORCMainSail):
        msg = (
            "a jib section cannot sail beside an ORC sail, which models its own jib: "
            "remove the jib section and use sail.model_type: orc_w_jib with sail.jib_area"
        )
        raise ValueError(msg)
    return {**parts, **optional}


def build_required(section: str, block: dict[str, Any]) -> Model:
    """Instantiate the model one of the four required parts asks for.

    Args:
        section (str): ``hull``, ``keel``, ``sail`` or ``rudder``.
        block (dict): That section of the config.

    Returns:
        Model: The part, built from ``block``.

    """
    models, default = REQUIRED_COMPONENTS[section]
    return build_model(section, block, models, default)


def build_optional(cfg: dict[str, Any]) -> dict[str, Model]:
    """Instantiate every optional part the config enables.

    Also warns about any section nothing reads. With parts built by presence, a
    misspelled ``windage:`` would otherwise sail the boat without windage and
    say nothing.

    Args:
        cfg (dict): The whole parsed config, not one section: which sections
            are present is exactly the thing being read.

    Returns:
        dict[str, Model]: The enabled parts keyed by section, in table order.
            Empty when the config names none of them.

    """
    unread = sorted(set(cfg) - HUB_SECTIONS - set(REQUIRED_COMPONENTS) - set(OPTIONAL_COMPONENTS))
    if unread:
        known = ", ".join(sorted(OPTIONAL_COMPONENTS))
        warnings.warn(
            f"config sections {unread} are not read by anything; optional parts are: {known}",
            stacklevel=2,
        )

    built: dict[str, Model] = {}
    for section, (models, default) in OPTIONAL_COMPONENTS.items():
        block = cfg.get(section)
        if not isinstance(block, dict):
            continue
        # Present but switched off: keep the measured numbers in the config
        # without putting the part on the boat.
        if not block.get("enabled", True):
            continue
        built[section] = build_model(section, block, models, default)
    return built
