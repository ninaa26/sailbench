"""`sail.model_type` selects the sail model, and every shipped config names one that exists."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from sailbench.foils.basic_sail import BasicSail
from sailbench.foils.hybrid_sail import HybridSail
from sailbench.foils.orc_sail import ORCMainSail, ORCWithJibSail
from sailbench.foils.sail_factory import DEFAULT_SAIL_MODEL, SAIL_MODELS, build_sail
from sailbench.sim.sailboat_hub import CONFIG_PATH, SailboatHub


def _boat_configs() -> list[str]:
    """Every config in configs/ that describes a boat, as opposed to an RL run."""
    return sorted(p.name for p in Path(CONFIG_PATH).glob("*.yaml") if "sail" in yaml.safe_load(p.read_text()))


BOAT_CONFIGS = _boat_configs()


def sail_block(**overrides: object) -> dict[str, Any]:
    """Build a minimal sail block."""
    return {"area": 1.0, "wind_speed": 5.0, "wind_dir_deg": 90.0, **overrides}


class TestModelSelection:
    """`model_type` names the model; nothing else does."""

    @pytest.mark.parametrize(
        ("model_type", "expected"),
        [
            ("basic", BasicSail),
            ("sail", BasicSail),  # legacy alias the existing configs use
            ("hybrid", HybridSail),
            ("orc_main", ORCMainSail),
            ("orc_w_jib", ORCWithJibSail),
        ],
    )
    def test_each_name_builds_its_model(self, model_type: str, expected: type) -> None:
        """Every name in the table builds the class it points at."""
        extra = {"jib_area": 0.4} if model_type == "orc_w_jib" else {}
        assert isinstance(build_sail(sail_block(model_type=model_type, **extra)), expected)

    def test_names_and_table_agree(self) -> None:
        """The parametrization above covers the whole table, so a new model needs a new row."""
        assert set(SAIL_MODELS) == {"basic", "sail", "hybrid", "orc_main", "orc_w_jib"}

    def test_missing_model_type_keeps_the_old_default(self) -> None:
        """A sail block with no model_type gets what the hub used to build unconditionally."""
        assert isinstance(build_sail(sail_block()), SAIL_MODELS[DEFAULT_SAIL_MODEL])
        assert isinstance(build_sail(sail_block()), BasicSail)

    def test_the_legacy_alias_is_the_same_model(self) -> None:
        """`sail` and `basic` must not drift into two different section models."""
        assert SAIL_MODELS["sail"] is SAIL_MODELS["basic"]

    def test_unknown_model_type_is_an_error(self) -> None:
        """A typo must not silently fall back to the default and sail a different boat."""
        with pytest.raises(ValueError, match="orc_main"):
            build_sail(sail_block(model_type="orc"))


class TestShippedConfigs:
    """Every boat config in configs/ builds a hub."""

    @pytest.mark.parametrize("config_file", BOAT_CONFIGS)
    def test_config_builds_a_hub(self, config_file: str) -> None:
        """A config that names a sail model must be able to construct one."""
        hub = SailboatHub(config_file)
        assert hub.sail is not None

    def test_flingo_full_is_a_sloop(self) -> None:
        """flingo_full.yaml is the measured boat, sailed as the main-and-jib rig it is."""
        hub = SailboatHub("flingo_full.yaml")
        assert isinstance(hub.sail, ORCWithJibSail)
        assert hub.sail.jib_area == pytest.approx(0.775)
        assert hub.sail.main_area == pytest.approx(1.971 - 0.775)

    def test_wpi_sails_a_section_polar(self) -> None:
        """The WPI boat is a rigid wingsail, so it stays on the section model, not ORC."""
        hub = SailboatHub("wpi_wild_goats.yaml")
        assert isinstance(hub.sail, BasicSail)

    def test_existing_configs_are_unmoved(self) -> None:
        """Wiring model_type up must not have changed what the pre-existing configs build."""
        for config_file in ("basic_sailbot.yaml", "flingo_floty.yaml", "real_boat.yaml", "fun_boat.yaml"):
            assert isinstance(SailboatHub(config_file).sail, BasicSail), config_file
