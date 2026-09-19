"""`sail.model_type` selects the sail model, and every shipped config names one that exists."""

import math
import warnings
from pathlib import Path
from typing import Any

import pytest
import yaml

from sailbench.foils.basic_sail import BasicSail
from sailbench.foils.hybrid_sail import HybridSail
from sailbench.foils.orc_sail import ORCMainSail, ORCWithJibSail
from sailbench.foils.sail_factory import (
    DEFAULT_SAIL_MODEL,
    SAIL_MODELS,
    UnreadSailKeyWarning,
    build_sail,
)
from sailbench.models.model import State
from sailbench.sim.sailboat_hub import CONFIG_PATH, SailboatHub
from sailbench.solvers.rk4 import rk4_step


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


class TestUnreadKeysAreReported:
    """A key the selected model cannot read is a number someone believes is in effect."""

    def test_orc_model_reports_section_and_analytic_keys(self) -> None:
        """Point an existing config at the ORC envelope and nine keys stop being read."""
        cfg = dict(yaml.safe_load(Path(CONFIG_PATH, "basic_sailbot.yaml").read_text())["sail"])
        cfg["model_type"] = "orc_main"
        with pytest.warns(UnreadSailKeyWarning, match="airfoil_name") as record:
            build_sail(cfg)
        message = str(record[0].message)
        for key in ("CL_max", "CD0", "CD1", "alpha_min", "alpha_max", "res", "luff_deg", "luff_ramp_deg"):
            assert key in message, key

    def test_section_model_reports_orc_keys(self) -> None:
        """And the mirror: heff and friends do nothing to a section polar."""
        cfg = dict(yaml.safe_load(Path(CONFIG_PATH, "flingo_full.yaml").read_text())["sail"])
        cfg["model_type"] = "basic"
        with pytest.warns(UnreadSailKeyWarning, match="heff"):
            build_sail(cfg)

    def test_jib_area_is_not_reported_against_the_orc_models(self) -> None:
        """Both ORC models read jib_area now, so neither should call it unread."""
        for model_type in ("orc_main", "orc_w_jib"):
            with warnings.catch_warnings():
                warnings.simplefilter("error", UnreadSailKeyWarning)
                build_sail(sail_block(model_type=model_type, jib_area=0.4, heff=2.0))

    @pytest.mark.parametrize("config_file", BOAT_CONFIGS)
    def test_no_shipped_config_warns(self, config_file: str) -> None:
        """The report is worth nothing if it fires on every boat in the tree.

        This is why the check is only across the ORC boundary. Every config
        here deliberately keeps the section and analytic parameters side by
        side so that switching between those two is a one-line edit, and
        reporting that habit would bury the case that matters.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error", UnreadSailKeyWarning)
            SailboatHub(config_file)


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

    @staticmethod
    def _sail_for(seconds: float, *, model_type: str) -> tuple[SailboatHub, State]:
        """Sail flingo_full on a beam reach under `model_type`, from a standstill."""
        hub = SailboatHub("flingo_full.yaml")
        hub.sail_cfg["model_type"] = model_type
        hub.boat_factory()  # rebuild the boat from the edited block
        psi = math.radians(90.0) - math.pi / 2.0
        state = State(x=0.0, y=0.0, psi=(math.cos(psi), math.sin(psi)), u=0.0, v=0.0, r=0.0)
        for _ in range(int(seconds / 0.02)):
            state = hub.step(state, 0.02, rk4_step, sail_angle=math.radians(45.0), rudder_angle=0.0)
        return hub, state

    def test_orc_main_is_flingo_with_the_jib_struck(self) -> None:
        """`orc_main` on this config must be the same boat under main alone.

        No config selects orc_main, so without this the model is only ever
        built in isolation. It is also the assertion that the one-line switch
        means what the config says it means: the mainsail keeps its 1.196 m2,
        the jib's 0.775 leaves the rig, and the boat is slower for it rather
        than sailing the combined area as one oversized main.
        """
        hub = SailboatHub("flingo_full.yaml")
        hub.sail_cfg["model_type"] = "orc_main"
        hub.boat_factory()
        assert isinstance(hub.sail, ORCMainSail)
        assert not isinstance(hub.sail, ORCWithJibSail)
        assert hub.sail.main_area == pytest.approx(1.971 - 0.775)
        assert hub.sail.area == pytest.approx(1.971 - 0.775)

        _, under_main = self._sail_for(6.0, model_type="orc_main")
        _, under_both = self._sail_for(6.0, model_type="orc_w_jib")
        assert under_main.u > 0.5  # main alone still drives the boat
        assert under_main.u < under_both.u  # but less sail is less speed

    def test_flingo_full_sails(self) -> None:
        """End to end: the ORC rig drives the boat through the hub, not just in isolation.

        Everything else here checks that the right class is constructed. This
        checks that the class the hub wires up, the hub's own sheeting and the
        transform tree agree well enough to make headway -- six seconds on a
        beam reach from a standstill.
        """
        hub = SailboatHub("flingo_full.yaml")
        # Wind blows TO 90 deg; heading 90 deg off it puts the boat on a beam reach.
        psi = math.radians(90.0) - math.pi / 2.0
        state = State(x=0.0, y=0.0, psi=(math.cos(psi), math.sin(psi)), u=0.0, v=0.0, r=0.0)
        for _ in range(300):
            state = hub.step(state, 0.02, rk4_step, sail_angle=math.radians(45.0), rudder_angle=0.0)
        assert state.u > 0.5
        assert math.isfinite(state.x)
        assert hub.last_forces["sail"][0] > 0.0  # the rig is driving, not braking

    def test_existing_configs_are_unmoved(self) -> None:
        """Wiring model_type up must not have changed what the pre-existing configs build."""
        for config_file in ("basic_sailbot.yaml", "flingo_floty.yaml", "real_boat.yaml", "fun_boat.yaml"):
            assert isinstance(SailboatHub(config_file).sail, BasicSail), config_file
