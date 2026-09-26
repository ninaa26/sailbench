"""Each part is the model its config section asks for, and optional parts come and go with their sections."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

import sailbench.sim.sailboat_hub as hub_module
from sailbench.dynamics.ballast import Ballast
from sailbench.dynamics.basic_hull_model import BasicHullModel
from sailbench.dynamics.component_factory import (
    HULL_MODELS,
    OPTIONAL_COMPONENTS,
    REQUIRED_COMPONENTS,
    build_components,
    build_optional,
    build_required,
)
from sailbench.dynamics.linear_hydro import LinearHydroModel
from sailbench.dynamics.quadratic_drag_hydro import QuadraticHydroModel
from sailbench.foils.basic_keel import BasicKeel
from sailbench.foils.basic_rudder import BasicRudder
from sailbench.foils.basic_sail import BasicSail
from sailbench.foils.hybrid_sail import HybridSail
from sailbench.models.model import State
from sailbench.sim.sailboat_hub import SailboatHub
from sailbench.solvers.rk4 import rk4_step
from sailbench.tf.tf_tree import TFTree2D

CONFIG_DIR = Path("configs")
BOAT_CONFIGS = sorted(p.name for p in CONFIG_DIR.glob("*.yaml") if "sail" in (yaml.safe_load(p.read_text()) or {}))
# Heading east at 1 m/s, so basic_sailbot's wind (5 m/s, blowing north) is on the beam.
REACHING = State.from_array(np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]))


def _config(name: str = "basic_sailbot.yaml") -> dict[str, Any]:
    """Load a real config, so every model gets the keys it actually reads."""
    with (CONFIG_DIR / name).open() as file:
        return dict(yaml.safe_load(file))


def _hub(cfg: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SailboatHub:
    """Build a hub from an edited config. The hub only reads files under CONFIG_PATH."""
    (tmp_path / "boat.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(hub_module, "CONFIG_PATH", f"{tmp_path}/")
    return SailboatHub("boat.yaml")


def _jib(**overrides: Any) -> dict[str, Any]:
    """basic_sailbot's own sail block as a jib section, minus the wind the hub is meant to supply."""
    block = {k: v for k, v in _config()["sail"].items() if k not in ("wind_speed", "wind_dir_deg")}
    return {**block, "model_type": "basic", **overrides}


def test_a_config_that_names_no_models_builds_the_boat_it_always_did() -> None:
    """Before model_type was read, the hub hard-wired these four and nothing else."""
    hub = SailboatHub("basic_sailbot.yaml")
    assert isinstance(hub.hull, BasicHullModel)
    assert isinstance(hub.keel, BasicKeel)
    assert isinstance(hub.sail, BasicSail)
    assert isinstance(hub.rudder, BasicRudder)
    assert hub.components == [hub.hull, hub.keel, hub.sail, hub.rudder]


@pytest.mark.parametrize(
    ("name", "model"),
    [("basic", BasicHullModel), ("linear", LinearHydroModel), ("quadratic", QuadraticHydroModel)],
)
def test_the_hull_is_the_one_the_config_asks_for(name: str, model: type) -> None:
    """The linear and quadratic hulls existed with no way to select them."""
    block = _config()["hull"]
    block["model_type"] = name
    assert isinstance(build_required("hull", block), model)


def test_the_hub_sails_the_hull_its_config_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Selecting a model in the config is enough; the hub needs no edit."""
    cfg = _config("fun_boat.yaml")
    cfg["hull"]["model_type"] = "quadratic"
    hub = _hub(cfg, tmp_path, monkeypatch)
    assert isinstance(hub.hull, QuadraticHydroModel)
    assert hub.components_by_name["hull"] is hub.hull


def test_an_unknown_model_names_the_section_and_the_choices() -> None:
    """A typo is an error, and the message says what could have been picked."""
    block = _config()["hull"]
    block["model_type"] = "catamaran"
    with pytest.raises(ValueError, match="hull model_type 'catamaran' does not exist") as excinfo:
        build_required("hull", block)
    for known in HULL_MODELS:
        assert known in str(excinfo.value)


def test_every_default_is_in_its_own_table() -> None:
    """A default that names nothing would make every config without model_type fail."""
    for models, default in [*REQUIRED_COMPONENTS.values(), *OPTIONAL_COMPONENTS.values()]:
        assert default in models


def test_windage_comes_and_goes_with_its_section() -> None:
    """Present builds it, `enabled: false` keeps the numbers without the part, absent is no part."""
    cfg = _config("flingo_floty.yaml")
    assert list(build_optional(cfg)) == ["windage"]

    cfg["windage"]["enabled"] = False
    assert build_optional(cfg) == {}

    del cfg["windage"]
    assert build_optional(cfg) == {}


def test_a_misspelled_section_is_warned_about() -> None:
    """Parts are built by presence, so a typo would otherwise drop one silently."""
    cfg = _config("flingo_floty.yaml")
    cfg["windgae"] = cfg.pop("windage")
    with pytest.warns(UserWarning, match="windgae"):
        assert build_optional(cfg) == {}


@pytest.mark.parametrize("config", BOAT_CONFIGS)
def test_shipped_configs_have_no_unread_sections(config: str) -> None:
    """Every section in a shipped boat config is read by something."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        build_optional(_config(config))


class TestBallast:
    """Ballast is weight: it changes how the boat accelerates, not what pushes it."""

    def test_adds_its_mass_and_parallel_axis_inertia(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mass adds straight on; inertia is its own plus m * r^2 about the origin."""
        cfg = _config()
        cfg["ballast"] = {"mass": 5.0, "x_pos": -0.2, "y_pos": 0.1, "inertia_z": 0.05}
        hub = _hub(cfg, tmp_path, monkeypatch)
        assert hub.m == pytest.approx(27.0 + 5.0)
        assert hub.iz == pytest.approx(10.0 + 0.05 + 5.0 * (0.2**2 + 0.1**2))

    def test_switched_off_adds_nothing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """`enabled: false` sails the boat as if the section were not there."""
        cfg = _config()
        cfg["ballast"] = {"mass": 5.0, "enabled": False}
        hub = _hub(cfg, tmp_path, monkeypatch)
        assert "ballast" not in hub.components_by_name
        assert hub.m == pytest.approx(27.0)

    def test_makes_no_force(self) -> None:
        """Weight acts out of the plane this simulator integrates in."""
        state = State.from_array(np.array([0.0, 0.0, 1.0, 0.0, 2.0, 0.5, 0.3]))
        assert np.array_equal(Ballast({"mass": 5.0}).compute(state, TFTree2D()), np.zeros(2))

    @pytest.mark.parametrize("params", [{}, {"mass": 0.0}, {"mass": -1.0}, {"mass": 5.0, "inertia_z": -0.1}])
    def test_rejects_a_weight_that_is_not_one(self, params: dict[str, float]) -> None:
        """No mass, or a negative one, is a config mistake rather than no ballast."""
        with pytest.raises(ValueError, match="ballast"):
            Ballast(params)

    def test_a_heavier_boat_answers_the_same_force_more_slowly(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The added mass reaches the equations of motion, not just the attribute."""
        start = State.from_array(np.array([0.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0]))
        # A tiny step, so both boats feel the same forces throughout it. Over a
        # normal 0.02 s step sway builds at a rate set by the mass and swings
        # the keel's leeway, so the forces themselves differ and the ratio
        # drifts from 1/m (0.90 against 0.67 here).
        dt = 1e-4

        def surge_change(cfg: dict[str, Any]) -> tuple[float, float]:
            hub = _hub(cfg, tmp_path, monkeypatch)
            return hub.step(start, dt, rk4_step, 0.6, 0.0).u - start.u, hub.m

        light, light_mass = surge_change(_config())
        cfg = _config()
        cfg["ballast"] = {"mass": 13.5}
        heavy, heavy_mass = surge_change(cfg)

        assert abs(light) > 0.0
        assert heavy == pytest.approx(light * light_mass / heavy_mass, rel=1e-3)


class TestJib:
    """A jib is a second sail on the main's sheet, set forward of the mast."""

    @pytest.mark.parametrize(("name", "model"), [("basic", BasicSail), ("hybrid", HybridSail)])
    def test_is_the_sail_model_its_section_asks_for(self, name: str, model: type) -> None:
        """The jib picks from the same section-polar and analytic sails the main can be."""
        cfg = _config()
        cfg["jib"] = _jib(model_type=name)
        assert isinstance(build_optional(cfg)["jib"], model)

    def test_sails_on_the_mains_wind(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Its section states no wind: it gets the sail's at build, and follows it after."""
        cfg = _config()
        cfg["jib"] = _jib()
        hub = _hub(cfg, tmp_path, monkeypatch)
        jib = hub.components_by_name["jib"]
        assert jib.p["wind_speed"] == hub.sail_cfg["wind_speed"]
        assert jib.p["wind_dir_deg"] == hub.sail_cfg["wind_dir_deg"]

        hub.sail_cfg["wind_speed"] = 9.0
        hub.step(REACHING, 0.02, rk4_step, 0.6, 0.0)
        assert jib.p["wind_speed"] == 9.0

    def test_is_trimmed_by_the_mains_sheet(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Built from the main's own numbers, it makes exactly the main's force: same wind, same angle."""
        cfg = _config()
        cfg["jib"] = _jib()
        hub = _hub(cfg, tmp_path, monkeypatch)
        hub._update_dynamic_frames(REACHING, np.radians(40.0), 0.0, 0.02)
        hub._forces(REACHING)
        assert np.hypot(*hub.last_forces["sail"]) > 0.0
        assert hub.last_forces["jib"] == pytest.approx(hub.last_forces["sail"])

    def test_turns_the_boat_from_where_it_is_set(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Moving it forward leaves the force alone and adds its arm times its side force to the moment."""

        def forces(x_pos: float) -> tuple[tuple[float, float, float], tuple[float, float]]:
            cfg = _config()
            cfg["jib"] = _jib(x_pos=x_pos)
            hub = _hub(cfg, tmp_path, monkeypatch)
            hub._update_dynamic_frames(REACHING, np.radians(40.0), 0.0, 0.02)
            return hub._forces(REACHING), hub.last_forces["jib"]

        (fx0, fy0, mz0), _ = forces(0.0)
        (fx1, fy1, mz1), (_, jib_fy) = forces(0.4)
        assert abs(jib_fy) > 0.0
        assert (fx1, fy1) == pytest.approx((fx0, fy0))
        assert mz1 - mz0 == pytest.approx(0.4 * jib_fy)

    @pytest.mark.parametrize("sail_model", ["orc_main", "orc_w_jib"])
    def test_cannot_sail_beside_an_orc_rig(self, sail_model: str) -> None:
        """ORC models its own jib, so a second one here would be counted twice."""
        cfg = _config()
        cfg["sail"].update(model_type=sail_model, jib_area=1.0)
        cfg["jib"] = _jib()
        with pytest.raises(ValueError, match="orc_w_jib"):
            build_components(cfg)

    def test_is_not_an_orc_model(self) -> None:
        """ORC's jib coefficients only hold blended into its collective rig."""
        cfg = _config()
        cfg["jib"] = _jib(model_type="orc_w_jib")
        with pytest.raises(ValueError, match="jib model_type 'orc_w_jib' does not exist"):
            build_optional(cfg)


def test_every_optional_part_sails_together(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Windage, ballast and a jib on one boat: each is built, weighed and summed, and the hub names none."""
    cfg = _config("flingo_floty.yaml")
    cfg["ballast"] = {"mass": 2.0}
    cfg["jib"] = _jib(x_pos=0.3)
    hub = _hub(cfg, tmp_path, monkeypatch)
    assert list(hub.components_by_name) == ["hull", "keel", "sail", "rudder", "windage", "ballast", "jib"]
    assert hub.m == pytest.approx(27.0 + 2.0)

    state = REACHING
    for _ in range(50):
        state = hub.step(state, 0.02, rk4_step, 0.6, 0.0)
    assert np.all(np.isfinite(state.to_array()))
    assert set(hub.last_forces) == {*hub.components_by_name, "total"}
    assert np.hypot(*hub.last_forces["windage"]) > 0.0
    assert np.hypot(*hub.last_forces["jib"]) > 0.0
