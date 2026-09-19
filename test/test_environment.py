"""Fluid properties written in a config reach the components that use them."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from sailbench.models.model import State
from sailbench.sim.sailboat_hub import SailboatHub

CONFIG = "flingo_floty.yaml"
WET = ("hull", "keel", "rudder")


def _hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **overrides: float) -> SailboatHub:
    """Build a hub from the shipped config with the environment block edited."""
    source = yaml.safe_load((Path("configs") / CONFIG).read_text())
    source.setdefault("environment", {}).update(overrides)
    (tmp_path / CONFIG).write_text(yaml.safe_dump(source, sort_keys=False))
    monkeypatch.setattr("sailbench.sim.sailboat_hub.CONFIG_PATH", f"{tmp_path}/")
    return SailboatHub(config_file=CONFIG)


def _force(hub: SailboatHub, part: str) -> np.ndarray:
    state = State(x=0.0, y=0.0, psi=(1.0, 0.0), u=1.5, v=0.12, r=0.0)
    hub._update_dynamic_frames(state, math.radians(30.0), 5.0, 0.02)  # noqa: SLF001
    return np.asarray(getattr(hub, part).compute(state, hub.tf), dtype=float)[:2]


def test_every_component_receives_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each component reads its own key name; none of them should have to be named here."""
    hub = _hub(tmp_path, monkeypatch)
    for cfg in (hub.hull_cfg, hub.keel_cfg, hub.rudder_cfg, hub.sail_cfg, hub.windage_cfg):
        assert "rho_water" in cfg
        assert "rho_air" in cfg


@pytest.mark.parametrize("part", WET)
def test_water_density_actually_drives_the_force(part: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The bug this fixes: the number in the config used to do nothing at all.

    Every one of these forces is linear in water density, so salt water at 1025
    must scale them by exactly 1.025. Before this, each component fell back to
    its own hardcoded default and the ratio was 1.0.
    """
    fresh = _force(_hub(tmp_path, monkeypatch, rho_water=1000.0), part)
    salt = _force(_hub(tmp_path, monkeypatch, rho_water=1025.0), part)
    assert np.linalg.norm(salt) / np.linalg.norm(fresh) == pytest.approx(1.025, rel=1e-6)


def test_air_density_drives_the_rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Same for the aerodynamic side."""
    thin = _force(_hub(tmp_path, monkeypatch, rho_air=1.225), "sail")
    thick = _force(_hub(tmp_path, monkeypatch, rho_air=1.225 * 1.1), "sail")
    assert np.linalg.norm(thick) / np.linalg.norm(thin) == pytest.approx(1.1, rel=1e-6)


def test_a_component_can_still_override_the_shared_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`environment` is a default, not a mandate."""
    source = yaml.safe_load((Path("configs") / CONFIG).read_text())
    source["environment"]["rho_water"] = 1000.0
    source["keel"]["rho_water"] = 1025.0
    (tmp_path / CONFIG).write_text(yaml.safe_dump(source, sort_keys=False))
    monkeypatch.setattr("sailbench.sim.sailboat_hub.CONFIG_PATH", f"{tmp_path}/")
    hub = SailboatHub(config_file=CONFIG)

    assert hub.keel_cfg["rho_water"] == 1025.0
    assert hub.hull_cfg["rho_water"] == 1000.0
