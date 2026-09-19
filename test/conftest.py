"""Pytest fixtures for sailbench."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from sailbench.foils.basic_keel import BasicKeel
from sailbench.foils.hybrid_sail import HybridSail
from sailbench.tf.tf_tree import TFTree2D, Transform2D
from sailbench.foils.basic_rudder import BasicRudder
from sailbench.dynamics.linear_hydro import LinearHydroModel


@pytest.fixture
def config() -> dict[str, Any]:
    """Load the basic sailboat congifiguration for testing."""
    with Path("configs/basic_sailbot.yaml").open() as file:
        return dict(yaml.safe_load(file))


@pytest.fixture
def tf_tree() -> TFTree2D:
    """Generate a simple transform tree for testing."""
    tf = TFTree2D()
    tf.add_frame(name="boat", parent="world", transform=Transform2D(x=0.0, y=0.0, c=1.0, s=0.0))
    tf.add_frame(name="keel", parent="boat", transform=Transform2D(x=0.0, y=-0.5, c=1.0, s=0.0))
    tf.add_frame(name="rudder", parent="boat", transform = Transform2D(x=0.0, y=0.0, c=1.0, s=0.0))
    tf.add_frame(name="sail", parent="boat", transform=Transform2D(x=0.0, y=0.0, c=1.0, s=0.0))
    return tf


@pytest.fixture
def keel(config: dict[str, Any]) -> BasicKeel:
    """Generate a BasicKeel instance for testing."""
    keel_cfg = config["keel"]
    return BasicKeel(keel_cfg)

@pytest.fixture
def sail(config: dict[str, Any]) -> HybridSail:
    """Create a hybrid sail with fixed wind.

    CL_max/CD0/CD1 are stated here rather than read from the boat config: the
    simulator never instantiates HybridSail, so carrying its coefficients in
    every boat config made them look like live tuning knobs when only this
    fixture read them.
    """
    sail_cfg = dict(config["sail"])
    sail_cfg.update({"CL_max": 1.0, "CD0": 0.1, "CD1": 0.8})
    return HybridSail(sail_cfg)


@pytest.fixture
def rudder(config: dict[str, Any]) -> BasicRudder:
    """Generate a BasicRudder instance for testing."""
    rudder_cfg = config["rudder"]
    return BasicRudder(rudder_cfg)
@pytest.fixture
def hull(config: dict[str, Any]) -> LinearHydroModel:
    """Create a linear hydro hull model.

    Same reasoning as the sail fixture: SailboatHub builds a BasicHullModel, so
    the xu1/yv1/nr1 damping coefficients are exercised only from here.
    """
    hull_cfg = dict(config["hull"])
    hull_cfg.update({"xu1": 20.0, "yv1": 40.0, "nr1": 15.0})
    return LinearHydroModel(hull_cfg)

