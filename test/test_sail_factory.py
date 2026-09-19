"""The sail model a config asks for is the one it gets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from sailbench.foils.basic_sail import BasicSail
from sailbench.foils.hybrid_sail import HybridSail
from sailbench.foils.sail_factory import SAIL_MODELS, build_sail
from sailbench.sim.sailboat_hub import SailboatHub

CONFIG_DIR = Path("configs")


def _sail_block(config_name: str = "basic_sailbot.yaml") -> dict[str, Any]:
    """A real sail block, so the models get the keys they actually read."""
    with (CONFIG_DIR / config_name).open() as file:
        return dict(yaml.safe_load(file)["sail"])


def test_no_model_type_still_builds_the_basic_sail() -> None:
    """Configs written before model_type was read must not move."""
    cfg = _sail_block()
    cfg.pop("model_type", None)
    assert isinstance(build_sail(cfg), BasicSail)


@pytest.mark.parametrize("name", ["basic", "sail", "BASIC"])
def test_names_for_the_basic_sail(name: str) -> None:
    """`sail` is the spelling the existing configs use; matching is case-insensitive."""
    cfg = _sail_block()
    cfg["model_type"] = name
    assert isinstance(build_sail(cfg), BasicSail)


def test_hybrid_is_now_reachable_from_a_config() -> None:
    """HybridSail has existed all along with no way to select it."""
    cfg = _sail_block()
    cfg["model_type"] = "hybrid"
    assert isinstance(build_sail(cfg), HybridSail)


def test_an_unknown_model_is_an_error_not_a_fallback() -> None:
    """The bug this exists to stop.

    A config asking for a model that is not there used to get BasicSail without
    a word. On a config whose sail block has no section-polar keys that is not
    a near-miss: the boat sails backwards.
    """
    cfg = _sail_block()
    cfg["model_type"] = "orc_w_jib"
    with pytest.raises(ValueError, match="does not exist") as excinfo:
        build_sail(cfg)
    # The message has to say what you could have picked instead.
    for known in SAIL_MODELS:
        assert known in str(excinfo.value)


@pytest.mark.parametrize("config", sorted(p.name for p in CONFIG_DIR.glob("*.yaml")))
def test_every_shipped_config_still_builds(config: str) -> None:
    """A config in the repo naming a model that does not exist is a broken config."""
    with (CONFIG_DIR / config).open() as file:
        loaded = yaml.safe_load(file) or {}
    if "sail" not in loaded:
        pytest.skip(f"{config} is not a boat config")
    assert SailboatHub(config_file=config).sail is not None
