"""Contract tests for waypoint Gym environment."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sailbench.models.model import State
from sailbench.rl.envs.waypoint_env import WaypointEnv, WaypointEnvConfig


def _make_env(max_episode_steps: int = 50) -> WaypointEnv:
    cfg = WaypointEnvConfig(
        simulator_config="basic_sailbot.yaml",
        max_episode_steps=max_episode_steps,
        waypoint_min_radius_m=8.0,
        waypoint_max_radius_m=12.0,
    )
    return WaypointEnv(config=cfg)


def test_reset_returns_valid_observation() -> None:
    env = _make_env()
    obs, _ = env.reset(seed=123)
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)


def test_step_contract_and_action_clipping() -> None:
    env = _make_env()
    env.reset(seed=0)
    action = np.array([5.0, -7.0], dtype=np.float32)  # intentionally out of bounds
    obs, reward, terminated, truncated, info = env.step(action)

    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert np.all(env.last_action <= 1.0)
    assert np.all(env.last_action >= -1.0)


def test_episode_truncates_at_horizon() -> None:
    env = _make_env(max_episode_steps=3)
    env.reset(seed=11)
    truncated = False
    for _ in range(3):
        _, _, _, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
    assert truncated


def test_control_mapping_extrema_are_expected() -> None:
    env = _make_env()
    env.reset(seed=17)

    captured: list[tuple[float, float]] = []
    original_step = env.hub.step

    def spy_step(*, state, dt, solver, sail_angle, rudder_angle):  # type: ignore[no-untyped-def]
        captured.append((float(rudder_angle), float(sail_angle)))
        return state

    env.hub.step = spy_step  # type: ignore[method-assign]
    try:
        env.step(np.array([-1.0, -1.0], dtype=np.float32))
        env.step(np.array([1.0, 1.0], dtype=np.float32))
    finally:
        env.hub.step = original_step  # type: ignore[method-assign]

    assert len(captured) == 2
    rudder_lo, sail_lo = captured[0]
    rudder_hi, sail_hi = captured[1]
    assert rudder_lo == -35.0
    assert sail_lo == 0.0
    assert rudder_hi == 35.0
    assert sail_hi == np.radians(85.0)


def test_stagnation_penalty_in_info_when_enabled() -> None:
    cfg = WaypointEnvConfig(
        simulator_config="basic_sailbot.yaml",
        max_episode_steps=10,
        waypoint_min_radius_m=8.0,
        waypoint_max_radius_m=12.0,
        stagnation_penalty=0.5,
        stagnation_u_threshold=0.5,
    )
    env = WaypointEnv(config=cfg)
    env.reset(seed=1)
    _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert "penalty_stagnation" in info


def test_surge_speed_bonus_in_info_when_enabled() -> None:
    cfg = WaypointEnvConfig(
        simulator_config="basic_sailbot.yaml",
        max_episode_steps=10,
        waypoint_min_radius_m=8.0,
        waypoint_max_radius_m=12.0,
        surge_speed_bonus=0.3,
        surge_speed_bonus_cap_m_s=3.0,
    )
    env = WaypointEnv(config=cfg)
    env.reset(seed=2)
    _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert "reward_surge_speed" in info


def test_observation_includes_previous_action_channels() -> None:
    env = _make_env()
    env.reset(seed=23)
    obs, *_ = env.step(np.array([0.25, -0.5], dtype=np.float32))
    # Observation channels 11/12 track the clipped previous action.
    assert np.isclose(obs[11], env.last_action[0])
    assert np.isclose(obs[12], env.last_action[1])


def test_vis_callback_receives_state_messages() -> None:
    payloads: list[dict[str, object]] = []
    cfg = WaypointEnvConfig(
        simulator_config="basic_sailbot.yaml",
        waypoint_min_radius_m=8.0,
        waypoint_max_radius_m=12.0,
        vis_callback=payloads.append,
        vis_stride_steps=1,
    )
    env = WaypointEnv(config=cfg)
    env.reset(seed=9)
    env.step(np.array([0.0, 0.0], dtype=np.float32))

    assert len(payloads) >= 2
    latest = payloads[-1]
    assert latest["type"] == "state"
    assert "boat" in latest
    assert "waypoint" in latest
    control = latest.get("control")
    assert isinstance(control, dict)
    assert control.get("mode") == "training"


def test_info_carries_is_success_for_sb3() -> None:
    """SB3's EvalCallback logs a success rate only under this exact key.

    Renaming or dropping it silently turns off `eval/success_rate` and the
    `successes` array in `evaluations.npz`, leaving a run with no measurement of
    whether the boat ever reaches the mark.
    """
    env = _make_env()
    _, info = env.reset(seed=3)
    assert info["is_success"] is info["success"]

    _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert info["is_success"] is info["success"]


def test_reaching_the_waypoint_pays_the_success_reward() -> None:
    """Arrival is worth `success_reward` on top of the shaping terms."""
    cfg = WaypointEnvConfig(
        simulator_config="basic_sailbot.yaml",
        vmg_multiplier=0.0,
        dist_multiplier=0.0,
        time_penalty=0.0,
        success_reward=200.0,
    )
    env = WaypointEnv(config=cfg)
    env.reset(seed=5)
    env.waypoint = np.array([env.state.x, env.state.y], dtype=np.float64)

    _, reward, terminated, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert info["is_success"]
    assert terminated
    assert reward == pytest.approx(200.0, abs=1.0)


def _no_go_env(tau: float) -> WaypointEnv:
    return WaypointEnv(
        config=WaypointEnvConfig(
            simulator_config="basic_sailbot.yaml",
            no_go_zone_penalty=15.0,
            no_go_zone_half_angle_deg=35.0,
            no_go_occupancy_tau_s=tau,
        )
    )


def _point(env: WaypointEnv, twa_deg: float) -> None:
    """Aim the boat `twa_deg` off the wind, without touching anything else."""
    wind_to = math.radians(float(env.hub.sail_cfg["wind_dir_deg"]))
    heading = wind_to + math.pi - math.radians(twa_deg)
    env.state = State(x=env.state.x, y=env.state.y, psi=(math.cos(heading), math.sin(heading)), u=1.0, v=0.0, r=0.0)


def test_no_go_charge_on_contact_is_unchanged_at_tau_zero() -> None:
    """Tau 0 keeps the old behaviour, so configs written before this are untouched."""
    env = _no_go_env(tau=0.0)
    env.reset(seed=1)
    _point(env, 0.0)
    expected = 15.0 * (1.0 - math.cos(math.radians(35.0)))
    assert env._no_go_zone_penalty() == pytest.approx(expected, rel=1e-6)


def test_passing_through_costs_far_less_than_settling_in() -> None:
    """A tack sweeps the zone in ~2.3 s; pinching lives there. Only the second is charged."""
    env = _no_go_env(tau=4.0)
    env.reset(seed=2)
    _point(env, 10.0)

    passing = sum(env._no_go_zone_penalty() for _ in range(115))  # ~2.3 s at dt 0.02
    env.reset(seed=2)
    _point(env, 10.0)
    settled = sum(env._no_go_zone_penalty() for _ in range(115, 1150))  # the rest of a beat

    assert passing < 0.1 * settled
    assert env.no_go_occupancy > 0.9  # sitting there saturates the charge


def test_occupancy_does_not_reset_by_stepping_outside_briefly() -> None:
    """A step counter could be gamed by dipping past the boundary; a lag cannot."""
    env = _no_go_env(tau=4.0)
    env.reset(seed=3)
    _point(env, 10.0)
    for _ in range(200):
        env._no_go_zone_penalty()
    inside = env.no_go_occupancy

    _point(env, 50.0)  # pop outside for a tenth of a second
    for _ in range(5):
        env._no_go_zone_penalty()
    _point(env, 10.0)
    assert env.no_go_occupancy > 0.95 * inside


def test_occupancy_clears_between_episodes() -> None:
    env = _no_go_env(tau=4.0)
    env.reset(seed=4)
    _point(env, 0.0)
    for _ in range(300):
        env._no_go_zone_penalty()
    assert env.no_go_occupancy > 0.5
    env.reset(seed=5)
    assert env.no_go_occupancy == 0.0
