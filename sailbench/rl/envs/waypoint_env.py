"""Gymnasium waypoint-navigation environment for SailBench."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from sailbench.models.model import State
from sailbench.sim.protocol import make_state_message
from sailbench.sim.sailboat_hub import SailboatHub
from sailbench.solvers.rk4 import rk4_step

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(slots=True)
class WaypointEnvConfig:
    """Configuration for waypoint navigation task."""

    simulator_config: str = "basic_sailbot.yaml"
    dt: float | None = None
    max_episode_steps: int = 1000
    max_rudder_deg: float = 35.0
    max_sail_deg: float = 85.0
    initial_heading_range_deg: float = 30.0
    initial_speed_max: float = 0.2
    spawn_radius_m: float = 10.0
    waypoint_min_radius_m: float = 12.0
    waypoint_max_radius_m: float = 25.0
    upwind_waypoint_bias: float = 0.0
    upwind_half_angle_deg: float = 35.0
    success_radius_m: float = 1.5
    fail_radius_m: float = 80.0
    speed_scale: float = 6.0
    yaw_rate_scale: float = 2.0
    wind_speed_scale: float = 15.0
    vmg_multiplier: float = 1.0
    joint_penalty: float = 0.0
    dist_multiplier: float = 5.0
    time_penalty: float = 3.0
    stagnation_penalty: float = 0.0
    stagnation_u_threshold: float = 0.35
    no_go_zone_penalty: float = 0.0
    no_go_zone_half_angle_deg: float = 45.0
    # Time constant for how long the boat has been sitting inside the no-go zone.
    # 0 charges the penalty on contact, which is the previous behaviour and which
    # punishes a tack -- a tack has to pass through the zone. Above 0 the charge
    # follows a lagged occupancy instead, so passing through is nearly free and
    # living there is not. 4 s: a tack is inside for about 2.3 s.
    no_go_occupancy_tau_s: float = 0.0
    jibe_penalty: float = 0.0
    jibe_threshold_deg: float = 150.0
    surge_speed_bonus: float = 0.0
    surge_speed_bonus_cap_m_s: float = 6.0
    success_reward: float = 25.0
    failure_penalty: float = -10.0
    vis_callback: Callable[[dict[str, Any]], None] | None = None
    vis_stride_steps: int = 1


class WaypointEnv(gym.Env[NDArray[np.float32], NDArray[np.float64]]):  # type: ignore[misc]
    """Continuous-control waypoint task built on SailboatHub dynamics."""

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(self, config: WaypointEnvConfig) -> None:
        super().__init__()
        self.cfg = config
        self.hub = SailboatHub(config_file=self.cfg.simulator_config)
        self.dt = float(self.cfg.dt or self.hub.simulation_cfg.get("dt", 0.02))
        self.max_sail_rad = math.radians(self.cfg.max_sail_deg)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(13,), dtype=np.float32)

        self.state = State(x=0.0, y=0.0, psi=(1.0, 0.0), u=0.0, v=0.0, r=0.0)
        self.waypoint = np.zeros(2, dtype=np.float64)
        self.steps = 0
        self.t = 0.0
        self.prev_distance = 0.0
        self.prev_action = np.zeros(2, dtype=np.float64)
        self.last_action = np.zeros(2, dtype=np.float64)
        self.no_go_occupancy = 0.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Reset environment and sample start/goal states."""
        super().reset(seed=seed)
        del options

        self.hub = SailboatHub(config_file=self.cfg.simulator_config)
        self.steps = 0
        self.t = 0.0
        self.prev_action = np.zeros(2, dtype=np.float64)
        self.last_action = np.zeros(2, dtype=np.float64)
        self.no_go_occupancy = 0.0

        start = self._sample_start()
        self.state = start
        self.waypoint = self._sample_waypoint(start)
        self.prev_distance = self._distance_to_waypoint()
        self._publish_vis_state(force=True)

        return self._get_observation(), self._build_info(
            vmg=0.0,
            vmg_term=0.0,
            joint_delta=0.0,
            joint_penalty_term=0.0,
            movement_penalty_term=0.0,
            stagnation_penalty_term=0.0,
            no_go_zone_penalty_term=0.0,
            jibe_penalty_term=0.0,
            surge_speed_bonus_term=0.0,
            dist_delta=0.0,
            dist_term=0.0,
            time_penalty_term=0.0,
            success=False,
            failure=False,
        )

    def step(self, action: NDArray[np.float64]) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Advance simulation with normalized rudder/sheet-limit actions."""
        clipped = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        rudder_deg = float(clipped[0] * self.cfg.max_rudder_deg)
        # Map sail action from [-1, 1] -> [0, 1] so sheet limit is [0, max_sail_deg].
        sail_cmd = 0.5 * (float(clipped[1]) + 1.0)
        sheet_limit_rad = float(sail_cmd * self.max_sail_rad)

        self.state = self.hub.step(
            state=self.state,
            dt=self.dt,
            solver=rk4_step,
            sail_angle=sheet_limit_rad,
            rudder_angle=rudder_deg,
        )
        self.steps += 1
        self.t += self.dt

        distance = self._distance_to_waypoint()
        vmg = self._velocity_made_good_to_waypoint()
        vmg_term = self.cfg.vmg_multiplier * vmg
        action_delta = np.abs(clipped - self.prev_action)
        
        joint_delta = float(np.sum(action_delta))
        joint_penalty_term = self.cfg.joint_penalty * joint_delta
        movement_penalty_term = joint_penalty_term
        dist_delta = self.prev_distance - distance
        dist_term = self.cfg.dist_multiplier * dist_delta
        stagnation_penalty_term = self._stagnation_penalty()
        no_go_zone_penalty_term = self._no_go_zone_penalty()
        jibe_penalty_term = self._jibe_penalty()
        surge_speed_bonus_term = self._surge_speed_bonus()
        time_penalty_term = self.cfg.time_penalty

        terminated = False
        truncated = self.steps >= self.cfg.max_episode_steps
        success = distance <= self.cfg.success_radius_m
        failure = distance >= self.cfg.fail_radius_m

        terminal_reward = 0.0
        if success:
            terminated = True
            terminal_reward = self.cfg.success_reward
        elif failure:
            terminated = True
            terminal_reward = self.cfg.failure_penalty

        reward = (
            vmg_term
            - movement_penalty_term
            + dist_term
            + surge_speed_bonus_term
            - stagnation_penalty_term
            - no_go_zone_penalty_term
            - jibe_penalty_term
            - time_penalty_term
            + terminal_reward
        )

        self.prev_distance = distance
        self.prev_action = clipped
        self.last_action = clipped

        info = self._build_info(
            vmg=vmg,
            vmg_term=vmg_term,
            joint_delta=joint_delta,
            joint_penalty_term=joint_penalty_term,
            movement_penalty_term=movement_penalty_term,
            dist_delta=dist_delta,
            dist_term=dist_term,
            stagnation_penalty_term=stagnation_penalty_term,
            no_go_zone_penalty_term=no_go_zone_penalty_term,
            jibe_penalty_term=jibe_penalty_term,
            surge_speed_bonus_term=surge_speed_bonus_term,
            time_penalty_term=time_penalty_term,
            success=success,
            failure=failure,
        )
        self._publish_vis_state(force=terminated or truncated)
        return self._get_observation(), float(reward), terminated, truncated, info

    def _publish_vis_state(self, *, force: bool = False) -> None:
        """Push the current simulation state to an optional visualization callback."""
        callback = self.cfg.vis_callback
        if callback is None:
            return
        stride = max(int(self.cfg.vis_stride_steps), 1)
        if not force and (self.steps % stride != 0):
            return

        wind_speed = float(self.hub.sail_cfg.get("wind_speed", 0.0))
        wind_dir_deg = float(self.hub.sail_cfg.get("wind_dir_deg", 0.0))
        sail_force = getattr(self.hub, "last_sail_force", (0.0, 0.0))
        forces = getattr(self.hub, "last_forces", {})
        sail_angle_deg = math.degrees(float(getattr(self.hub, "last_sail_angle_rad", 0.0)))
        msg = make_state_message(
            self.state,
            self.t,
            wind_speed=wind_speed,
            wind_dir_deg=wind_dir_deg,
            sail_force=sail_force,
            forces=forces,
            sail_angle_deg=sail_angle_deg,
            rudder_angle_deg=float(self.last_action[0] * self.cfg.max_rudder_deg),
            waypoint=(float(self.waypoint[0]), float(self.waypoint[1])),
            control_mode="training",
        )
        callback(msg)

    def _stagnation_penalty(self) -> float:
        w = float(self.cfg.stagnation_penalty)
        if w <= 0.0:
            return 0.0
        thr = float(self.cfg.stagnation_u_threshold)
        deficit = max(0.0, thr - abs(float(self.state.u)))
        return w * deficit

    def _no_go_zone_penalty(self) -> float:
        """Charge for making way inside the no-go zone, once the boat has settled there.

        This is a task constraint standing in for physics the simulator does not
        have. Flingo holds 22 degrees true and drives at 1.46 m/s there; a real
        boat of her type stops somewhere between 30 and 45. A policy trained
        against the simulator as it stands learns to pinch -- 41% of its steps
        inside 25 degrees, at 62% of the boat speed it could have had -- and that
        habit does not survive contact with water. Raising the weight is not a
        fix for the physics and is not claimed to be one; see the note in
        configs/flingo_rl.yaml.

        The charge follows a lagged occupancy rather than mere contact, because a
        tack has to pass through the zone: penalising contact prices the manoeuvre
        the beat is made of, and the policy stops tacking instead of starting to
        foot. Sweeping through at 30 deg/s spends about 2.3 s inside, which at a
        4 s time constant reaches roughly 0.44 of the full rate and then decays.
        Sitting there drives it to 1. The lag also cannot be reset by dipping a
        degree outside the boundary and coming back, which a step counter could.
        """
        w = float(self.cfg.no_go_zone_penalty)
        cos_threshold = math.cos(math.radians(float(self.cfg.no_go_zone_half_angle_deg)))
        wind_dir_deg = float(self.hub.sail_cfg.get("wind_dir_deg", 90.0))
        wind_dir_rad = math.radians(wind_dir_deg)
        c, s = self.state.psi
        # wind_boat_x: x-component of the wind-blows-to vector in boat frame.
        # Negative means wind blows aft → wind SOURCE is ahead → boat is in no-go zone.
        wind_boat_x = c * math.cos(wind_dir_rad) + s * math.sin(wind_dir_rad)
        # penetration: 0 at zone boundary, positive deeper in the zone
        penetration = max(0.0, -wind_boat_x - cos_threshold)

        tau = float(self.cfg.no_go_occupancy_tau_s)
        if tau > 0.0:
            inside = 1.0 if penetration > 0.0 else 0.0
            self.no_go_occupancy += (inside - self.no_go_occupancy) * min(self.dt / tau, 1.0)
        else:
            self.no_go_occupancy = 1.0

        if w <= 0.0:
            return 0.0
        return w * penetration * self.no_go_occupancy

    def _jibe_penalty(self) -> float:
        w = float(self.cfg.jibe_penalty)
        if w <= 0.0:
            return 0.0
        # jibe_threshold_deg: TWA beyond which we penalize (e.g. 150° → penalise deep downwind).
        # cos(180° - threshold) gives the wind_boat_x value at the boundary.
        cos_threshold = math.cos(math.radians(180.0 - float(self.cfg.jibe_threshold_deg)))
        wind_dir_deg = float(self.hub.sail_cfg.get("wind_dir_deg", 90.0))
        wind_dir_rad = math.radians(wind_dir_deg)
        c, s = self.state.psi
        # wind_boat_x > cos_threshold means wind source is astern → deep downwind / jibe zone.
        wind_boat_x = c * math.cos(wind_dir_rad) + s * math.sin(wind_dir_rad)
        penetration = max(0.0, wind_boat_x - cos_threshold)
        return w * penetration

    def _surge_speed_bonus(self) -> float:
        w, cap = self.cfg.surge_speed_bonus, self.cfg.surge_speed_bonus_cap_m_s
        if w <= 0.0 or cap <= 0.0:
            return 0.0
        u = max(0.0, float(self.state.u))  # body-frame surge: forward only, no astern reward
        return w * min(u, cap) / cap

    def _velocity_made_good_to_waypoint(self) -> float:
        dx = self.waypoint[0] - self.state.x
        dy = self.waypoint[1] - self.state.y
        distance = max(math.hypot(dx, dy), 1e-9)
        goal_dir_x = dx / distance
        goal_dir_y = dy / distance

        c, s = self.state.psi
        vx = c * self.state.u - s * self.state.v
        vy = s * self.state.u + c * self.state.v
        return float(vx * goal_dir_x + vy * goal_dir_y)

    def _sample_start(self) -> State:
        theta = float(self.np_random.uniform(-math.pi, math.pi))
        radius = float(self.np_random.uniform(0.0, self.cfg.spawn_radius_m))
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)

        heading_jitter = math.radians(self.cfg.initial_heading_range_deg)
        heading = float(self.np_random.uniform(-heading_jitter, heading_jitter))
        u = float(self.np_random.uniform(0.0, self.cfg.initial_speed_max))
        v = float(self.np_random.uniform(-0.05, 0.05))
        r = float(self.np_random.uniform(-0.05, 0.05))
        return State(x=x, y=y, psi=(math.cos(heading), math.sin(heading)), u=u, v=v, r=r)

    def _sample_waypoint(self, start: State) -> NDArray[np.float64]:
        theta = float(self.np_random.uniform(-math.pi, math.pi))
        bias = float(np.clip(self.cfg.upwind_waypoint_bias, 0.0, 1.0))
        if float(self.np_random.random()) < bias:
            # Wind config stores direction the wind blows toward; upwind is the opposite bearing.
            wind_to_deg = float(self.hub.sail_cfg.get("wind_dir_deg", 90.0))
            upwind_heading = math.radians(wind_to_deg) + math.pi
            half_width = math.radians(max(float(self.cfg.upwind_half_angle_deg), 0.0))
            theta = float(self.np_random.uniform(upwind_heading - half_width, upwind_heading + half_width))
        radius = float(self.np_random.uniform(self.cfg.waypoint_min_radius_m, self.cfg.waypoint_max_radius_m))
        return np.array(
            [
                start.x + radius * math.cos(theta),
                start.y + radius * math.sin(theta),
            ],
            dtype=np.float64,
        )

    def _distance_to_waypoint(self) -> float:
        dx = self.waypoint[0] - self.state.x
        dy = self.waypoint[1] - self.state.y
        return float(math.hypot(dx, dy))

    def _heading_alignment(self) -> float:
        dx = self.waypoint[0] - self.state.x
        dy = self.waypoint[1] - self.state.y
        dist = max(math.hypot(dx, dy), 1e-9)
        target_unit = np.array([dx / dist, dy / dist], dtype=np.float64)
        heading_unit = np.array([self.state.psi[0], self.state.psi[1]], dtype=np.float64)
        return float(np.sum(target_unit * heading_unit))

    def _relative_waypoint_boat_frame(self) -> NDArray[np.float64]:
        dx_world = self.waypoint[0] - self.state.x
        dy_world = self.waypoint[1] - self.state.y
        c = self.state.psi[0]
        s = self.state.psi[1]
        dx_boat = c * dx_world + s * dy_world
        dy_boat = -s * dx_world + c * dy_world
        return np.array([dx_boat, dy_boat], dtype=np.float64)

    def _get_observation(self) -> NDArray[np.float32]:
        rel_wp = self._relative_waypoint_boat_frame()
        distance = max(self._distance_to_waypoint(), 1e-9)
        rel_dir = rel_wp / distance

        wind_speed = float(self.hub.sail_cfg.get("wind_speed", 0.0))
        wind_dir_deg = float(self.hub.sail_cfg.get("wind_dir_deg", 90.0))
        wind_dir_rad = math.radians(wind_dir_deg)
        wind_world = np.array([math.cos(wind_dir_rad), math.sin(wind_dir_rad)], dtype=np.float64)
        c = self.state.psi[0]
        s = self.state.psi[1]
        wind_boat = np.array(
            [
                c * wind_world[0] + s * wind_world[1],
                -s * wind_world[0] + c * wind_world[1],
            ],
            dtype=np.float64,
        )

        observation = np.array(
            [
                np.clip(rel_wp[0] / self.cfg.waypoint_max_radius_m, -1.0, 1.0),
                np.clip(rel_wp[1] / self.cfg.waypoint_max_radius_m, -1.0, 1.0),
                np.clip(distance / self.cfg.waypoint_max_radius_m, 0.0, 1.0),
                np.clip(rel_dir[0], -1.0, 1.0),
                np.clip(rel_dir[1], -1.0, 1.0),
                np.clip(self.state.u / self.cfg.speed_scale, -1.0, 1.0),
                np.clip(self.state.v / self.cfg.speed_scale, -1.0, 1.0),
                np.clip(self.state.r / self.cfg.yaw_rate_scale, -1.0, 1.0),
                np.clip(wind_boat[0], -1.0, 1.0),
                np.clip(wind_boat[1], -1.0, 1.0),
                np.clip(wind_speed / self.cfg.wind_speed_scale, 0.0, 1.0),
                float(self.last_action[0]),
                float(self.last_action[1]),
            ],
            dtype=np.float32,
        )
        return observation

    def _build_info(
        self,
        vmg: float,
        vmg_term: float,
        joint_delta: float,
        joint_penalty_term: float,
        movement_penalty_term: float,
        stagnation_penalty_term: float,
        no_go_zone_penalty_term: float,
        jibe_penalty_term: float,
        surge_speed_bonus_term: float,
        dist_delta: float,
        dist_term: float,
        time_penalty_term: float,
        success: bool,
        failure: bool,
    ) -> dict[str, Any]:
        return {
            "distance_to_waypoint": self._distance_to_waypoint(),
            "waypoint_x": float(self.waypoint[0]),
            "waypoint_y": float(self.waypoint[1]),
            "sim_time_s": self.t,
            "step_count": self.steps,
            "vmg": vmg,
            "reward_vmg": vmg_term,
            "joint_delta": joint_delta,
            "penalty_joint": joint_penalty_term,
            "penalty_movement_total": movement_penalty_term,
            "dist_delta": dist_delta,
            "reward_dist": dist_term,
            "penalty_stagnation": stagnation_penalty_term,
            "penalty_no_go_zone": no_go_zone_penalty_term,
            "no_go_occupancy": self.no_go_occupancy,
            "penalty_jibe": jibe_penalty_term,
            "reward_surge_speed": surge_speed_bonus_term,
            "penalty_time": time_penalty_term,
            "success": success,
            # SB3's EvalCallback logs `eval/success_rate` and fills
            # `evaluations.npz["successes"]` only when it finds this exact key in
            # the info dict. Without it nothing in a training run answers "does
            # the boat reach the mark?" -- episode length is a proxy, not an
            # answer. `success` above is kept for callers that already read it.
            "is_success": success,
            "failure": failure,
        }

