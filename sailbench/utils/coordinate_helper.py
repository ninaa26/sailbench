"""Helper functions for coordinate transforms."""

import numpy as np

from sailbench.models.model import State
from sailbench.tf.tf_tree import TFTree2D, Transform2D


def get_global_track(state: State) -> float:
    """Get the boat's track angle in degrees.

    Args:
        state (np.ndarray): Current body state of the sailboat -> [x, y, psi, u, v, r]

    Returns:
        float: Boat track angle in degrees.

    """
    u = state.u
    v = state.v
    track_rad = np.arctan2(v, u)
    return float(np.degrees(track_rad))


def get_local_track(state: State) -> float:
    """Get the boat's local track angle in degrees.

    Args:
        state (State): Current body state of the sailboat -> [x, y, psi, u, v, r]

    Returns:
        float: Boat local track angle in degrees from -180 to 180.

    """
    global_deg = get_global_track(state)
    psi_deg = state.get_heading

    diff = global_deg - psi_deg

    # wrap to [-180, 180]
    return ((diff + 180.0) % 360.0) - 180.0

def get_local_track_vector(state: State) -> np.ndarray:
    """Return unit track vector in boat frame."""
    vec = np.array([state.u, state.v], dtype=np.float64)
    mag = np.linalg.norm(vec)

    return vec / mag

def get_velocity_magnitude(state: State) -> float:
    """Get the boat's velocity magnitude.

    Args:
        state (State): Current body state of the sailboat -> [x, y, psi, u, v, r]

    Returns:
        float: Boat velocity magnitude in m/s.

    """
    u = state.u
    v = state.v
    return float(np.hypot(u, v))


def apparent_wind_boat(
    state: State,
    tf_tree: TFTree2D,
    wind_speed: float,
    wind_dir_deg: float,
) -> np.ndarray:
    """Apparent wind in the boat frame, as the vector the air travels along.

    Shared by every above-water model and by the hub's sheeting logic, so the
    two cannot disagree about the wind. `wind_dir_deg` is the direction the
    true wind blows *to*, in the world frame.

    Args:
        state (State): Current body state of the sailboat.
        tf_tree (TFTree2D): Transform tree, used for the boat's heading.
        wind_speed (float): True wind speed [m/s].
        wind_dir_deg (float): Direction the true wind blows toward [deg].

    Returns:
        np.ndarray: Apparent wind vector in the boat frame [m/s].

    """
    wind_rad = np.radians(float(wind_dir_deg))
    wind_world = float(wind_speed) * np.array([np.cos(wind_rad), np.sin(wind_rad)], dtype=float)
    v_boat_world = tf_tree.vector_to_frame(np.array([state.u, state.v], dtype=float), "boat", "world")
    return np.asarray(tf_tree.vector_to_frame(wind_world - v_boat_world, "world", "boat"), dtype=float)


def fluid_transform_from_state(state: State) -> Transform2D:
    """Build the boat->fluid transform for a given state.

    The fluid frame's +x axis points along the direction the water travels
    relative to the boat, which is opposite the boat's velocity. Foils resolve
    their drag along +x of this frame, so it must be rebuilt whenever the state
    changes; a stale fluid frame silently flips the sign of foil drag.

    Args:
        state (State): Current body state of the sailboat.

    Returns:
        Transform2D: Rotation-only transform from the boat frame to the fluid frame.

    """
    speed = get_velocity_magnitude(state)
    if speed <= 1e-6:
        return Transform2D(x=0.0, y=0.0, c=1.0, s=0.0)
    return Transform2D(x=0.0, y=0.0, c=-state.u / speed, s=-state.v / speed)


def fluid_frame_to_body_frame(forces_fluid: np.ndarray, local_track_deg: float) -> np.ndarray:
    """Convert forces from fluid frame to body frame.

    Args:
        forces_fluid (np.ndarray): Forces in fluid frame (x: lift, y: drag).
        local_track (np.ndarray): Local track angle in degrees.

    Returns:
        np.ndarray: Forces in body frame (+X, +Y).

    """
    local_track_deg_360 = local_track_deg % 360
    local_track_rad = np.radians(local_track_deg_360)
    rotation_matrix = np.array(
        [[np.cos(local_track_rad), np.sin(local_track_rad)], [-np.sin(local_track_rad), np.cos(local_track_rad)]]
    )
    return np.asarray(rotation_matrix @ forces_fluid, dtype=np.float64)


def global_to_local(vec_global: np.ndarray, psi: tuple[float, float]) -> np.ndarray:
    """Rotate vector from global frame to body/local frame."""
    c, s = psi

    rot_t = np.array(
        [
            [c, s],
            [-s, c],
        ]
    )

    return rot_t @ vec_global
