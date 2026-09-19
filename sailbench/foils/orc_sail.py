"""Sail models using the ORC VPP aerodynamic formulation.

Reference: ORC VPP Documentation 2023, sections 5.1-5.5. CLmax and CD0 come
from Table 5.1 (mainsail) and Table 5.4 (jib), "low" set, which is the set for
a rig with no adjustable check stays or forestay. Those coefficients were last
revised in 2016; the 2026 VPP update only touched hull residuary resistance.

Coefficients are functions of apparent wind angle, not of a geometric angle of
attack in a rotating sail frame, and drive and heel come out in one step:

    CR = CL sin(beta) - CD cos(beta)      (drive, boat +x)
    CH = CL cos(beta) + CD sin(beta)      (heel/side)

BasicSail instead looks up a symmetric NACA section. A soft sail is a cambered
membrane, and a symmetric section has no camber, so it makes no lift at zero
incidence.

Sheet trim enters through ORC's `flat`. The table gives the maximum achievable
lift at each apparent wind angle; `flat` scales it down when trim is not
optimal, and reaches zero when the sail is eased until it luffs.

Depowering: ORC picks `flat` (and `reef`) so that heeling moment stays inside
the righting moment. Without it the rig sails at full power all the time, and
in a 3-DOF simulator with no heel nothing pays for the side force it makes.
flat_for_righting_moment does this, and is off unless a righting moment is
configured.

Sloops (5.4.1): ORC combines main and jib into one "collective" rig by
area-weighting each coefficient and normalising by the reference area.

    CLmax = sum_i CLmax_i * bk_i * A_i / Aref            (5.35)
    CD0   = sum_i CD0_i   * bk_i * A_i / Aref            (5.36)
    KPP   = sum_i kp_i * CLmax_i^2 * bk_i * A_i / (Aref * CLmax^2)   (5.41)

bk_i is a blanketing factor, 1 for both sails here. ORC's mainsail blanketing
only differs from 1 with a mizzen staysail, and the jib's only for an
overlapping genoa (fj in 5.6.2 is zero when the jib fits inside the
foretriangle). A model sloop carries neither.

ORCWithJibSail requires a jib_area. ORCMainSail reads the same key as the sail
to strike, so on one config the two models are the boat with and without its
jib. With no jib_area, ORCMainSail is a mainsail of the configured area.
Otherwise the two differ only in which sails envelope() sums over.

The jib out-lifts the main below about 30 degrees apparent and does nothing
past 150, so a sloop points higher than the same hull under main alone.

BasicSail and HybridSail read a section polar and an analytic CL/CD curve.
These read neither, and ignore airfoil_name, res, CL_max and the rest, the same
way those models ignore the ORC keys. sail_factory warns if a block mixes them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import sailbench.utils.coordinate_helper as utils
from sailbench.models.model import Model, State
from sailbench.tf.tf_tree import TFTree2D

# --- ORC VPP 2023 Table 5.1, mainsail, "low" coefficient set ----------------
MAIN_AWA_DEG = np.array([0.0, 7.0, 9.0, 12.0, 28.0, 60.0, 90.0, 120.0, 150.0, 180.0])
MAIN_CL = np.array([0.00000, 0.86207, 1.05172, 1.16379, 1.34698, 1.35345, 1.26724, 0.93103, 0.38793, -0.11207])
MAIN_CD0 = np.array([0.04310, 0.02586, 0.02328, 0.02328, 0.03259, 0.11302, 0.38250, 0.96888, 1.31578, 1.34483])
# Two-dimensional quadratic viscous drag coefficient (ORC "kpm").
KPM = 0.01379

# --- ORC VPP 2023 Table 5.4, jib, "low" coefficient set --------------------
# The table starts at 7 degrees; np.interp holds the first value below that,
# which is the CL = 0 luff the mainsail table states explicitly.
JIB_AWA_DEG = np.array([7.0, 15.0, 20.0, 27.0, 50.0, 60.0, 100.0, 150.0, 180.0])
JIB_CL = np.array([0.00000, 1.00000, 1.37500, 1.45000, 1.45000, 1.25000, 0.40000, 0.00000, -0.10000])
JIB_CD0 = np.array([0.05000, 0.03200, 0.03100, 0.03700, 0.25000, 0.35000, 0.73000, 0.95000, 0.90000])
# ORC "kpj".
KPJ = 0.016


@dataclass(frozen=True)
class SailTable:
    """One sail's ORC coefficient table: CLmax and CD0 against apparent wind angle, plus its ``kp``."""

    awa_deg: np.ndarray
    cl: np.ndarray
    cd0: np.ndarray
    kp: float

    def at(self, awa_deg: float) -> tuple[float, float]:
        """Return ``(CLmax, CD0)`` interpolated at an apparent wind angle."""
        return float(np.interp(awa_deg, self.awa_deg, self.cl)), float(np.interp(awa_deg, self.awa_deg, self.cd0))


MAIN_TABLE = SailTable(MAIN_AWA_DEG, MAIN_CL, MAIN_CD0, KPM)
JIB_TABLE = SailTable(JIB_AWA_DEG, JIB_CL, JIB_CD0, KPJ)

# --- ORC Figure 5.14, kheff against apparent wind angle -------------------
# Effective rig height is not the masthead. Close-hauled the jib seals against
# the deck and the two sails act as one taller wing, so the rig sheds less tip
# vortex than its height suggests. Eased onto a reach that seal is lost and the
# interaction turns unfavourable.
#
# ORC publishes the curve only as a figure. These values are traced off the
# plots at 5-degree intervals and held at 0.80 past 80 degrees, where both
# figures end flat.
#
# Both editions are here because the peak is a rating parameter, not a
# measurement. ORC raised it from 1.22 to 1.4513 in 2023 together with deeper
# depowering (minimum flat 0.62 to 0.42) and a stronger twist function, and
# inside the VPP those offset. With no righting-moment limit configured only
# the power-adding half applies, so prefer the 2022 curve until one is set.
# On the reaching side (below 1.0) the two are nearly identical.
KHEFF_AWA_DEG = np.arange(0.0, 81.0, 5.0)
KHEFF_2022 = np.array(
    [1.000, 1.093, 1.169, 1.210, 1.2200, 1.178, 1.118, 1.059, 0.999, 0.939, 0.899, 0.871, 0.845, 0.824, 0.809, 0.801,
     0.800],
)
KHEFF_2023 = np.array(
    [1.000, 1.195, 1.350, 1.433, 1.4513, 1.365, 1.248, 1.133, 1.028, 0.948, 0.899, 0.868, 0.844, 0.825, 0.810, 0.802,
     0.800],
)
KHEFF_CURVES = {"orc-2022": KHEFF_2022, "orc-2023": KHEFF_2023}


def _checked_jib_area(jib_area: float, area: float) -> float:
    """Return ``jib_area`` as a float, or raise if it is not part of ``area``.

    Args:
        jib_area (float): The configured ``jib_area``.
        area (float): The rig's configured area [m^2].

    Returns:
        float: The validated jib area [m^2].

    Raises:
        ValueError: If the jib is not a positive part of the rig.

    """
    value = float(jib_area)
    if not 0.0 < value <= area:
        msg = f"sail jib_area must lie within (0, area]: got jib_area={value}, area={area}"
        raise ValueError(msg)
    return value


class ORCMainSail(Model):
    """Mainsail-only model on the ORC VPP coefficient envelope.

    Selected by ``model_type: orc_main``. On a config carrying a ``jib_area``
    this is the same boat with the jib struck: the mainsail keeps its area and
    the jib's share leaves the rig. Use :class:`ORCWithJibSail` for both sails.

    Config keys, all optional except ``area``:
        area: the rig as rigged [m^2]. With a ``jib_area`` the reference area
            becomes ``area - jib_area``, otherwise it is ``area``.
        jib_area: [m^2] the part of ``area`` this model drops. Read rather than
            refused so one config can be sailed either way.
        heff: rig height [m], top of the sail plan above the waterline (ORC's
            ``b + HBI``). Defaults to ``1.8 * sqrt(area)``.
        heff_model: ``orc-2022`` or ``orc-2023``. Scales ``heff`` by that
            edition's kheff curve against apparent wind angle (Figure 5.14), so
            the rig is taller close-hauled and shorter on a reach. The editions
            differ only in the peak, 1.22 against 1.45. See :data:`KHEFF_CURVES`.
            Absent, ``heff`` is constant.
        eff_span_corr: ORC eq. 5.42 sail-plan correction to effective span,
            from roach, fractionality and overlap. Default 1.0.
        wind_speed, wind_dir_deg: true wind, direction it blows *to*.
        rho_air: [kg/m^3], default 1.225. Usually supplied by the config's
            `environment` block rather than set here.
        alpha_opt_deg: angle of attack of peak lift, default 22.
        flat_stall_floor: lift left when badly over-sheeted.
        max_heeling_moment_nm, heel_arm_m: the righting-moment limit to depower
            against. Both together, or neither.

    """

    def __init__(self, params: dict[str, Any]) -> None:
        """Initialize the ORC sail model.

        Args:
            params (dict): Dictionary of parameters for the sail model.

        """
        super().__init__(params)
        # `area` is the rig as rigged; `_rig` says which of those sails are
        # actually set. ORC's reference area is the sum of the ones that are,
        # so striking the jib shrinks it.
        self.area = float(self.p.get("area", 1.0))
        self.sails = self._rig()
        self.area = sum(area for _, area in self.sails)
        # Parasitic part of CD0: the least drag the rig makes at any angle,
        # i.e. skin friction and windage on the sail. Everything above it is
        # form drag from projected area, which is the part trim changes. See
        # `form_drag_trim_factor`.
        self.cd0_floor = min(self.envelope(float(b))[1] for b in np.arange(0.0, 181.0, 1.0))
        self.heff = float(self.p.get("heff", 1.8 * np.sqrt(max(self.area, 1e-6))))
        heff_model = str(self.p.get("heff_model", "constant")).lower()
        if heff_model != "constant" and heff_model not in KHEFF_CURVES:
            known = ", ".join(["constant", *sorted(KHEFF_CURVES)])
            msg = f"{type(self).__name__} heff_model must be one of {known}: got {heff_model!r}"
            raise ValueError(msg)
        self.kheff = KHEFF_CURVES.get(heff_model)
        self.eff_span_corr = float(self.p.get("eff_span_corr", 1.0))
        self.alpha_opt = np.radians(float(self.p.get("alpha_opt_deg", 22.0)))
        self.flat_floor = float(self.p.get("flat_stall_floor", 0.55))
        # Righting-moment limit. Both keys or neither: a limit with no arm
        # cannot be turned into a force, an arm with no limit does nothing.
        # Absent, the sail carries whatever the trim curve asks for.
        limit = self.p.get("max_heeling_moment_nm")
        arm = self.p.get("heel_arm_m")
        if (limit is None) != (arm is None):
            msg = (
                f"{type(self).__name__} needs max_heeling_moment_nm and heel_arm_m together: "
                f"got max_heeling_moment_nm={limit!r}, heel_arm_m={arm!r}"
            )
            raise ValueError(msg)
        self.max_heeling_moment = None if limit is None else float(limit)
        self.heel_arm = 0.0 if arm is None else float(arm)
        # Diagnostics for the HUD / debugging.
        self.last_awa_deg = 0.0
        self.last_flat = 0.0
        self.last_cl = 0.0
        self.last_cd = 0.0
        self.last_heff = self.heff

    # --- rig ------------------------------------------------------------
    def _rig(self) -> list[tuple[SailTable, float]]:
        """Return the rig's sails as ``(table, area)`` pairs.

        A ``jib_area`` here means strike the jib. The mainsail keeps its area,
        the jib's share leaves the rig, and the reference area drops with it,
        so switching model_type on a sloop config lowers a headsail instead of
        handing the mainsail the jib's square metres to sail as one big main.
        """
        jib_area = self.p.get("jib_area")
        if jib_area is None:
            self.main_area = self.area
            return [(MAIN_TABLE, self.main_area)]

        self.main_area = self.area - _checked_jib_area(jib_area, self.area)
        if self.main_area <= 0.0:
            msg = (
                "sail model_type: orc_main strikes the jib and is left with no mainsail: "
                f"jib_area={float(jib_area)} is the whole of area={self.area}"
            )
            raise ValueError(msg)
        return [(MAIN_TABLE, self.main_area)]

    # --- coefficient envelope ------------------------------------------
    def envelope(self, awa_deg: float) -> tuple[float, float, float]:
        """Return the collective ``(CLmax, CD0, kpp)`` of the rig at an apparent wind angle.

        ORC eqs. 5.35, 5.36 and 5.41: each sail's table value weighted by its
        share of the reference area. ``kpp`` is weighted by lift squared as
        well, so the sail doing the lifting sets the quadratic viscous drag.
        With a single mainsail the result is the mainsail table unchanged.

        Args:
            awa_deg (float): Apparent wind angle off the bow [deg].

        Returns:
            tuple[float, float, float]: Collective ``(CLmax, CD0, kpp)``.

        """
        b = float(np.clip(abs(awa_deg), 0.0, 180.0))
        cl = cd0 = 0.0
        lift_weight = kp_weight = kp_mean = 0.0
        for table, area in self.sails:
            w = area / self.area
            cl_i, cd_i = table.at(b)
            cl += w * cl_i
            cd0 += w * cd_i
            lift_weight += w * cl_i * cl_i
            kp_weight += table.kp * w * cl_i * cl_i
            kp_mean += w * table.kp
        # Eq. 5.41 divides by CLmax^2, which is zero head to wind. kpp there
        # multiplies CL^2 = 0, so its value does not matter; the area-weighted
        # mean keeps it finite and continuous.
        kpp = kp_weight / lift_weight if lift_weight > 1e-12 else kp_mean
        return cl, cd0, kpp

    def effective_height(self, awa_deg: float) -> float:
        """Return the effective rig height for induced drag at an apparent wind angle.

        ORC eqs. 5.43 and 5.45: ``heff = eff_span_corr * kheff(beta) * (b + HBI)``.
        With ``heff_model`` unset ``kheff`` is 1 and this is the configured
        height scaled by ``eff_span_corr`` alone.

        Args:
            awa_deg (float): Apparent wind angle off the bow [deg].

        Returns:
            float: Effective rig height [m].

        """
        k = 1.0 if self.kheff is None else float(np.interp(abs(awa_deg), KHEFF_AWA_DEG, self.kheff))
        return self.eff_span_corr * k * self.heff

    def form_drag_trim_factor(self, beta: float, alpha: float) -> float:
        """How much of the table's form drag this trim presents.

        ORC's CD0 is the drag of a correctly trimmed sail. Its VPP picks the
        trim, so it never models a badly set one. Here the helm sets a sheet
        limit instead, and without this factor the full table is charged
        downwind whatever the boom is doing, which lets a sail strapped flat
        amidships run dead downwind at full speed.

        Running, a sail is a drag device, and the drag of a bluff surface goes
        with projected area: sin^2 of the angle between chord and flow, i.e.
        sin^2(alpha). That peaks at alpha = 90 deg, the sail square to the
        apparent wind, which is the trim the table represents, so normalise
        projected area against 1.

        Do not normalise against the fully-eased boom. sin^2 is not monotonic
        in alpha: easing sweeps alpha from beta down through 90 deg, where
        sin^2 is at its maximum, so every trim between hard in and fully out
        computes above 1 and clips back to it. At AWA 139 deg, sheet limits of
        10, 20, 40, 60 and 80 deg gave byte-identical drive, and only a boom
        within a few degrees of the centreline was charged anything. A policy
        trained against that pinned the sheet at one end for every step.

        The penalty ramps in across the second quadrant rather than switching
        on at the beam. At beta = 90 the weight is zero, so the result is 1 and
        meets the lifting-surface branch with no step; by a dead run it is
        charged in full. That gradient is also about right physically: a sail
        at 100 deg apparent is still mostly a lifting surface, one at 170 is
        not.

        Clipped at 1 because trim can only be worse than the table's optimum,
        never better, which is the same rule `flat` follows.

        Args:
            beta (float): Apparent wind angle off the bow, magnitude [rad].
            alpha (float): Sail angle of attack, ``beta`` minus the boom angle [rad].

        Returns:
            float: Fraction of the table's form drag this trim presents, in [0, 1].

        """
        if beta <= 0.5 * np.pi:
            return 1.0
        projected = float(np.sin(alpha) ** 2)
        weight = float(np.clip((beta - 0.5 * np.pi) / (0.5 * np.pi), 0.0, 1.0))
        return float(np.clip(1.0 - weight * (1.0 - projected), 0.0, 1.0))

    def flat_from_trim(self, alpha_rad: float) -> float:
        """ORC ``flat`` depowering factor from the sail's angle of attack.

        Peaks at ``alpha_opt``. Falls to 0 when the sail is eased until it luffs,
        and decays to ``flat_stall_floor`` when over-sheeted past stall.

        Args:
            alpha_rad (float): Sail angle of attack [rad].

        Returns:
            float: The ``flat`` factor, in [0, 1].

        """
        a = abs(float(alpha_rad))
        x = a / self.alpha_opt
        if x <= 1.0:
            return float(np.sin(0.5 * np.pi * np.clip(x, 0.0, 1.0)))
        decay = float(np.clip((x - 1.0) / 1.5, 0.0, 1.0))
        return float(1.0 - (1.0 - self.flat_floor) * decay)

    def flat_for_righting_moment(  # noqa: PLR0913
        self,
        flat: float,
        beta: float,
        cl_max: float,
        cd0: float,
        q_area: float,
        kpp: float = KPM,
        heff: float | None = None,
    ) -> float:
        """Largest flat, up to the one asked for, that fits the righting moment.

        The heel coefficient is a quadratic in flat::

            CH(f) = k cl_max^2 sin(beta) f^2 + cl_max cos(beta) f + cd0 sin(beta)

        It opens upward, so the f satisfying CH(f) <= CH_max form an interval
        and the answer is closed form. No iteration, and no assumption that CH
        rises with f.

        That assumption fails downwind. Past 90 degrees cos(beta) is negative,
        so more lift reduces heel, and the heel force is dominated by the
        cd0 sin(beta) term that easing does not touch. The constraint can then
        be infeasible with flat alone; ORC reefs for this case, which is not
        modelled here. When that happens, return the least-heeling trim
        available rather than easing further and making it worse.

        Returns flat unchanged when no righting moment is configured.

        Args:
            flat (float): The ``flat`` the trim curve asked for.
            beta (float): Apparent wind angle off the bow, magnitude [rad].
            cl_max (float): Collective maximum lift coefficient at ``beta``.
            cd0 (float): Collective parasitic drag coefficient at ``beta``.
            q_area (float): Dynamic pressure times reference area [N].
            kpp (float): Collective quadratic viscous drag coefficient.
            heff (float | None): Effective rig height [m]; ``self.heff`` if None.

        Returns:
            float: The ``flat`` to sail at, never above the one asked for.

        """
        if self.max_heeling_moment is None or q_area <= 0.0 or self.heel_arm <= 0.0:
            return flat

        ch_max = self.max_heeling_moment / (q_area * self.heel_arm)
        h = self.heff if heff is None else heff
        k = kpp + self.area / (np.pi * h**2)
        a = float(k * cl_max * cl_max * np.sin(beta))
        b = float(cl_max * np.cos(beta))
        c = float(cd0 * np.sin(beta))

        def ch(f: float) -> float:
            return a * f * f + b * f + c

        if ch(flat) <= ch_max:
            return flat  # the trim the helm asked for already fits

        if abs(a) < 1e-12:
            if abs(b) < 1e-12:
                return flat  # no lift and no induced drag: nothing to depower
            eased = (ch_max - c) / b
            return float(np.clip(eased, 0.0, flat)) if b > 0.0 else flat

        disc = b * b - 4.0 * a * (c - ch_max)
        if disc > 0.0:
            root = np.sqrt(disc)
            lo = (-b - root) / (2.0 * a)
            hi = (-b + root) / (2.0 * a)
            # Feasible trims are [lo, hi]; intersect with what the helm can ease to.
            if max(lo, 0.0) <= min(hi, flat):
                return float(min(hi, flat))

        # Infeasible with flat alone: give the least heel available on [0, flat].
        vertex = -b / (2.0 * a)
        candidates = [0.0, flat]
        if 0.0 < vertex < flat:
            candidates.append(float(vertex))
        return float(min(candidates, key=ch))

    def compute(self, state: State, tf_tree: TFTree2D) -> np.ndarray:
        """Compute the aerodynamic force the rig makes, in the boat frame.

        Args:
            state (State): Current boat state.
            tf_tree (TFTree2D): Transform tree with boat and sail frames.

        Returns:
            np.ndarray: X and Y forces in newtons (boat frame).

        """
        rho = float(self.p.get("rho_air", 1.225))

        aw_boat = utils.apparent_wind_boat(
            state,
            tf_tree,
            float(self.p.get("wind_speed", 0.0)),
            float(self.p.get("wind_dir_deg", 0.0)),
        )

        aw_speed = float(np.hypot(aw_boat[0], aw_boat[1]))
        if aw_speed < 1e-6:
            self.last_awa_deg = self.last_flat = self.last_cl = self.last_cd = 0.0
            return np.zeros(2, dtype=float)

        # Apparent wind angle: direction the wind blows FROM, off the bow.
        # Positive = wind from the +y side.
        awa = float(np.arctan2(-aw_boat[1], -aw_boat[0]))
        beta = abs(awa)
        wind_side = 1.0 if awa >= 0.0 else -1.0

        # Sail angle of attack = apparent wind angle minus boom angle.
        boom = abs(float(np.arctan2(tf_tree.transforms["sail"].s, tf_tree.transforms["sail"].c)))
        alpha = beta - boom

        cl_max, cd0, kpp = self.envelope(np.degrees(beta))
        heff = self.effective_height(np.degrees(beta))
        flat = self.flat_from_trim(alpha) if alpha > 0.0 else 0.0

        # Only the form-drag part of CD0 changes with trim; the parasitic floor
        # is there whatever the boom does. Applied before the righting-moment
        # solve so that depowering sees the drag the rig actually makes.
        cd0 = self.cd0_floor + (cd0 - self.cd0_floor) * self.form_drag_trim_factor(beta, alpha)

        q = 0.5 * rho * aw_speed * aw_speed * self.area
        flat = self.flat_for_righting_moment(flat, beta, cl_max, cd0, q, kpp, heff)

        cl = cl_max * flat
        # ORC eq. 5.46: CDi = [KPP + Aref / (pi * heff^2)] * (CLmax * flat)^2
        cd_induced = (kpp + self.area / (np.pi * heff**2)) * cl * cl
        cd = cd0 + cd_induced

        # ORC drive / heel resolution.
        cr = cl * np.sin(beta) - cd * np.cos(beta)
        ch = cl * np.cos(beta) + cd * np.sin(beta)

        self.last_awa_deg = float(np.degrees(awa))
        self.last_flat = float(flat)
        self.last_cl = float(cl)
        self.last_cd = float(cd)
        self.last_heff = float(heff)

        # Wind from +y pushes the boat toward -y.
        return np.array([q * cr, -wind_side * q * ch], dtype=float)


class ORCWithJibSail(ORCMainSail):
    """Main and jib: ORC's collective rig, section 5.4.1.

    Selected by ``model_type: orc_w_jib``. Same config as :class:`ORCMainSail`,
    plus:
        jib_area: [m^2] the jib's part of ``area``. Required. The envelope is
            the area-weighted blend of the main and jib tables, and ``area``
            stays the reference area they are normalised by.

    """

    def _rig(self) -> list[tuple[SailTable, float]]:
        """Return the main and jib, split by ``jib_area``."""
        jib_area = self.p.get("jib_area")
        if jib_area is None:
            msg = "sail model_type: orc_w_jib needs jib_area; for a single mainsail use model_type: orc_main"
            raise ValueError(msg)
        self.jib_area = _checked_jib_area(jib_area, self.area)
        self.main_area = self.area - self.jib_area
        return [(MAIN_TABLE, self.main_area), (JIB_TABLE, self.jib_area)]
