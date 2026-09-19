"""ORC sail models: the VPP coefficient envelope, trim depowering and force split."""

import itertools
import math

import numpy as np
import pytest

from sailbench.foils.orc_sail import (
    JIB_AWA_DEG,
    JIB_CD0,
    JIB_CL,
    JIB_TABLE,
    KHEFF_2022,
    KHEFF_2023,
    KHEFF_AWA_DEG,
    KPJ,
    KPM,
    MAIN_AWA_DEG,
    MAIN_CD0,
    MAIN_CL,
    MAIN_TABLE,
    ORCMainSail,
    ORCWithJibSail,
)
from sailbench.models.model import State
from sailbench.tf.tf_tree import TFTree2D, Transform2D

WIND_TO_DEG = 90.0
AREA = 1.971
HEFF = 2.592
JIB = 0.775


def make_sail(**overrides: float) -> ORCMainSail:
    """Build an ORC sail with Flingo's measured rig."""
    return ORCMainSail(
        {
            "area": AREA,
            "heff": HEFF,
            "wind_speed": 5.0,
            "wind_dir_deg": WIND_TO_DEG,
            "air_density": 1.225,
            **overrides,
        },
    )


def make_sloop(**overrides: float) -> ORCWithJibSail:
    """Build an ORC sloop with Flingo's measured rig and jib."""
    return ORCWithJibSail(
        {
            "area": AREA,
            "jib_area": JIB,
            "heff": HEFF,
            "wind_speed": 5.0,
            "wind_dir_deg": WIND_TO_DEG,
            "air_density": 1.225,
            **overrides,
        },
    )


def make_state(u: float = 1.7, psi: float = 0.0) -> State:
    """Generate a test state."""
    return State.from_array(np.array([0.0, 0.0, np.cos(psi), np.sin(psi), u, 0.0, 0.0]))


def tree(boom_rad: float, psi: float = 0.0) -> TFTree2D:
    """Transform tree with the boat on a heading and the boom trimmed."""
    tf = TFTree2D()
    tf.add_frame(name="boat", parent="world", transform=Transform2D(x=0.0, y=0.0, c=math.cos(psi), s=math.sin(psi)))
    tf.add_frame(
        name="sail", parent="boat", transform=Transform2D(x=0.0, y=0.0, c=math.cos(boom_rad), s=math.sin(boom_rad)),
    )
    return tf


def beat(twa_deg: float) -> float:
    """Heading `twa_deg` off the true wind, wind on the port side."""
    return math.radians(WIND_TO_DEG) - math.pi + math.radians(twa_deg)


class TestEnvelope:
    """ORC Table 5.1 lookup."""

    def test_no_lift_head_to_wind(self) -> None:
        """CL is zero at 0 degrees, so luffing is built into the table."""
        assert make_sail().envelope(0.0)[0] == 0.0

    def test_peak_lift_is_upwind(self) -> None:
        """A sail makes its most lift on a beat or close reach, not running."""
        sail = make_sail()
        assert sail.envelope(45.0)[0] > sail.envelope(135.0)[0]
        assert sail.envelope(28.0)[0] > sail.envelope(150.0)[0]

    def test_parasitic_drag_grows_downwind(self) -> None:
        """CD0 rises steeply as the sail squares off to the wind."""
        sail = make_sail()
        assert sail.envelope(150.0)[1] > sail.envelope(90.0)[1] > sail.envelope(28.0)[1]

    def test_matches_the_table_at_its_own_nodes(self) -> None:
        """Interpolation must reproduce the published values exactly."""
        sail = make_sail()
        for awa, cl in zip(MAIN_AWA_DEG, MAIN_CL, strict=True):
            assert sail.envelope(float(awa))[0] == pytest.approx(float(cl))

    def test_symmetric_in_wind_side(self) -> None:
        """The envelope depends on the magnitude of the apparent wind angle."""
        sail = make_sail()
        assert sail.envelope(-45.0) == sail.envelope(45.0)

    def test_stays_within_the_table_beyond_its_ends(self) -> None:
        """Past 180 degrees the lookup clamps rather than extrapolating."""
        sail = make_sail()
        assert sail.envelope(250.0) == sail.envelope(180.0)


class TestTrim:
    """The ORC `flat` depowering factor."""

    def test_luffing_at_zero_incidence(self) -> None:
        """An over-eased sail carries no lift at all."""
        assert make_sail().flat_from_trim(0.0) == 0.0

    def test_full_power_at_optimum(self) -> None:
        """Flat reaches 1 at alpha_opt."""
        assert make_sail().flat_from_trim(math.radians(22.0)) == pytest.approx(1.0)

    def test_rises_monotonically_up_to_the_optimum(self) -> None:
        """Sheeting in from luffing must never lose power."""
        sail = make_sail()
        flats = [sail.flat_from_trim(math.radians(a)) for a in (0, 5, 10, 15, 22)]
        assert flats == sorted(flats)

    def test_over_sheeting_depowers_towards_the_floor(self) -> None:
        """Past the optimum the sail stalls back towards flat_stall_floor."""
        sail = make_sail()
        assert sail.flat_from_trim(math.radians(22.0)) > sail.flat_from_trim(math.radians(45.0))
        assert sail.flat_from_trim(math.radians(90.0)) == pytest.approx(sail.flat_floor)


class TestForces:
    """Drive/heel resolution in the boat frame."""

    def test_no_apparent_wind_no_force(self) -> None:
        """Zero wind with the boat stopped means no force."""
        f = make_sail(wind_speed=0.0).compute(make_state(u=0.0), tree(0.0))
        assert np.allclose(f, 0.0)

    @pytest.mark.parametrize("twa_deg", [35, 45, 60, 90, 135])
    def test_drives_the_boat_forward(self, twa_deg: float) -> None:
        """Correctly trimmed, the sail pushes the boat forwards."""
        sail = make_sail()
        psi = beat(twa_deg)
        best = max(sail.compute(make_state(psi=psi), tree(math.radians(t), psi))[0] for t in range(5, 90, 5))
        assert best > 0.0

    def test_heel_force_opposes_the_wind_side(self) -> None:
        """Wind from one side pushes the boat toward the other."""
        sail = make_sail()
        psi = beat(45.0)
        assert sail.compute(make_state(psi=psi), tree(math.radians(20.0), psi))[1] > 0.0
        mirrored = math.radians(WIND_TO_DEG) + math.pi - math.radians(45.0)
        assert sail.compute(make_state(psi=mirrored), tree(math.radians(-20.0), mirrored))[1] < 0.0

    def test_force_scales_with_air_density(self) -> None:
        """Force is proportional to air_density, the same key the other sail models read."""
        psi = beat(45.0)
        args = (make_state(psi=psi), tree(math.radians(20.0), psi))
        light = make_sail(air_density=1.0).compute(*args)
        heavy = make_sail(air_density=2.0).compute(*args)
        assert heavy[0] == pytest.approx(2.0 * light[0])

    def test_taller_rig_pays_less_induced_drag(self) -> None:
        """Induced drag goes as area / (pi * heff^2), so height buys efficiency."""
        psi = beat(40.0)
        args = (make_state(psi=psi), tree(math.radians(18.0), psi))
        short = make_sail(heff=1.5).compute(*args)[0]
        tall = make_sail(heff=3.5).compute(*args)[0]
        assert tall > short

    def test_luffed_sail_makes_no_lift(self) -> None:
        """A boom eased past the apparent wind angle produces no lift."""
        sail = make_sail()
        psi = beat(30.0)
        sail.compute(make_state(psi=psi), tree(math.radians(80.0), psi))
        assert sail.last_flat == 0.0
        assert sail.last_cl == 0.0


class TestRightingMomentDepower:
    """ORC chooses `flat` against a righting-moment limit.

    Without it the rig sails at permanent full power, which a 3-DOF hull with no
    heel degree of freedom never pays for.
    """

    ARM = 1.47  # CE above the centre of lateral resistance

    def limited(self, limit: float) -> ORCMainSail:
        """Build a sail depowered to `limit` newton-metres of heeling moment."""
        return make_sail(max_heeling_moment_nm=limit, heel_arm_m=self.ARM)

    def heeling_moment(self, sail: ORCMainSail, twa_deg: float = 35.0, trim_deg: float = 20.0) -> float:
        """Return the heeling moment the rig actually generates, in newton-metres."""
        psi = beat(twa_deg)
        return abs(sail.compute(make_state(psi=psi), tree(math.radians(trim_deg), psi))[1]) * self.ARM

    def test_inactive_by_default(self) -> None:
        """A sail with no righting moment configured is untouched."""
        assert make_sail().max_heeling_moment is None
        assert make_sail().flat_for_righting_moment(0.8, 0.6, 1.3, 0.03, 100.0) == 0.8

    @pytest.mark.parametrize("limit", [40.0, 25.0, 15.0, 8.0])
    def test_caps_the_heeling_moment(self, limit: float) -> None:
        """Whatever the trim asks for, the rig stays inside the limit."""
        assert self.heeling_moment(self.limited(limit)) <= limit * 1.001

    def test_slack_limit_changes_nothing(self) -> None:
        """A limit above what the rig generates must not depower it."""
        free = self.heeling_moment(make_sail())
        assert self.heeling_moment(self.limited(free * 2.0)) == pytest.approx(free)

    def test_tighter_limit_costs_drive(self) -> None:
        """Depowering trades drive away; that is the point of the constraint."""
        psi = beat(35.0)
        args = (make_state(psi=psi), tree(math.radians(20.0), psi))
        drives = [self.limited(lim).compute(*args)[0] for lim in (15.0, 25.0, 40.0)]
        assert drives == sorted(drives)

    def test_flat_never_exceeds_the_trim_value(self) -> None:
        """The constraint only ever removes power, never adds it."""
        psi = beat(35.0)
        args = (make_state(psi=psi), tree(math.radians(20.0), psi))
        loose = make_sail()
        loose.compute(*args)
        tight = self.limited(15.0)
        tight.compute(*args)
        assert 0.0 <= tight.last_flat <= loose.last_flat

    def test_impossible_limit_fully_depowers(self) -> None:
        """If even a luffing rig cannot satisfy the limit, flat goes to zero."""
        sail = self.limited(1e-6)
        psi = beat(35.0)
        sail.compute(make_state(psi=psi), tree(math.radians(20.0), psi))
        assert sail.last_flat == 0.0

    def test_downwind_constraint_can_be_infeasible(self) -> None:
        """Deep downwind, heel is drag-dominated and easing cannot fix it.

        Past 90 degrees cos(beta) is negative, so more lift *reduces* heel, and
        cd0*sin(beta) sets a floor that trim cannot touch. ORC reefs for this.
        The requirement here is that the solve returns the least-heeling trim
        available rather than easing further and making things worse.
        """
        free = self.heeling_moment(make_sail(), twa_deg=150.0, trim_deg=80.0)
        capped = self.heeling_moment(self.limited(10.0), twa_deg=150.0, trim_deg=80.0)
        assert capped <= free + 1e-9

    @pytest.mark.parametrize("twa", [100.0, 120.0, 150.0, 170.0])
    def test_downwind_never_eases_into_more_heel(self, twa: float) -> None:
        """Across the whole downwind range the constraint never increases heel."""
        free = self.heeling_moment(make_sail(), twa_deg=twa, trim_deg=70.0)
        capped = self.heeling_moment(self.limited(5.0), twa_deg=twa, trim_deg=70.0)
        assert capped <= free + 1e-9

    def test_half_the_arm_allows_twice_the_force(self) -> None:
        """The limit is a moment, so it scales with the arm."""
        long_arm = make_sail(max_heeling_moment_nm=25.0, heel_arm_m=self.ARM)
        short_arm = make_sail(max_heeling_moment_nm=25.0, heel_arm_m=self.ARM / 2.0)
        psi = beat(35.0)
        args = (make_state(psi=psi), tree(math.radians(20.0), psi))
        assert abs(short_arm.compute(*args)[1]) > abs(long_arm.compute(*args)[1])

    @pytest.mark.parametrize(("limit", "arm"), [(25.0, None), (None, 1.47)])
    def test_both_keys_required(self, limit: float | None, arm: float | None) -> None:
        """Half a constraint is a configuration error, not a silent no-op."""
        params: dict[str, float] = {"area": AREA, "heff": HEFF}
        if limit is not None:
            params["max_heeling_moment_nm"] = limit
        if arm is not None:
            params["heel_arm_m"] = arm
        with pytest.raises(ValueError, match="together"):
            ORCMainSail(params)


class TestRigSelection:
    """`orc_main` is a mainsail and `orc_w_jib` is main plus jib; neither guesses from the config."""

    def test_mainsail_model_strikes_the_jib(self) -> None:
        """`orc_main` on a sloop config is that boat with the jib lowered.

        The mainsail keeps its own area; the jib's share leaves the rig, so the
        reference area the coefficients are normalised by shrinks with it.
        Handing the mainsail the combined area instead would sail a rig the
        boat does not have.
        """
        struck = make_sail(jib_area=JIB)
        assert struck.main_area == pytest.approx(AREA - JIB)
        assert struck.area == pytest.approx(AREA - JIB)
        assert make_sloop().main_area == pytest.approx(struck.main_area)

    def test_striking_the_jib_is_the_only_difference(self) -> None:
        """Same config, same mainsail: only the jib and its area are gone."""
        struck, whole = make_sail(jib_area=JIB), make_sloop()
        assert struck.heff == whole.heff
        assert [t for t, _ in struck.sails] == [MAIN_TABLE]
        assert [t for t, _ in whole.sails] == [MAIN_TABLE, JIB_TABLE]
        # Main-alone is exactly the mainsail table; the sloop is a blend.
        assert struck.envelope(27.0)[0] == pytest.approx(float(np.interp(27.0, MAIN_AWA_DEG, MAIN_CL)))

    def test_no_jib_area_is_a_plain_mainsail(self) -> None:
        """With nothing to strike, the configured area is the mainsail's."""
        assert make_sail().area == pytest.approx(AREA)
        assert make_sail().main_area == pytest.approx(AREA)

    def test_striking_the_whole_rig_leaves_no_sail(self) -> None:
        """A jib that is the entire rig cannot be struck and leave a mainsail."""
        with pytest.raises(ValueError, match="no mainsail"):
            make_sail(jib_area=AREA)

    def test_less_drive_under_main_alone(self) -> None:
        """Dropping the jib drops sail area, so the boat is slower where the jib drew."""
        psi = beat(45.0)
        args = (make_state(psi=psi), tree(math.radians(20.0), psi))
        assert make_sail(jib_area=JIB).compute(*args)[0] < make_sloop().compute(*args)[0]

    def test_sloop_model_requires_a_jib(self) -> None:
        """The sloop model without a jib_area is a config error pointing at `orc_main`."""
        with pytest.raises(ValueError, match="jib_area"):
            ORCWithJibSail({"area": AREA, "heff": HEFF})

    def test_sloop_is_a_sail(self) -> None:
        """Everything but the rig table is shared, so the sloop is substitutable for the main."""
        assert isinstance(make_sloop(), ORCMainSail)


class TestSloop:
    """ORC's collective rig: main and jib tables blended by area share."""

    def test_jib_table_matches_the_published_values(self) -> None:
        """A jib-only rig reproduces Table 5.4 at its own nodes."""
        sail = make_sloop(jib_area=AREA)
        for awa, cl, cd in zip(JIB_AWA_DEG, JIB_CL, JIB_CD0, strict=True):
            got = sail.envelope(float(awa))
            assert got[0] == pytest.approx(float(cl))
            assert got[1] == pytest.approx(float(cd))
            assert got[2] == pytest.approx(KPJ)

    def test_mainsail_model_is_the_mainsail_table(self) -> None:
        """The single-sail model is exactly Table 5.1."""
        sail = make_sail()
        for awa in (0.0, 12.0, 28.0, 60.0, 120.0, 180.0):
            cl, cd, kpp = sail.envelope(awa)
            assert cl == pytest.approx(float(np.interp(awa, MAIN_AWA_DEG, MAIN_CL)))
            assert cd == pytest.approx(float(np.interp(awa, MAIN_AWA_DEG, MAIN_CD0)))
            assert kpp == pytest.approx(KPM)

    def test_blend_is_weighted_by_area_share(self) -> None:
        """Eqs. 5.35 and 5.36: CL and CD0 are the area-weighted means."""
        sail = make_sloop()
        main = make_sail()
        jib = make_sloop(jib_area=AREA)
        wj = JIB / AREA
        for awa in (10.0, 27.0, 45.0, 90.0, 150.0):
            expected_cl = (1 - wj) * main.envelope(awa)[0] + wj * jib.envelope(awa)[0]
            expected_cd = (1 - wj) * main.envelope(awa)[1] + wj * jib.envelope(awa)[1]
            assert sail.envelope(awa)[0] == pytest.approx(expected_cl)
            assert sail.envelope(awa)[1] == pytest.approx(expected_cd)

    def test_kpp_is_weighted_by_lift_squared(self) -> None:
        """Eq. 5.41: the sail carrying the lift sets the viscous drag slope."""
        sail = make_sloop()
        wj = JIB / AREA
        cl_m = float(np.interp(27.0, MAIN_AWA_DEG, MAIN_CL))
        cl_j = float(np.interp(27.0, JIB_AWA_DEG, JIB_CL))
        expected = (KPM * (1 - wj) * cl_m**2 + KPJ * wj * cl_j**2) / ((1 - wj) * cl_m**2 + wj * cl_j**2)
        assert sail.envelope(27.0)[2] == pytest.approx(expected)
        assert KPM < sail.envelope(27.0)[2] < KPJ

    def test_kpp_is_finite_head_to_wind(self) -> None:
        """Both tables give CL = 0 at 0 degrees; the division must not blow up."""
        kpp = make_sloop().envelope(0.0)[2]
        assert np.isfinite(kpp)
        assert KPM <= kpp <= KPJ

    def test_jib_adds_lift_upwind_and_removes_it_downwind(self) -> None:
        """The stated reason for the blend: a headsail fills early and dies late."""
        sloop, main = make_sloop(), make_sail()
        assert sloop.envelope(20.0)[0] > main.envelope(20.0)[0]
        assert sloop.envelope(27.0)[0] > main.envelope(27.0)[0]
        assert sloop.envelope(150.0)[0] < main.envelope(150.0)[0]

    def test_more_drive_close_hauled(self) -> None:
        """At the same trim on a beat the sloop out-drives the main-only rig."""
        psi = beat(35.0)
        args = (make_state(psi=psi), tree(math.radians(15.0), psi))
        assert make_sloop().compute(*args)[0] > make_sail().compute(*args)[0]

    @pytest.mark.parametrize("bad", [2.5, 0.0, -0.1])
    def test_jib_area_must_fit_inside_the_rig(self, bad: float) -> None:
        """A jib bigger than the rig, zero, or negative, is a config error."""
        with pytest.raises(ValueError, match="jib_area"):
            make_sloop(jib_area=bad)

    @pytest.mark.parametrize("bad", [2.5, 0.0, -0.1])
    def test_the_mainsail_model_validates_it_too(self, bad: float) -> None:
        """Now that `orc_main` reads jib_area, it has to check it on the same terms."""
        with pytest.raises(ValueError, match="jib_area"):
            make_sail(jib_area=bad)

    def test_whole_rig_as_jib_is_allowed(self) -> None:
        """jib_area == area is the degenerate but valid jib-only rig."""
        assert make_sloop(jib_area=AREA).main_area == 0.0


class TestEffectiveHeight:
    """ORC 5.4.3: effective rig height varies with apparent wind angle."""

    def test_constant_by_default(self) -> None:
        """Without heff_model the configured height is used at every angle."""
        sail = make_sail()
        for awa in (0.0, 20.0, 45.0, 80.0, 150.0):
            assert sail.effective_height(awa) == pytest.approx(HEFF)

    @pytest.mark.parametrize(("model", "peak"), [("orc-2022", 1.22), ("orc-2023", 1.4513)])
    def test_orc_curve_matches_the_published_anchors(self, model: str, peak: float) -> None:
        """Each edition's text pins kheff at 1.0 (0 deg), its peak (20 deg) and 0.80 (80 on)."""
        sail = make_sail(heff_model=model)
        assert sail.effective_height(0.0) == pytest.approx(HEFF * 1.0)
        assert sail.effective_height(20.0) == pytest.approx(HEFF * peak)
        assert sail.effective_height(80.0) == pytest.approx(HEFF * 0.80)
        assert sail.effective_height(150.0) == pytest.approx(HEFF * 0.80)

    @pytest.mark.parametrize(("model", "table"), [("orc-2022", KHEFF_2022), ("orc-2023", KHEFF_2023)])
    def test_orc_curve_matches_the_table_at_its_nodes(self, model: str, table: np.ndarray) -> None:
        """Interpolation reproduces the traced figure exactly."""
        sail = make_sail(heff_model=model)
        for awa, k in zip(KHEFF_AWA_DEG, table, strict=True):
            assert sail.effective_height(float(awa)) == pytest.approx(HEFF * float(k))

    def test_editions_differ_only_close_hauled(self) -> None:
        """2023 raised the peak; the reaching side of the curve was left alone."""
        old, new = make_sail(heff_model="orc-2022"), make_sail(heff_model="orc-2023")
        assert new.effective_height(20.0) > old.effective_height(20.0)
        for awa in (50.0, 60.0, 70.0, 80.0):
            assert new.effective_height(awa) == pytest.approx(old.effective_height(awa), rel=0.01)

    def test_taller_on_a_beat_than_on_a_reach(self) -> None:
        """The whole point: jib-hull sealing makes the rig act taller close-hauled."""
        sail = make_sail(heff_model="orc-2023")
        assert sail.effective_height(20.0) > sail.effective_height(0.0)
        assert sail.effective_height(20.0) > sail.effective_height(60.0) > sail.effective_height(90.0)

    def test_symmetric_in_wind_side(self) -> None:
        """Height depends on the magnitude of the apparent wind angle."""
        sail = make_sail(heff_model="orc-2023")
        assert sail.effective_height(-25.0) == sail.effective_height(25.0)

    def test_span_correction_scales_the_height(self) -> None:
        """eff_span_corr multiplies the height whatever the model."""
        assert make_sail(eff_span_corr=1.057).effective_height(45.0) == pytest.approx(HEFF * 1.057)
        orc = make_sail(heff_model="orc-2023", eff_span_corr=1.057)
        assert orc.effective_height(20.0) == pytest.approx(HEFF * 1.057 * 1.4513)

    def test_less_induced_drag_on_a_beat(self) -> None:
        """A taller effective rig sheds less tip vortex for the same lift."""
        psi = beat(35.0)
        args = (make_state(psi=psi), tree(math.radians(15.0), psi))
        const, orc = make_sail(), make_sail(heff_model="orc-2023")
        const.compute(*args)
        orc.compute(*args)
        assert orc.last_cl == pytest.approx(const.last_cl)
        assert orc.last_cd < const.last_cd
        assert orc.last_heff > const.last_heff

    @pytest.mark.parametrize("bad", ["fancy", "orc"])
    def test_unknown_model_raises(self, bad: str) -> None:
        """A typo, or the unversioned name, must not silently fall back."""
        with pytest.raises(ValueError, match="heff_model"):
            make_sail(heff_model=bad)


class TestDownwindTrimAnswers:
    """Running, the sheet has to do something: form drag follows projected area.

    ORC's CD0 is the drag of a correctly trimmed sail. Charged in full whatever
    the boom does, moving the sheet across its entire range at TWA 160-180 would
    change the drive force by 0.00 N, leaving a policy no gradient on sail trim
    there.
    """

    def test_beam_and_upwind_are_untouched(self) -> None:
        """The factor is 1 at or inside a beam reach, so no upwind polar moves."""
        sail = make_sloop()
        for beta_deg in (0.0, 30.0, 60.0, 89.0, 90.0):
            beta = math.radians(beta_deg)
            for boom_deg in (0.0, 20.0, 45.0, 90.0):
                alpha = beta - math.radians(boom_deg)
                assert sail.form_drag_trim_factor(beta, alpha) == pytest.approx(1.0)

    def test_continuous_across_the_beam(self) -> None:
        """No step at 90 degrees: the penalty ramps in from nothing at the beam."""
        sail = make_sloop()
        at_beam = sail.form_drag_trim_factor(math.radians(90.0), math.radians(45.0))
        assert at_beam == pytest.approx(1.0)
        # Approaching the beam from outside, the factor returns to the inside branch.
        for past in (10.0, 1.0, 0.1, 0.01):
            beta = math.radians(90.0 + past)
            assert sail.form_drag_trim_factor(beta, math.radians(45.0)) == pytest.approx(1.0, abs=past / 90.0)

    def test_square_to_the_wind_gets_the_full_table(self) -> None:
        """The table's drag belongs to the sail square to the flow, not to any one boom.

        Square to the apparent wind is `alpha = 90 deg`, which is a boom at
        `beta - 90`. A boom fully eased to 90 degrees is only that trim on a dead
        run; at 120 degrees apparent it is 30 degrees past square and by the lee,
        so it is charged less, not the full table.
        """
        sail = make_sloop()
        for beta_deg in (120.0, 150.0, 180.0):
            beta = math.radians(beta_deg)
            assert sail.form_drag_trim_factor(beta, 0.5 * math.pi) == pytest.approx(1.0)
        over_eased = sail.form_drag_trim_factor(math.radians(120.0), math.radians(120.0 - 90.0))
        assert over_eased < 0.9

    def test_intermediate_sheets_are_not_all_identical(self) -> None:
        """The regression that hid the bug for a whole training run.

        `sin^2(alpha)` peaks at 90 degrees, so normalising against the fully-eased
        boom made every trim between hard in and fully out compute above 1 and
        clip back to it. At 139.5 degrees apparent, sheet limits of 10 through 80
        degrees returned byte-identical drive.
        """
        sail = make_sloop()
        beta = math.radians(139.5)
        factors = [sail.form_drag_trim_factor(beta, beta - math.radians(b)) for b in (10.0, 20.0, 40.0, 60.0, 80.0)]
        assert len({round(f, 6) for f in factors}) == len(factors)
        assert max(factors) - min(factors) > 0.1

    def test_centreline_sail_presents_nothing_dead_downwind(self) -> None:
        """Sheeted flat amidships on a run, the sail is edge-on and makes no form drag."""
        sail = make_sloop()
        assert sail.form_drag_trim_factor(math.pi, math.pi) == pytest.approx(0.0, abs=1e-9)

    def test_never_exceeds_the_optimum(self) -> None:
        """Trim can only be worse than the table, never better -- as `flat` cannot exceed 1."""
        sail = make_sloop()
        for beta_deg in np.arange(91.0, 180.1, 1.0):
            for boom_deg in np.arange(0.0, 90.1, 5.0):
                beta = math.radians(float(beta_deg))
                f = sail.form_drag_trim_factor(beta, beta - math.radians(float(boom_deg)))
                assert 0.0 <= f <= 1.0

    def test_drive_rises_monotonically_with_sheet_on_a_run(self) -> None:
        """The behaviour this factor exists for: easing the sheet downwind makes drive."""
        drives = [_drive_at(twa_deg=180.0, boom_deg=b) for b in (0.0, 30.0, 60.0, 90.0)]
        assert all(b > a for a, b in itertools.pairwise(drives))
        assert drives[0] < 0.1 * drives[-1]  # centreline makes almost nothing

    @pytest.mark.parametrize("twa", [160.0, 170.0, 180.0])
    def test_sheet_range_changes_drive_downwind(self, twa: float) -> None:
        """Regression guard: without the factor the spread here is 0.00 N."""
        drives = [_drive_at(twa_deg=twa, boom_deg=b) for b in np.arange(0.0, 90.1, 15.0)]
        assert max(drives) - min(drives) > 0.25 * max(drives)


def _drive_at(twa_deg: float, boom_deg: float) -> float:
    """Return the boat-frame drive force from the Flingo rig at a wind and boom angle."""
    sail = make_sloop()
    psi = math.radians(270.0 - twa_deg)
    return float(sail.compute(make_state(u=1.5, psi=psi), tree(math.radians(boom_deg), psi))[0])
