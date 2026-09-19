# Pulling Flingo Floty's physics data out of SolidWorks

What `configs/flingo_floty.yaml` needs, where each number actually comes from,
and what the CAD can never tell us.

## Frame convention the sim uses

Read off `sailbench/sim/sailboat_hub.py` and the configs:

- Origin = the boat's **centre of gravity**. The rigid-body equations
  (`du = fx/m + r*v`, `dv = fy/m - r*u`, `dr = mz/Iz`) carry no `m*x_g` coupling
  terms, so every `x_pos` / `y_pos` must be measured **from the CG**, not from
  the SolidWorks assembly origin.
- `+x` = forward (bow). The rudder sits at `x_pos: -0.46`, i.e. aft.
- `+y` = port. `dc = -r*s`, `ds = r*c` makes heading increase counter-clockwise.
- Yaw axis = vertical, so `inertia_z` is the **Lzz about the vertical axis
  through the CG**, whichever of SolidWorks' X/Y/Z that happens to be in this
  assembly.
- Units are SI throughout: kg, m, kg·m², m².

## Straight from the macro

Run `ExportBoatParams.bas` (instructions in its header) against the open
assembly. It writes `boat_params.yaml` beside the assembly file.

| Config key | Macro output |
|---|---|
| `boat.mass` | `total.mass_kg` — add battery/electronics mass if they aren't modelled |
| `boat.inertia_z` | `total.inertia_at_cg_kg_m2.Lzz` (or Lxx/Lyy — whichever is the vertical axis) |
| `rudder.x_pos`, `rudder.y_pos` | rudder component `cg_m`, minus `total.cg_m` |
| `keel.x_pos`, `keel.y_pos` | keel component `cg_m`, minus `total.cg_m` |
| `sail.x_pos`, `sail.y_pos` | mast/sail centre of effort, minus `total.cg_m` |

Tell me which SolidWorks axis is forward and which is up and I'll do the
subtraction and the axis mapping into the YAML.

## Needs the Measure / Section Properties tools

- `hull.L` — waterline length, not overall hull length.
- `hull.B` — max beam **at the waterline**.
- `hull.T` — draft at the floating waterline. Find it by cutting the hull at
  candidate heights until the submerged volume equals `mass / 1000` m³
  (27 kg → 0.027 m³). That check is also the honest test of whether the CAD
  mass and the config mass agree.
- `keel.area`, `rudder.area` — planform area (one side), from the flat pattern
  or a projected view, **not** the wetted surface area the macro prints.
- `sail.area` — from the sail plan, not the boat assembly.
- Keel and rudder **chord length**, which sets `res` (Reynolds):
  `Re = V * chord / 1e-6` in water, `Re = V * chord / 1.5e-5` in air.
  The current `5e5` implies a ~0.25 m chord at 2 m/s — worth confirming.

## The CAD cannot give these

Don't pretend otherwise; they come from tow tests, CFD, or tuning against
recorded sailing logs:

- `hull.xu1 / yv1 / nr1` and `xu2 / yv2 / nr2` — hydrodynamic damping.
- `sail.CL_max / CD0 / CD1` — sail aerodynamics.
- `rudder.effectiveness`, `moment_coeff`, `cl_max`, `cd_max` — fudge factors.
- `rudder.max_rate_deg_s`, `deadband_deg`, `center_tau_s` — from the servo
  datasheet (a 0.12 s/60° servo is 500 °/s, not the 120 °/s in the config).
- Added mass. The sim uses one scalar `mass` for surge and sway, but a hull's
  sway added mass is on the order of its own displacement. Until that's
  modelled, `yv1`/`yv2` are absorbing it.

## Known issues found while mapping this

- `hull.rho_water` in the configs is dead: `BasicHullModel` reads `self.p["rho"]`
  and `BasicKeel` reads `water_density`. Both silently default to 1000, so the
  numbers are right by luck.
- `boat.inertia_z: 1.0` is too small for a 27 kg boat — a rough `m*(0.25L)²`
  gives ~4 kg·m² at L = 1.5 m. Expect the CAD Lzz to be several times the
  current value, and expect yaw response to slow down noticeably once it's in.
- `linear_hydro.py` comments "positive r = starboard turn"; the kinematics in
  `sailboat_hub.py` make positive r a **port** turn.
