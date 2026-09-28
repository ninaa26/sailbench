# sailbench
Sailbench is an end-to-end sailing physics simulator made by [Cornell Autonomous Sailboat Team](https://cusail.com/). It is to be used for testing and RL model training for autonomous sailboats.

<img width="755" height="454" alt="image" src="https://github.com/user-attachments/assets/8149ee33-4ae5-45b6-ae8e-e1e794e2095b" />

## First-Time Setup

`sailbench` uses the [uv project manager](https://docs.astral.sh/uv/) for dependency and environment management. Follow the steps below to set up a local development environment.

### Prerequisites
- Python 3.12+
- [UV project manager](https://docs.astral.sh/uv/)

### Setup Instructions

1. **Install Python**  
   If you don’t already have Python installed, download it from `https://www.python.org/downloads/`.

2. **Install `uv`**  
   Follow the installation instructions here: `https://docs.astral.sh/uv/getting-started/installation/`.

3. **Install project dependencies**

   In the terminal, navigate to ```sailbench/```

   For users who just want to run the simulation, run:
   ```bash
   uv sync
   ```

   For developers (linters/type-checking), install the `dev` dependency group:

   ```bash
   uv sync --group dev
   ```

   If you’re working on RL training/evaluation, also install the `rl` group:

   ```bash
   uv sync --group dev --group rl
   ```

5. **[Optional but recommended] Setup VSCode Extensions**   
    I would recommend utilizing VSCode for developing in this project. The two extensions to install are [Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) (Python formatter and linter) as well as [MyPy](https://marketplace.visualstudio.com/items?itemName=ms-python.mypy-type-checker) (Python type checker). This will help keep consistent code quality and style across sailbench.

## Configuring a Boat

A boat is one YAML file in `configs/`, one section per part. `simulation` and `boat` (mass and yaw inertia) are read by the hub itself; every other section builds one part, and picks its model with `model_type`.

Every boat has these four:

| Section  | `model_type` choices (default first)                         |
|----------|--------------------------------------------------------------|
| `hull`   | `basic`, `linear`, `quadratic`                                |
| `keel`   | `basic` (alias `keel`)                                        |
| `sail`   | `basic` (alias `sail`), `hybrid`, `orc_main`, `orc_w_jib`     |
| `rudder` | `basic` (alias `keel`)                                        |

Optional parts are added by adding their section, and left off by leaving it out or by setting `enabled: false`, which keeps the numbers in the file without putting the part on the boat:

```yaml
windage:          # above-water drag on mast, rigging and topsides
  frontal_area_m2: 0.18
  lateral_area_m2: 0.37
  drag_coefficient: 0.8

ballast:          # dead weight: adds mass and yaw inertia, makes no force
  mass: 5.0       # [kg] on top of boat.mass, so leave it out of that
  x_pos: -0.1     # [m] off-centre ballast adds mass * r^2 to the inertia

jib:              # a second sail on the main's sheet, forward of the mast
  enabled: false
  model_type: hybrid   # basic or hybrid
  area: 0.4
  x_pos: 0.35
  CL_max: 1.0
  CD0: 0.1
  CD1: 0.8
```

Wind is set once, in the `sail` section; windage and the jib are given the same wind. The ORC sails model their own jib, so with `orc_main` or `orc_w_jib` put the jib in the `sail` section (`model_type: orc_w_jib` plus `jib_area`) instead of a `jib` section.

A misspelled model name is an error, and a section nothing reads is a warning. To add a model or a new optional part, add its class to the tables in `sailbench/dynamics/component_factory.py`; the hub needs no changes.

## Running the Web Simulation

Once you've installed the packages, you can play sailbench with manual control with the following instructions.

1. **Start the sim backend** (from ```sailbench/```):
   ```bash
   uv run python -m sailbench.sim.web_runner --config basic_sailbot.yaml --fps 60
   ```

2. **Start the frontend** (in a separate terminal):
   ```bash
   cd web
   python -m http.server 8000
   ```

3. Open http://localhost:8000 in your browser.

### Watch a trained RL policy in the web simulation

To run a trained RL model in sailbench, perform the following.

1. Start the backend with an RL checkpoint:
   ```bash
   uv run python -m sailbench.sim.web_runner \
     --config basic_sailbot.yaml \
     --policy-model runs/<run_name>/best_model/best_model.zip \
     --policy-config configs/rl_waypoint_sb3.yaml
   ```
2. In a second terminal:
   ```bash
   cd web
   python -m http.server 8000
   ```
3. Open http://localhost:8000. The red/yellow marker shows the current waypoint.


## RL Training (Gymnasium + SB3)

SailBench includes a Gymnasium continuous-control task (`sailbench.rl.envs.WaypointEnv`) and Stable-Baselines3 scripts for training/evaluating PPO on waypoint navigation.

Install the optional RL dependencies:

```bash
uv sync --group rl
```

### Train (PPO)

Use the default waypoint RL config (`configs/rl_waypoint_sb3.yaml`):

```bash
uv run python scripts/train_waypoint_sb3.py --config configs/rl_waypoint_sb3.yaml
```

To watch training live in the web visualizer, enable the training websocket stream:

```bash
uv run python scripts/train_waypoint_sb3.py \
  --config configs/rl_waypoint_sb3.yaml \
  --watch-web
```

Then run the frontend in another terminal and open `http://localhost:8000`:

```bash
cd web
python -m http.server 8000
```

Notes:

- The live stream is served on `ws://127.0.0.1:8765/sim` by default (same frontend URL as the normal backend).
- Training visualization publishes from env-0 only, so it works with vectorized training (`train.n_envs > 1`).
- You can reduce browser update load with `--watch-stride <N>` (or `train.watch_stride` in YAML).

To continue a stopped run from a checkpoint:

```bash
uv run python scripts/train_waypoint_sb3.py \
  --config configs/rl_waypoint_sb3.yaml \
  --resume-from runs/<run_name>/checkpoints/ppo_waypoint_<steps>_steps.zip
```

When `--resume-from` is provided, training continues from that model state and keeps timestep counting continuous.

Training outputs are written under `runs/` by default:

- `runs/waypoint_ppo_<timestamp>/best_model/best_model.zip`: best checkpoint per eval callback
- `runs/waypoint_ppo_<timestamp>/checkpoints/`: periodic checkpoints
- `runs/waypoint_ppo_<timestamp>/final_model.zip`: final model after training
- `runs/waypoint_ppo_<timestamp>/vecnormalize.pkl`: VecNormalize stats (only if enabled via config)
- `runs/waypoint_ppo_<timestamp>/config_used.yaml`: the exact config used for the run

Notes for resume:

- Keep `--config` consistent with the original training setup (especially env settings and `train.n_envs`).
- If normalization is enabled, the trainer automatically attempts to load checkpoint stats from the matching file `.../<checkpoint_stem>_vecnormalize.pkl`.

### TensorBoard

If you keep `train.tensorboard_log: runs/tensorboard` (the default), you can launch TensorBoard with:

```bash
uv run tensorboard --logdir runs/tensorboard
```

When training, logs will be written into subdirectories under `runs/tensorboard/`, one per run. You can view your training progress, hyperparameters, and evaluation metrics in TensorBoard at [http://localhost:6006](http://localhost:6006) after launching the command above.


### Evaluate a trained checkpoint

```bash
uv run python scripts/eval_waypoint_sb3.py \
  --config configs/rl_waypoint_sb3.yaml \
  --model runs/<run_name>/best_model/best_model.zip \
  --vecnormalize runs/<run_name>/vecnormalize.pkl
```

Notes:

- Pass `--vecnormalize` only if your training run produced `vecnormalize.pkl` (i.e., you enabled observation/reward normalization in the training config).
- `eval_waypoint_sb3.py` prints a JSON blob of summary metrics to stdout; use `--output-json <path>` to save them.
