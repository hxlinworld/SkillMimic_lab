# SkillMimic on Isaac Lab

This document is the Isaac Lab counterpart of the commands in
[`README.md`](README.md). It keeps the same two user-facing workflows:

1. train or run the demonstration-conditioned basketball skill policy;
2. train or run one of the four high-level controllers that reuse the skill
   policy: Circling, Heading, Throwing, and Scoring.

The original Isaac Gym implementation remains in `skillmimic/`. The migrated
implementation is in `skillmimic_lab/` and does not import `isaacgym`.

## 1. Supported environment

The two implementations require different environments. Do not install the
Isaac Gym `requirements.txt` into the Isaac Lab environment.

| Component | Original README | Isaac Lab migration |
| --- | --- | --- |
| Python | 3.8 | 3.10 |
| Simulator | Isaac Gym Preview 4 | Isaac Sim 4.1.0 |
| Framework | Isaac Gym | Isaac Lab v1.1.0 |
| PyTorch | legacy project environment | 2.2.2 + CUDA 11.8 |
| RL library | legacy RL-Games | RL-Games 1.6.1 |

The migration expects:

- Linux with a supported NVIDIA GPU and driver;
- the Isaac Sim 4.1 standalone package;
- an Isaac Lab checkout at tag `v1.1.0`;
- the BallPlay-M subset and released checkpoints already present under
  `skillmimic/data/`.

The default repository-local paths are:

```text
.external/isaac-sim-4.1.0/
.external/IsaacLab/
```

Set `ISAAC_SIM_ROOT` if Isaac Sim is installed elsewhere. Set `PYTHON_BIN` if
the desired Python executable is not the `python` currently on `PATH`.

### Install Python dependencies

Create and activate a Python 3.10 environment, then install the migration
requirements from the project root:

```bash
conda create -n skillmimic_lab python=3.10
conda activate skillmimic_lab
pip install -r requirements_isaaclab.txt
```

`requirements_isaaclab.txt` installs the repository-local Isaac Lab extension
in editable mode. SkillMimic uses its own task registration and RL-Games
adapter, so the optional `omni.isaac.lab_tasks` package is not required. The
requirements file does not install Isaac Sim itself.

The launcher never installs, upgrades, or removes packages. It only sources
Isaac Sim's `setup_conda_env.sh` when that file exists and then uses the active
Python environment.

### First validation

Run the headless, four-environment zero-action smoke test before inference or
training:

```bash
bash scripts/run_isaaclab.sh smoke
```

A successful run ends with a line beginning with:

```text
[SkillMimic Lab] PASS mode=smoke task=ballplay
```

Runtime output is written to `logs/isaaclab/latest.log`. The file is replaced
on each launch.

## 2. Source layout

```text
skillmimic_lab/
├── env/tasks/
│   ├── skillmimic.py            # SkillMimicBallPlay DirectRLEnv
│   ├── hrl_base.py               # shared HLC environment
│   ├── hrl_circling.py
│   ├── hrl_heading_easy.py
│   ├── hrl_throwing.py
│   └── hrl_scoring_layup.py
├── learning/
│   ├── policy.py                 # released-checkpoint compatibility loaders
│   └── train.py                  # RL-Games training entry point
├── utils/                        # motion and quaternion/tensor utilities
├── agents/                       # migrated RL-Games configurations
└── run.py                        # smoke tests and policy inference
```

The shell entry point is `scripts/run_isaaclab.sh`. Its modes are:

```text
smoke
play
circling
heading
throwing
scoring
train
train-circling
train-heading
train-throwing
train-scoring
```

Arguments following the mode are forwarded to the corresponding Python entry
point. Common shell settings are:

| Setting | Default | Meaning |
| --- | ---: | --- |
| `NUM_ENVS` | 4 for inference, 2048 for training | parallel environments per GPU |
| `NUM_GPUS` | 1 | training processes/GPUs; values above 1 enable torchrun |
| `STEPS` | 600 | inference loop steps |
| `HEADLESS` | 1 | set to 0 to create a viewer |
| `DEVICE_ID` | 0 | CUDA device passed to Isaac Sim |
| `PYTHON_BIN` | `python` | Python executable |
| `ISAAC_SIM_ROOT` | `.external/isaac-sim-4.1.0` | Isaac Sim standalone root |
| `SKILLMIMIC_KIT_RUNTIME_ROOT` | node-local temporary directory | writable Kit data/cache root; each distributed rank gets its own subdirectory |
| `TENSORBOARD_LOG_DIR` | `logs/rl_games` | directory served by TensorBoard |
| `TENSORBOARD_PORT` | 6006 | TensorBoard server port |

### Running without a graphical interface

The launcher is headless by default (`HEADLESS=1`), so no command change is
required on a server without a desktop, X server, or physical display. Do not
use `HEADLESS=0` on such a machine.

Examples:

```bash
# Smoke test.
bash scripts/run_isaaclab.sh smoke

# Skill-policy inference.
NUM_ENVS=16 bash scripts/run_isaaclab.sh play

# High-level controller inference.
NUM_ENVS=16 bash scripts/run_isaaclab.sh scoring

# Skill-policy training.
NUM_ENVS=2048 bash scripts/run_isaaclab.sh train
```

Headless mode can also be selected explicitly:

```bash
HEADLESS=1 NUM_ENVS=16 bash scripts/run_isaaclab.sh play
```

For a complete training command, either the default `HEADLESS=1` or the
forwarded `--headless` argument may be used:

```bash
NUM_ENVS=2048 bash scripts/run_isaaclab.sh train \
  --motion_path skillmimic/data/motions/BallPlay-M/layup \
  --max_iterations 50000 \
  --headless
```

Isaac Sim may still print `GLFW initialization failed` or `No windowing`
warnings during headless startup. These warnings alone do not mean the run
failed. The launcher shows output in the terminal and writes the same stream to
`logs/isaaclab/latest.log`, replacing that file on each launch. It also starts
TensorBoard in the background at `http://localhost:6006`. For training,
confirm that RL-Games has started and is writing below `logs/rl_games/`.

The launcher keeps writable Kit state under a unique node-local temporary
directory. This is required when the standalone Isaac Sim archive is installed
on shared storage: Kit's driver/shader cache supports only one writer per
directory and cache files are not safe to reuse across nodes with different
driver stacks. On NVIDIA cluster nodes, the launcher also selects the NVIDIA
Vulkan ICD rather than allowing Kit to enumerate unrelated Intel or Radeon
ICDs.

## 3. Pre-trained models

The migration uses the same released models listed by the README. No checkpoint
conversion step is required for inference.

```text
skillmimic/data/models/mixedskills/nn/skillmimic_llc.pth
skillmimic/data/models/hlc_circling/nn/SkillMimic.pth
skillmimic/data/models/hlc_heading/nn/SkillMimic.pth
skillmimic/data/models/hlc_throwing/nn/SkillMimic.pth
skillmimic/data/models/hlc_scoring/nn/SkillMimic.pth
```

The compatibility loaders reconstruct the deterministic actor and its running
observation statistics from the legacy RL-Games checkpoint. Legacy optimizer
state is not resumed. `--checkpoint` in a training command accepts only a
checkpoint produced by the Isaac Lab training path.

## 4. Skill Policy

The README's `SkillMimicBallPlay` task maps to the Isaac Lab task
`SkillMimic-BallPlay-Direct-v0` and the `play`/`train` launcher modes.

### Inference

The direct Isaac Lab equivalent of the README's layup example is:

```bash
NUM_ENVS=16 STEPS=140 bash scripts/run_isaaclab.sh play \
  --motion_path skillmimic/data/motions/BallPlay-M/layup \
  --checkpoint skillmimic/data/models/mixedskills/nn/skillmimic_llc.pth \
  --state_init 20 \
  --episode_length 140
```

The released mixed-skills model and the full BallPlay-M directory are the
defaults, so the shortest inference command is:

```bash
bash scripts/run_isaaclab.sh play
```

On a machine with a graphical display, use `HEADLESS=0 NUM_ENVS=1` for a local
viewer:

```bash
HEADLESS=0 NUM_ENVS=1 bash scripts/run_isaaclab.sh play
```

`--motion_path` accepts either one `.pt` motion file or a directory. Directory
loading is recursive. `--state_init -1` selects a random valid reference frame;
an integer greater than or equal to 2 selects a deterministic frame.

### Training

The README's skill-policy training command maps to:

```bash
NUM_ENVS=2048 bash scripts/run_isaaclab.sh train \
  --motion_path skillmimic/data/motions/BallPlay-M/layup \
  --max_iterations 50000 \
  --headless
```

To train on a larger skill set, point `--motion_path` at that directory:

```bash
NUM_ENVS=16384 bash scripts/run_isaaclab.sh train \
  --motion_path skillmimic/data/motions/BallPlay-M \
  --max_iterations 50000 \
  --headless
```

For distributed training on four GPUs while keeping the same total of 16384
environments, run 4096 environments in each worker:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_GPUS=4 NUM_ENVS=4096 \
  bash scripts/run_isaaclab.sh train \
  --motion_path skillmimic/data/motions/BallPlay-M \
  --max_iterations 50000 \
  --minibatch_size 4096 \
  --headless
```

Distributed training launches one Isaac Sim process per GPU and synchronizes
RL-Games gradients through NCCL. `NUM_ENVS` is per GPU, so the total number of
environments is `NUM_GPUS * NUM_ENVS`. The per-GPU minibatch is set to 4096 so
the synchronized global minibatch remains 16384, matching the single-GPU
command above.

The README also uses `skillmimic/data/motions/skillset_1` as an example for a
separately downloaded larger dataset. That directory is not part of the
released BallPlay-M subset; use it only after supplying the corresponding data.

Training output is written below:

```text
logs/rl_games/ballplay_isaaclab/<timestamp>/
```

### Resuming training

Resume from an Isaac Lab-era RL-Games checkpoint by passing it through
`--checkpoint`. For example:

```bash
RUN_NAME="2026-07-24_02-20-48"
CHECKPOINT="logs/rl_games/ballplay_isaaclab/${RUN_NAME}/nn/last_skillmimic_isaaclab_ep_3700_rew_8.512084.pth"

NUM_ENVS=2048 SKILLMIMIC_RUN_NAME="${RUN_NAME}" \
  bash scripts/run_isaaclab.sh train \
  --checkpoint "${CHECKPOINT}" \
  --max_iterations 50000
```

Replace `RUN_NAME` and `CHECKPOINT` with the run and checkpoint to resume.
Setting `SKILLMIMIC_RUN_NAME` reuses the previous output directory so the new
TensorBoard events and checkpoints stay with that run; omit it to create a new
timestamped directory. `--max_iterations` is the total target iteration count,
not the number of additional iterations. For distributed training, also pass
the same `CUDA_VISIBLE_DEVICES`, `NUM_GPUS`, `NUM_ENVS`, and task-specific
arguments used by the original run. HLC checkpoints use the corresponding
`train-circling`, `train-heading`, `train-throwing`, or `train-scoring` mode.

The default PPO minibatch is 16384, which is 8 times the default 2048
environments. If `NUM_ENVS` is changed substantially, update `minibatch_size`
in `skillmimic_lab/agents/rl_games_ppo_cfg.yaml` as recommended by the README.

## 5. High-Level Controller

Each high-level environment loads the released low-level mixed-skills policy.
The task mapping is:

| README task | Launcher mode | Isaac Lab task ID | Default motion directory | HLC actions |
| --- | --- | --- | --- | ---: |
| `HRLCircling` | `circling` | `SkillMimic-Circling-Direct-v0` | `BallPlay-M/run` | 3 |
| `HRLHeadingEasy` | `heading` | `SkillMimic-Heading-Direct-v0` | `BallPlay-M/run` | 3 |
| `HRLThrowing` | `throwing` | `SkillMimic-Throwing-Direct-v0` | `BallPlay-M/turnhook` | 3 |
| `HRLScoringLayup` | `scoring` | `SkillMimic-Scoring-Direct-v0` | `BallPlay-M/run` | 7 |

The HLC action is discrete. It is mapped to the original 64-D skill condition,
and the loaded LLC produces the 156-D humanoid action. One HLC choice is held
for three low-level control steps, matching the original task wrapper.

### Inference

Circling:

```bash
NUM_ENVS=1 bash scripts/run_isaaclab.sh circling \
  --motion_path skillmimic/data/motions/BallPlay-M/run \
  --checkpoint skillmimic/data/models/hlc_circling/nn/SkillMimic.pth \
  --llc_checkpoint skillmimic/data/models/mixedskills/nn/skillmimic_llc.pth
```

Heading:

```bash
NUM_ENVS=1 bash scripts/run_isaaclab.sh heading \
  --motion_path skillmimic/data/motions/BallPlay-M/run \
  --checkpoint skillmimic/data/models/hlc_heading/nn/SkillMimic.pth \
  --llc_checkpoint skillmimic/data/models/mixedskills/nn/skillmimic_llc.pth
```

Throwing:

```bash
NUM_ENVS=1 bash scripts/run_isaaclab.sh throwing \
  --motion_path skillmimic/data/motions/BallPlay-M/turnhook \
  --checkpoint skillmimic/data/models/hlc_throwing/nn/SkillMimic.pth \
  --llc_checkpoint skillmimic/data/models/mixedskills/nn/skillmimic_llc.pth
```

Scoring:

```bash
NUM_ENVS=1 bash scripts/run_isaaclab.sh scoring \
  --motion_path skillmimic/data/motions/BallPlay-M/run \
  --checkpoint skillmimic/data/models/hlc_scoring/nn/SkillMimic.pth \
  --llc_checkpoint skillmimic/data/models/mixedskills/nn/skillmimic_llc.pth
```

The checkpoint, LLC checkpoint, and motion paths shown above are already the
defaults for their modes. For example, this is sufficient:

```bash
NUM_ENVS=1 bash scripts/run_isaaclab.sh scoring
```

### Training

Circling:

```bash
NUM_ENVS=2048 bash scripts/run_isaaclab.sh train-circling \
  --motion_path skillmimic/data/motions/BallPlay-M/run \
  --llc_checkpoint skillmimic/data/models/mixedskills/nn/skillmimic_llc.pth \
  --max_iterations 6000 \
  --headless
```

Heading:

```bash
NUM_ENVS=2048 bash scripts/run_isaaclab.sh train-heading \
  --motion_path skillmimic/data/motions/BallPlay-M/run \
  --llc_checkpoint skillmimic/data/models/mixedskills/nn/skillmimic_llc.pth \
  --max_iterations 6000 \
  --headless
```

Throwing:

```bash
NUM_ENVS=2048 bash scripts/run_isaaclab.sh train-throwing \
  --motion_path skillmimic/data/motions/BallPlay-M/turnhook \
  --llc_checkpoint skillmimic/data/models/mixedskills/nn/skillmimic_llc.pth \
  --max_iterations 6000 \
  --headless
```

Scoring:

```bash
NUM_ENVS=2048 bash scripts/run_isaaclab.sh train-scoring \
  --motion_path skillmimic/data/motions/BallPlay-M/run \
  --llc_checkpoint skillmimic/data/models/mixedskills/nn/skillmimic_llc.pth \
  --max_iterations 6000 \
  --headless
```

HLC training output is written below
`logs/rl_games/<task>_isaaclab/<timestamp>/`.

## 6. README option mapping

The Isaac Gym YAML files remain available for the original implementation, but
the Isaac Lab path uses Python config classes and the YAML files under
`skillmimic_lab/agents/`.

| Original README option | Isaac Lab equivalent |
| --- | --- |
| `--test` | `play`, `circling`, `heading`, `throwing`, or `scoring` mode |
| `--task SkillMimicBallPlay` | `play` or `train` mode |
| `--task HRLCircling` | `circling` or `train-circling` mode |
| `--task HRLHeadingEasy` | `heading` or `train-heading` mode |
| `--task HRLThrowing` | `throwing` or `train-throwing` mode |
| `--task HRLScoringLayup` | `scoring` or `train-scoring` mode |
| `--num_envs N` | `NUM_ENVS=N` |
| `--motion_file PATH` | `--motion_path PATH` |
| `--checkpoint PATH` | `--checkpoint PATH` |
| `--llc_checkpoint PATH` | `--llc_checkpoint PATH` |
| `--state_init FRAME` | `--state_init FRAME` |
| `--episode_length N` | `--episode_length N` for inference |
| `--headless` | `HEADLESS=1` or forwarded `--headless` |
| `--cfg_env` | `skillmimic_lab/env/tasks/*.py` config classes |
| `--cfg_train` | `skillmimic_lab/agents/*.yaml` |

## 7. Compatibility contract

The migration preserves the data and policy interfaces needed by the README's
released assets:

- skill-policy observations: 902 values;
- skill-policy actions: 156 joint targets;
- legacy HOI reference observation: 380 values;
- HLC observations: 843 for Circling, 842 for Heading, 838 for Throwing, and
  843 for Scoring;
- original 53-body and 156-joint MJCF name ordering;
- original BallPlay-M `.pt` motion layout;
- original RL-Games actor weights and running observation statistics.

BallPlay-M tensors and legacy checkpoints use XYZW quaternions. Isaac Lab uses
WXYZ simulator quaternions. Conversion occurs only when values cross the
simulator boundary, so the stored dataset and checkpoints are not modified.

Isaac Lab v1.1 does not provide `MjcfFileCfg`. The environment enables Isaac
Sim 4.1's `omni.importer.mjcf` extension, imports
`skillmimic/data/assets/mjcf/mocap_humanoid.xml`, and validates body and joint
names before using observations or actions.

## 8. Dataset and rendering

The migration reuses the BallPlay-M subset in:

```text
skillmimic/data/motions/BallPlay-M/
```

No dataset conversion is required. The loader accepts the existing `.pt`
files directly.

The Blender rendering workflow described in the README remains independent of
the simulator migration. See `blender_for_SkillMimic/README.md`. The original
`skillmimic/utils/make_video.py` utility is also retained.

## 9. Differences from the original interactive demo

The README documents several Isaac Gym viewer features that are not exposed by
the current Isaac Lab launcher:

- keyboard skill switching (`Q`, `W`, arrow keys, `E`, and `R`);
- mouse target placement through `--projtype Mouse`;
- `--play_dataset`;
- `--save_images`.

In the migrated skill-policy demo, select the demonstration skill with
`--motion_path`. HLC targets are generated by the task environment at reset.
Use the original Isaac Gym implementation when one of the viewer-only features
above is required.

## 10. Additional checks

CPU-side tensor, motion, and checkpoint compatibility tests can be run with:

```bash
python -m unittest tests/test_isaaclab_compat.py
```

Shell and Python syntax checks are:

```bash
bash -n scripts/run_isaaclab.sh
python -m compileall -q skillmimic_lab tests
```

For simulator failures, inspect `logs/isaaclab/latest.log` first. The most
useful phase markers are `simulation_app_ready`, `creating_environment`,
`environment_created`, and `environment_reset_complete`.

## 11. Original implementation backup

The migration does not overwrite `skillmimic/`. A compressed backup of the
original tracked Isaac Gym source is also stored at:

```text
backups/skillmimic_isaacgym.gz
```

See `backups/README.md` for its checksum and restore command.

## References

- [SkillMimic README](README.md)
- [Isaac Lab v1.1 binary installation](https://isaac-sim.github.io/IsaacLab/v1.1.0/source/setup/installation/binaries_installation.html)
- [Migrating from IsaacGymEnvs](https://isaac-sim.github.io/IsaacLab/main/source/migration/migrating_from_isaacgymenvs.html)
- [Direct workflow environment tutorial](https://isaac-sim.github.io/IsaacLab/v1.1.0/source/tutorials/03_envs/create_direct_rl_env.html)
