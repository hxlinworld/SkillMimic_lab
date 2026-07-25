# SkillMimic Lab

[Original Repository](https://github.com/wyhuai/SkillMimic) |
[Original Isaac Gym README](skillmimic/README.md) |
[Paper](https://arxiv.org/abs/2408.15270) |
[Project Page](https://ingrid789.github.io/SkillMimic/) |
[Video](https://youtu.be/j1smsXilUGM)

> [!IMPORTANT]
> **This repository is an Isaac Lab implementation of SkillMimic.** It ports
> the original Isaac Gym basketball skill policy and high-level controllers to
> Isaac Sim 4.1 and Isaac Lab 1.1 while retaining compatibility with the
> released BallPlay-M motions and pretrained checkpoints.

![SkillMimic basketball skills](https://github.com/user-attachments/assets/ac75c9be-f144-4b6d-980f-272c6f657627)

SkillMimic learns basketball interaction skills from human demonstrations and
reuses those skills in high-level tasks. This repository provides the migrated
Isaac Lab runtime in `skillmimic_lab/`. The original Isaac Gym implementation
is retained in `skillmimic/` for reference.

## What is included

- an Isaac Lab `DirectRLEnv` implementation of the SkillMimic skill policy;
- Isaac Lab environments for Circling, Heading, Throwing, and Scoring;
- loaders for the released legacy RL-Games checkpoints;
- direct loading of the released BallPlay-M `.pt` motion data;
- RL-Games training configurations for the skill policy and HLC tasks;
- single-GPU and distributed multi-GPU training launchers;
- headless inference, smoke tests, TensorBoard logging, and compatibility tests.

The migration does not import `isaacgym` and does not require checkpoint or
dataset conversion.

## Runtime support

| Component | Isaac Lab implementation | Original implementation |
| --- | --- | --- |
| Python | 3.10 | 3.8 |
| Simulator | Isaac Sim 4.1.0 | Isaac Gym Preview 4 |
| Framework | Isaac Lab v1.1.0 | Isaac Gym |
| PyTorch | 2.2.2 + CUDA 11.8 | Legacy project environment |
| RL library | RL-Games 1.6.1 | Legacy RL-Games |
| Source | `skillmimic_lab/` | `skillmimic/` |

The original Isaac Gym setup and commands are documented separately in the
[original README](skillmimic/README.md). Do not install `requirements.txt` in
the Isaac Lab environment.

## Installation

### Prerequisites

- Linux with a supported NVIDIA GPU and driver;
- the Isaac Sim 4.1 standalone package;
- an Isaac Lab checkout at tag `v1.1.0`;
- Conda or another Python 3.10 environment manager.

The default local layout is:

```text
SkillMimic_lab/
├── .external/
│   ├── isaac-sim-4.1.0/
│   └── IsaacLab/
├── skillmimic/                  # original Isaac Gym implementation and assets
├── skillmimic_lab/              # Isaac Lab implementation
└── scripts/run_isaaclab.sh
```

`.external/` is intentionally excluded from Git. Extract the Isaac Sim
standalone archive to `.external/isaac-sim-4.1.0/` and clone Isaac Lab 1.1:

```bash
mkdir -p .external
git clone --branch v1.1.0 https://github.com/isaac-sim/IsaacLab.git .external/IsaacLab
```

Create the Python environment and install the migration dependencies:

```bash
conda create -n skillmimic_lab python=3.10
conda activate skillmimic_lab
pip install -r requirements_isaaclab.txt
```

`requirements_isaaclab.txt` installs the repository-local Isaac Lab extension
in editable mode. It does not install Isaac Sim. If Isaac Sim is stored
elsewhere, set `ISAAC_SIM_ROOT` before launching. Use `PYTHON_BIN` to select a
Python executable other than the active `python`.

### Validate the installation

Run the four-environment, zero-action smoke test:

```bash
bash scripts/run_isaaclab.sh smoke
```

A successful run ends with:

```text
[SkillMimic Lab] PASS mode=smoke task=ballplay
```

The launcher is headless by default and mirrors terminal output to
`logs/isaaclab/latest.log`.

## Quick start

### Skill-policy inference

```bash
bash scripts/run_isaaclab.sh play
```

Use more parallel environments or select a specific motion directory:

```bash
NUM_ENVS=16 STEPS=140 bash scripts/run_isaaclab.sh play \
  --motion_path skillmimic/data/motions/BallPlay-M/layup \
  --checkpoint skillmimic/data/models/mixedskills/nn/skillmimic_llc.pth \
  --state_init 20 \
  --episode_length 140
```

For a local graphical viewer:

```bash
HEADLESS=0 NUM_ENVS=1 bash scripts/run_isaaclab.sh play
```

Do not set `HEADLESS=0` on a server without a desktop or X server.

### Skill-policy training

```bash
NUM_ENVS=2048 bash scripts/run_isaaclab.sh train \
  --motion_path skillmimic/data/motions/BallPlay-M/layup \
  --max_iterations 50000 \
  --headless
```

Training output is written to:

```text
logs/rl_games/ballplay_isaaclab/<timestamp>/
```

## High-level controllers

The migrated high-level controllers load the released mixed-skills policy as
their low-level controller. Their task mapping is:

| Task | Inference mode | Training mode | Isaac Lab task ID | Default motion |
| --- | --- | --- | --- | --- |
| Circling | `circling` | `train-circling` | `SkillMimic-Circling-Direct-v0` | `BallPlay-M/run` |
| Heading | `heading` | `train-heading` | `SkillMimic-Heading-Direct-v0` | `BallPlay-M/run` |
| Throwing | `throwing` | `train-throwing` | `SkillMimic-Throwing-Direct-v0` | `BallPlay-M/turnhook` |
| Scoring | `scoring` | `train-scoring` | `SkillMimic-Scoring-Direct-v0` | `BallPlay-M/run` |

Run any released HLC checkpoint with its short launcher mode:

```bash
NUM_ENVS=1 bash scripts/run_isaaclab.sh circling
NUM_ENVS=1 bash scripts/run_isaaclab.sh heading
NUM_ENVS=1 bash scripts/run_isaaclab.sh throwing
NUM_ENVS=1 bash scripts/run_isaaclab.sh scoring
```

Train the corresponding controllers with:

```bash
NUM_ENVS=2048 bash scripts/run_isaaclab.sh train-circling --max_iterations 6000
NUM_ENVS=2048 bash scripts/run_isaaclab.sh train-heading --max_iterations 6000
NUM_ENVS=2048 bash scripts/run_isaaclab.sh train-throwing --max_iterations 6000
NUM_ENVS=2048 bash scripts/run_isaaclab.sh train-scoring --max_iterations 6000
```

HLC training output is stored under
`logs/rl_games/<task>_isaaclab/<timestamp>/`.

## Launcher reference

The complete launcher mode list is:

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

Arguments following the mode are forwarded to the Python entry point. Common
environment settings are:

| Setting | Default | Purpose |
| --- | ---: | --- |
| `NUM_ENVS` | 4 inference, 2048 training | Environments per GPU |
| `NUM_GPUS` | 1 | Training processes and GPUs |
| `STEPS` | 600 | Inference loop steps |
| `HEADLESS` | 1 | Set to 0 for a local viewer |
| `DEVICE_ID` | 0 | CUDA device passed to Isaac Sim |
| `PYTHON_BIN` | `python` | Python executable |
| `ISAAC_SIM_ROOT` | `.external/isaac-sim-4.1.0` | Isaac Sim standalone root |
| `TENSORBOARD_LOG_DIR` | `logs/rl_games` | TensorBoard log root |
| `TENSORBOARD_PORT` | 6006 | TensorBoard port |

The launcher starts TensorBoard during training at
`http://localhost:6006`. Runtime and training logs are excluded from Git.

## Multi-GPU training

`NUM_ENVS` is the number of environments per GPU. This example runs 4096
environments on each of four GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_GPUS=4 NUM_ENVS=4096 \
  bash scripts/run_isaaclab.sh train \
  --motion_path skillmimic/data/motions/BallPlay-M \
  --max_iterations 50000 \
  --minibatch_size 4096 \
  --headless
```

The launcher starts one Isaac Sim process per GPU and RL-Games synchronizes
gradients through NCCL.

## Resume training

Only checkpoints produced by this Isaac Lab training path can resume optimizer
state. The original released checkpoints remain valid for inference.

```bash
RUN_NAME="<existing-run-name>"
CHECKPOINT="logs/rl_games/ballplay_isaaclab/${RUN_NAME}/nn/<checkpoint>.pth"

NUM_ENVS=2048 SKILLMIMIC_RUN_NAME="${RUN_NAME}" \
  bash scripts/run_isaaclab.sh train \
  --checkpoint "${CHECKPOINT}" \
  --max_iterations 50000
```

`--max_iterations` is the total target iteration count, not the number of
additional iterations.

## Released assets

The migration directly reuses the released checkpoints:

```text
skillmimic/data/models/mixedskills/nn/skillmimic_llc.pth
skillmimic/data/models/hlc_circling/nn/SkillMimic.pth
skillmimic/data/models/hlc_heading/nn/SkillMimic.pth
skillmimic/data/models/hlc_throwing/nn/SkillMimic.pth
skillmimic/data/models/hlc_scoring/nn/SkillMimic.pth
```

BallPlay-M motions are loaded directly from:

```text
skillmimic/data/motions/BallPlay-M/
```

## Compatibility

The Isaac Lab implementation preserves the interfaces required by the released
assets:

- 902-value skill-policy observations and 156 joint actions;
- 380-value legacy HOI reference observations;
- the original 53-body and 156-joint MJCF ordering;
- HLC observation dimensions for all four released tasks;
- legacy RL-Games actor weights and running observation statistics;
- the original BallPlay-M tensor layout.

Legacy motions and checkpoints store XYZW quaternions, while Isaac Lab uses
WXYZ simulator quaternions. Conversion occurs only at the simulator boundary;
stored files are not modified.

## Current differences from the Isaac Gym demo

The following original viewer interactions are not exposed by the current
Isaac Lab launcher:

- keyboard skill switching;
- mouse target placement with `--projtype Mouse`;
- `--play_dataset`;
- `--save_images`.

Use `--motion_path` to select the demonstrated skill. Use the
[original Isaac Gym implementation](skillmimic/README.md) when one of the
viewer-only workflows is required.

## Repository layout

```text
skillmimic_lab/
├── env/tasks/                   # skill policy and four HLC environments
├── learning/                    # checkpoint loaders and RL-Games integration
├── agents/                      # RL-Games training configurations
├── utils/                       # motion and quaternion utilities
└── run.py                       # smoke tests and inference

skillmimic/                      # original Isaac Gym source and assets
scripts/run_isaaclab.sh          # Isaac Lab launcher
requirements_isaaclab.txt        # Isaac Lab Python dependencies
tests/test_isaaclab_compat.py     # CPU compatibility tests
```

## Tests

CPU-side data, tensor, and checkpoint checks:

```bash
python -m unittest tests.test_isaaclab_compat
```

Syntax checks:

```bash
bash -n scripts/run_isaaclab.sh
python -m compileall -q skillmimic_lab tests
```

For simulator startup failures, inspect `logs/isaaclab/latest.log`.

## Original SkillMimic

The original Isaac Gym code, pretrained models, motion subset, and assets are
kept under `skillmimic/`. See the
[original SkillMimic README](skillmimic/README.md) for its installation,
inference, training, rendering, and dataset instructions.

## Citation

If this repository is useful for your research, cite the original SkillMimic
work:

```bibtex
@InProceedings{Wang_2025_CVPR,
    author    = {Wang, Yinhuai and Zhao, Qihan and Yu, Runyi and Tsui, Hok Wai and Zeng, Ailing and Lin, Jing and Luo, Zhengyi and Yu, Jiwen and Li, Xiu and Chen, Qifeng and Zhang, Jian and Zhang, Lei and Tan, Ping},
    title     = {SkillMimic: Learning Basketball Interaction Skills from Demonstrations},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {17540-17549}
}
```

## Acknowledgements

This repository builds on the original
[SkillMimic](https://github.com/wyhuai/SkillMimic) implementation and its
dependencies, including [ASE](https://github.com/nv-tlabs/ASE) and
[PhysHOI](https://github.com/wyhuai/PhysHOI). The simulator migration targets
[Isaac Lab](https://github.com/isaac-sim/IsaacLab).
