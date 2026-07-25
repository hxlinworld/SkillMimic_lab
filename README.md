# SkillMimic Lab

[Original Repository](https://github.com/wyhuai/SkillMimic) |
[Original README](skillmimic/README.md) |
[Paper](https://arxiv.org/abs/2408.15270) |
[Project Page](https://ingrid789.github.io/SkillMimic/) |
[Video](https://youtu.be/j1smsXilUGM)

> [!IMPORTANT]
> **This repository provides an Isaac Lab implementation of SkillMimic.** It
> implements the SkillMimic skill policy and high-level controllers with Isaac
> Sim 4.1 and Isaac Lab 1.1, while reusing the released BallPlay-M motions and
> pretrained checkpoints.

![SkillMimic basketball skills](https://github.com/user-attachments/assets/ac75c9be-f144-4b6d-980f-272c6f657627)

The Isaac Lab implementation is in `skillmimic_lab/`. The original Isaac Gym
code and assets remain in `skillmimic/`.

## Installation

### Step 1: clone this repository

```bash
git clone https://github.com/hxlinworld/SkillMimic_lab.git
cd SkillMimic_lab
```

### Step 2: download Isaac Sim 4.1.0 and Isaac Lab 1.1

The following command downloads the official 7.8 GB Linux standalone archive
listed in the
[NVIDIA Isaac Sim download archive](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/download.html#download-archive):

```bash
mkdir -p .external/isaac-sim-4.1.0
wget -c \
  "https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone%404.1.0-rc.7%2B4.1.14801.71533b68.gl.linux-x86_64.release.zip" \
  -O .external/isaac-sim-4.1.0.zip
unzip -q .external/isaac-sim-4.1.0.zip \
  -d .external/isaac-sim-4.1.0
```

Clone the matching Isaac Lab release:

```bash
git clone --branch v1.1.0 --depth 1 \
  https://github.com/isaac-sim/IsaacLab.git .external/IsaacLab
```

The resulting local layout is:

```text
.external/isaac-sim-4.1.0/
.external/IsaacLab/
```

### Step 3: create the Python environment

```bash
conda create -n skillmimic_lab python=3.10
conda activate skillmimic_lab
pip install -r requirements_isaaclab.txt
```

This setup uses Python 3.10, PyTorch 2.2.2 with CUDA 11.8, and RL-Games 1.6.1.
If Isaac Sim is installed at another location, set `ISAAC_SIM_ROOT` before
running the launcher.

## Verify the Environment

Run the smoke test before inference or training:

```bash
bash scripts/run_isaaclab.sh smoke
```

The command creates four environments, applies zero actions for 600 simulation
steps, and checks observations and rewards for invalid values. A successful run
ends with:

```text
[SkillMimic Lab] PASS mode=smoke task=ballplay
```

## Pre-Trained Models

Pre-trained models are available at
[`skillmimic/data/models/`](skillmimic/data/models/).

## Skill Policy

### Inference

Run the released mixed-skills policy with its default checkpoint and BallPlay-M
motion directory:

```bash
bash scripts/run_isaaclab.sh play
```

To select a motion, checkpoint, number of environments, and rollout length
explicitly:

```bash
NUM_ENVS=16 STEPS=140 bash scripts/run_isaaclab.sh play \
  --motion_path skillmimic/data/motions/BallPlay-M/layup \
  --checkpoint skillmimic/data/models/mixedskills/nn/skillmimic_llc.pth \
  --state_init 20 \
  --episode_length 140
```

### Training

Train the skill policy from demonstrations:

```bash
NUM_ENVS=2048 bash scripts/run_isaaclab.sh train \
  --motion_path skillmimic/data/motions/BallPlay-M/layup \
  --max_iterations 50000
```

`NUM_ENVS` controls the number of parallel environments. `--motion_path`
selects either one `.pt` motion file or a directory, and `--max_iterations`
sets the total RL-Games training iterations.

Training settings:

- **Environment:** [`skillmimic_lab/env/tasks/`](skillmimic_lab/env/tasks/)
- **PPO:** [`skillmimic_lab/agents/rl_games_ppo_cfg.yaml`](skillmimic_lab/agents/rl_games_ppo_cfg.yaml)
- **Motion data:** `--motion_path <file-or-directory>`
- **Visualization:** disabled by default; set `HEADLESS=0` to enable it

- It is strongly encouraged to use large "--num_envs" when training on a large dataset, e.g., use "NUM_ENVS=16384" for `--motion_path skillmimiclab/data/motions/BallPlay-M` (Meanwhile, `--minibatch_size` is recommended to be set as 8×`num_envs`)

```bash
NUM_ENVS=16384 bash scripts/run_isaaclab.sh train \
  --motion_path skillmimiclab/data/motions/BallPlay-M \
  --minibatch_size 131072
```

Training output:

- The best skill-policy checkpoint (`skillmimic_isaaclab.pth`) and periodic
  checkpoints are saved to `logs/rl_games/<task>_isaaclab/<run>/nn/`.
- TensorBoard events are saved to the adjacent `summaries/` directory. The
  launcher starts TensorBoard at `http://localhost:6006`.
- The latest terminal output is saved to `logs/isaaclab/latest.log`; follow it
  with `tail -f logs/isaaclab/latest.log`.


## High-Level Controller

### Inference

The released controllers and their required low-level policy are selected by
the launcher:

```bash
NUM_ENVS=1 bash scripts/run_isaaclab.sh circling
NUM_ENVS=1 bash scripts/run_isaaclab.sh heading
NUM_ENVS=1 bash scripts/run_isaaclab.sh throwing
NUM_ENVS=1 bash scripts/run_isaaclab.sh scoring
```

### Training

```bash
NUM_ENVS=2048 bash scripts/run_isaaclab.sh train-circling \
  --max_iterations 6000
NUM_ENVS=2048 bash scripts/run_isaaclab.sh train-heading \
  --max_iterations 6000
NUM_ENVS=2048 bash scripts/run_isaaclab.sh train-throwing \
  --max_iterations 6000
NUM_ENVS=2048 bash scripts/run_isaaclab.sh train-scoring \
  --max_iterations 6000
```

For distributed training, `NUM_ENVS` is the number of environments per GPU:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_GPUS=4 NUM_ENVS=4096 \
  bash scripts/run_isaaclab.sh train --max_iterations 50000
```

Pass `--checkpoint <path>` to resume a checkpoint produced by this Isaac Lab
training path. The released Isaac Gym checkpoints are supported for inference,
but their optimizer state cannot be resumed.

## Notes

- Released motions and checkpoints under `skillmimic/data/` load directly;
  no conversion step is required.
- Legacy XYZW quaternions are converted to Isaac Lab's WXYZ convention only at
  the simulator boundary.
- Keyboard skill switching, mouse target placement, `--play_dataset`, and
  `--save_images` remain available only in the
  [original Isaac Gym implementation](skillmimic/README.md).

## Citation

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

Built on [SkillMimic](https://github.com/wyhuai/SkillMimic),
[ASE](https://github.com/nv-tlabs/ASE),
[PhysHOI](https://github.com/wyhuai/PhysHOI), and
[Isaac Lab](https://github.com/isaac-sim/IsaacLab).
