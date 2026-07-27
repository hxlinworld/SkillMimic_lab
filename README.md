# SkillMimic Lab

[Original Repository](https://github.com/wyhuai/SkillMimic) |
[Original README](skillmimic/README.md) |
[Paper](https://arxiv.org/abs/2408.15270) |
[Project Page](https://ingrid789.github.io/SkillMimic/) |
[Video](https://youtu.be/j1smsXilUGM)

> [!IMPORTANT]
> **This repository provides an Isaac Lab implementation of SkillMimic.** It
> implements the SkillMimic skill policy and high-level controllers with Isaac
> Sim 5.1 and Isaac Lab 2.3.2, while reusing the released BallPlay-M motions and
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

### Step 2: use the NVIDIA Isaac Lab container

Install Docker, the NVIDIA Container Toolkit, and an NVIDIA driver supported by
Isaac Sim 5.1, then pull the matching pre-built Isaac Lab image once:

```bash
docker pull nvcr.io/nvidia/isaac-lab:2.3.2
docker run --rm --gpus all --entrypoint nvidia-smi \
  nvcr.io/nvidia/isaac-lab:2.3.2
```

The launcher uses that image directly, bind-mounts this checkout at
`/workspace/skillmimic-lab`, and keeps simulator caches in
`.docker/isaaclab/`. If the current user cannot access the Docker socket, it
automatically falls back to `sudo -n docker`.

No Conda environment or host-side `pip install` is required. The image provides
Python 3.11, PyTorch 2.7 with CUDA 12.8, RL-Games 1.6.1, Isaac Sim 5.1, and
Isaac Lab 2.3.2. `requirements_isaaclab.txt` is intentionally empty so legacy
pins cannot overwrite the image runtime.

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

### WebRTC training

WebRTC must be enabled in the training process itself. Stop any standalone
Isaac Sim streaming container that is using port 49100, then launch:

```bash
PUBLIC_IP=<server-public-ip> NUM_ENVS=1024 \
  bash scripts/run_isaaclab.sh train-webrtc \
  --motion_path skillmimic/data/motions/BallPlay-M/layup \
  --minibatch_size 8192 \
  --max_iterations 50000
```

The launcher enables Isaac Lab's public-network WebRTC mode and passes TCP
49100 and UDP 47998 through host networking. As a local convenience,
`PUBLIC_IP` may instead be stored as the first line of
`.docker/webrtc_public_ip`. Only one WebRTC client can connect at a time, and
viewport rendering reduces training throughput.

`NUM_ENVS` controls the number of parallel environments. `--motion_path`
selects either one `.pt` motion file or a directory, and `--max_iterations`
sets the total RL-Games training iterations.

Training containers use the stable name
`skillmimic-<environment>-seed-<seed>`. If `--seed` is omitted, the launcher
generates a random positive seed and passes it to the training process. By
default, a second container for an environment that is already training is
rejected. Set `ALLOW_PARALLEL_TRAINING=1` only when parallel runs with
different seeds are intentional; pass `--seed 43` to reproduce a known run.

Training settings:

- **Environment:** [`skillmimic_lab/env/tasks/`](skillmimic_lab/env/tasks/)
- **PPO:** [`skillmimic_lab/agents/rl_games_ppo_cfg.yaml`](skillmimic_lab/agents/rl_games_ppo_cfg.yaml)
- **Motion data:** `--motion_path <file-or-directory>`
- **Visualization:** the pre-built Isaac Lab image is intended for headless use

For large datasets, use many parallel environments and set the minibatch size
to roughly eight times `NUM_ENVS`, subject to available GPU memory:

```bash
NUM_ENVS=16384 bash scripts/run_isaaclab.sh train \
  --motion_path skillmimic/data/motions/BallPlay-M \
  --minibatch_size 131072
```

Training output:

- The best skill-policy checkpoint (`skillmimic_isaaclab.pth`) and periodic
  checkpoints are saved to `logs/rl_games/<task>_isaaclab/<run>/nn/`.
- TensorBoard events are saved to the adjacent `summaries/` directory. The
  launcher starts TensorBoard at `http://localhost:6006`.
- The latest terminal output is saved to `logs/isaaclab/latest.log`; follow it
  with `tail -f logs/isaaclab/latest.log`.
- Each epoch prints a `[training-diagnostics]` line. Fall and timeout rates are
  calculated over completed episodes; action mean/standard deviation and
  periodic joint-tracking RMSE are sampled every environment step and then
  aggregated over the epoch.


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
