# SkillMimic Lab

[Original Repository](https://github.com/wyhuai/SkillMimic) |
[Original README](skillmimic/README.md) |
[Paper](https://arxiv.org/abs/2408.15270) |
[Project Page](https://ingrid789.github.io/SkillMimic/) |
[Video](https://youtu.be/j1smsXilUGM)

> [!IMPORTANT]
> **This repository ports SkillMimic from Isaac Gym to Isaac Lab.** It uses
> Isaac Sim 4.1 and Isaac Lab 1.1 while retaining the released BallPlay-M
> motions and pretrained checkpoints.

![SkillMimic basketball skills](https://github.com/user-attachments/assets/ac75c9be-f144-4b6d-980f-272c6f657627)

The port is in `skillmimic_lab/`. The original Isaac Gym code and assets remain
in `skillmimic/`.

## Setup

Required versions: Python 3.10, Isaac Sim 4.1.0 standalone, Isaac Lab v1.1.0,
PyTorch 2.2.2 with CUDA 11.8, and RL-Games 1.6.1.

Place Isaac Sim and Isaac Lab under the ignored `.external/` directory:

```text
.external/isaac-sim-4.1.0/
.external/IsaacLab/
```

```bash
mkdir -p .external
git clone --branch v1.1.0 https://github.com/isaac-sim/IsaacLab.git .external/IsaacLab

conda create -n skillmimic_lab python=3.10
conda activate skillmimic_lab
pip install -r requirements_isaaclab.txt
```

Extract the Isaac Sim standalone package to `.external/isaac-sim-4.1.0/`. If
it is installed elsewhere, set `ISAAC_SIM_ROOT`. Do not install the original
`requirements.txt` in this environment.

## Run

Validate the environment, then run the released skill policy:

```bash
bash scripts/run_isaaclab.sh smoke
bash scripts/run_isaaclab.sh play
```

The smoke test succeeds when it prints:

```text
[SkillMimic Lab] PASS mode=smoke task=ballplay
```

Available policy modes are:

| Task | Inference | Training |
| --- | --- | --- |
| Skill policy | `play` | `train` |
| Circling | `circling` | `train-circling` |
| Heading | `heading` | `train-heading` |
| Throwing | `throwing` | `train-throwing` |
| Scoring | `scoring` | `train-scoring` |

For example:

```bash
NUM_ENVS=1 bash scripts/run_isaaclab.sh scoring
```

The launcher is headless by default. Use a local viewer only on a machine with
a graphical display:

```bash
HEADLESS=0 NUM_ENVS=1 bash scripts/run_isaaclab.sh play
```

## Train

```bash
NUM_ENVS=2048 bash scripts/run_isaaclab.sh train \
  --motion_path skillmimic/data/motions/BallPlay-M/layup \
  --max_iterations 50000

NUM_ENVS=2048 bash scripts/run_isaaclab.sh train-scoring \
  --max_iterations 6000
```

Arguments after the mode are forwarded to the Python entry point. For multiple
GPUs, set `NUM_GPUS`; `NUM_ENVS` is the number of environments per GPU:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_GPUS=4 NUM_ENVS=4096 \
  bash scripts/run_isaaclab.sh train --max_iterations 50000
```

Use `--checkpoint` to resume an Isaac Lab training run. Released Isaac Gym
checkpoints support inference, but their optimizer state cannot be resumed.

## Configuration

| Variable | Default | Meaning |
| --- | ---: | --- |
| `NUM_ENVS` | 4 inference, 2048 training | Environments per GPU |
| `NUM_GPUS` | 1 | Training processes and GPUs |
| `STEPS` | 600 | Inference steps |
| `HEADLESS` | 1 | Set to 0 for a local viewer |
| `DEVICE_ID` | 0 | CUDA device |
| `ISAAC_SIM_ROOT` | `.external/isaac-sim-4.1.0` | Isaac Sim path |

Training output and TensorBoard events are written below `logs/`, which is
excluded from Git. TensorBoard uses `http://localhost:6006` by default.

## Notes

- Released motions and checkpoints under `skillmimic/data/` load directly;
  no conversion step is required.
- The port converts legacy XYZW quaternions to Isaac Lab's WXYZ convention only
  at the simulator boundary.
- Keyboard skill switching, mouse target placement, `--play_dataset`, and
  `--save_images` remain available only in the
  [original Isaac Gym implementation](skillmimic/README.md).

## Tests

```bash
python -m unittest tests.test_isaaclab_compat
bash -n scripts/run_isaaclab.sh
python -m compileall -q skillmimic_lab tests
```

For simulator startup failures, inspect `logs/isaaclab/latest.log`.

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
