# Isaac Lab Quadruped Tasks Extension

![image](https://github.com/user-attachments/assets/e2a0a26e-0f06-4eb7-8478-d726585dac94)

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.x-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

## Overview

This is the **Airbotix Technology** fork of [felipemohr/IsaacLab-Quadruped-Tasks](https://github.com/felipemohr/IsaacLab-Quadruped-Tasks), migrated to the **Isaac Lab 2.x API** (`isaaclab.*` namespace) and extended with support for the **Deep Robotics Lite3** quadruped robot.

This repository contains an Isaac Lab extension with tasks for training quadruped robots using Reinforcement Learning.

### Supported Robots & Tasks

|                                      | Flat (Blind)                                  | Rough (Blind)                                  | Stairs (Blind)                                  | Rough (Vision)                               | Stairs (Vision)                                  |
| ------------------------------------ | --------------------------------------------- | ---------------------------------------------- | ----------------------------------------------- | -------------------------------------------- | ------------------------------------------------- |
| **Lite3 (Deep Robotics)**            | `Isaac-Quadruped-Lite3-Blind-Flat-v0`         | `Isaac-Quadruped-Lite3-Blind-Rough-v0`         | `Isaac-Quadruped-Lite3-Blind-Stairs-v0`         | `Isaac-Quadruped-Lite3-Vision-v0`            | `Isaac-Quadruped-Lite3-Vision-Stairs-v0`          |
| **ANYmal D (ANYbotics)**             | `Isaac-Quadruped-AnymalD-Blind-Flat-v0`       | `Isaac-Quadruped-AnymalD-Blind-Rough-v0`       | `Isaac-Quadruped-AnymalD-Blind-Stairs-v0`       | `Isaac-Quadruped-AnymalD-Vision-v0`          | `Isaac-Quadruped-AnymalD-Vision-Stairs-v0`        |
| **Go2 (Unitree)**                    | `Isaac-Quadruped-Go2-Blind-Flat-v0`           | `Isaac-Quadruped-Go2-Blind-Rough-v0`           | `Isaac-Quadruped-Go2-Blind-Stairs-v0`           | `Isaac-Quadruped-Go2-Vision-v0`              | `Isaac-Quadruped-Go2-Vision-Stairs-v0`            |
| **Spot (Boston Dynamics)**           | `Isaac-Quadruped-Spot-Blind-Flat-v0`          | `Isaac-Quadruped-Spot-Blind-Rough-v0`          | `Isaac-Quadruped-Spot-Blind-Stairs-v0`          | `Isaac-Quadruped-Spot-Vision-v0`             | `Isaac-Quadruped-Spot-Vision-Stairs-v0`           |

## Key Changes from Upstream

- ✅ **Isaac Lab 2.x migration**: all `omni.isaac.lab.*` imports replaced with `isaaclab.*`
- ✅ **isaaclab_assets 0.2.x**: robot assets now imported from `isaaclab_assets.robots.*`
- ✅ **Gym entry points**: updated from `omni.isaac.lab.envs` to `isaaclab.envs`
- ✅ **RSL-RL wrappers**: migrated to `isaaclab_rl.rsl_rl`
- ✅ **Lite3 robot support**: `DelayedPDActuator`, custom reward shaping, USD asset integration

## Installation

1. Install **Isaac Sim 4.5** and **Isaac Lab 2.x** (follow [official docs](https://isaac-sim.github.io/IsaacLab)).

2. Clone this repo and install the extension:

```bash
git clone git@github.com:Airbotix-Technology-Pvt-Ltd/IsaacLab-Quadruped-Tasks.git
cd IsaacLab-Quadruped-Tasks
conda activate env_isaaclab
python -m pip install -e exts/omni.isaac.lab_quadruped_tasks
```

> **Note:** The Lite3 USD asset path is configured in `robots/lite3/lite3_env_cfg.py`. Update `LITE3_USD_PATH` to point to your local USD file.

## Training

Use the `rsl_rl/train.py` script, specifying the task:

```bash
python scripts/rsl_rl/train.py --task Isaac-Quadruped-Lite3-Blind-Flat-v0 --headless
```

Optional arguments:

- `--num_envs` — Number of parallel environments (default: `1024`)
- `--max_iterations` — Training iterations (default: `8000`)
- `--save_interval` — Checkpoint save interval (default: `500`)
- `--seed` — Random seed (default: `42`)

Resume from checkpoint:

```bash
python scripts/rsl_rl/train.py --task Isaac-Quadruped-Lite3-Blind-Stairs-v0 \
  --resume True --checkpoint_path logs/rsl_rl/lite3_blind_stairs/RUN_DIR/model_XXXX.pt
```

Visualise training with TensorBoard:

```bash
tensorboard --logdir logs/rsl_rl/
```

## Playing a Trained Agent

```bash
python scripts/rsl_rl/play.py \
  --task Isaac-Quadruped-Lite3-Blind-Flat-Play-v0 \
  --num_envs 64 \
  --checkpoint_path logs/rsl_rl/lite3_blind_flat/RUN_DIR/model_8000.pt
```

Tasks ending with `-Play-v0` disable domain randomisation for cleaner evaluation.

## Results

Below are videos recorded during Go2 training (upstream demos):

### Blind locomotion, flat terrain:

https://github.com/user-attachments/assets/21ae573a-2c04-4a7e-ae92-aed17b525ade

### Blind locomotion, rough terrain:

https://github.com/user-attachments/assets/8c5d5d6d-40ab-4345-b046-95352a208968

## Credits

Original repository by [Felipe Mohr](https://github.com/felipemohr).  
Lite3 support and Isaac Lab 2.x migration by [Airbotix Technology Pvt. Ltd.](https://github.com/Airbotix-Technology-Pvt-Ltd)
