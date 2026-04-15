"""
License: MIT License
Copyright (c) 2024, Felipe Mohr Santos

Lite3 (Deep Robotics) robot support added by Airbotix Technology Pvt. Ltd.
"""

import gymnasium as gym

from omni.isaac.lab_quadruped_tasks.agents.rsl_rl_cfg import QuadrupedPPORunnerCfg
from . import lite3_env_cfg


##
# Create PPO runners for RSL-RL
##

lite3_blind_flat_runner_cfg = QuadrupedPPORunnerCfg()
lite3_blind_flat_runner_cfg.experiment_name = "lite3_blind_flat"

lite3_blind_rough_runner_cfg = QuadrupedPPORunnerCfg()
lite3_blind_rough_runner_cfg.experiment_name = "lite3_blind_rough"

lite3_blind_stairs_runner_cfg = QuadrupedPPORunnerCfg()
lite3_blind_stairs_runner_cfg.experiment_name = "lite3_blind_stairs"

lite3_vision_runner_cfg = QuadrupedPPORunnerCfg()
lite3_vision_runner_cfg.experiment_name = "lite3_vision"
lite3_vision_runner_cfg.policy.actor_hidden_dims = [512, 256, 128]
lite3_vision_runner_cfg.policy.critic_hidden_dims = [512, 256, 128]

lite3_vision_stairs_runner_cfg = QuadrupedPPORunnerCfg()
lite3_vision_stairs_runner_cfg.experiment_name = "lite3_vision_stairs"
lite3_vision_stairs_runner_cfg.policy.actor_hidden_dims = [512, 256, 128]
lite3_vision_stairs_runner_cfg.policy.critic_hidden_dims = [512, 256, 128]

##
# Register Gym environments
##

#############################
# Lite3 Blind Flat Environment
#############################

gym.register(
    id="Isaac-Quadruped-Lite3-Blind-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": lite3_env_cfg.Lite3BlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": lite3_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Quadruped-Lite3-Blind-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": lite3_env_cfg.Lite3BlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": lite3_blind_flat_runner_cfg,
    },
)

##############################
# Lite3 Blind Rough Environment
##############################

gym.register(
    id="Isaac-Quadruped-Lite3-Blind-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": lite3_env_cfg.Lite3BlindRoughEnvCfg,
        "rsl_rl_cfg_entry_point": lite3_blind_rough_runner_cfg,
    },
)

gym.register(
    id="Isaac-Quadruped-Lite3-Blind-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": lite3_env_cfg.Lite3BlindRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": lite3_blind_rough_runner_cfg,
    },
)

###############################
# Lite3 Blind Stairs Environment
###############################

gym.register(
    id="Isaac-Quadruped-Lite3-Blind-Stairs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": lite3_env_cfg.Lite3BlindStairsEnvCfg,
        "rsl_rl_cfg_entry_point": lite3_blind_stairs_runner_cfg,
    },
)

gym.register(
    id="Isaac-Quadruped-Lite3-Blind-Stairs-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": lite3_env_cfg.Lite3BlindStairsEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": lite3_blind_stairs_runner_cfg,
    },
)

#########################
# Lite3 Vision Environment
#########################

gym.register(
    id="Isaac-Quadruped-Lite3-Vision-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": lite3_env_cfg.Lite3VisionEnvCfg,
        "rsl_rl_cfg_entry_point": lite3_vision_runner_cfg,
    },
)

gym.register(
    id="Isaac-Quadruped-Lite3-Vision-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": lite3_env_cfg.Lite3VisionEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": lite3_vision_runner_cfg,
    },
)

################################
# Lite3 Vision Stairs Environment
################################

gym.register(
    id="Isaac-Quadruped-Lite3-Vision-Stairs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": lite3_env_cfg.Lite3VisionStairsEnvCfg,
        "rsl_rl_cfg_entry_point": lite3_vision_stairs_runner_cfg,
    },
)

gym.register(
    id="Isaac-Quadruped-Lite3-Vision-Stairs-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": lite3_env_cfg.Lite3VisionStairsEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": lite3_vision_stairs_runner_cfg,
    },
)
