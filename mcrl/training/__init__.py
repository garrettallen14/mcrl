"""
MCRL Training Infrastructure.

PureJaxRL-style PPO implementation for voxel environments.
"""

from mcrl.training.config import (
    TrainConfig, 
    PPOConfig, 
    NetworkConfig,
    get_fast_debug_config,
    get_baseline_config,
    get_high_explore_config,
    get_large_scale_config,
    get_rtx4090_config,
)
from mcrl.training.networks import VoxelEncoder, ActorCritic
from mcrl.training.networks_fast import FastActorCritic, FastVoxelEncoder
from mcrl.training.ppo import ppo_loss, compute_gae
from mcrl.training.train import train, create_train_state

__all__ = [
    "TrainConfig",
    "PPOConfig", 
    "NetworkConfig",
    "VoxelEncoder",
    "ActorCritic",
    "FastActorCritic",
    "FastVoxelEncoder",
    "ppo_loss",
    "compute_gae",
    "train",
    "create_train_state",
    "get_fast_debug_config",
    "get_baseline_config",
    "get_high_explore_config",
    "get_large_scale_config",
    "get_rtx4090_config",
]
