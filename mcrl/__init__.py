"""
MinecraftRL - A JAX-native Minecraft environment for reinforcement learning.

High-performance voxel environment designed for:
- Fast iteration (100k+ steps/sec with vectorization)
- Sample-efficient RL research
- Diamond acquisition task

Quick start:
    import jax
    from mcrl import MinecraftEnv
    
    env = MinecraftEnv()
    key = jax.random.PRNGKey(0)
    
    # Single environment
    state, obs = env.reset(key)
    state, obs, reward, done, info = env.step(state, 1)  # Forward
    
    # Vectorized (parallel environments)
    vec_reset, vec_step = env.make_vectorized()
    keys = jax.random.split(key, 1024)
    states, obs = vec_reset(keys)
"""

__version__ = "0.1.0"

from mcrl.env import MinecraftEnv, EnvConfig, make_env
from mcrl.core.types import BlockType, ItemType, ToolType
from mcrl.core.state import GameState, WorldState, PlayerState
from mcrl.systems.actions import Action

__all__ = [
    "MinecraftEnv",
    "EnvConfig", 
    "make_env",
    "BlockType",
    "ItemType",
    "ToolType",
    "Action",
    "GameState",
    "WorldState",
    "PlayerState",
]
