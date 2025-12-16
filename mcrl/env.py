"""
MinecraftRL Environment.

A JAX-native Minecraft environment for reinforcement learning.
Designed for high throughput via vectorization with jax.vmap.
"""

from __future__ import annotations
from typing import Optional, NamedTuple
from functools import partial

import jax
import jax.numpy as jnp
from flax import struct

from mcrl.core.types import BlockType, ItemType
from mcrl.core.state import (
    GameState, WorldState, PlayerState,
    create_initial_player_state, create_initial_game_state,
)
from mcrl.core.constants import (
    DEFAULT_WORLD_SIZE,
    MAX_EPISODE_TICKS,
    TICK_DURATION,
    NUM_ACTIONS,
    LOCAL_OBS_SIZE,
)
from mcrl.systems.world_gen import generate_world, find_spawn_position
from mcrl.systems.physics import apply_physics
from mcrl.systems.actions import process_action, Action
from mcrl.utils.observations import get_full_observation, get_local_voxels, encode_inventory, get_player_state_vector
from mcrl.utils.rewards import calculate_milestone_reward, check_episode_done


class StepResult(NamedTuple):
    """Result of environment step."""
    state: GameState
    obs: dict
    reward: jnp.ndarray
    done: jnp.ndarray
    info: dict


@struct.dataclass
class EnvConfig:
    """Environment configuration."""
    world_size: tuple = DEFAULT_WORLD_SIZE
    max_episode_ticks: int = MAX_EPISODE_TICKS
    death_penalty: float = -100.0
    milestone_scale: float = 1.0


class MinecraftEnv:
    """
    JAX-native Minecraft environment.
    
    Features:
    - Pure JAX implementation for JIT compilation
    - Vectorizable via jax.vmap for parallel environments
    - Voxel-based observations (no pixel rendering)
    - 25 discrete actions matching diamond_env
    - Sparse milestone rewards for diamond progression
    
    Usage:
        env = MinecraftEnv()
        state, obs = env.reset(key)
        state, obs, reward, done, info = env.step(state, action)
    
    Vectorized usage:
        vec_reset, vec_step = env.make_vectorized()
        states, obs = vec_reset(keys)  # keys: [num_envs]
        states, obs, rewards, dones, infos = vec_step(states, actions)
    """
    
    def __init__(self, config: Optional[EnvConfig] = None):
        self.config = config or EnvConfig()
        
        # Pre-compile JIT functions
        self._reset_jit = jax.jit(self._reset)
        self._step_jit = jax.jit(self._step)
    
    @property
    def num_actions(self) -> int:
        return NUM_ACTIONS
    
    @property
    def observation_shape(self) -> dict:
        """Shape of observations."""
        return {
            "local_voxels": (LOCAL_OBS_SIZE, LOCAL_OBS_SIZE, LOCAL_OBS_SIZE),
            "facing_blocks": (8,),
            "inventory": (16,),
            "player_state": (14,),
            "tick": (),
        }
    
    def _reset(self, key: jax.random.PRNGKey) -> tuple[GameState, dict]:
        """Initialize new episode (internal, JIT-compiled)."""
        key, k_world, k_spawn, k_yaw = jax.random.split(key, 4)
        
        # Generate world
        W, H, D = self.config.world_size
        world = generate_world(k_world, W, H, D)
        
        # Find spawn position
        spawn_pos = find_spawn_position(k_spawn, world)
        spawn_yaw = jax.random.uniform(k_yaw, (), minval=0.0, maxval=360.0)
        
        # Create player
        player = create_initial_player_state(spawn_pos, spawn_yaw)
        
        # Create game state
        state = create_initial_game_state(world, player)
        
        # Get initial observation
        obs = get_full_observation(state)
        
        return state, obs
    
    def _step(
        self, 
        state: GameState, 
        action: jnp.ndarray,
    ) -> tuple[GameState, dict, jnp.ndarray, jnp.ndarray, dict]:
        """Execute one step (internal, JIT-compiled)."""
        # Store previous state for reward calculation
        prev_flags = state.reward_flags
        
        # Process action
        state = process_action(state, action)
        
        # Apply physics
        state = apply_physics(state, TICK_DURATION)
        
        # Increment tick
        new_world = state.world.replace(tick=state.world.tick + 1)
        state = state.replace(world=new_world)
        
        # Calculate reward
        reward, new_flags = calculate_milestone_reward(state)
        state = state.replace(reward_flags=new_flags)
        
        # Check done
        done, success = check_episode_done(state)
        
        # Apply death penalty
        death_penalty = jnp.where(
            state.player.health <= 0,
            self.config.death_penalty,
            0.0
        )
        reward = reward + death_penalty
        
        # Update done flag
        state = state.replace(done=done)
        
        # Get observation
        obs = get_full_observation(state)
        
        # Info dict
        info = {
            "tick": state.world.tick,
            "success": success,
            "milestones": state.reward_flags,
            "health": state.player.health,
        }
        
        return state, obs, reward, done, info
    
    def reset(self, key: jax.random.PRNGKey) -> tuple[GameState, dict]:
        """Reset environment and return initial state and observation."""
        return self._reset_jit(key)
    
    def step(
        self,
        state: GameState,
        action: jnp.ndarray,
    ) -> tuple[GameState, dict, jnp.ndarray, jnp.ndarray, dict]:
        """
        Execute one environment step.
        
        Args:
            state: Current game state
            action: Action to take (int, 0-24)
        
        Returns:
            state: New game state
            obs: Observation dict
            reward: Reward for this step
            done: Whether episode is finished
            info: Additional information
        """
        return self._step_jit(state, action)
    
    def make_vectorized(self, num_envs: Optional[int] = None):
        """
        Create vectorized reset and step functions.
        
        Returns:
            vec_reset: Function (keys) -> (states, obs)
            vec_step: Function (states, actions) -> (states, obs, rewards, dones, infos)
        
        Usage:
            vec_reset, vec_step = env.make_vectorized()
            keys = jax.random.split(key, num_envs)
            states, obs = vec_reset(keys)
            actions = jnp.zeros(num_envs, dtype=jnp.int32)
            states, obs, rewards, dones, infos = vec_step(states, actions)
        """
        @jax.jit
        def vec_reset(keys: jnp.ndarray):
            return jax.vmap(self._reset)(keys)
        
        @jax.jit
        def vec_step(states: GameState, actions: jnp.ndarray):
            return jax.vmap(self._step)(states, actions)
        
        return vec_reset, vec_step


def make_env(
    world_size: tuple = DEFAULT_WORLD_SIZE,
    max_ticks: int = MAX_EPISODE_TICKS,
) -> MinecraftEnv:
    """Create environment with specified configuration."""
    config = EnvConfig(
        world_size=world_size,
        max_episode_ticks=max_ticks,
    )
    return MinecraftEnv(config)
