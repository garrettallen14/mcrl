"""
Game state definitions for MinecraftRL.

All state is immutable and JAX-compatible using flax.struct.
This enables efficient vectorization with jax.vmap.
"""

from __future__ import annotations
import jax.numpy as jnp
from flax import struct

from mcrl.core.constants import INVENTORY_SIZE, MAX_HEALTH, MAX_HUNGER


@struct.dataclass
class WorldState:
    """
    Immutable world state.
    
    Attributes:
        blocks: 3D array of block types [W, H, D], uint8
        tick: Current game tick
        seed: World seed for procedural generation
    """
    blocks: jnp.ndarray  # [W, H, D] uint8
    tick: jnp.int32
    seed: jnp.uint32
    
    @property
    def shape(self) -> tuple[int, int, int]:
        return self.blocks.shape


@struct.dataclass 
class PlayerState:
    """
    Immutable player state.
    
    Inventory format: [INVENTORY_SIZE, 2] where each row is (item_type, count)
    """
    # Position and movement
    pos: jnp.ndarray          # [3] float32 - x, y, z
    vel: jnp.ndarray          # [3] float32 - vx, vy, vz
    rot: jnp.ndarray          # [2] float32 - pitch, yaw (degrees)
    
    # Inventory: each slot is (item_id, count)
    inventory: jnp.ndarray    # [INVENTORY_SIZE, 2] uint16
    equipped_slot: jnp.int32  # 0-8 hotbar selection
    
    # Status
    health: jnp.float32
    hunger: jnp.float32
    on_ground: jnp.bool_
    
    # Mining state (for progressive breaking)
    mining_block: jnp.ndarray   # [3] int32 - block being mined (-1 if none)
    mining_progress: jnp.int32  # Ticks spent mining current block
    
    # Sprint/sneak state
    is_sprinting: jnp.bool_
    is_sneaking: jnp.bool_


@struct.dataclass
class GameState:
    """
    Complete game state for one environment instance.
    
    Future extensions:
        - entities: Array of mob/item entity states
        - weather: Weather state
        - difficulty: Game difficulty setting
    """
    world: WorldState
    player: PlayerState
    
    # Episode tracking
    done: jnp.bool_
    reward_flags: jnp.uint32  # Bitmask of collected milestones
    
    # Optional: for multi-agent (future)
    # players: PlayerState  # Would be batched [num_players, ...]


def create_initial_player_state(
    spawn_pos: jnp.ndarray,
    spawn_yaw: float = 0.0,
) -> PlayerState:
    """Create a fresh player state at spawn position."""
    return PlayerState(
        pos=spawn_pos.astype(jnp.float32),
        vel=jnp.zeros(3, dtype=jnp.float32),
        rot=jnp.array([0.0, spawn_yaw], dtype=jnp.float32),
        inventory=jnp.zeros((INVENTORY_SIZE, 2), dtype=jnp.uint16),
        equipped_slot=jnp.int32(0),
        health=jnp.float32(MAX_HEALTH),
        hunger=jnp.float32(MAX_HUNGER),
        on_ground=jnp.bool_(True),
        mining_block=jnp.array([-1, -1, -1], dtype=jnp.int32),
        mining_progress=jnp.int32(0),
        is_sprinting=jnp.bool_(False),
        is_sneaking=jnp.bool_(False),
    )


def create_initial_game_state(
    world: WorldState,
    player: PlayerState,
) -> GameState:
    """Create initial game state."""
    return GameState(
        world=world,
        player=player,
        done=jnp.bool_(False),
        reward_flags=jnp.uint32(0),
    )
