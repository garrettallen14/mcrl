"""
Optimized observation extraction for MCRL.

Key optimizations:
1. Use jax.lax.dynamic_slice instead of fancy indexing
2. Pre-compute tracked items array
3. Vectorized operations throughout
"""

import jax
import jax.numpy as jnp
from functools import partial

from mcrl.core.types import BlockType, ItemType
from mcrl.core.state import GameState
from mcrl.core.constants import LOCAL_OBS_RADIUS, LOCAL_OBS_SIZE


# Pre-computed tracked items array (avoid list comprehension)
TRACKED_ITEMS = jnp.array([
    ItemType.OAK_LOG,
    ItemType.BIRCH_LOG,
    ItemType.SPRUCE_LOG,
    ItemType.OAK_PLANKS,
    ItemType.STICK,
    ItemType.COBBLESTONE,
    ItemType.COAL,
    ItemType.IRON_ORE,
    ItemType.IRON_INGOT,
    ItemType.DIAMOND,
    ItemType.CRAFTING_TABLE,
    ItemType.FURNACE,
    ItemType.WOODEN_PICKAXE,
    ItemType.STONE_PICKAXE,
    ItemType.IRON_PICKAXE,
    ItemType.DIAMOND_PICKAXE,
], dtype=jnp.int32)


def pad_world_blocks(blocks: jnp.ndarray, padding: int = LOCAL_OBS_RADIUS) -> jnp.ndarray:
    """
    Pad world blocks for fast observation extraction.
    
    Call this once at episode reset, store padded blocks in state.
    """
    return jnp.pad(
        blocks,
        ((padding, padding), (padding, padding), (padding, padding)),
        mode='constant',
        constant_values=BlockType.BEDROCK
    )


def get_local_voxels_fast(
    padded_blocks: jnp.ndarray,
    player_pos: jnp.ndarray,
    radius: int = LOCAL_OBS_RADIUS,
) -> jnp.ndarray:
    """
    Fast local voxel extraction using dynamic_slice.
    
    Requires pre-padded world blocks.
    
    Args:
        padded_blocks: World blocks padded by `radius` on all sides
        player_pos: Player position (will be floored to int)
        radius: Observation radius (default 8 for 17³ obs)
    
    Returns:
        voxels: (2*radius+1, 2*radius+1, 2*radius+1) block types
    """
    # Player position in padded coordinates (padding offsets to non-negative)
    pos = jnp.floor(player_pos).astype(jnp.int32)
    
    # Start position for slice (no bounds check needed due to padding)
    size = 2 * radius + 1
    
    # Single contiguous memory read - much faster than fancy indexing
    voxels = jax.lax.dynamic_slice(
        padded_blocks,
        (pos[0], pos[1], pos[2]),  # Start indices
        (size, size, size)          # Slice sizes
    )
    
    return voxels


def get_facing_blocks_fast(
    blocks: jnp.ndarray,
    player_pos: jnp.ndarray,
    player_rot: jnp.ndarray,
    num_samples: int = 8,
    max_dist: float = 4.0,
) -> jnp.ndarray:
    """
    Fast facing block extraction - vectorized ray sampling.
    
    Args:
        blocks: World blocks (unpadded)
        player_pos: Player position
        player_rot: Player rotation (pitch, yaw) in degrees
        num_samples: Number of points along ray
        max_dist: Maximum ray distance
    
    Returns:
        facing: (num_samples,) block types along view ray
    """
    W, H, D = blocks.shape
    
    # Eye position
    eye = player_pos + jnp.array([0.0, 1.62, 0.0])
    
    # Direction from rotation
    pitch_rad = jnp.deg2rad(player_rot[0])
    yaw_rad = jnp.deg2rad(player_rot[1])
    
    direction = jnp.array([
        jnp.cos(pitch_rad) * jnp.sin(yaw_rad),
        -jnp.sin(pitch_rad),
        jnp.cos(pitch_rad) * jnp.cos(yaw_rad),
    ])
    
    # Sample distances (vectorized)
    distances = jnp.linspace(0.5, max_dist, num_samples)
    
    # All sample positions at once: (num_samples, 3)
    positions = eye[None, :] + direction[None, :] * distances[:, None]
    block_positions = jnp.floor(positions).astype(jnp.int32)
    
    # Bounds check (vectorized)
    in_bounds = (
        (block_positions[:, 0] >= 0) & (block_positions[:, 0] < W) &
        (block_positions[:, 1] >= 0) & (block_positions[:, 1] < H) &
        (block_positions[:, 2] >= 0) & (block_positions[:, 2] < D)
    )
    
    # Clamp for safe indexing
    safe_pos = jnp.clip(
        block_positions,
        jnp.array([0, 0, 0]),
        jnp.array([W-1, H-1, D-1])
    )
    
    # Single gather operation
    facing = blocks[safe_pos[:, 0], safe_pos[:, 1], safe_pos[:, 2]]
    
    # Out of bounds = AIR
    facing = jnp.where(in_bounds, facing, BlockType.AIR)
    
    return facing


def encode_inventory_fast(inventory: jnp.ndarray) -> jnp.ndarray:
    """
    Fast inventory encoding - fully vectorized.
    
    Args:
        inventory: (36, 2) array of (item_id, count) pairs
    
    Returns:
        counts: (16,) counts of tracked items
    """
    # Vectorized comparison: (36, 1) == (1, 16) -> (36, 16) boolean
    matches = inventory[:, 0:1] == TRACKED_ITEMS[None, :]
    
    # Sum counts where matches: (36, 1) * (36, 16) -> sum -> (16,)
    counts = jnp.sum(inventory[:, 1:2] * matches, axis=0)
    
    return counts.astype(jnp.float32)


def get_player_state_fast(player) -> jnp.ndarray:
    """
    Fast player state encoding - single array creation.
    
    Returns:
        state: (14,) player state features
    """
    return jnp.array([
        player.pos[0] / 64.0,    # Normalized x
        player.pos[1] / 64.0,    # Normalized y (depth)
        player.pos[2] / 64.0,    # Normalized z
        player.vel[0],           # Velocity x
        player.vel[1],           # Velocity y
        player.vel[2],           # Velocity z
        player.rot[0] / 90.0,    # Normalized pitch
        player.rot[1] / 180.0,   # Normalized yaw
        player.health / 20.0,    # Normalized health
        player.hunger / 20.0,    # Normalized hunger
        player.on_ground * 1.0,  # On ground flag
        player.is_sprinting * 1.0,
        player.is_sneaking * 1.0,
        player.mining_progress,  # Current mining progress
    ], dtype=jnp.float32)


def get_full_observation_fast(
    padded_blocks: jnp.ndarray,
    blocks: jnp.ndarray,
    player,
    tick: int,
) -> dict:
    """
    Fast full observation extraction.
    
    Args:
        padded_blocks: Pre-padded world blocks
        blocks: Original world blocks (for facing ray)
        player: PlayerState
        tick: Current game tick
    
    Returns:
        obs: Dict with local_voxels, facing_blocks, inventory, player_state
    """
    return {
        "local_voxels": get_local_voxels_fast(padded_blocks, player.pos),
        "facing_blocks": get_facing_blocks_fast(blocks, player.pos, player.rot),
        "inventory": encode_inventory_fast(player.inventory),
        "player_state": get_player_state_fast(player),
        "tick": tick,
    }


# ============================================================================
# Ultra-fast fused observation (minimizes allocations)
# ============================================================================

# Pre-computed normalization constants
_POS_NORM = jnp.float32(1.0 / 64.0)
_PITCH_NORM = jnp.float32(1.0 / 90.0)
_YAW_NORM = jnp.float32(1.0 / 180.0)
_HEALTH_NORM = jnp.float32(1.0 / 20.0)

# Pre-computed ray sample distances
_RAY_DISTANCES = jnp.linspace(0.5, 4.0, 8, dtype=jnp.float32)


def get_observation_fused(state) -> dict:
    """
    Ultra-fast fused observation extraction.
    
    Optimizations:
    - Pre-computed normalization constants
    - Minimized intermediate allocations
    - Pre-computed ray distances
    - Direct struct field access
    
    Args:
        state: GameState
    
    Returns:
        obs: Dict with all observation components
    """
    player = state.player
    world = state.world
    
    # === Local voxels (17³) - single dynamic_slice ===
    pos_int = jnp.floor(player.pos).astype(jnp.int32)
    local_voxels = jax.lax.dynamic_slice(
        world.padded_blocks,
        (pos_int[0], pos_int[1], pos_int[2]),
        (LOCAL_OBS_SIZE, LOCAL_OBS_SIZE, LOCAL_OBS_SIZE)
    )
    
    # === Facing blocks - vectorized ray ===
    W, H, D = world.blocks.shape
    eye = player.pos + jnp.array([0.0, 1.62, 0.0])
    
    pitch_rad = player.rot[0] * (jnp.pi / 180.0)
    yaw_rad = player.rot[1] * (jnp.pi / 180.0)
    
    cos_pitch = jnp.cos(pitch_rad)
    direction = jnp.array([
        cos_pitch * jnp.sin(yaw_rad),
        -jnp.sin(pitch_rad),
        cos_pitch * jnp.cos(yaw_rad),
    ])
    
    # All ray positions at once
    positions = eye[None, :] + direction[None, :] * _RAY_DISTANCES[:, None]
    block_pos = jnp.floor(positions).astype(jnp.int32)
    
    # Bounds check and gather
    in_bounds = (
        (block_pos[:, 0] >= 0) & (block_pos[:, 0] < W) &
        (block_pos[:, 1] >= 0) & (block_pos[:, 1] < H) &
        (block_pos[:, 2] >= 0) & (block_pos[:, 2] < D)
    )
    safe_pos = jnp.clip(block_pos, 0, jnp.array([W-1, H-1, D-1]))
    facing = world.blocks[safe_pos[:, 0], safe_pos[:, 1], safe_pos[:, 2]]
    facing = jnp.where(in_bounds, facing, BlockType.AIR)
    
    # === Inventory - vectorized matching ===
    matches = player.inventory[:, 0:1] == TRACKED_ITEMS[None, :]
    inventory = jnp.sum(player.inventory[:, 1:2] * matches, axis=0).astype(jnp.float32)
    
    # === Player state - fused array construction ===
    player_state = jnp.array([
        player.pos[0] * _POS_NORM,
        player.pos[1] * _POS_NORM,
        player.pos[2] * _POS_NORM,
        player.vel[0],
        player.vel[1],
        player.vel[2],
        player.rot[0] * _PITCH_NORM,
        player.rot[1] * _YAW_NORM,
        player.health * _HEALTH_NORM,
        player.hunger * _HEALTH_NORM,
        player.on_ground.astype(jnp.float32),
        player.is_sprinting.astype(jnp.float32),
        player.is_sneaking.astype(jnp.float32),
        player.mining_progress.astype(jnp.float32),
    ], dtype=jnp.float32)
    
    return {
        "local_voxels": local_voxels,
        "facing_blocks": facing,
        "inventory": inventory,
        "player_state": player_state,
        "tick": world.tick,
    }


# ============================================================================
# Optimized raycast for mining
# ============================================================================

def raycast_fast(
    blocks: jnp.ndarray,
    eye: jnp.ndarray,
    direction: jnp.ndarray,
    max_dist: float = 4.5,
    num_samples: int = 20,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Fast vectorized raycast - no scan loop.
    
    Returns:
        hit_pos: (3,) int32 block position
        hit: bool whether a block was hit
        block_type: uint8 type of hit block
    """
    W, H, D = blocks.shape
    
    # Sample all points at once
    t = jnp.linspace(0.1, max_dist, num_samples)
    positions = eye[None, :] + direction[None, :] * t[:, None]
    block_positions = jnp.floor(positions).astype(jnp.int32)
    
    # Vectorized bounds check
    in_bounds = (
        (block_positions[:, 0] >= 0) & (block_positions[:, 0] < W) &
        (block_positions[:, 1] >= 0) & (block_positions[:, 1] < H) &
        (block_positions[:, 2] >= 0) & (block_positions[:, 2] < D)
    )
    
    # Safe indexing
    safe_pos = jnp.clip(
        block_positions,
        jnp.array([0, 0, 0]),
        jnp.array([W-1, H-1, D-1])
    )
    
    # Gather all block types at once
    block_types = blocks[safe_pos[:, 0], safe_pos[:, 1], safe_pos[:, 2]]
    
    # Solid check (not air, water, leaves)
    is_solid = (
        (block_types != BlockType.AIR) &
        (block_types != BlockType.WATER) &
        (block_types != BlockType.OAK_LEAVES) &
        in_bounds
    )
    
    # Find first hit (argmax on boolean gives first True)
    # If no hit, argmax returns 0, but is_solid.any() will be False
    first_idx = jnp.argmax(is_solid)
    hit = is_solid.any()
    
    hit_pos = block_positions[first_idx]
    block_type = block_types[first_idx]
    
    return hit_pos, hit, block_type
