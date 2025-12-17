"""
Observation generation for MinecraftRL.

Provides voxel-based observations (no pixel rendering).
Observations are JAX arrays for efficient batching.
"""

import jax
import jax.numpy as jnp

from mcrl.core.types import BlockType, ItemType
from mcrl.core.state import GameState
from mcrl.core.constants import LOCAL_OBS_RADIUS, LOCAL_OBS_SIZE, INVENTORY_SIZE


def get_local_voxels(
    state: GameState,
    radius: int = LOCAL_OBS_RADIUS,
) -> jnp.ndarray:
    """
    Get local voxel grid around player.
    
    Returns a cube of block types centered on player position.
    Shape: [2*radius+1, 2*radius+1, 2*radius+1] uint8
    
    This is the primary observation for RL - much faster than rendering.
    The agent learns to interpret the 3D block structure directly.
    """
    world = state.world
    player_pos = jnp.floor(state.player.pos).astype(jnp.int32)
    
    W, H, D = world.shape
    size = radius * 2 + 1
    
    # Create offset grid
    offsets = jnp.arange(-radius, radius + 1)
    ox, oy, oz = jnp.meshgrid(offsets, offsets, offsets, indexing='ij')
    
    # Sample positions relative to player
    sample_x = player_pos[0] + ox
    sample_y = player_pos[1] + oy  
    sample_z = player_pos[2] + oz
    
    # Bounds check
    valid = (
        (sample_x >= 0) & (sample_x < W) &
        (sample_y >= 0) & (sample_y < H) &
        (sample_z >= 0) & (sample_z < D)
    )
    
    # Clamp for safe indexing
    safe_x = jnp.clip(sample_x, 0, W - 1)
    safe_y = jnp.clip(sample_y, 0, H - 1)
    safe_z = jnp.clip(sample_z, 0, D - 1)
    
    # Sample blocks
    blocks = world.blocks[safe_x, safe_y, safe_z]
    
    # Out of bounds = bedrock (solid wall)
    blocks = jnp.where(valid, blocks, BlockType.BEDROCK)
    
    return blocks


def get_facing_blocks(state: GameState, distance: int = 8) -> jnp.ndarray:
    """
    Get blocks along the ray the player is looking at.
    Useful for understanding what the player is targeting.
    
    Returns: [distance] array of block types
    """
    world = state.world
    pos = state.player.pos + jnp.array([0.0, 1.62, 0.0])  # Eye level
    rot = state.player.rot
    
    # Direction from pitch/yaw
    pitch_rad = jnp.deg2rad(rot[0])
    yaw_rad = jnp.deg2rad(rot[1])
    
    direction = jnp.array([
        jnp.cos(pitch_rad) * jnp.sin(yaw_rad),
        -jnp.sin(pitch_rad),
        jnp.cos(pitch_rad) * jnp.cos(yaw_rad),
    ])
    
    # Sample along ray
    W, H, D = world.shape
    blocks = []
    
    def sample_at_distance(d):
        sample_pos = pos + direction * d
        block_pos = jnp.floor(sample_pos).astype(jnp.int32)
        
        in_bounds = (
            (block_pos[0] >= 0) & (block_pos[0] < W) &
            (block_pos[1] >= 0) & (block_pos[1] < H) &
            (block_pos[2] >= 0) & (block_pos[2] < D)
        )
        
        block = jnp.where(
            in_bounds,
            world.blocks[
                jnp.clip(block_pos[0], 0, W-1),
                jnp.clip(block_pos[1], 0, H-1),
                jnp.clip(block_pos[2], 0, D-1),
            ],
            BlockType.AIR
        )
        return block
    
    distances = jnp.arange(1, distance + 1).astype(jnp.float32)
    facing = jax.vmap(sample_at_distance)(distances)
    
    return facing


def encode_inventory(inventory: jnp.ndarray) -> jnp.ndarray:
    """
    Encode inventory as a flat feature vector.
    
    Returns counts for key items relevant to diamond progression.
    Shape: [num_tracked_items]
    """
    # Items to track for diamond task
    tracked_items = [
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
    ]
    
    def get_count(item_type):
        matches = inventory[:, 0] == item_type
        counts = jnp.where(matches, inventory[:, 1], 0)
        return counts.sum()
    
    counts = jnp.array([get_count(item) for item in tracked_items])
    return counts


def get_log_compass(state: GameState, search_radius: int = 20) -> jnp.ndarray:
    """
    Get direction vector pointing toward nearest log.
    
    Returns: [4] array:
        - dx, dy, dz: normalized direction to nearest log (or 0 if none)
        - distance: normalized distance (0-1, where 1 = far away)
    
    This helps the agent navigate toward trees even when they're outside
    the local voxel observation range.
    """
    world = state.world
    player_pos = state.player.pos
    W, H, D = world.blocks.shape
    
    # Search for logs in a radius around player
    pos_int = jnp.floor(player_pos).astype(jnp.int32)
    
    # Create search grid (pre-allocated in practice, inline here for clarity)
    offsets = jnp.arange(-search_radius, search_radius + 1)
    ox, oy, oz = jnp.meshgrid(offsets, jnp.arange(-2, 15), offsets, indexing='ij')
    ox, oy, oz = ox.flatten(), oy.flatten(), oz.flatten()
    
    # Sample positions
    check_x = pos_int[0] + ox
    check_y = pos_int[1] + oy
    check_z = pos_int[2] + oz
    
    # Bounds check
    valid = (
        (check_x >= 0) & (check_x < W) &
        (check_y >= 0) & (check_y < H) &
        (check_z >= 0) & (check_z < D)
    )
    
    # Safe indexing
    safe_x = jnp.clip(check_x, 0, W - 1)
    safe_y = jnp.clip(check_y, 0, H - 1)
    safe_z = jnp.clip(check_z, 0, D - 1)
    
    # Get block types
    blocks = world.blocks[safe_x, safe_y, safe_z]
    blocks = jnp.where(valid, blocks, 0)
    
    # Check if log
    is_log = (
        (blocks == BlockType.OAK_LOG) |
        (blocks == BlockType.BIRCH_LOG) |
        (blocks == BlockType.SPRUCE_LOG)
    )
    
    # Compute distances
    dx = ox.astype(jnp.float32)
    dy = oy.astype(jnp.float32)
    dz = oz.astype(jnp.float32)
    distances = jnp.sqrt(dx**2 + dy**2 + dz**2)
    
    # Mask non-logs with large distance
    distances = jnp.where(is_log, distances, 1000.0)
    
    # Find nearest
    nearest_idx = jnp.argmin(distances)
    nearest_dist = distances[nearest_idx]
    
    # Direction to nearest log
    dir_x = dx[nearest_idx]
    dir_y = dy[nearest_idx]
    dir_z = dz[nearest_idx]
    
    # Normalize direction
    dir_len = jnp.maximum(nearest_dist, 0.01)
    dir_x = dir_x / dir_len
    dir_y = dir_y / dir_len
    dir_z = dir_z / dir_len
    
    # If no log found, zero direction
    found_log = nearest_dist < 999.0
    dir_x = jnp.where(found_log, dir_x, 0.0)
    dir_y = jnp.where(found_log, dir_y, 0.0)
    dir_z = jnp.where(found_log, dir_z, 0.0)
    
    # Normalized distance (0 = at log, 1 = far away)
    norm_dist = jnp.clip(nearest_dist / 30.0, 0.0, 1.0)
    
    return jnp.array([dir_x, dir_y, dir_z, norm_dist])


def get_player_state_vector(state: GameState) -> jnp.ndarray:
    """
    Get player state as a feature vector.
    
    Includes position, rotation, velocity, health, hunger, etc.
    Shape: [num_features]
    """
    player = state.player
    
    features = jnp.concatenate([
        player.pos,                          # 3: x, y, z
        player.vel,                          # 3: vx, vy, vz
        player.rot / 180.0,                  # 2: normalized pitch, yaw
        jnp.array([player.health / 20.0]),   # 1: normalized health
        jnp.array([player.hunger / 20.0]),   # 1: normalized hunger
        jnp.array([player.on_ground * 1.0]), # 1: on ground flag
        jnp.array([player.is_sprinting * 1.0]),  # 1: sprinting flag
        jnp.array([player.is_sneaking * 1.0]),   # 1: sneaking flag
        jnp.array([player.equipped_slot / 8.0]), # 1: equipped slot
    ])
    
    return features


def get_full_observation(state: GameState) -> dict:
    """
    Get complete observation dictionary.
    
    Returns dict with:
        - local_voxels: [17, 17, 17] block types around player
        - facing_blocks: [8] blocks along view ray
        - inventory: [16] item counts for key items
        - player_state: [14] player features
        - log_compass: [4] direction & distance to nearest log
        - tick: current game tick
    """
    return {
        "local_voxels": get_local_voxels(state),
        "facing_blocks": get_facing_blocks(state),
        "inventory": encode_inventory(state.player.inventory),
        "player_state": get_player_state_vector(state),
        "log_compass": get_log_compass(state),
        "tick": state.world.tick,
    }


def flatten_observation(obs: dict) -> jnp.ndarray:
    """Flatten observation dict to single vector for simple policies."""
    return jnp.concatenate([
        obs["local_voxels"].flatten().astype(jnp.float32) / 255.0,
        obs["facing_blocks"].astype(jnp.float32) / 255.0,
        obs["inventory"].astype(jnp.float32) / 64.0,  # Normalize by max stack
        obs["player_state"],
        jnp.array([obs["tick"] / 36000.0]),  # Normalize by max ticks
    ])
